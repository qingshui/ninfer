#include "ops/linear_attention/causal_conv/varlen.h"
#include "ops/linear_attention/kernels/causal_conv.cuh"
#include "core/device.h"
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kVarlenBlock = 256;

__device__ __forceinline__ int lower_bound_rows(const std::int32_t* values, int count, int key) {
    int low = 0, high = count;
    while (low < high) {
        const int mid = (low + high) >> 1;
        if (values[mid] < key)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

// Token-parallel outputs. A block covers one column and 2*kVarlenBlock channels; history before
// the row start comes from the row's source window. The arithmetic is that of
// causal_conv1d_prefill_pairs_kernel (and, bit for bit, of the small-T route).
template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
__global__ void __launch_bounds__(kVarlenBlock)
    varlen_output_kernel(CausalConvVarlenOperands ops) {
    __shared__ std::int32_t row_of_column;
    const std::int64_t pairs    = ops.channels / 2;
    const std::int64_t c_blocks = common::div_up(pairs, static_cast<std::int64_t>(kVarlenBlock));
    const std::int64_t block    = blockIdx.x;
    const std::int32_t column   = static_cast<std::int32_t>(block / c_blocks);
    const std::int64_t pair     = (block - static_cast<std::int64_t>(column) * c_blocks) * kVarlenBlock +
                              threadIdx.x;
    if (threadIdx.x == 0) {
        int low = 0, high = ops.rows;
        while (high - low > 1) {
            const int mid = (low + high) >> 1;
            if (ops.token_offsets[mid] <= column)
                low = mid;
            else
                high = mid;
        }
        row_of_column = low;
    }
    __syncthreads();
    if (pair >= pairs) return;

    const std::int32_t row   = row_of_column;
    const std::int32_t local = column - ops.token_offsets[row];
    const auto* x2           = reinterpret_cast<const __nv_bfloat162*>(ops.x);
    const auto* weight2      = reinterpret_cast<const __nv_bfloat162*>(ops.weight);
    const auto* state2       = reinterpret_cast<const __nv_bfloat162*>(ops.source_base +
                                                                 ops.source_offsets[row]);
    const std::int64_t index = static_cast<std::int64_t>(column) * pairs + pair;
    const __nv_bfloat162 x0  = local >= 3 ? x2[index - 3 * pairs] : state2[local * pairs + pair];
    const __nv_bfloat162 x1  = local >= 2 ? x2[index - 2 * pairs] : state2[(local + 1) * pairs + pair];
    const __nv_bfloat162 x2v = local >= 1 ? x2[index - pairs] : state2[(local + 2) * pairs + pair];
    const __nv_bfloat162 x3  = x2[index];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    causal_conv1d_acc_pair(weight2[pair], x0, acc0, acc1);
    causal_conv1d_acc_pair(weight2[pairs + pair], x1, acc0, acc1);
    causal_conv1d_acc_pair(weight2[2 * pairs + pair], x2v, acc0, acc1);
    causal_conv1d_acc_pair(weight2[3 * pairs + pair], x3, acc0, acc1);
    const CausalConvSplitOutput3<__nv_bfloat162, Rows0 / 2, Rows1 / 2, Rows2 / 2> out_map{
        reinterpret_cast<__nv_bfloat162*>(ops.out0), reinterpret_cast<__nv_bfloat162*>(ops.out1),
        reinterpret_cast<__nv_bfloat162*>(ops.out2)};
    *out_map.column(pair).at(column) =
        __floats2bfloat162_rn(common::silu(acc0), common::silu(acc1));
}

// One thread owns (row, channel). It captures the row's source window before writing any state, so
// an exact source/destination alias is safe, then publishes the checkpoint windows and the final
// window. The window after `end` consumed columns is positions end..end+2 of concat(source, x).
__global__ void __launch_bounds__(kVarlenBlock) varlen_state_kernel(CausalConvVarlenOperands ops) {
    __shared__ std::int32_t first_boundary, end_boundary;
    const std::int32_t row = blockIdx.y;
    if (threadIdx.x == 0) {
        first_boundary = lower_bound_rows(ops.boundary_rows, ops.boundaries, row);
        end_boundary   = lower_bound_rows(ops.boundary_rows, ops.boundaries, row + 1);
    }
    __syncthreads();
    const std::int64_t channels = ops.channels;
    const std::int64_t channel  = static_cast<std::int64_t>(blockIdx.x) * kVarlenBlock + threadIdx.x;
    if (channel >= channels) return;

    const std::int32_t begin  = ops.token_offsets[row];
    const std::int32_t length = ops.token_offsets[row + 1] - begin;
    const auto* source =
        reinterpret_cast<const __nv_bfloat16*>(ops.source_base + ops.source_offsets[row]);
    const __nv_bfloat16 old[3] = {source[channel], source[channels + channel],
                                  source[2 * channels + channel]};
    const auto window = [&](std::int32_t consumed, std::int32_t slot) {
        const std::int32_t position = consumed - 3 + slot;
        return position < 0 ? old[position + 3]
                            : ops.x[static_cast<std::int64_t>(begin + position) * channels + channel];
    };
    for (std::int32_t i = first_boundary; i < end_boundary; ++i) {
        const std::int64_t offset = ops.boundary_offsets[i];
        if (offset < 0) continue;
        auto* record = reinterpret_cast<__nv_bfloat16*>(ops.checkpoint_base + offset);
        const std::int32_t column = ops.boundary_columns[i];
        for (std::int32_t slot = 0; slot < 3; ++slot)
            record[slot * channels + channel] = window(column, slot);
    }
    auto* destination =
        reinterpret_cast<__nv_bfloat16*>(ops.destination_base + ops.destination_offsets[row]);
    for (std::int32_t slot = 0; slot < 3; ++slot)
        destination[slot * channels + channel] = window(length, slot);
}

template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
void launch_geometry(const CausalConvVarlenOperands& ops, cudaStream_t stream) {
    static_assert((Rows0 % 2) == 0 && (Rows1 % 2) == 0 && (Rows2 % 2) == 0,
                  "a pair must not straddle two destinations");
    if (Rows0 + Rows1 + Rows2 != ops.channels)
        throw std::invalid_argument("causal_conv1d: split geometry does not cover the channels");
    const std::int64_t c_blocks = common::div_up(static_cast<std::int64_t>(ops.channels / 2),
                                                 static_cast<std::int64_t>(kVarlenBlock));
    const std::int64_t grid     = c_blocks * ops.tokens;
    if (grid > std::numeric_limits<int>::max())
        throw std::overflow_error("causal_conv1d: varlen output grid exceeds CUDA limit");
    varlen_output_kernel<Rows0, Rows1, Rows2>
        <<<static_cast<unsigned>(grid), kVarlenBlock, 0, stream>>>(ops);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_causal_conv_split_varlen(const CausalConvVarlenOperands& ops,
                                     CausalConvSplitGeometry geometry, cudaStream_t stream) {
    switch (geometry) {
    case CausalConvSplitGeometry::Rows1024x1024x3072:
        launch_geometry<1024, 1024, 3072>(ops, stream);
        break;
    case CausalConvSplitGeometry::Rows2048x2048x4096:
        launch_geometry<2048, 2048, 4096>(ops, stream);
        break;
    case CausalConvSplitGeometry::Rows2048x2048x6144:
        launch_geometry<2048, 2048, 6144>(ops, stream);
        break;
    case CausalConvSplitGeometry::Rows2048x2048x8192:
        launch_geometry<2048, 2048, 8192>(ops, stream);
        break;
    default:
        throw std::logic_error("causal_conv1d: unhandled split geometry");
    }
    const dim3 grid(static_cast<unsigned>(common::div_up(ops.channels, kVarlenBlock)),
                    static_cast<unsigned>(ops.rows));
    varlen_state_kernel<<<grid, kVarlenBlock, 0, stream>>>(ops);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
