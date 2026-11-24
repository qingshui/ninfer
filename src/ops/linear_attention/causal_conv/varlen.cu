#include "ops/linear_attention/causal_conv/varlen.h"
#include "ops/linear_attention/kernels/causal_conv.cuh"
#include "core/device.h"
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kVarlenBlock     = 256;
constexpr int kVarlenTokens    = 32; // packed columns one block walks
constexpr int kVarlenBatch     = 8;  // independent column loads issued before any is consumed
constexpr int kVarlenMinBlocks = 5;  // caps registers so enough warps hide the load latency

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

// Column-parallel outputs. A thread owns one channel pair and walks kVarlenTokens consecutive
// columns, so the row lookup, the weights and the three-column history are paid once per walk and
// the loads of several columns are in flight together. History before the row start comes from the
// row's source window. The arithmetic is that of causal_conv1d_prefill_pairs_kernel (and, bit for
// bit, of the small-T route). A block never straddles two destinations, so the output stride of
// each walk is a compile-time constant.
template <std::int64_t Pairs, std::int32_t Stride>
__device__ __forceinline__ void varlen_walk(const CausalConvVarlenOperands& ops, std::int64_t pair,
                                            __nv_bfloat162* out) {
    const std::int32_t first = static_cast<std::int32_t>(blockIdx.y) * kVarlenTokens;
    const std::int32_t last  = min(first + kVarlenTokens, ops.tokens);

    const auto* x2      = reinterpret_cast<const __nv_bfloat162*>(ops.x) + pair;
    const auto* weight2 = reinterpret_cast<const __nv_bfloat162*>(ops.weight) + pair;
    const float2 tap0   = __bfloat1622float2(weight2[0]);
    const float2 tap1   = __bfloat1622float2(weight2[Pairs]);
    const float2 tap2   = __bfloat1622float2(weight2[2 * Pairs]);
    const float2 tap3   = __bfloat1622float2(weight2[3 * Pairs]);
    const auto convolve = [&](const float2& h0, const float2& h1, const float2& h2,
                              const float2& x, __nv_bfloat162* destination) {
        float acc0 = fmaf(tap0.x, h0.x, 0.0f);
        float acc1 = fmaf(tap0.y, h0.y, 0.0f);
        acc0       = fmaf(tap1.x, h1.x, acc0);
        acc1       = fmaf(tap1.y, h1.y, acc1);
        acc0       = fmaf(tap2.x, h2.x, acc0);
        acc1       = fmaf(tap2.y, h2.y, acc1);
        acc0       = fmaf(tap3.x, x.x, acc0);
        acc1       = fmaf(tap3.y, x.y, acc1);
        *destination = __floats2bfloat162_rn(common::silu_approx(acc0), common::silu_approx(acc1));
    };

    // The row owning `first` is the last one that starts at or before it.
    int low = 0, high = ops.rows;
    while (high - low > 1) {
        const int mid = (low + high) >> 1;
        if (ops.token_offsets[mid] <= first)
            low = mid;
        else
            high = mid;
    }
    std::int32_t row    = low;
    std::int32_t column = first;
    while (column < last) {
        const std::int32_t row_begin = ops.token_offsets[row];
        const std::int32_t row_end = row + 1 < ops.rows ? ops.token_offsets[row + 1] : INT_MAX;
        if (column >= row_end) {
            ++row;
            continue;
        }
        const std::int32_t stop = min(last, row_end);
        const auto* state2 =
            reinterpret_cast<const __nv_bfloat162*>(ops.source_base + ops.source_offsets[row]) +
            pair;
        const auto history = [&](std::int32_t position) {
            return __bfloat1622float2(
                position >= 0 ? x2[static_cast<std::int64_t>(row_begin + position) * Pairs]
                              : state2[static_cast<std::int64_t>(position + 3) * Pairs]);
        };
        const std::int32_t local = column - row_begin;
        float2 h0                = history(local - 3);
        float2 h1                = history(local - 2);
        float2 h2                = history(local - 1);
        for (; stop - column >= kVarlenBatch; column += kVarlenBatch) {
            const auto* source = x2 + static_cast<std::int64_t>(column) * Pairs;
            auto* destination  = out + static_cast<std::int64_t>(column) * Stride;
            __nv_bfloat162 current[kVarlenBatch];
#pragma unroll
            for (int i = 0; i < kVarlenBatch; ++i) {
                current[i] = source[i * Pairs];
            }
#pragma unroll
            for (int i = 0; i < kVarlenBatch; ++i) {
                const float2 x = __bfloat1622float2(current[i]);
                convolve(h0, h1, h2, x, destination + i * Stride);
                h0 = h1;
                h1 = h2;
                h2 = x;
            }
        }
        for (; column < stop; ++column) {
            const float2 x = __bfloat1622float2(x2[static_cast<std::int64_t>(column) * Pairs]);
            convolve(h0, h1, h2, x, out + static_cast<std::int64_t>(column) * Stride);
            h0 = h1;
            h1 = h2;
            h2 = x;
        }
        ++row;
    }
}

template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
__global__ void __launch_bounds__(kVarlenBlock, kVarlenMinBlocks)
    varlen_output_kernel(CausalConvVarlenOperands ops) {
    constexpr std::int32_t Stride0 = Rows0 / 2, Stride1 = Rows1 / 2, Stride2 = Rows2 / 2;
    constexpr std::int64_t Pairs   = Stride0 + Stride1 + Stride2;
    static_assert(Stride0 % kVarlenBlock == 0 && Stride1 % kVarlenBlock == 0 &&
                      Stride2 % kVarlenBlock == 0,
                  "a block must not straddle two destinations");
    const std::int64_t pair = static_cast<std::int64_t>(blockIdx.x) * kVarlenBlock + threadIdx.x;
    if (pair >= Pairs) return;
    if (pair < Stride0)
        varlen_walk<Pairs, Stride0>(ops, pair, reinterpret_cast<__nv_bfloat162*>(ops.out0) + pair);
    else if (pair < Stride0 + Stride1)
        varlen_walk<Pairs, Stride1>(ops, pair,
                                    reinterpret_cast<__nv_bfloat162*>(ops.out1) + (pair - Stride0));
    else
        varlen_walk<Pairs, Stride2>(
            ops, pair,
            reinterpret_cast<__nv_bfloat162*>(ops.out2) + (pair - Stride0 - Stride1));
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
        for (std::int32_t slot = 0; slot < 3; ++slot) {
            record[slot * channels + channel] = window(column, slot);
        }
    }
    auto* destination =
        reinterpret_cast<__nv_bfloat16*>(ops.destination_base + ops.destination_offsets[row]);
    for (std::int32_t slot = 0; slot < 3; ++slot) {
        destination[slot * channels + channel] = window(length, slot);
    }
}

template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
void launch_geometry(const CausalConvVarlenOperands& ops, cudaStream_t stream) {
    static_assert((Rows0 % 2) == 0 && (Rows1 % 2) == 0 && (Rows2 % 2) == 0,
                  "a pair must not straddle two destinations");
    if (Rows0 + Rows1 + Rows2 != ops.channels)
        throw std::invalid_argument("causal_conv1d: split geometry does not cover the channels");
    const std::int64_t c_blocks = common::div_up(static_cast<std::int64_t>(ops.channels / 2),
                                                 static_cast<std::int64_t>(kVarlenBlock));
    const std::int64_t walks    = common::div_up(static_cast<std::int64_t>(ops.tokens),
                                                 static_cast<std::int64_t>(kVarlenTokens));
    if (walks > std::numeric_limits<std::uint16_t>::max())
        throw std::overflow_error("causal_conv1d: varlen output grid exceeds CUDA limit");
    varlen_output_kernel<Rows0, Rows1, Rows2><<<dim3(static_cast<unsigned>(c_blocks),
                                                     static_cast<unsigned>(walks)),
                                                kVarlenBlock, 0, stream>>>(ops);
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
