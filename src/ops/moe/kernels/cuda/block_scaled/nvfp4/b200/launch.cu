#include "ops/moe/backend.h"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/b200/grouped_tcgen05.cuh"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/compute.cuh"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/launch.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4 {
namespace {

template <int Hidden, int Experts>
__global__ void quantize_input(const __nv_bfloat16* x, std::uint8_t* codes, std::uint8_t* scales,
                               float divisor, int tokens, int* counts, float* routed_sum,
                               bool clear_counts) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (routed_sum) {
        for (int i = group; i < tokens * Hidden; i += gridDim.x * blockDim.x) {
            routed_sum[i] = 0;
        }
    }
    if (counts && clear_counts && group < Experts) { counts[group] = 0; }
    if (group < tokens * (Hidden / 16)) {
        const auto packed = common::quantize_nvfp4_k16(x + std::int64_t(group) * 16, divisor);
        *reinterpret_cast<uint2*>(codes + std::int64_t(group) * 8) =
            make_uint2(packed.codes_lo, packed.codes_hi);
        scales[group] = packed.scale;
    }
}

// Copies the hidden-state codes and scales of token assignments[slot] / TopK into packed slot
// `slot`, so the gate/up GEMM reads its activation tiles with plain 2D TMA loads. A gather4 load
// of the token rows keeps the producer thread busy for ~100 cycles per four rows, several times
// longer than the tile's MMAs.
template <int Hidden, int TopK>
__global__ void gather_input(const std::uint8_t* codes, const std::uint8_t* scales,
                             const int* assignments, std::uint8_t* gathered_codes,
                             std::uint8_t* gathered_scales, std::int64_t slots) {
    constexpr int kCodeVectors  = Hidden / 2 / 16;
    constexpr int kScaleVectors = Hidden / 16 / 16;
    constexpr int kVectors      = kCodeVectors + kScaleVectors;
    const std::int64_t i        = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= slots * kVectors) return;
    const std::int64_t slot = i / kVectors;
    const int vector        = static_cast<int>(i - slot * kVectors);
    const std::int64_t token = assignments[slot] / TopK;
    if (vector < kCodeVectors) {
        reinterpret_cast<uint4*>(gathered_codes + slot * (Hidden / 2))[vector] =
            reinterpret_cast<const uint4*>(codes + token * (Hidden / 2))[vector];
    } else {
        const int v = vector - kCodeVectors;
        reinterpret_cast<uint4*>(gathered_scales + slot * (Hidden / 16))[v] =
            reinterpret_cast<const uint4*>(scales + token * (Hidden / 16))[v];
    }
}

// The compiled MoE shapes share one launcher; the entry plan carries the geometry.
template <class Function>
void with_shape(int hidden, Function&& function) {
    if (hidden == 3072) {
        function.template operator()<SwiGLUH3072I1024E256K8>();
    } else {
        function.template operator()<SwiGLUH2560I640E512K10>();
    }
}

[[noreturn]] void reject(const std::string& what) {
    throw std::invalid_argument("nvfp4 moe tcgen05: " + what);
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

// The grouped kernels address an expert as codes + expert * code_stride and
// scales + expert * scale_stride (one TMA map over the bank), so the bank must be affine.
void check_bank(const ExpertMatrixDeviceView& bank, const char* name, int experts) {
    const std::string label = name;
    if (bank.offsets || bank.expert_indices)
        reject(label + " bank must be affine (expert-major planes), not indexed");
    if (!aligned(bank.codes, 128) || !aligned(bank.scales, 16) || !bank.matrix_scales)
        reject(label + " bank needs 128-byte aligned codes, 16-byte aligned scales and matrix "
                       "scalars");
    if (bank.code_stride % 16 || bank.scale_stride % 16)
        reject(label + " bank expert strides must be 16-byte multiples");
    if (bank.physical_count != 0 && static_cast<int>(bank.physical_count) != experts)
        reject(label + " bank holds " + std::to_string(bank.physical_count) + " experts, expected " +
               std::to_string(experts));
}

using Geometry = SwiGLUH3072I1024E256K8;

GroupedArgs base_args(const Storage& s) {
    GroupedArgs args{};
    args.offsets     = s.groups.offsets;
    args.assignments = s.groups.assignments;
    args.experts     = Geometry::experts;
    args.top_k       = Geometry::top_k;
    return args;
}

// gate/up: B = hidden codes of the packed slots, two weight parts (gate rows, up rows).
template <int TileN>
void run_gate_up(const Tensor& x, const CudaSwiGLUExperts& experts, const Storage& s,
                 CudaExecutionView execution) {
    constexpr int H = Geometry::hidden, I = Geometry::intermediate, E = Geometry::experts;
    static_assert(H % 256 == 0 && I % 256 == 0 && I % 128 == 0 && H % 128 == 0);
    const int tokens      = static_cast<int>(x.ne[1]);
    const int assignments = tokens * Geometry::top_k;
    const auto& gu        = experts.bank.gate_up;
    using Cfg             = sm::GemmConfig<sm::GemmKind::Nvfp4, TileN, 2, (TileN >= 32 ? 8 : 4)>;

    gather_input<H, Geometry::top_k>
        <<<static_cast<unsigned>((std::int64_t(assignments) * (H / 2 / 16 + H / 16 / 16) + 255) / 256),
           256, 0, execution.stream()>>>(
            s.input_codes, static_cast<const std::uint8_t*>(s.input_scales), s.groups.assignments,
            s.gathered_codes, static_cast<std::uint8_t*>(s.gathered_scales), assignments);
    CUDA_CHECK(cudaGetLastError());

    GroupedArgs a        = base_args(s);
    a.a_scales           = reinterpret_cast<const std::uint8_t*>(gu.scales);
    a.a_scale_stride     = gu.scale_stride;
    a.b_scales           = static_cast<const std::uint8_t*>(s.gathered_scales);
    a.b_scale_row_bytes  = H / 16;
    a.tiles_m            = I / 128;
    a.k_iters            = H / 256;
    a.scale_k64          = H / 64;
    a.part_row_offset    = I;
    const auto a_map = make_expert_bank_map(gu.codes, H / 2, 2 * I, E, gu.code_stride, 128,
                                            "moe tcgen05: gate/up weight codes map");
    const auto b_map = sm::make_tensor_map_2d(
        s.gathered_codes, CU_TENSOR_MAP_DATA_TYPE_UINT8, H / 2, assignments, H / 2, 128, TileN,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        "moe tcgen05: hidden codes map");
    const GateUpEpilogue epilogue{s.product_codes,
                                  static_cast<std::uint8_t*>(s.product_scales),
                                  gu.matrix_scales,
                                  gu.matrix_scale_stride,
                                  experts.activation_calibration.gate_up.scalar_divisor(),
                                  experts.activation_calibration.down.scalar_divisor(),
                                  I};
    launch_grouped<Cfg>(a_map, b_map, a, epilogue, assignments,
                                         execution.stream());
}

// down: B = product codes of the same packed slots, contiguous per expert.
template <int TileN>
void run_down(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
              const Plan& plan, const Storage& s, CudaExecutionView execution) {
    constexpr int H = Geometry::hidden, I = Geometry::intermediate, E = Geometry::experts,
                  K = Geometry::top_k;
    const int tokens      = static_cast<int>(x.ne[1]);
    const int assignments = tokens * K;
    const auto& dn        = experts.bank.down;
    const auto stream     = execution.stream();
    using Cfg             = sm::GemmConfig<sm::GemmKind::Nvfp4, TileN, 1, (TileN >= 32 ? 8 : 4)>;

    GroupedArgs a        = base_args(s);
    a.a_scales           = reinterpret_cast<const std::uint8_t*>(dn.scales);
    a.a_scale_stride     = dn.scale_stride;
    a.b_scales           = static_cast<const std::uint8_t*>(s.product_scales);
    a.b_scale_row_bytes  = I / 16;
    a.tiles_m            = H / 128;
    a.k_iters            = I / 256;
    a.scale_k64          = I / 64;
    a.part_row_offset    = 0;
    const auto a_map = make_expert_bank_map(dn.codes, I / 2, H, E, dn.code_stride, 128,
                                            "moe tcgen05: down weight codes map");
    const auto b_map = sm::make_tensor_map_2d(
        s.product_codes, CU_TENSOR_MAP_DATA_TYPE_UINT8, I / 2, assignments, I / 2, 128, TileN,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        "moe tcgen05: product codes map");
    const float divisor      = experts.activation_calibration.down.scalar_divisor();
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    if (plan.atomic_output) {
        if (!s.routed_sum) reject("atomic output needs the per-token sum buffer");
        if (!aligned(s.routed_sum, 16)) reject("the per-token sum buffer must be 16-byte aligned");
        const DownEpilogue<true> epilogue{s.routed_sum, nullptr, s.groups.assignments,
                                          coefficients, dn.matrix_scales, dn.matrix_scale_stride,
                                          divisor,      H,       K};
        launch_grouped<Cfg>(a_map, b_map, a, epilogue, assignments, stream);
    } else {
        if (!s.contributions) reject("ordered output needs the contribution buffer");
        const DownEpilogue<false> epilogue{nullptr, s.contributions, s.groups.assignments,
                                           coefficients, dn.matrix_scales, dn.matrix_scale_stride,
                                           divisor,      H,              K};
        launch_grouped<Cfg>(a_map, b_map, a, epilogue, assignments, stream);
    }
}

template <class Function>
void with_tile(int tile_tokens, Function&& function) {
    switch (tile_tokens) {
    case 16:
        return function(std::integral_constant<int, 16>{});
    case 32:
        return function(std::integral_constant<int, 32>{});
    case 64:
        return function(std::integral_constant<int, 64>{});
    case 128:
        return function(std::integral_constant<int, 128>{});
    }
    reject("unsupported token tile");
}

void launch_grouped_tcgen05(const Tensor& x, const CudaSwiGLUExperts& experts,
                            const ExpertRoutes& routes, const Plan& plan, const Storage& s,
                            CudaExecutionView execution) {
    constexpr int E = Geometry::experts;
    if (!s.groups.offsets || !s.groups.assignments) reject("expert groups are missing");
    if (!aligned(s.input_codes, 16) || !aligned(s.input_scales, 16) ||
        !aligned(s.gathered_codes, 16) || !aligned(s.gathered_scales, 16) ||
        !aligned(s.product_codes, 16) || !aligned(s.product_scales, 16))
        reject("activation buffers must be 16-byte aligned");
    check_bank(experts.bank.gate_up, "gate/up", E);
    check_bank(experts.bank.down, "down", E);
    const int tokens  = static_cast<int>(x.ne[1]);
    const auto tuning = resolve_nvfp4_sm100_tuning(plan.nvfp4.sm100, tokens);
    with_tile(tuning.gate_up_tile_tokens, [&](auto tile) {
        run_gate_up<decltype(tile)::value>(x, experts, s, execution);
    });
    with_tile(tuning.down_tile_tokens, [&](auto tile) {
        run_down<decltype(tile)::value>(x, experts, routes, plan, s, execution);
    });
}
} // namespace

void launch_input(const Tensor& x, const CudaSwiGLUExperts& experts, const Storage& s,
                  CudaExecutionView execution, bool clear_counts) {
    const auto stream = execution.stream();
    with_shape(x.ne[0], [&]<class Shape>() {
        quantize_input<Shape::hidden, Shape::experts>
            <<<(std::max(x.ne[1] * (Shape::hidden / 16), Shape::experts) + 255) / 256, 256, 0,
               stream>>>(static_cast<const __nv_bfloat16*>(x.data), s.input_codes,
                         static_cast<std::uint8_t*>(s.input_scales),
                         experts.activation_calibration.gate_up.scalar_divisor(), x.ne[1],
                         s.groups.counts, s.routed_sum, clear_counts);
    });
    CUDA_CHECK(cudaGetLastError());
}

void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& s, CudaExecutionView execution) {
    switch (moe_backend(plan.hidden, plan.experts, plan.top_k, static_cast<int>(x.ne[1]))) {
    case MoeBackend::Tcgen05:
        launch_grouped_tcgen05(x, experts, routes, plan, s, execution);
        return;
    case MoeBackend::MmaSync:
        // Block-scaled mma.sync is an SM120 instruction; the 2560/640/512/10 profile has no
        // B200 kernel. The shape dispatch rejects that geometry, so this only guards the entry.
        throw std::invalid_argument("moe block scaled: mma.sync is not available on b200");
    case MoeBackend::Vector:
        throw std::logic_error("nvfp4 moe: single-token calls take the vector path");
    }
}

void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, bool shared, MoeEpilogue mode, Tensor& destination,
                   const Storage& s, const Plan& plan, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const auto* ids          = static_cast<const int*>(routes.ids.data);
    const auto* mask         = static_cast<const int*>(subset.mask.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    with_shape(plan.hidden, [&]<class Shape>() {
        constexpr int I = Shape::intermediate, K = Shape::top_k, H = Shape::hidden;
        vector_gate<I, H, K><<<dim3(I / 4, x.ne[1] * K), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind,
            mask, s.vector_activation);
        CUDA_CHECK(cudaGetLastError());
        const auto down = [&](auto output) {
            vector_down<I, 4, K, H, K><<<dim3(H / 4, x.ne[1]), K * 32, 0, stream>>>(
                s.vector_activation, experts.bank.down, ids, coefficients, subset.kind, mask,
                output);
            CUDA_CHECK(cudaGetLastError());
        };
        if (shared) {
            down(FloatContribution{s.routed_sum});
        } else {
            with_output(destination, mode, down);
        }
    });
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4
