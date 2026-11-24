// B200 shared expert of the Qwen3.5-122B geometry (hidden 3072, intermediate 1024): the BF16
// gate/up and down projections run the sm100 dense tcgen05 GEMMs (T01) instead of mma.sync.
//
// gate/up writes silu(x * gate^T) * (x * up^T) straight into the shared activation. down folds
// the per-token coefficient into the activation rows, stages the routed (+residual) terms in the
// destination and lets the dense GEMM add its product with the Add epilogue (residual may alias
// the output). FP32 contribution outputs and the single-token vector plans keep the common path.

#include "core/device.h"
#include "ops/moe/kernels/cuda/block_scaled/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/output.cuh"
#include "ops/linear/kernels/sm100/gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::moe_cuda::block_scaled {
namespace {

using Shape = SwiGLUH3072I1024E256K8;
constexpr int H = Shape::hidden, I = Shape::intermediate;

[[noreturn]] void reject(const std::string& what) {
    throw std::invalid_argument("nvfp4 moe shared tcgen05: " + what);
}

// h'[t, i] = beta[t] * (hi[t, i] + lo[t, i]). Folding the coefficient into the activation rows
// makes the down GEMM produce already-scaled products, so its epilogue only adds.
template <bool Split>
__global__ void scale_shared(__nv_bfloat16* h, const float* beta, int tokens) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = tokens * I;
    if (index >= count) return;
    float value = __bfloat162float(h[index]);
    if constexpr (Split) { value += __bfloat162float(h[count + index]); }
    h[index] = __float2bfloat16_rn(value * beta[index / I]);
}

// destination[t, r] = initial(t, r) + routed(t, r); the dense down GEMM adds beta * h * down on
// top of this staged value.
template <class Output>
__global__ void stage_routed(const float* coefficients, Storage storage, Output output,
                             int tokens) {
    const int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (index >= tokens * H) return;
    const float4 routed = routed_value4<Shape>(index / H, index % H, storage, coefficients);
    output.store(index, output.initial(index) + routed.x);
    output.store(index + 1, output.initial(index + 1) + routed.y);
    output.store(index + 2, output.initial(index + 2) + routed.z);
    output.store(index + 3, output.initial(index + 3) + routed.w);
}

} // namespace

bool launch_shared_dense(const Tensor& x, const ScaledSwiGLUTerm& shared, const ExpertRoutes& routes,
                         MoeEpilogue mode, Tensor& destination, const Storage& storage,
                         const Plan& plan, CudaExecutionView execution, bool prepared) {
    // The vector algorithm keeps its GEMV kernels; FP32 contribution outputs keep the float
    // path because the dense GEMM writes BF16.
    if (plan.shared_algorithm != SharedAlgorithm::Mma || destination.dtype != DType::BF16)
        return false;
    const int tokens = static_cast<int>(x.ne[1]);
    if (tokens <= 0) reject("token count must be positive");
    const auto stream   = execution.stream();
    auto* h             = static_cast<__nv_bfloat16*>(storage.shared_activation);
    const auto* input   = static_cast<const __nv_bfloat16*>(x.data);
    const auto* beta    = static_cast<const float*>(shared.coefficients.data);
    const auto* gate_up = static_cast<const __nv_bfloat16*>(shared.weights.gate_up.weight.qdata);
    const auto* down    = static_cast<const __nv_bfloat16*>(shared.weights.down.weight.qdata);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    auto* y                  = static_cast<__nv_bfloat16*>(destination.data);
    if (!h || !input || !beta || !gate_up || !down) reject("missing shared operands");
    const auto& gu = shared.weights.gate_up.weight;
    const auto& dn = shared.weights.down.weight;
    if (gu.qtype != QType::BF16 || dn.qtype != QType::BF16 || gu.n != 2 * I || gu.k != H ||
        dn.n != H || dn.k != I)
        reject("expected BF16 gate/up [2i, h] and down [h, i] shared weights");

    if (!prepared) {
        // One dense SwiGLU GEMM covers both halves. The dense epilogue rounds once, so the
        // split plane carries no correction; zero it for the uniform hi + lo read below.
        if (plan.shared_split_activation)
            CUDA_CHECK(cudaMemsetAsync(h + std::int64_t(tokens) * I, 0,
                                       std::size_t(tokens) * I * sizeof(__nv_bfloat16), stream));
        const Sm100Epilogue epilogue{.kind = Sm100Epilogue::SwiGLU};
        launch_sm100_bf16_gemm({input, gate_up, 2 * I, H, tokens}, epilogue, h, stream);
    }
    const int activation_elements = tokens * I;
    if (plan.shared_split_activation)
        scale_shared<true><<<(activation_elements + 255) / 256, 256, 0, stream>>>(h, beta, tokens);
    else
        scale_shared<false><<<(activation_elements + 255) / 256, 256, 0, stream>>>(h, beta,
                                                                                    tokens);
    CUDA_CHECK(cudaGetLastError());
    with_output(destination, mode, [&](auto output) {
        stage_routed<<<(tokens * H / 4 + 255) / 256, 256, 0, stream>>>(coefficients, storage, output,
                                                                        tokens);
    });
    CUDA_CHECK(cudaGetLastError());
    const Sm100Epilogue epilogue{.kind = Sm100Epilogue::Add, .residual = y};
    launch_sm100_bf16_gemm({h, down, H, I, tokens}, epilogue, y, stream);
    return true;
}

bool launch_control_shared_gate_up_dense(const Tensor& x, const MoeWeights& weights,
                                         const Storage& storage, const Plan& plan,
                                         CudaExecutionView execution) {
    // Below this size the single mma.sync launch is cheaper than one more GEMM launch.
    constexpr int kMinimumDenseTokens = 128;
    const int tokens                  = static_cast<int>(x.ne[1]);
    const auto& gu                    = weights.shared.gate_up.weight;
    if (plan.shared_algorithm != SharedAlgorithm::Mma || plan.hidden != H || tokens < kMinimumDenseTokens ||
        gu.qtype != QType::BF16 || gu.n != 2 * I || gu.k != H || !storage.shared_activation)
        return false;
    const auto stream = execution.stream();
    auto* h           = static_cast<__nv_bfloat16*>(storage.shared_activation);
    // The dense epilogue rounds once, so the split plane carries no correction.
    if (plan.shared_split_activation)
        CUDA_CHECK(cudaMemsetAsync(h + std::int64_t(tokens) * I, 0,
                                   std::size_t(tokens) * I * sizeof(__nv_bfloat16), stream));
    const Sm100Epilogue epilogue{.kind = Sm100Epilogue::SwiGLU};
    launch_sm100_bf16_gemm({static_cast<const __nv_bfloat16*>(x.data),
                            static_cast<const __nv_bfloat16*>(gu.qdata), 2 * I, H, tokens},
                           epilogue, h, stream);
    return true;
}

bool launch_control_router_dense(const Tensor& x, const MoeWeights& weights, float* scores,
                                 const Storage& storage, const Plan& plan,
                                 CudaExecutionView execution) {
    const auto& control = weights.control;
    const int tokens    = static_cast<int>(x.ne[1]);
    if (!scores || plan.hidden != H || control.qtype != QType::BF16 ||
        control.n != Shape::control_rows || control.k != H)
        return false;
    const auto stream = execution.stream();
    // The router rows are the experts and the shared gate; the select kernel counts into zeroed
    // per-expert totals.
    if (storage.groups.counts)
        CUDA_CHECK(cudaMemsetAsync(storage.groups.counts, 0, Shape::experts * sizeof(int), stream));
    const Sm100Epilogue epilogue{.kind = Sm100Epilogue::Scores,
                                 .scores = scores,
                                 .score_stride = Shape::score_rows};
    launch_sm100_bf16_gemm({static_cast<const __nv_bfloat16*>(x.data),
                            static_cast<const __nv_bfloat16*>(control.qdata), Shape::control_rows, H,
                            tokens},
                           epilogue, nullptr, stream);
    return true;
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
