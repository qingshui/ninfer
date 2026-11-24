#include "ops/linear/backend.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include "ops/linear/kernels/fp8/fp8_operands.h"
#include "ops/linear/kernels/sm100/gemm.h"
#include "ops/linear/split/dispatch.h"

#include <array>
#include <stdexcept>

namespace ninfer::ops::detail::linear_split {
namespace {
constexpr int kRows = 20480;
constexpr int kK    = 3072;

// The tcgen05 route multiplies E4M3 codes, so the activation is always quantized. A16-only
// permission has no route on this shape.
void require_quantized(LinearPolicy policy) {
    if (!allows_a8(policy))
        throw std::invalid_argument(
            "fp8 linear_split n=20480 k=3072: tcgen05 requires an A8 activation permission");
}

std::size_t capacity(LinearPolicy policy, std::int32_t, std::int32_t last) {
    require_quantized(policy);
    return fp8_a8_workspace_capacity_bytes(last, kK, kFp8SplitN20480K3072PartialBytes);
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    require_quantized(weights.policy);
    if (workspace == nullptr) throw std::invalid_argument("fp8 linear_split requires workspace");
    const auto scratch =
        allocate_fp8_a8_workspace(*workspace, x.ne[1], kK, kFp8SplitN20480K3072PartialBytes);
    fp8_split_n20480_k3072_dispatch(x, weights.first, outputs, scratch, stream);
}
} // namespace

// GDN input projection of the 122B geometry: one [qkv, z] parent cut into {12288, 8192}. The
// activation is quantized per token into the caller's scratch and the tcgen05 GEMM writes each
// segment straight into its output tensor.
void fp8_split_n20480_k3072_dispatch(const Tensor& x, const Weight& weight,
                                     std::span<const Tensor> outputs, Fp8A8Workspace workspace,
                                     cudaStream_t stream) {
    if (linear_backend(weight.qtype, kRows, kK, x.ne[1]) != LinearBackend::Tcgen05)
        throw std::invalid_argument("fp8 linear_split: unsupported token extent");
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const auto p                               = fp8_a8_operands<float>(weight, workspace, x.ne[1]);
    const std::array<__nv_bfloat16*, 2> output = {static_cast<__nv_bfloat16*>(outputs[0].data),
                                                  static_cast<__nv_bfloat16*>(outputs[1].data)};
    Sm100Epilogue epilogue;
    epilogue.kind     = Sm100Epilogue::Split;
    epilogue.segments = kSplitRows<12288, 8192>;
    epilogue.outputs  = output;
    launch_sm100_fp8_gemm({p.x, p.x_scales, p.codes, p.scales, p.rows, p.k, p.tokens}, epilogue,
                          nullptr, stream, {workspace.partials, workspace.partial_bytes});
}

const SplitShape kFp8RowFp32SplitN20480K3072{kRows, kK, kSplitRows<12288, 8192>, launch, capacity};
} // namespace ninfer::ops::detail::linear_split
