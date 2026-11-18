#include "ops/linear/add/fp8/dispatch.h"
#include "ops/linear/backend.h"
#include "ops/linear/kernels/fp8/fp8_operands.h"
#include "ops/linear/kernels/sm100/gemm.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
// GDN output projection of the 122B geometry with the residual added in the epilogue. The
// residual is the output tensor itself; the activation arrives already quantized in `scratch`.
void launch(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
            cudaStream_t stream) {
    const auto p = fp8_a8_operands<float>(w, scratch, x.ne[1]);
    auto* data   = static_cast<__nv_bfloat16*>(out.data);
    Sm100Epilogue epilogue;
    epilogue.kind     = Sm100Epilogue::Add;
    epilogue.residual = data;
    launch_sm100_fp8_gemm({p.x, p.x_scales, p.codes, p.scales, p.rows, p.k, p.tokens}, epilogue,
                          data, stream);
}
} // namespace

Fp8LinearAddPlan select_fp8_row_fp32_add_n3072_k8192(LinearPolicy policy, std::int32_t tokens) {
    if (linear_backend(QType::FP8_E4M3FN_ROW_FP32, 3072, 8192, tokens) != LinearBackend::Tcgen05)
        throw std::invalid_argument("fp8 linear_add: unsupported token extent");
    if (!allows_a8(policy))
        throw std::invalid_argument(
            "fp8 linear_add n=3072 k=8192: tcgen05 requires an A8 activation permission");
    return {nullptr, launch, 0};
}
} // namespace ninfer::ops::detail
