#include "ops/linear/add/dispatch.h"
#include "ops/linear/backend.h"
#include "ops/linear/kernels/bf16/bf16_operands.h"
#include "ops/linear/kernels/sm100/gemm.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
// GDN output projection of the 122B geometry with the residual added in the epilogue; B200
// serves every token count on tcgen05. The residual is the output tensor itself.
void launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const Bf16A16Operands p = bf16_a16_operands(x, w);
    auto* data              = static_cast<__nv_bfloat16*>(out.data);
    Sm100Epilogue epilogue;
    epilogue.kind     = Sm100Epilogue::Add;
    epilogue.residual = data;
    launch_sm100_bf16_gemm({p.x, p.weight, p.rows, p.k, p.tokens}, epilogue, data, stream);
}
} // namespace

LinearAddLaunch select_bf16_linear_add_n3072_k8192(std::int32_t tokens) {
    if (linear_backend(QType::BF16, 3072, 8192, tokens) != LinearBackend::Tcgen05)
        throw std::invalid_argument("bf16 linear_add: unsupported token extent");
    return launch;
}
} // namespace ninfer::ops::detail
