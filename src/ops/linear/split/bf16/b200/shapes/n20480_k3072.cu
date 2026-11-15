#include "ops/linear/split/dispatch.h"
#include "ops/linear/backend.h"
#include "ops/linear/kernels/bf16/bf16_operands.h"
#include "ops/linear/kernels/sm100/gemm.h"

#include <array>
#include <stdexcept>

namespace ninfer::ops::detail::linear_split {
namespace {
// GDN input projection of the 122B geometry: one [qkv, z] parent cut into {12288, 8192}. B200
// serves every token count on tcgen05, writing each segment straight into its output tensor.
void launch(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
            cudaStream_t stream) {
    if (linear_backend(QType::BF16, 20480, 3072, x.ne[1]) != LinearBackend::Tcgen05)
        throw std::invalid_argument("bf16 linear_split: unsupported token extent");
    const Bf16A16Operands p                    = bf16_a16_operands(x, weight);
    const std::array<__nv_bfloat16*, 2> output = {static_cast<__nv_bfloat16*>(outputs[0].data),
                                                  static_cast<__nv_bfloat16*>(outputs[1].data)};
    Sm100Epilogue epilogue;
    epilogue.kind     = Sm100Epilogue::Split;
    epilogue.segments = kSplitRows<12288, 8192>;
    epilogue.outputs  = output;
    launch_sm100_bf16_gemm({p.x, p.weight, p.rows, p.k, p.tokens}, epilogue, nullptr, stream);
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
                  cudaStream_t stream) {
    launch(x, weights.first, outputs, stream);
}
} // namespace

void bf16_split_n20480_k3072_dispatch(const Tensor& x, const Weight& weight,
                                      std::span<const Tensor> outputs, cudaStream_t stream) {
    launch(x, weight, outputs, stream);
}

const SplitShape kBf16SplitN20480K3072{20480, 3072, kSplitRows<12288, 8192>, launch_split};
} // namespace ninfer::ops::detail::linear_split
