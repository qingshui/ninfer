#include "ops/linear/add/fp8/dispatch.h"
#include "core/platform.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
Fp8LinearAddPlan select_plan(QType type, std::int32_t rows, std::int32_t k, LinearPolicy policy,
                             std::int32_t tokens) {
#if NINFER_TARGET_B200
    if (type == QType::FP8_E4M3FN_ROW_FP32 && tokens > 0 && rows == 3072 && k == 8192)
        return select_fp8_row_fp32_add_n3072_k8192(policy, tokens);
#endif
    if (type == QType::FP8_E4M3FN_ROW_BF16 && tokens > 0 && rows == 5120) {
        if (k == 6144) return select_fp8_add_n5120_k6144(policy, tokens);
        if (k == 17408) return select_fp8_add_n5120_k17408(policy, tokens);
    }
    throw std::invalid_argument("fp8 linear_add: unsupported shape or token extent");
}
} // namespace

std::size_t fp8_linear_add_workspace_capacity_bytes(QType type, std::int32_t rows, std::int32_t k,
                                                    LinearPolicy policy, std::int32_t first,
                                                    std::int32_t last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("fp8 linear_add: invalid token interval");
    (void)select_plan(type, rows, k, policy, first);
    const auto plan = select_plan(type, rows, k, policy, last);
    return plan.a16 ? 0 : fp8_a8_workspace_capacity_bytes(last, k, plan.partial_bytes);
}

void fp8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                             MemoryArena& arena, cudaStream_t stream) {
    const auto plan = select_plan(w.qtype, w.n, w.k, policy, x.ne[1]);
    if (plan.a16) return plan.a16(x, w, out, stream);
    auto scope         = arena.scope();
    const auto scratch = allocate_fp8_a8_workspace(arena, x.ne[1], w.k, plan.partial_bytes);
    launch_fp8_a8_quantize(x, w, scratch, stream);
    plan.quantized(x, w, out, scratch, stream);
}
} // namespace ninfer::ops::detail
