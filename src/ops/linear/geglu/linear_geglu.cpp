#include "core/weight.h"
#include "ninfer/ops/linear_geglu.h"

#include "ops/common/host/operands.h"
#include "ops/linear/common/operands.h"
#include "ops/linear/geglu/dispatch.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {

void validate_policy(LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("linear_geglu: invalid compute policy");
}

} // namespace

std::size_t linear_geglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                  std::int32_t input_rows, LinearPolicy policy,
                                                  std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_geglu workspace: invalid profile or token interval");
    }
    detail::validate_geglu_geometry(qtype, gate_up_rows, input_rows);
    if (qtype == QType::BF16) return 0;
    return detail::fp8_row_fp32_geglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
}

void linear_geglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                  MemoryArena& ws, CudaExecutionView execution) {
    validate_policy(policy);
    if (x.device != execution.device() || out.device != execution.device() ||
        gate_up_weight.device != execution.device() || ws.device() != execution.device()) {
        throw std::invalid_argument(
            "linear_geglu: operands and workspace must belong to execution");
    }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_geglu: x/out must be BF16");
    }
    const auto& w = gate_up_weight;
    detail::validate_geglu_geometry(w.qtype, w.n, w.k);
    const int t = x.ne[1];
    if (t <= 0 || x.ne[0] != w.k || out.ne[0] != w.n / 2 || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[1] != t || out.ne[2] != 1 || out.ne[3] != 1 || w.padded_shape[0] != w.n ||
        w.padded_shape[1] != w.k)
        throw std::invalid_argument("linear_geglu: expected [K,T] and [M,T] for weight [2*M,K]");
    if (!x.is_contiguous() || !out.is_contiguous() || !common::operand::aligned(x.data, 16) ||
        !common::operand::aligned(out.data, 16))
        throw std::invalid_argument("linear_geglu: x/out must be contiguous and 16-byte aligned");

    if (w.qtype == QType::BF16) {
        detail::validate_bf16_projection(x, w, {&out, 1}, nullptr, execution, "linear_geglu");
        detail::launch_bf16_geglu_n13312_k2048(x, w, out, execution.stream());
        return;
    }
    (void)common::validate_fp8_weight(w, "fp8 linear_geglu");
    detail::validate_projection_operands(x, w, {&out, 1}, nullptr, execution, "linear_geglu");
    detail::fp8_row_fp32_geglu_dispatch(x, w, out, policy, ws, execution.stream());
}

} // namespace ninfer::ops
