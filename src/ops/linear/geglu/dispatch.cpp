#include "ops/linear/geglu/dispatch.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr std::int32_t kRows = 13312;
constexpr std::int32_t kK    = 2048;
// A8 starts where the measured A16 SIMT/sliced-K routes stop being competitive.
constexpr std::int32_t kA8MinTokens = 5;
} // namespace

void validate_geglu_geometry(QType type, std::int32_t rows, std::int32_t k) {
    if ((type == QType::BF16 || type == QType::FP8_E4M3FN_ROW_FP32) && rows == kRows && k == kK)
        return;
    throw std::invalid_argument("linear_geglu: unsupported weight geometry");
}

std::size_t fp8_row_fp32_geglu_workspace_capacity_bytes(LinearPolicy policy,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear_geglu workspace: invalid token interval");
    return allows_a8(policy) && max_tokens >= kA8MinTokens
               ? fp8_a8_workspace_capacity_bytes(max_tokens, kK)
               : 0;
}

void fp8_row_fp32_geglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                 LinearPolicy policy, MemoryArena& workspace, cudaStream_t stream) {
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear_geglu: T must be positive");
    if (!allows_a8(policy) || x.ne[1] < kA8MinTokens)
        return launch_fp8_row_fp32_geglu_a16(x, weight, out, stream);
    launch_fp8_row_fp32_geglu_a8(x, weight, out, workspace, stream);
}
} // namespace ninfer::ops::detail
