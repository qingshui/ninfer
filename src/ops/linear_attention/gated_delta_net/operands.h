#pragma once

#include "core/dtype.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr std::int32_t kStateDim = 128;

[[nodiscard]] constexpr bool are_head_counts_valid(std::int64_t qk_heads,
                                                   std::int64_t value_heads) noexcept {
    return qk_heads > 0 && value_heads >= qk_heads && (value_heads % qk_heads) == 0;
}

struct RecurrenceGeometry {
    std::int32_t qk_heads, value_heads, tokens;
};

struct RecurrenceOperands {
    const __nv_bfloat16 *q, *k, *v;
    const float *g, *beta;
    const void* state_in; // state_dtype words
    void* state_out;
    __nv_bfloat16* out;
    int qk_heads, value_heads, tokens;
    float scale;
    DType state_dtype = DType::FP32;
};

// Recurrent state fields are FP32 or BF16; every state field of one call shares the dtype.
inline void require_state_dtype(DType dtype, DType expected, const char* op, const char* name) {
    if ((dtype != DType::FP32 && dtype != DType::BF16) || dtype != expected)
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be FP32 or BF16 and match the other state fields");
}

inline void require_scale(float scale, const char* op) {
    const float expected = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected) > 1.0e-6f)
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(128)");
}

} // namespace ninfer::ops::detail::gated_delta_net
