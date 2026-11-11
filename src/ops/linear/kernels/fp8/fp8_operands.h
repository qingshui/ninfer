#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
// Row-scale words are BF16 (FP8_E4M3FN_ROW_BF16) or FP32 (FP8_E4M3FN_ROW_FP32); every kernel
// is shared and parameterized by the scale word type.
template <class ScaleT>
struct Fp8A16OperandsT {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const ScaleT* scales;
    int rows, k, tokens;
};

template <class ScaleT>
struct Fp8A8OperandsT {
    const std::uint8_t* x;
    const float* x_scales;
    const std::uint8_t* codes;
    const ScaleT* scales;
    int rows, k, tokens;
};

using Fp8A16Operands = Fp8A16OperandsT<__nv_bfloat16>;
using Fp8A8Operands  = Fp8A8OperandsT<__nv_bfloat16>;

__host__ __device__ __forceinline__ float fp8_row_scale_value(__nv_bfloat16 scale) {
    return __bfloat162float(scale);
}

__host__ __device__ __forceinline__ float fp8_row_scale_value(float scale) { return scale; }

template <class ScaleT = __nv_bfloat16>
inline Fp8A16OperandsT<ScaleT> fp8_a16_operands(const Tensor& x, const Weight& w) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const ScaleT*>(w.scales),
            w.n,
            w.k,
            x.ne[1]};
}

template <class ScaleT = __nv_bfloat16>
inline Fp8A8OperandsT<ScaleT> fp8_a8_operands(const Weight& w, Fp8A8Workspace x, int tokens) {
    return {x.codes,
            x.scales,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const ScaleT*>(w.scales),
            w.n,
            w.k,
            tokens};
}

// Private template admission is independent of the public finite shape registry.
template <class Schedule, class Operands>
inline void validate_fp8_operands(const Operands& p) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    if (p.rows <= 0 || p.k <= 0 || p.tokens <= 0 || p.k % 32 || !aligned(p.x, 16) ||
        !aligned(p.codes, 16) || !aligned(p.scales, sizeof(*p.scales)))
        throw std::invalid_argument("FP8 templates require positive N/K/T and aligned operands");
    if constexpr (requires { p.x_scales; }) {
        if (!aligned(p.x_scales, 4) || !aligned(p.scales, 4))
            throw std::invalid_argument("FP8 A8 requires scales aligned to four bytes");
    }
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK)
            throw std::invalid_argument("FP8 template static K does not match operands");
    }
}
} // namespace ninfer::ops::detail
