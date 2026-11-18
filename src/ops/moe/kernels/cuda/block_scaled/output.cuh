#pragma once

#include "ops/moe/kernels/cuda/output.cuh"
#include "ops/moe/kernels/cuda/block_scaled/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled {

template <class Shape>
__device__ __forceinline__ float routed_value(int token, int row, const Storage& storage,
                                              const float* coefficients) {
    if (storage.routed_sum) {
        return storage.routed_sum[std::int64_t(token) * Shape::hidden + row];
    }
    float value = 0;
    if (storage.contributions) {
#pragma unroll
        for (int j = 0; j < Shape::top_k; ++j) {
            const int assignment = token * Shape::top_k + j;
            const int column     = storage.groups.inverse[assignment];
            if (column >= 0) {
                value = fmaf(coefficients[assignment],
                             storage.contributions[std::int64_t(column) * Shape::hidden + row],
                             value);
            }
        }
    }
    return value;
}

// Four consecutive columns of one token, with the arithmetic of routed_value per column. The
// hidden size is a multiple of four, so a quad never straddles two tokens.
template <class Shape>
__device__ __forceinline__ float4 routed_value4(int token, int row, const Storage& storage,
                                                const float* coefficients) {
    static_assert(Shape::hidden % 4 == 0);
    if (storage.routed_sum) {
        return *reinterpret_cast<const float4*>(storage.routed_sum +
                                                std::int64_t(token) * Shape::hidden + row);
    }
    float4 value{0.0F, 0.0F, 0.0F, 0.0F};
    if (storage.contributions) {
#pragma unroll
        for (int j = 0; j < Shape::top_k; ++j) {
            const int assignment = token * Shape::top_k + j;
            const int column     = storage.groups.inverse[assignment];
            if (column >= 0) {
                const float weight = coefficients[assignment];
                const float4 term  = *reinterpret_cast<const float4*>(
                    storage.contributions + std::int64_t(column) * Shape::hidden + row);
                value.x = fmaf(weight, term.x, value.x);
                value.y = fmaf(weight, term.y, value.y);
                value.z = fmaf(weight, term.z, value.z);
                value.w = fmaf(weight, term.w, value.w);
            }
        }
    }
    return value;
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
