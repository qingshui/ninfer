#pragma once

#include "ops/linear_attention/causal_conv/plan.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kCausalConvVarlenMaxRows       = 1024;
inline constexpr std::int32_t kCausalConvVarlenMaxBoundaries = 1024;

// Packed rows of one column axis. Every table is a device pointer; none is read on the host.
struct CausalConvVarlenOperands {
    const __nv_bfloat16* x;
    const __nv_bfloat16* weight;
    const std::int32_t* token_offsets; // [rows+1]
    const std::byte* source_base;      // source storage + field offset
    const std::int64_t* source_offsets;
    std::byte* destination_base; // destination storage + field offset
    const std::int64_t* destination_offsets;
    std::byte* checkpoint_base; // destination storage + checkpoint field offset
    const std::int32_t* boundary_rows;
    const std::int32_t* boundary_columns;
    const std::int64_t* boundary_offsets;
    __nv_bfloat16* out0;
    __nv_bfloat16* out1;
    __nv_bfloat16* out2;
    std::int32_t channels, tokens, rows, boundaries;
};

void launch_causal_conv_split_varlen(const CausalConvVarlenOperands& operands,
                                     CausalConvSplitGeometry geometry, cudaStream_t stream);

} // namespace ninfer::ops::detail
