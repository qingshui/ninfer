#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// qtype is FP8_E4M3FN_ROW_BF16 or FP8_E4M3FN_ROW_FP32; each owns its registered shapes.
[[nodiscard]] std::size_t fp8_linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  MemoryArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
