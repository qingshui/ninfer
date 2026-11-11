#pragma once
#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Registered gate/up geometry: [13312,2048] BF16 or FP8_E4M3FN_ROW_FP32. Throws otherwise.
void validate_geglu_geometry(QType type, std::int32_t rows, std::int32_t k);

void launch_bf16_geglu_n13312_k2048(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);

[[nodiscard]] std::size_t fp8_row_fp32_geglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);
void fp8_row_fp32_geglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                 LinearPolicy policy, MemoryArena& workspace, cudaStream_t stream);

void launch_fp8_row_fp32_geglu_a16(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream);
void launch_fp8_row_fp32_geglu_a8(const Tensor& x, const Weight& weight, Tensor& out,
                                  MemoryArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
