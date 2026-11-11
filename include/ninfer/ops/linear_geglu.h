#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include "core/device.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Capacity for every extent in the inclusive interval; invalid profiles or intervals throw.
[[nodiscard]] std::size_t
linear_geglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                      std::int32_t input_rows, LinearPolicy policy,
                                      std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * out[i,t] = GELU_erf((W_gate * x)[i,t]) * (W_up * x)[i,t],
 * GELU_erf(z) = 0.5*z*(1 + erf(z/sqrt(2))).
 *
 * x [K,T] and out [M,T] are contiguous, 16-byte-aligned BF16 matrices.
 * gate_up_weight [2*M,K] stores all M gate rows followed by their matching M up rows.
 * Weights use their native stored representation. Inputs and weight storage must not overlap out.
 * The product consumes FP32 gate/up accumulators and rounds once to BF16.
 * Writes the entire output and has no persistent state.
 *
 * Policy controls permitted activation quantization. Caller workspace covers the complete
 * operation, is scoped to the call and overlaps no operand.
 */
void linear_geglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                  MemoryArena& ws, CudaExecutionView execution);

} // namespace ninfer::ops
