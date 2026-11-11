#pragma once

#include "core/device.h"
#include "core/tensor.h"


namespace ninfer::ops {

/**
 * Elementwise sigmoid gate:
 *
 *   ideal[i] = x[i] * (1 / (1 + exp(-gate[i]))).
 *
 * gate/x are same-shaped, contiguous, non-overlapping BF16 tensors.
 * Updates x in place and preserves gate. Empty tensors perform no device work.
 */
void sigmoid_mul(const Tensor& gate, Tensor& x, CudaExecutionView execution);

/**
 * Per-head sigmoid gate shared by the D features of each head:
 *
 *   ideal[d + D*h, t] = x[d + D*h, t] * (1 / (1 + exp(-gate[h, t]))).
 *
 * gate is contiguous BF16 [H,T]; x is contiguous BF16 [D*H,T] and is read as [D,H,T], so the
 * feature index of head h occupies [D*h, D*(h+1)). D is derived as x.ne[0]/H and must be a whole
 * number. gate and x are non-overlapping. Updates x in place and preserves gate. Empty tensors
 * perform no device work. The Op is not fused with any projection or other elementwise stage.
 */
void headwise_sigmoid_mul(const Tensor& gate, Tensor& x, CudaExecutionView execution);

} // namespace ninfer::ops
