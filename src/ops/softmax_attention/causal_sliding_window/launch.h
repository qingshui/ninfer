#pragma once

#include "core/arena.h"
#include "ops/softmax_attention/causal_sliding_window/plan.h"
#include "ops/softmax_attention/kernels/causal/operands.h"

namespace ninfer::ops::detail {

// Operands are validated by the owning Op. partial is ignored when plan.splits == 1.
void launch_causal_sliding_window_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                            const Tensor& positions, const Tensor& valid_columns,
                                            float scale, const ConstCyclicKVCacheLayerView& context,
                                            const CausalSlidingWindowPlan& plan,
                                            CausalPartialView partial, Tensor& out,
                                            cudaStream_t stream);

} // namespace ninfer::ops::detail
