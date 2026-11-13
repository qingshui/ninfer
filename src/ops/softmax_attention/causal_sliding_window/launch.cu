#include "ops/softmax_attention/causal_sliding_window/launch.h"

#include "core/device.h"
#include "ops/common/host/launch.h"
#include "ops/common/host/operands.h"
#include "ops/softmax_attention/kernels/causal_sliding_window/kernel.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Geometry = CausalD256H8Kv2;
static_assert(Geometry::KVHeads == kCausalSlidingWindowKvHeads &&
              Geometry::GroupSize == kCausalSlidingWindowGroup);

template <class S, bool Direct>
void launch_slice(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                  const Tensor& valid_columns, float scale,
                  const ConstCyclicKVCacheLayerView& context, const CausalSlidingWindowPlan& plan,
                  CausalPartialView partial, Tensor& out, cudaStream_t stream) {
    constexpr auto kernel = causal_swa_kernel<Geometry, S, Direct>;
    constexpr int bytes   = sizeof(Bf16KvGroupedStorage<Geometry, S>);
    const int dynamic     = common::prepare_shared<bytes, kernel>();
    const int width       = q.ne[2];
    const dim3 grid(Geometry::KVHeads * common::div_up(width * Geometry::GroupSize, S::kQueryRows),
                    plan.splits, plan.batch);
    kernel<<<grid, S::kThreads, dynamic, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const int*>(positions.data),
        static_cast<const int*>(valid_columns.data),
        static_cast<const std::int64_t*>(context.k.records.byte_offsets.data),
        static_cast<const std::int64_t*>(context.v.records.byte_offsets.data),
        common::operand::state_field_base<const __nv_bfloat16>(context.k),
        common::operand::state_field_base<const __half>(context.v),
        static_cast<int>(context.padded_capacity), width, scale, plan.splits,
        static_cast<__nv_bfloat16*>(out.data), partial);
    CUDA_CHECK(cudaGetLastError());
}

template <class S>
void launch_instance(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                     const Tensor& valid_columns, float scale,
                     const ConstCyclicKVCacheLayerView& context,
                     const CausalSlidingWindowPlan& plan, CausalPartialView partial, Tensor& out,
                     cudaStream_t stream) {
    if (plan.splits == 1) {
        launch_slice<S, true>(q, k, v, positions, valid_columns, scale, context, plan, partial, out,
                              stream);
        return;
    }
    launch_slice<S, false>(q, k, v, positions, valid_columns, scale, context, plan, partial, out,
                           stream);
    const int width = q.ne[2];
    const dim3 grid(width * plan.batch, Geometry::QHeads);
    causal_swa_merge_kernel<Geometry><<<grid, Geometry::kHeadDim, 0, stream>>>(
        partial.acc, partial.maximum, partial.sum, static_cast<const int*>(valid_columns.data),
        width, plan.splits, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_causal_sliding_window_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                            const Tensor& positions, const Tensor& valid_columns,
                                            float scale, const ConstCyclicKVCacheLayerView& context,
                                            const CausalSlidingWindowPlan& plan,
                                            CausalPartialView partial, Tensor& out,
                                            cudaStream_t stream) {
    if (plan.splits < 1 || plan.splits > kCausalSlidingWindowMaxSplits ||
        (plan.splits > 1 && (!partial.acc || !partial.maximum || !partial.sum)) ||
        context.capacity != static_cast<std::uint32_t>(kCausalSlidingWindow)) {
        throw std::invalid_argument("causal_sliding_window_attention: inconsistent plan");
    }
    switch (plan.instance) {
    case CausalSlidingWindowInstance::Rows16:
        return launch_instance<Bf16KvGroupedMmaSchedule<16, 32, 2, 2>>(
            q, k, v, positions, valid_columns, scale, context, plan, partial, out, stream);
    case CausalSlidingWindowInstance::Rows32:
        return launch_instance<Bf16KvGroupedMmaSchedule<32, 32>>(
            q, k, v, positions, valid_columns, scale, context, plan, partial, out, stream);
    case CausalSlidingWindowInstance::Rows64:
        return launch_instance<Bf16KvGroupedMmaSchedule<64, 32>>(
            q, k, v, positions, valid_columns, scale, context, plan, partial, out, stream);
    }
    throw std::logic_error("causal_sliding_window_attention: unknown instance");
}

} // namespace ninfer::ops::detail
