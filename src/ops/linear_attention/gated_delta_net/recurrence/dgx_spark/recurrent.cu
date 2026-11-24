#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/recurrent_io.cuh"
#include "ops/linear_attention/gated_delta_net/kernels/recurrent_subgroup.cuh"
#include "core/device.h"
#include "ops/common/host/operands.h"

namespace ninfer::ops::detail::gated_delta_net {
namespace {

template <bool Normalize, bool Prefetch, class State>
__global__ __launch_bounds__(128, 2) void direct_kernel(DirectAccess<State> access) {
    extern __shared__ __align__(16) char backing[];
    subgroup::run<16, Normalize, subgroup::Mode::Update, Prefetch>(
        access, access.coordinates(), access.width,
        reinterpret_cast<subgroup::Window<subgroup::kWindowSize>*>(backing));
}

template <bool Normalize, class State>
__global__ __launch_bounds__(128, 2) void batch_kernel(BatchUpdateAccess<State> access) {
    extern __shared__ __align__(16) char backing[];
    subgroup::run<16, Normalize, subgroup::Mode::Update, true>(
        access, access.coordinates(), 1, reinterpret_cast<subgroup::Window<1>*>(backing));
}

template <bool Normalize, class State>
void launch_direct(const DirectAccess<State>& access, cudaStream_t stream) {
    const dim3 grid(access.heads.H_v, 1, kStateDim / kBlockDv), block(32, kNumWarps);
    if (access.width <= subgroup::kWindowSize) {
        direct_kernel<Normalize, true, State>
            <<<grid, block, sizeof(subgroup::Window<subgroup::kWindowSize>), stream>>>(access);
    } else {
        direct_kernel<Normalize, false, State><<<grid, block, 0, stream>>>(access);
    }
}

template <class State>
void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& state_in, Tensor& state_out, Tensor& out,
                            cudaStream_t stream) {
    const DirectAccess<State> access{static_cast<const __nv_bfloat16*>(q.data),
                                     static_cast<const __nv_bfloat16*>(k.data),
                                     static_cast<const __nv_bfloat16*>(v.data),
                                     static_cast<const float*>(g.data),
                                     static_cast<const float*>(beta.data),
                                     static_cast<const State*>(state_in.data),
                                     static_cast<State*>(state_out.data),
                                     static_cast<__nv_bfloat16*>(out.data),
                                     HeadMap::of(q.ne[1], v.ne[1]),
                                     q.ne[2],
                                     scale};
    if (normalize_qk)
        launch_direct<true>(access, stream);
    else
        launch_direct<false>(access, stream);
    CUDA_CHECK(cudaGetLastError());
}

template <class State>
void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, ConstStateFieldView source_state,
                                   StateFieldView destination_state, Tensor& out,
                                   cudaStream_t stream) {
    const BatchUpdateAccess<State> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        common::operand::state_field_base<const State>(source_state),
        common::operand::state_field_base<State>(destination_state),
        static_cast<const std::int64_t*>(source_state.records.byte_offsets.data),
        static_cast<const std::int64_t*>(destination_state.records.byte_offsets.data),
        static_cast<__nv_bfloat16*>(out.data),
        HeadMap::of(q.ne[1], v.ne[1]),
        scale};
    const dim3 grid(v.ne[1], q.ne[3], kStateDim / kBlockDv), block(32, kNumWarps);
    if (normalize_qk) {
        batch_kernel<true, State><<<grid, block, sizeof(subgroup::Window<1>), stream>>>(access);
    } else {
        batch_kernel<false, State><<<grid, block, sizeof(subgroup::Window<1>), stream>>>(access);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& state_in, Tensor& state_out, Tensor& out,
                            cudaStream_t stream) {
    if (state_out.dtype == DType::BF16)
        launch_recurrent_inout<__nv_bfloat16>(q, k, v, g, beta, scale, normalize_qk, state_in,
                                              state_out, out, stream);
    else
        launch_recurrent_inout<float>(q, k, v, g, beta, scale, normalize_qk, state_in, state_out,
                                      out, stream);
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, ConstStateFieldView source_state,
                                   StateFieldView destination_state, Tensor& out,
                                   cudaStream_t stream) {
    if (destination_state.field.tensor.dtype == DType::BF16)
        launch_recurrent_batch_update<__nv_bfloat16>(q, k, v, g, beta, scale, normalize_qk,
                                                     source_state, destination_state, out, stream);
    else
        launch_recurrent_batch_update<float>(q, k, v, g, beta, scale, normalize_qk, source_state,
                                             destination_state, out, stream);
}

} // namespace ninfer::ops::detail::gated_delta_net
