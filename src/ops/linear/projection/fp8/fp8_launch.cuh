#pragma once
#include "ops/linear/kernels/fp8/fp8_launch.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include <algorithm>

namespace ninfer::ops::detail {
template <class Geometry, class Schedule, class ScaleT = __nv_bfloat16>
void fp8_linear_a16_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands<ScaleT>(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

template <class Geometry, int Capacity, class Schedule, bool Exact = false,
          class ScaleT = __nv_bfloat16>
void fp8_linear_a16_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, Capacity, Exact>>(
        fp8_a16_operands<ScaleT>(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

template <class Geometry, class Schedule, class ScaleT = __nv_bfloat16>
void fp8_linear_a16_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands<ScaleT>(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

template <class Geometry, class Schedule, class ScaleT = __nv_bfloat16>
void fp8_linear_a16_sliced_k(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands<ScaleT>(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

template <class Geometry, class Schedule, class ScaleT = __nv_bfloat16>
void launch_fp8_a8(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
                   cudaStream_t stream) {
    launch_fp8_a8_quantize(x, w, scratch, stream);
    launch_fp8_a8_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a8_operands<ScaleT>(w, scratch, x.ne[1]),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

template <class Geometry, class Schedule, class ScaleT = __nv_bfloat16>
void launch_fp8_a8_tma(const Tensor& x, const Weight& w, Tensor& out, Fp8A8Workspace scratch,
                       cudaStream_t stream) {
    launch_fp8_a8_quantize(x, w, scratch, stream);
    launch_fp8_a8_tma_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a8_operands<ScaleT>(w, scratch, x.ne[1]),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{},
        stream, scratch.partials);
}
} // namespace ninfer::ops::detail
