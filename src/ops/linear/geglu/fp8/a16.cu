#include "ops/linear/geglu/dispatch.h"
#include "ops/linear/geglu/product_epilogue.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"

#include "core/device.h"

namespace ninfer::ops::detail {
namespace {
constexpr int kK            = 2048;
constexpr int kIntermediate = 6656;
using ScaleType             = float;

void launch_small(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    using Rows = SwiGluBranchRows<Schedule::kRowsPerWarp / 2, kIntermediate>;
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, kK, 4>>(
        fp8_a16_operands<ScaleType>(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate}, GeGluPairEpilogue{},
        stream, Rows{});
}

void launch_matrix(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const auto p = fp8_a16_operands<ScaleType>(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const auto sliced = [&]<int T, int W, int Stages>() {
        using S = Fp8ScheduleInstance<Fp8SlicedInstance<T, W, Stages>, kK>;
        launch_fp8_a16_sliced_k_mma<S>(p, output, GeGluPairEpilogue{}, stream,
                                       SwiGluRowMajorMmaRows<S>{});
    };
    const auto mma = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kK>;
        launch_fp8_a16_mma<S>(p, output, GeGluRowMajorMmaEpilogue{}, stream,
                              SwiGluRowMajorMmaRows<S>{});
    };
    if (x.ne[1] <= 8) return sliced.template operator()<8, 4, 1>();
    if (x.ne[1] <= 16) return sliced.template operator()<16, 4, 1>();
    if (x.ne[1] <= 24) return sliced.template operator()<32, 8, 1>();
    if (x.ne[1] <= 32) return sliced.template operator()<32, 4, 1>();
    if (x.ne[1] <= 64)
        return mma.template operator()<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>();
    if (x.ne[1] <= 96)
        return mma.template operator()<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>();
    mma.template operator()<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>();
}
} // namespace

void launch_fp8_row_fp32_geglu_a16(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream) {
    if (x.ne[1] <= 4) return launch_small(x, weight, out, stream);
    launch_matrix(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
