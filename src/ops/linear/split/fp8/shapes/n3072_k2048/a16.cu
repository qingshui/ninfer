#include "ops/linear/split/fp8/shapes/n3072_k2048/launch.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "core/device.h"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail::linear_split::fp8_n3072_k2048 {
namespace {
constexpr int kK = 2048;
using Output     = LinearBf16SegmentedOutput<2048, 512, 512>;

Output make_output(std::span<const Tensor> outputs) {
    return {{static_cast<__nv_bfloat16*>(outputs[0].data),
             static_cast<__nv_bfloat16*>(outputs[1].data),
             static_cast<__nv_bfloat16*>(outputs[2].data)}};
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
            cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, kK>>(
        fp8_a16_operands<float>(x, weight), make_output(outputs), LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void mma(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
         cudaStream_t stream) {
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, kK>>(
        fp8_a16_operands<float>(x, weight), make_output(outputs), LinearIdentityEpilogue{}, stream);
}
} // namespace

void launch_a16(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        using Gemv = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
        launch_fp8_a16_gemv<Fp8ScheduleInstance<Gemv, kK>>(fp8_a16_operands<float>(x, weight),
                                                           make_output(outputs),
                                                           LinearIdentityEpilogue{}, stream);
        return;
    }
    if (tokens <= 16) return sliced<Fp8SlicedInstance<16, 4, 1>>(x, weight, outputs, stream);
    if (tokens <= 32) return sliced<Fp8SlicedInstance<32, 4, 1>>(x, weight, outputs, stream);
    if (tokens <= 64)
        return mma<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, outputs, stream);
    if (tokens <= 128)
        return mma<Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>>(x, weight, outputs, stream);
    mma<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, outputs, stream);
}
} // namespace ninfer::ops::detail::linear_split::fp8_n3072_k2048
