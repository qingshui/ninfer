#include "ops/linear/split/fp8/shapes/n3072_k2048/launch.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail::linear_split::fp8_n3072_k2048 {
namespace {
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma128x128 = Fp8A8TmaMmaSchedule<128, 128, 128, 4, 4, 2, 1, Fp8MmaRaster::RowFast>;
} // namespace

void launch_a8(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
               Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const LinearBf16SegmentedOutput<2048, 512, 512> output{
        {static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
         static_cast<__nv_bfloat16*>(outputs[2].data)}};
    const auto operands = fp8_a8_operands<float>(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, 2048>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                     workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    };
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (x.ne[1] <= 96) return launch.template operator()<Fp8A8T32R128K128>();
    if (x.ne[1] <= 128) return launch.template operator()<Tma64x128>();
    launch.template operator()<Tma128x128>();
}
} // namespace ninfer::ops::detail::linear_split::fp8_n3072_k2048
