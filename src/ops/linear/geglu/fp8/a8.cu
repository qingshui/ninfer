#include "ops/linear/geglu/dispatch.h"
#include "ops/linear/geglu/product_epilogue.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"

namespace ninfer::ops::detail {
namespace {
constexpr int kK            = 2048;
constexpr int kIntermediate = 6656;
using Tma64x128             = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma128x128            = Fp8A8TmaMmaSchedule<128, 128, 128, 4, 4, 2, 1, Fp8MmaRaster::RowFast>;
} // namespace

void launch_fp8_row_fp32_geglu_a8(const Tensor& x, const Weight& weight, Tensor& out,
                                  MemoryArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], kK);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    const auto operands = fp8_a8_operands<float>(weight, scratch, x.ne[1]);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const auto launch = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kK>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, GeGluTokenMajorMmaEpilogue{}, stream,
                                     scratch.partials, SwiGluTokenMajorMmaRows<S>{});
        else
            launch_fp8_a8_mma<S>(operands, output, GeGluTokenMajorMmaEpilogue{}, stream,
                                 SwiGluTokenMajorMmaRows<S>{});
    };
    if (x.ne[1] <= 16) return launch.template operator()<Fp8A8T16R64K128>();
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R128K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (x.ne[1] <= 128) return launch.template operator()<Tma64x128>();
    launch.template operator()<Tma128x128>();
}
} // namespace ninfer::ops::detail
