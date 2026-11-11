#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Eight output rows form one row group: warps split K and reduce within the CTA.
constexpr int kK = 2048;
using S0         = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 4, 8, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 1, 4>,
    kK>;
using S1 = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<4, 4, 8, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 4>,
    kK>;
} // namespace

Bf16Launch select_bf16_n8_k2048(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    return launch_bf16_simt<S1>;
}
} // namespace ninfer::ops::detail
