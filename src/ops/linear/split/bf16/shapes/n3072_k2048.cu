#include "ops/linear/split/dispatch.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/split/bf16/launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {

constexpr int kK = 2048;
using S0         = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 1, 2, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    kK>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 3>, kK>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 8, 4, 64, 2>, kK>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 24, 4, 64, 2>, kK>;
using S4 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    kK>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 8, 3, 1>, kK>;
using S6 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 32, 128, 32, 16, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    kK>;
using S7 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::TokenFast, 1>, kK>;
using S8 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, kK>;

template <class Routes>
auto select_route(int tokens, const Routes& routes) {
    if (tokens <= 1) return routes.template gemv<S0>();
    if (tokens <= 8) return routes.template sliced<S1>();
    if (tokens <= 32) return routes.template sliced<S2>();
    if (tokens <= 48) return routes.template sliced<S3>();
    if (tokens <= 64) return routes.template mma<S4>();
    if (tokens <= 96) return routes.template tma<S5>();
    if (tokens <= 128) return routes.template mma<S6>();
    if (tokens <= 512) return routes.template tma<S7>();
    return routes.template tma<S8>();
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    using Output = LinearBf16SegmentedOutput<2048, 512, 512>;
    const Output output{{static_cast<__nv_bfloat16*>(out[0].data),
                         static_cast<__nv_bfloat16*>(out[1].data),
                         static_cast<__nv_bfloat16*>(out[2].data)}};
    select_route(x.ne[1],
                 Bf16SplitRoutes<Output>{bf16_a16_operands(x, weights.first), output, stream});
}
} // namespace

const SplitShape kBf16SplitN3072K2048{3072, 2048, kSplitRows<2048, 512, 512>, launch_split};

} // namespace ninfer::ops::detail::linear_split
