#include "ops/linear/add/nvfp4/launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Full-attention o_proj of the 122B geometry (also reached with A16Only policy by the A16
// sanity artifact): 32 q heads x 256 head_dim contracted back to the 3072-wide hidden state.
// The schedule ladder mirrors the plain projection shape for the same geometry; only the
// epilogue adds the residual.
template <class S, int T = 0, bool Exact = false>
using Instance = Nvfp4ScheduleInstance<S, 8192, T, Exact>;

LinearAddLaunch select_a16(int tokens) {
    if (tokens == 1)
        return launch_nvfp4_add_gemv<Instance<Nvfp4A16GemvSchedule<
            8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>>;
    if (tokens == 2)
        return launch_nvfp4_add_simt<
            Instance<Nvfp4A16SimtSchedule<8, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                          Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                          Nvfp4SimtBlockOrder::RowsContiguous, 1>,
                     2, true>>;
    if (tokens <= 8) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<8, 8, 2>>>;
    if (tokens <= 16)
        return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<16, 8, 2>>>;
    if (tokens <= 24)
        return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<32, 4, 1>>>;
    if (tokens <= 32)
        return launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>>;
    if (tokens <= 64)
        return launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>>;
    if (tokens <= 128)
        return launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>>;
    return launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>>;
}
} // namespace

Nvfp4LinearAddPlan select_nvfp4_add_n3072_k8192(LinearPolicy policy, std::int32_t tokens) {
    if (!allows_a4(policy) || tokens < 17) return {select_a16(tokens)};
    if (tokens >= 1024)
        return {nullptr, launch_nvfp4_add_n3072_k8192_tma, Nvfp4ScaleLayout::Tiled256};
    if (tokens <= 64)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>>};
    if (tokens <= 96)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>>};
    if (tokens <= 128)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<128, 128, 256, 4, 4, 2, 1>>>};
    if (tokens <= 192)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>>>};
    if (tokens <= 384)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>>};
    if (tokens <= 512)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<128, 128, 256, 4, 4, 2, 1>>>};
    return {nullptr, launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>>};
}
} // namespace ninfer::ops::detail
