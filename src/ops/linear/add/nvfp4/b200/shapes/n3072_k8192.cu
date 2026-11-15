#include "ops/linear/add/nvfp4/launch.cuh"
#include "ops/linear/kernels/sm100/gemm.h"
#include "ops/linear/backend.h"

namespace ninfer::ops::detail {
namespace {
// Full-attention o_proj of the 122B geometry with the residual added in the epilogue. Quantized
// calls run on tcgen05 at every token count (the activation scales use the K16M128x4 atom layout
// that kernel reads); the mma.sync ladder below serves A16-only policy calls.
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

void launch_a4(const Tensor& x, const Weight& w, Tensor& out, Nvfp4A4Workspace scratch,
               cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(out.data);
    Sm100Epilogue epilogue;
    epilogue.kind     = Sm100Epilogue::Add;
    epilogue.residual = data;
    launch_sm100_nvfp4_gemm(nvfp4_a4_operands(w, scratch, x.ne[1], Nvfp4ScaleLayout::K16M128x4),
                            epilogue, data, stream);
}
} // namespace

Nvfp4LinearAddPlan select_nvfp4_add_n3072_k8192(LinearPolicy policy, std::int32_t tokens) {
    if (!allows_a4(policy) ||
        linear_backend(QType::NVFP4, 3072, 8192, tokens) != LinearBackend::Tcgen05)
        return {select_a16(tokens)};
    return {nullptr, launch_a4, Nvfp4ScaleLayout::K16M128x4};
}
} // namespace ninfer::ops::detail
