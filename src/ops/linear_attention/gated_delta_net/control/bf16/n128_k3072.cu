#include "ops/linear_attention/gated_delta_net/control/kernels/mma_launch.cuh"

namespace ninfer::ops::detail::gdn_control {
void launch_n128_k3072(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                       CudaExecutionView e) {
    using Geometry = MmaGeometry<64, 3072, 64>;
    bool launched  = false;
    switch (plan.split_k) {
    case 16:
        launched = launch_mma<Geometry, 16>(in, out, scratch, e);
        break;
    case 8:
        launched = launch_mma<Geometry, 8>(in, out, scratch, e);
        break;
    case 4:
        launched = launch_mma<Geometry, 4>(in, out, scratch, e);
        break;
    case 2:
        launched = launch_mma<Geometry, 2>(in, out, scratch, e);
        break;
    case 1:
        launched = launch_mma<Geometry, 1>(in, out, {}, e);
        break;
    default:
        throw std::logic_error("gdn_control: invalid 128x3072 plan");
    }
    if (!launched) (void)launch_mma<Geometry, 1>(in, out, {}, e);
}
} // namespace ninfer::ops::detail::gdn_control
