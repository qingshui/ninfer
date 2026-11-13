#include "ops/linear/split/nvfp4/shapes/n17408_k3072/launch.h"

namespace ninfer::ops::detail::linear_split::nvfp4_n17408_k3072 {
namespace {
bool quantized(LinearPolicy policy, int columns) { return allows_a4(policy) && columns >= 4; }

std::size_t capacity(LinearPolicy policy, std::int32_t, std::int32_t last) {
    return quantized(policy, last) ? nvfp4_a4_workspace_capacity_bytes(last, 3072) : 0;
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    if (!quantized(weights.policy, x.ne[1])) {
        launch_a16(x, weights.first, outputs, stream);
        return;
    }
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], 3072);
    launch_a4(x, weights.first, outputs, scratch, stream);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::nvfp4_n17408_k3072

namespace ninfer::ops::detail::linear_split {
const SplitShape kNvfp4SplitN17408K3072{17408, 3072, kSplitRows<8192, 512, 8192, 512>,
                                        nvfp4_n17408_k3072::launch, nvfp4_n17408_k3072::capacity};
} // namespace ninfer::ops::detail::linear_split
