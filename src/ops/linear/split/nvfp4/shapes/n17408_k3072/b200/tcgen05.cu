#include "ops/linear/split/nvfp4/shapes/n17408_k3072/launch.h"
#include "ops/linear/kernels/sm100/gemm.h"

#include <array>

namespace ninfer::ops::detail::linear_split::nvfp4_n17408_k3072 {

// Full-attention qkvz of the 122B geometry, cut into {8192, 512, 8192, 512}. B200 quantizes the
// activation into the K16M128x4 atom layout the tcgen05 GEMM reads and writes each segment
// straight into its output tensor.
void launch_a4(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
               Nvfp4A4Workspace workspace, cudaStream_t stream) {
    constexpr auto kLayout = Nvfp4ScaleLayout::K16M128x4;
    launch_nvfp4_a4_quantize(x, weight, workspace, kLayout, stream);
    const std::array<__nv_bfloat16*, 4> output = {static_cast<__nv_bfloat16*>(outputs[0].data),
                                                  static_cast<__nv_bfloat16*>(outputs[1].data),
                                                  static_cast<__nv_bfloat16*>(outputs[2].data),
                                                  static_cast<__nv_bfloat16*>(outputs[3].data)};
    Sm100Epilogue epilogue;
    epilogue.kind     = Sm100Epilogue::Split;
    epilogue.segments = kSplitRows<8192, 512, 8192, 512>;
    epilogue.outputs  = output;
    launch_sm100_nvfp4_gemm(nvfp4_a4_operands(weight, workspace, x.ne[1], kLayout), epilogue,
                            nullptr, stream);
}

} // namespace ninfer::ops::detail::linear_split::nvfp4_n17408_k3072
