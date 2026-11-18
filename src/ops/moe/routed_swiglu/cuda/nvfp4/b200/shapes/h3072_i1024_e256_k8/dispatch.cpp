#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select_plan(const SwiGLUProfile&, int tokens, int) {
    Plan plan;
    plan.implementation = Implementation::BlockScaled;
    const int columns   = tokens <= 16 ? 8 : tokens <= 128 ? 32 : 64;
    plan.block_scaled   = {.format                = QType::NVFP4,
                           .intermediate          = 1024,
                           .hidden                = 3072,
                           .experts               = 256,
                           .top_k                 = 8,
                           .gate_columns          = tokens >= 3584 ? 128 : columns,
                           .down_columns          = columns,
                           .vector                = tokens == 1,
                           .single_block_grouping = tokens <= 64,
                           .shared_algorithm      = tokens < 8
                                                        ? kernel::block_scaled::SharedAlgorithm::Vector
                                                        : kernel::block_scaled::SharedAlgorithm::Mma,
                           .shared_columns        = tokens >= 512 ? 64 : 32,
                           .queued_ctas_per_sm    = 6};
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h3072_i1024_e256_k8() {
    static constexpr int boundaries[]{2, 17, 65, 128, 129, 3584};
    static const ShapePolicy policy{.select                = select_plan,
                                    .allocation_boundaries = boundaries,
                                    .all_slice_tokens      = 8192,
                                    .masked_slice_tokens   = 8192};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
