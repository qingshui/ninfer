#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"

namespace ninfer::ops::detail::moe_complete {
namespace {
// Decode sizes split the mma.sync control and shared gate/up product over K: 8 splits keep about
// 300-600 CTAs in flight while a call has one or two 32-token column tiles, 4 once it has three
// or four. From 128 tokens the dense tcgen05 GEMMs take the shared gate/up product.
int shared_splits(int tokens) {
    if (tokens < 8 || tokens >= 128) return 1;
    return tokens <= 64 ? 8 : 4;
}

Plan select_plan(int tokens, int, ExpertSubsetKind) {
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::BlockScaled;
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
                           // The sorting network keeps a token's ranked run in registers; the
                           // insertion order indexes it and spills it to local memory.
                           .network_topk          = true,
                           .shared_columns        = tokens >= 512 ? 64 : 32,
                           .shared_splits         = shared_splits(tokens),
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
} // namespace ninfer::ops::detail::moe_complete
