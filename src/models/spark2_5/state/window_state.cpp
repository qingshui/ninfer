#include "models/spark2_5/state/window_state.h"

#include "models/spark2_5/state/kv_cache.h"

namespace ninfer::models::spark2_5 {

CyclicKVCacheLayout plan_window_state(StateImageBuilder& builder, std::uint32_t sliding_layers) {
    return plan_cyclic_kv_cache(builder, sliding_layers, kWindowCapacity, kKvHeads,
                                kAttentionHeadDim);
}

} // namespace ninfer::models::spark2_5
