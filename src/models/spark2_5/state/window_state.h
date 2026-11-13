#pragma once

#include "core/cyclic_kv_cache.h"

#include <cstdint>

namespace ninfer::models::spark2_5 {

// A sliding layer attends the 512 most recent positions (distance <= 511), so its history is a
// ring of 512 slots with absolute position p at slot p % 512.
inline constexpr std::uint32_t kWindowCapacity = 512;

// Declares the rings of all sliding layers in a StateImage: K BF16 and V FP16, [256, 512, 2]
// per layer. The caller finishes the builder and owns record placement.
[[nodiscard]] CyclicKVCacheLayout plan_window_state(StateImageBuilder& builder,
                                                    std::uint32_t sliding_layers);

} // namespace ninfer::models::spark2_5
