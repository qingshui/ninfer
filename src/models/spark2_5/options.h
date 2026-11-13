#pragma once

#include "ninfer/types.h"

#include <cstdint>

namespace ninfer::models::spark2_5 {

// FP8 E4M3 with a UE8M0 scale per 128 channels. BF16 does not hold the judged 8x256K workload
// beside the weights on a 32 GiB device.
inline constexpr KvCacheStorage kDefaultKvCacheStorage =
    KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
inline constexpr std::uint32_t kMaximumSupportedConcurrency = 16;

// An explicit user choice wins over the model default.
[[nodiscard]] inline KvCacheStorage resolve_kv_cache(const EngineOptions& options) noexcept {
    return options.kv_cache.value_or(kDefaultKvCacheStorage);
}

} // namespace ninfer::models::spark2_5
