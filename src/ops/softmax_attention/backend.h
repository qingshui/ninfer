#pragma once

// Which execution engine serves a causal softmax attention call. The decision belongs to the
// attention Op: dispatch and plan code ask here instead of keeping their own list of tcgen05
// geometries.

#include "core/platform.h"
#include "ninfer/kv_cache_format.h"

#include <cstdint>

namespace ninfer::ops::detail {

// KV-cache representation as the attention kernels see it (K and V together).
enum class KvDType : std::uint8_t { Bf16, Int8, Fp8, K8V4, Nvfp4 };

enum class AttentionBackend { Tcgen05, MmaSync };

constexpr KvDType kv_dtype(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return KvDType::Bf16;
    case KvCacheStorage::Int8Group64:
        return KvDType::Int8;
    case KvCacheStorage::Fp8E4m3fnGroup128Ue8m0:
        return KvDType::Fp8;
    case KvCacheStorage::Nvfp4Group16:
        return KvDType::Nvfp4;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return KvDType::K8V4;
    }
    return KvDType::Bf16;
}

// Tcgen05 serves the Qwen3.5-122B full-attention geometry (32 query heads, 2 KV heads, D256) on
// B200 for every KV dtype, prefill and decode. Other geometries and targets run mma.sync.
constexpr AttentionBackend attention_backend(KvDType kv, int q_heads, int kv_heads, int head_dim) {
#if NINFER_TARGET_B200
    (void)kv;
    if (q_heads == 32 && kv_heads == 2 && head_dim == 256) return AttentionBackend::Tcgen05;
#else
    (void)kv;
    (void)q_heads;
    (void)kv_heads;
    (void)head_dim;
#endif
    return AttentionBackend::MmaSync;
}

} // namespace ninfer::ops::detail
