#pragma once

#include "ninfer/types.h"

namespace ninfer::models::spark2_5 {

// Values every layer of the Spark2_5 runtime derives from EngineOptions. The model never
// speculates, so the shape carries no draft width.
struct ExecutionShape {
    std::uint32_t max_context, max_concurrency, prefill_chunk;
    KvCacheStorage kv_storage;
    EnginePurpose purpose;
    bool context_cache = false;
    bool cuda_graph    = false;
    bool gpu_timing    = false;

    bool operator==(const ExecutionShape&) const = default;
};

} // namespace ninfer::models::spark2_5
