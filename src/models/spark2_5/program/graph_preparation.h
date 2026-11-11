#pragma once

#include "models/common/graph_preparation.h"
#include "models/spark2_5/profile.h"

namespace ninfer::models::spark2_5 {

// What a CUDA graph calibration measurement depends on. Two options with equal plans share one
// measurement.
struct GraphPreparationPlan {
    bool enabled                   = false;
    std::uint32_t max_context      = 0;
    std::uint32_t max_concurrency  = 0;
    std::uint32_t prefill_chunk    = 0;
    KvCacheStorage kv_cache{};

    friend bool operator==(const GraphPreparationPlan&, const GraphPreparationPlan&) = default;
};

struct GraphCalibration {
    GraphPreparationPlan plan;
    GraphMemorySummary memory;
};

[[nodiscard]] GraphPreparationPlan describe_graph_preparation(const EngineOptions&);
[[nodiscard]] GraphCalibration calibrate_graphs(const RankParameters&, DeviceContext&,
                                                const EngineOptions&);

} // namespace ninfer::models::spark2_5
