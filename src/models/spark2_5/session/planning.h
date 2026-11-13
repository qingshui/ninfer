#pragma once
#include "models/common/context/directory.h"
#include "models/spark2_5/execution_shape.h"

namespace ninfer::models::spark2_5 {

struct ExecutionPlan;

struct SessionPlan {
    ExecutionShape shape;
    context::ContextDirectorySpec context;
    std::uint32_t checkpoint_interval = 8192;

    bool operator==(const SessionPlan&) const = default;
};

// The single rank's State and full-layer KV pool schema derived from the finalized plan.
[[nodiscard]] SessionPlan plan_session(const ExecutionPlan&);
[[nodiscard]] runtime::ContextMetadataPlan session_metadata_plan(const ExecutionPlan&,
                                                                 const ContextPoolGeometry&);

} // namespace ninfer::models::spark2_5
