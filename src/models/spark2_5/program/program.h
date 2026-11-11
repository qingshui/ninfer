#pragma once

#include "models/common/graph_preparation.h"
#include "models/spark2_5/execution/parameters.h"
#include "models/spark2_5/profile.h"
#include "models/spark2_5/program/planning.h"
#include "models/spark2_5/program/resources.h"
#include "models/spark2_5/work.h"

namespace ninfer::models::spark2_5 {
namespace detail {
class RankProgramImpl;
}

// Owns one rank's physical execution resources. Work and commits carry already-resolved context
// identities; logical sequence, checkpoint and allocation ownership stays in Session.
class RankProgram {
public:
    RankProgram(const RankParameters&, ExecutionPlan, DeviceContext&,
                const StartupObserver& observer = {});
    ~RankProgram();
    RankProgram(const RankProgram&)            = delete;
    RankProgram& operator=(const RankProgram&) = delete;

    [[nodiscard]] static constexpr runtime::ExecutionCapacity capacity() noexcept { return {1, 1}; }

    GraphPreparationSample prepare_execution(GraphPreparationMode = GraphPreparationMode::Complete);
    void release_execution();
    void validate_work(const Work&, runtime::FrameKind) const;
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void prepare_commit(runtime::ExecutionTicket, Commit&) const;
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] bool can_discard(runtime::ExecutionTicket) const;
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();
    [[nodiscard]] RankMemoryUsage observe_resources() const;
    [[nodiscard]] std::optional<GraphMemorySummary> graph_memory_observation() const;
    [[nodiscard]] RankMemoryBudget resource_budget() const;
    void reset_measurement_peaks() noexcept;
    // Spark2_5 has no per-request input payloads.
    [[nodiscard]] MemorySpan input_storage() { return {}; }

private:
    std::unique_ptr<detail::RankProgramImpl> impl_;
};

} // namespace ninfer::models::spark2_5
