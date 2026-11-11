#pragma once

#include "models/common/graph.h"
#include "models/spark2_5/program/planning.h"

#include "core/context_pool.h"

#include <vector>

namespace ninfer::models::spark2_5::detail {
class RankProgramImpl;

// Context records that stand in for live sequences while graphs are prepared: one State image
// and one KV page per decode row, with every logical page of the row's table naming that page.
class GraphInputs {
public:
    explicit GraphInputs(RankProgramImpl&);
    ~GraphInputs();
    void prepare();
    // Fills the persistent inputs of a `batch`-row decode whose longest history has `visible`
    // keys, and uploads them.
    void set(std::uint32_t batch, std::uint32_t visible);

private:
    RankProgramImpl& owner_;
    std::unique_ptr<ContextPool> records_;
    std::vector<std::int64_t> states_;
    std::vector<AllocationId> allocations_;
    PinnedHostBuffer mapping_;
};

[[nodiscard]] std::size_t graph_context_bytes(const ExecutionPlan&);

// Whole decode rounds, one executable per batch size and attention envelope, plus one sampling
// graph per batch size. Replay selects by the batch maximum of visible keys.
class Graphs {
public:
    explicit Graphs(RankProgramImpl&);
    void prepare(GraphInputs&, GraphPreparation&);
    // Traversal, hidden hand-off and head; leaves logits in the persistent buffer.
    void replay(std::uint32_t batch, std::uint32_t visible);
    // Sampling and result copy.
    void decide(std::uint32_t batch);

    [[nodiscard]] std::size_t workspace_peak() const { return workspace_peak_; }

    void reset_workspace_peak() { workspace_peak_ = 0; }

private:
    struct Batch {
        std::vector<models::GraphSegment> rounds;
        models::GraphSegment decision;
    };

    [[nodiscard]] std::size_t index(std::uint32_t visible) const;
    RankProgramImpl& owner_;
    std::vector<std::uint32_t> bounds_;
    std::vector<Batch> batches_;
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models::spark2_5::detail
