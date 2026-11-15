#pragma once

#include "core/decode_graph.h"
#include "models/common/graph_preparation.h"
#include "models/qwen3_5/program/resources.h"
#include "models/qwen3_5/program/text_graph.h"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5 {

// Prefill is the phase of the prebuilt prefill launch plans; it is prepared through PrefillPlanCache
// and replayed through Graphs::start_prefill, never through the batch-bucketed start().
enum class GraphPhase : std::uint8_t { Main, Verify, Draft, Sample, Accept, Prefill };

struct GraphProfile {
    std::uint32_t min                                                = 0;
    std::uint32_t max                                                = 0;
    std::uint32_t topology_class                                     = 0;
    friend bool operator==(const GraphProfile&, const GraphProfile&) = default;
};

struct GraphInvocation {
    GraphPhase phase;
    std::uint32_t batch;
    std::uint32_t width;
    GraphProfile profile;
    ops::CausalAttentionExecutionEnvelope envelope;
};

struct GraphSpec {
    std::uint32_t capacity       = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    SpeculativeBackend backend   = SpeculativeBackend::None;
    // Prefill plan geometry: one plan per prefill_plan_segment_counts(prefill_chunk, pack_tokens,
    // batch_capacity). Zero staging State rows (Resources) mean no plan is prepared regardless.
    std::uint32_t prefill_chunk  = 0;
    std::uint32_t pack_tokens    = 0;
};

// Borrowed only during preparation. Bodies operate on the fixed execution frame;
// they do not acquire contexts, publish results or alter a Session.
struct GraphBodies {
    std::function<void(const GraphInvocation&)> prologue;
    std::function<std::unique_ptr<execution::TextRun>(const GraphInvocation&)> text;
    std::function<void(const GraphInvocation&)> project;
    std::function<void(const GraphInvocation&)> draft;
    std::function<void(const GraphInvocation&)> decision;
    std::function<void(const GraphInvocation&)> readback;
    std::function<void(std::uint32_t, DecodeGraphSection*, DecodeGraphSection*)> mtp_commit;
    // The packed text traversal of one prefill plan: `segments` are the plan's canonical segments
    // (borrowed for the traversal), each full-chunk window bound to its staging State row.
    std::function<std::unique_ptr<execution::TextRun>(const GraphInvocation&,
                                                      std::span<const execution::SequenceSegment>)>
        prefill_text;
};

// Key of one prefill launch plan: the phase and the summed token width of the flight it covers.
enum class PrefillPhase : std::uint8_t { Text };

struct PrefillPlanKey {
    PrefillPhase phase = PrefillPhase::Text;
    std::uint32_t width = 0;
    friend bool operator==(const PrefillPlanKey&, const PrefillPlanKey&) = default;
};

// Prebuilt prefill launch plans keyed by (phase, width bucket). A plan is the captured Main
// traversal of a flight of `segments` full-chunk windows: the weights views, kernel parameters,
// workspace layout and attention profiles are fixed at capture, and a replay patches only the
// per-flight device inputs (frame row tables, positions, ids) and the staging State rows its
// segments run on. The buckets are the prefill_plan_segment_counts widths; plans are built while
// the graphs are prepared (a capture is a startup resource decision), so a lookup that finds none
// is a miss the program reports and serves through the eager traversal.
class PrefillPlanCache {
public:
    struct Plan {
        PrefillPlanKey key;
        std::uint32_t segments = 0;
        execution::TextGraph graph;
    };

    // Bucket widths for the geometry, ascending; empty when no plan is possible.
    [[nodiscard]] static std::vector<std::uint32_t>
    bucket_widths(std::uint32_t prefill_chunk, std::uint32_t pack_tokens, std::uint32_t lanes);

    // The prebuilt plan of the bucket, or null when none was built.
    [[nodiscard]] Plan* acquire(PrefillPhase, std::uint32_t width) noexcept;
    [[nodiscard]] const Plan* acquire(PrefillPhase, std::uint32_t width) const noexcept;
    [[nodiscard]] Plan& build(PrefillPlanKey key, std::uint32_t segments);
    [[nodiscard]] std::size_t size() const noexcept { return plans_.size(); }
    void clear() noexcept { plans_.clear(); }

private:
    std::vector<std::unique_ptr<Plan>> plans_;
};

// Capture bucket sequence of one batch capacity: the powers of two up to the capacity, then the
// capacity itself (not repeated when it is a power of two). Graphs are captured once per bucket
// and replayed for every batch a bucket covers.
[[nodiscard]] std::vector<std::uint32_t> graph_capture_buckets(std::uint32_t batch_capacity);

class Graphs {
public:
    explicit Graphs(GraphSpec spec);
    void prepare(Resources&, const execution::Parameters&, DeviceContext&, const GraphBodies&,
                 GraphPreparation&);
    [[nodiscard]] bool start(GraphPhase, std::uint32_t batch, std::uint32_t frontier,
                             CudaExecutionView);
    // Smallest captured bucket whose graph can replay `batch` rows of the phase; zero when the
    // phase must run its eager path instead. Every batch of a captured phase replays at its
    // covering bucket: the padding rows stay inert through the zeroed host row state, and their
    // GDN State destinations point at the scratch State row (see ExecutionFrame and
    // StateImageViews::scratch_row), so no replay writes a pooled record.
    [[nodiscard]] std::uint32_t captured_batch(GraphPhase, std::uint32_t batch) const;
    // Width of the prebuilt prefill plan that replays a flight of `width` summed tokens; zero when
    // no plan was built for that width (no captured bucket: the flight runs eagerly). The caller
    // owns the shape test: a flight replays only when its segments are exactly the full-chunk
    // windows the plan was captured for.
    [[nodiscard]] std::uint32_t captured_prefill_width(std::uint32_t width) const;
    // Replays the plan of `width` at the profile that covers `frontier`, the largest first-token
    // position of its segments. Returns true once the whole traversal is submitted.
    [[nodiscard]] bool start_prefill(std::uint32_t width, std::uint32_t frontier,
                                     CudaExecutionView);
    [[nodiscard]] bool poll();
    void cancel();
    void clear();
    void commit_mtp(std::uint32_t batch, bool repair, bool counts, CudaExecutionView);

    [[nodiscard]] std::size_t workspace_peak() const noexcept { return workspace_peak_; }

private:
    struct SegmentProfile {
        GraphProfile profile;
        DecodeGraphDefinition definition;
    };

    struct SegmentTopology {
        std::uint32_t id      = 0;
        std::size_t installed = 0;
        DecodeGraphExecutable executable;
    };

    struct SegmentFamily {
        std::vector<SegmentProfile> profiles;
        std::vector<SegmentTopology> topologies;
        void instantiate(CudaExecutionView);
        void launch(std::uint32_t frontier, CudaExecutionView);
    };

    struct Batch {
        execution::TextGraph main, verify;
        SegmentFamily draft, sample, accept;

        struct MtpCommit {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            DecodeGraphSection repair, counts;
        };

        MtpCommit mtp_commit;
    };

    GraphSpec spec_;
    // One entry per capture bucket, in graph_capture_buckets order; start and commit_mtp resolve
    // a batch into its covering bucket before indexing it.
    std::vector<std::uint32_t> buckets_;
    std::vector<Batch> batches_;
    PrefillPlanCache prefill_;
    execution::TextGraph* pending_         = nullptr;
    std::size_t workspace_peak_            = 0;

    [[nodiscard]] std::uint32_t covering_bucket(std::uint32_t batch) const noexcept;
    [[nodiscard]] std::size_t bucket_index(std::uint32_t bucket) const noexcept;
};

[[nodiscard]] std::vector<GraphProfile> target_graph_profiles(std::uint32_t capacity,
                                                              std::uint32_t width);
[[nodiscard]] std::vector<GraphProfile> draft_graph_profiles(const GraphSpec&);

} // namespace ninfer::models::qwen3_5
