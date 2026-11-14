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

enum class GraphPhase : std::uint8_t { Main, Verify, Draft, Sample, Accept };

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
    // phase must run its eager path instead. Every batch of a captured phase replays at a bucket,
    // with its padding rows held inert by the zeroed host row state; the one exception is a
    // width-one Main graph over GDN layers, whose per-row State update has no row mask, so only
    // its exact bucket sizes are covered and other batches replay eager.
    [[nodiscard]] std::uint32_t captured_batch(GraphPhase, std::uint32_t batch) const;
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
    // Whether a width-one Main graph tolerates padded rows: without GDN layers its per-row update
    // is masked entirely by the zeroed host row state.
    bool pads_main_                        = false;
    std::vector<Batch> batches_;
    execution::TextGraph* pending_         = nullptr;
    std::size_t workspace_peak_            = 0;

    [[nodiscard]] std::uint32_t covering_bucket(std::uint32_t batch) const noexcept;
    [[nodiscard]] std::size_t bucket_index(std::uint32_t bucket) const noexcept;
};

[[nodiscard]] std::vector<GraphProfile> target_graph_profiles(std::uint32_t capacity,
                                                              std::uint32_t width);
[[nodiscard]] std::vector<GraphProfile> draft_graph_profiles(const GraphSpec&);

} // namespace ninfer::models::qwen3_5
