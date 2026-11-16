#pragma once

#include "runtime/observability/gpu_timing.h"

#include "ninfer/ops/kv_cache_append.h"

#include "models/common/context/transfer.h"
#include "models/common/qwen/vision/vision.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/program/resources.h"
#include "models/qwen3_5/program/graphs.h"
#include "models/qwen3_5/program/planning/startup.h"

#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct Residency {
    runtime::ResidencyId id;
    runtime::SequenceId sequence;
    bool installed = false;
    bool penalties = false;
};

enum class ComputeStage : std::uint8_t {
    Inputs,
    Forward,
    History,
    AwaitDecision,
    Readback,
    AwaitCommit,
    Commit,
    Retiring,
};

// A State image that receives the sequence State at a window column. Checkpoints inside a packed
// traversal are written by the segment Ops; `end` marks the one at the window end, which copies
// the row's destination image once the traversal and its continuation hidden are complete.
// Every checkpoint image also receives the continuation hidden of its column.
struct PendingCheckpoint {
    std::int64_t source = -1;       // destination image of the row (end checkpoints)
    std::int64_t destination = -1;  // checkpoint image
    std::uint32_t hidden_column = 0; // token-axis column of the last token before the checkpoint
    bool end = false;
};

struct ComputeOperation {
    runtime::ExecutionRequest<ExecutionTypes> request;
    runtime::CommandSequence last_command;
    std::vector<ExecutionRow> rows;
    std::vector<SamplingInput> sampling;
    std::uint32_t width          = 0;
    std::uint32_t batch          = 0;
    std::uint32_t drafts         = 0;
    std::uint32_t forward_row    = 0;
    std::uint32_t forward_offset = 0;
    std::uint32_t score_cursor   = 0;
    std::uint32_t score_count    = 0;
    ComputeStage stage           = ComputeStage::Inputs;
    std::uint32_t decision_stage = 0;
    bool decision_requested      = false;
    bool decision_supplied       = false;
    bool result_published        = false;
    bool cancelled               = false;
    bool speculative             = false;
    bool row_finished            = false;
    // Prefill rows share one TextRun over the summed token axis; `segments` lists them in row
    // order and `width` is their total length.
    bool packed        = false;
    // Prefill segments and decode rows share one token axis [prefill segments | verify block].
    // Decode rows occupy compact rows [0, batch) and segments the rows after them; `width` is the
    // verify width, `prefill_tokens` the column where the verify block starts and `axis_tokens`
    // the extent of the whole token axis.
    bool mixed                       = false;
    std::uint32_t prefill_tokens     = 0;
    std::uint32_t axis_tokens        = 0;
    bool graph_started = false;
    bool projected     = false;
    // Dead-position bitmap of constrained decode rows; one entry per lane of the resolved
    // concurrency.
    std::vector<std::uint32_t> dead_positions;
    // Operation row of every decision row of the pending batch, in row order: the identity of a
    // decode flight, the decode rows of a mixed one followed by its final segments, and the
    // producing segments of a packed prefill. Empty until sampling begins.
    std::vector<std::uint32_t> sample_members;
    std::optional<runtime::DecisionInput> decision;
    std::optional<Commit> commit;
    std::optional<Result> result;
    std::vector<float> scores;
    GpuExecutionObservation gpu;
    AttentionWorkCounters attention;
    std::optional<Tensor> input_embeddings;
    std::vector<execution::SequenceSegment> segments;
    // Boundaries (execution splits) of the packed segments, written to the segment table.
    std::uint32_t boundaries = 0;
    std::vector<PendingCheckpoint> checkpoints;
    std::unique_ptr<execution::TextRun> text;

    ComputeOperation(runtime::ExecutionRequest<ExecutionTypes>&&, std::uint32_t lanes);
    ~ComputeOperation();
};

// The decode rows of a decode or mixed operation.
[[nodiscard]] inline const DecodeWork& decode_part(const ComputeOperation& operation) {
    if (const auto* mixed = std::get_if<MixedWork>(&operation.request.work.operation))
        return mixed->decode;
    return std::get<DecodeWork>(operation.request.work.operation);
}

[[nodiscard]] inline bool has_decode(const ComputeOperation& operation) {
    return std::holds_alternative<DecodeWork>(operation.request.work.operation) || operation.mixed;
}

struct ContextOperation {
    runtime::ExecutionRequest<ExecutionTypes> request;
    runtime::CommandSequence last_command;
};

// Everything one compute frame owns beside its Resources storage: the flight on the frame, the
// context actions of that flight, the completion events and timers its stages record, and the
// CUDA graphs captured against the frame's tables and workspace. Flights on different slots
// share no host-written state and no replay pointer.
struct ComputeSlot {
    std::unique_ptr<ComputeOperation> compute;
    std::unique_ptr<context::ContextMigration> compute_actions;
    CudaCompletionEvent result_ready;
    CudaCompletionEvent drafts_ready;
    CudaCompletionEvent compute_retired;
    CudaEventTimer prefill_timer;
    runtime::ExecutionGpuTimer gpu_timer;
    std::optional<Graphs> graphs;

    ComputeSlot(DeviceContext&, context::ContextStorage&, bool gpu_timing);
};

class RankProgramImpl {
public:
    RankProgramImpl(const execution::Parameters&, const SequencePlanImpl&, DeviceContext&,
                    const StartupObserver&);
    ~RankProgramImpl();

    void validate_work(const Work&, runtime::FrameKind) const;
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void prepare_commit(runtime::ExecutionTicket, Commit&) const;
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();
    [[nodiscard]] RankMemoryUsage observe_resources() const;
    [[nodiscard]] RankExecutionObservation retirement_observation() const noexcept;
    void reset_measurement_peaks() noexcept;

    void validate_row(const ExecutionRow&, std::span<const ResidencyBinding> installs = {}) const;
    void validate_checkpoints(const InputWindow&, std::vector<std::int64_t>& images) const;
    void install_residencies(const ContextWork&);
    // Makes one residency current and uploads its token counts on `stream`.
    void install_binding(const ResidencyBinding&, cudaStream_t stream);
    // Writes the segment table of the prefill windows listed in `operation.segments` into the
    // host ingress: token offsets, and the boundaries and checkpoints of the windows' splits.
    void bind_segment_table(ComputeOperation&, const ForwardWork&);
    // The device view of that table in the selected frame; the State offsets of segment i are
    // the ingress row `first_row + i`.
    [[nodiscard]] execution::SegmentTable segment_table(std::uint32_t first_row,
                                                        std::uint32_t segments,
                                                        std::uint32_t boundaries) const;
    void publish_checkpoints(ComputeOperation&);
    void initialize_inputs(ComputeOperation&);
    void initialize_decode(ComputeOperation&, const DecodeWork&);
    void initialize_mixed(ComputeOperation&, const MixedWork&);
    void validate_mixed(const MixedWork&) const;
    void initialize_forward(ComputeOperation&, const ForwardWork&);
    void initialize_packed_forward(ComputeOperation&, const ForwardWork&);
    void validate_packed_forward(const ForwardWork&) const;
    void publish_continuation_hidden(ComputeOperation&);
    void initialize_score(ComputeOperation&, const ScoreWork&);
    void upload_input(std::uint32_t width, std::uint32_t batch);
    void upload_positions(std::uint32_t width, std::uint32_t batch);
    void upload_ingress();
    void begin_target(ComputeOperation&);
    void begin_mixed_target(ComputeOperation&);
    // Classifies one prefill traversal against the prebuilt prefill plans: replays the plan when
    // the flight is exactly the full-chunk shape of a built bucket (true), otherwise records the
    // eager fallback (false). Both outcomes are counted in `prefill_plan`.
    [[nodiscard]] bool replay_prefill_plan(ComputeOperation&);
    bool advance_target(ComputeOperation&);
    void begin_draft(ComputeOperation&);
    void begin_mtp_draft(ComputeOperation&);
    void finish_mixed_forward(ComputeOperation&);
    void prepare_mtp_draft(ComputeOperation&);
    void run_mtp_draft(std::uint32_t batch, std::uint32_t steps, std::uint32_t frontier);
    void run_dflash_draft(std::uint32_t batch, std::uint32_t minimum, std::uint32_t maximum);
    // `column` is where the rows' hidden starts on the token axis: zero for a decode execution,
    // the summed prefill length for a mixed one.
    void project_decode(std::uint32_t batch, std::uint32_t width, std::uint32_t column = 0);
    void decide_decode(std::uint32_t batch, std::uint32_t width);
    // Samples the producing segments of a packed prefill or a mixed execution: gathers their
    // last hidden columns, projects them and samples one decision row each into the egress rows
    // [first_row, first_row + count).
    void sample_forward_rows(ComputeOperation&, std::uint32_t first_row);
    [[nodiscard]] std::uint32_t forward_row_count(const ComputeOperation&) const noexcept;
    void readback_decode(std::uint32_t batch, bool speculative);
    GraphPreparationSample initialize_graphs(GraphPreparationMode = GraphPreparationMode::Complete);
    void prepare_frame_graphs(GraphPreparation&);
    void release_graphs();
    void append_draft_history(ComputeOperation&);
    void append_draft_history(std::uint32_t batch, ops::KVCacheAppendPrefixExecutionEnvelope);
    void commit_draft_history(ComputeOperation&);
    void align_mtp_history(ComputeOperation&);
    void align_packed_mtp_history(ComputeOperation&);
    void rebuild_mtp_history(std::uint32_t batch, std::uint32_t column = 0);
    void execute_mtp_commit(std::uint32_t batch, bool repair, bool counts,
                            DecodeGraphSection* repair_section = nullptr,
                            DecodeGraphSection* counts_section = nullptr,
                            std::uint32_t column               = 0);
    void request_decision(ComputeOperation&);
    void launch_decision(ComputeOperation&);
    void launch_commit(ComputeOperation&);
    void launch_score(ComputeOperation&);
    void publish_result(ComputeOperation&);
    void retire_compute();
    void progress();
    // Advances the flight of one compute slot as far as its host-visible stages allow.
    void progress_compute(std::uint32_t frame);
    // The flight a ticket names; the non-const form also selects the ticket's compute frame, so
    // every member the flight touches (frame tables, workspace, events, graphs) is its own.
    [[nodiscard]] ComputeOperation& require_compute(runtime::ExecutionTicket);
    [[nodiscard]] const ComputeOperation& require_compute(runtime::ExecutionTicket) const;
    // Binds the compute frame that Resources::frame()/work() and slot() address.
    void select_frame(std::uint32_t frame);
    [[nodiscard]] ComputeSlot& slot() noexcept { return *slots[resources.selected()]; }
    [[nodiscard]] const ComputeSlot& slot() const noexcept { return *slots[resources.selected()]; }
    [[nodiscard]] bool computing() const noexcept;
    [[nodiscard]] ops::SamplingConfig sampling_config(const ExecutionRow&, const SamplingInput&,
                                                      std::size_t compact_row) const;

    const execution::Parameters& parameters;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend backend;
    const ProposalHead proposal_head;
    const bool use_cuda_graph;
    const KvCacheStorage kv_storage;
    const WorkspaceSpec workspace_plan;
    // Compute frames of the rank: the executor reserves a flight on one of them and the ticket's
    // frame index names its slot and its storage in `resources`.
    const std::uint32_t compute_frames;
    Resources resources;
    std::unique_ptr<context::ContextStorage> storage;
    std::unique_ptr<context::ContextMigration> context_actions;
    std::unique_ptr<ContextOperation> context_operation;
    PinnedHostBuffer context_counts;
    // One residency slot per lane of the resolved concurrency.
    std::vector<Residency> residencies;
    std::size_t workspace_peak = 0;
    std::vector<std::unique_ptr<ComputeSlot>> slots;
    CudaCompletionEvent context_ready;
    std::deque<runtime::ExecutionEvent<ExecutionTypes>> events;
    // Cumulative prefill launch-plan outcomes of this rank (see PrefillPlanCounters).
    PrefillPlanCounters prefill_plan;
    std::optional<GraphMemorySummary> graph_memory;
    std::size_t workspace_lifetime_peak = 0;
};

} // namespace ninfer::models::qwen3_5::detail
