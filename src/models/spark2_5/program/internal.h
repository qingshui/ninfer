#pragma once

#include "models/spark2_5/execution/text.h"
#include "models/spark2_5/program/graphs.h"
#include "models/spark2_5/program/program.h"

#include "ninfer/ops/sampling.h"
#include "runtime/observability/gpu_timing.h"

#include <deque>

namespace ninfer::models::spark2_5::detail {

enum class ComputePhase : std::uint8_t {
    Forward,
    AwaitForward,
    Decision,
    AwaitResult,
    Score,
    AwaitScore,
    PendingCommit,
    Retiring,
};

struct ComputeFrame {
    runtime::ExecutionRequest<ExecutionTypes> request;
    runtime::CommandSequence retirement;
    ComputePhase phase = ComputePhase::Forward;
    ComputeResult result;
    std::optional<runtime::DecisionInput> decision;
    bool decision_requested = false;
    // The window of a control row currently in flight; control rows run one window at a time.
    std::optional<WorkRow> window;
    std::uint32_t control_row = 0, control_offset = 0, tokens = 0;
    std::uint32_t score_at = 0, score_count = 0;
    std::vector<float> logprobs;
    // Final-norm hidden of the traversal: [hidden, T] for a sequence, [hidden, B] for decode.
    Tensor hidden;
    // Largest visible key count of the traversal; selects the attention envelope.
    std::uint32_t visible = 0;
    GpuExecutionObservation gpu;
    AttentionWorkCounters attention;
    bool sequence = false, timed_prefill = false, graph_round = false, head_ready = false;
    bool counts_pending = false;
};

struct ContextFrame {
    runtime::ExecutionRequest<ExecutionTypes> request;
    bool submitted = false;
};

class RankProgramImpl {
public:
    RankProgramImpl(const RankParameters&, ExecutionPlan, DeviceContext&,
                    const StartupObserver& observer = {});
    ~RankProgramImpl();

    void validate_work(const Work&, runtime::FrameKind) const;
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void prepare_commit(runtime::ExecutionTicket, const Commit&) const;
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] bool can_discard(runtime::ExecutionTicket) const;
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();

    GraphPreparationSample prepare_graphs(GraphPreparationMode);
    void release_graphs();

    void advance_compute();
    void advance_context();
    void forward();
    void finish_forward();
    void request_decision();
    void decide();
    void result();
    void score();
    void control_counts();
    void select_window();

    // Fills the host mirror of the inputs for `rows` of `tokens` columns each.
    void prepare_inputs(std::span<const WorkRow>, std::uint32_t tokens);
    void upload_inputs(std::uint32_t columns);
    void upload_sampling();
    // One traversal of the batch bound to the persistent inputs, then the hidden hand-off.
    void run_text(std::int32_t tokens, std::int32_t batch,
                  ops::CausalAttentionExecutionEnvelope, Tensor hidden);
    // A decode round of `batch` single-token rows: traversal, hidden hand-off and head. With
    // `captured` the stream timers are recorded as graph nodes.
    void decode_round(std::int32_t batch, ops::CausalAttentionExecutionEnvelope, bool captured,
                      bool head);
    void project_head(const Tensor& hidden);
    void sample(std::int32_t batch, bool first_token);
    void copy_result(std::int32_t batch);
    [[nodiscard]] TextInputs inputs(std::int32_t tokens, std::int32_t batch);
    // The decode envelope that holds `visible` keys.
    [[nodiscard]] std::size_t envelope_index(std::uint32_t visible) const;
    [[nodiscard]] Tensor row(const TensorRegion&, std::int32_t rows);
    [[nodiscard]] Tensor device_tensor(const TensorRegion&);
    [[nodiscard]] Tensor host_tensor(const TensorRegion&);
    [[nodiscard]] WorkContract contract() const;
    [[nodiscard]] RankExecutionObservation observation() const;
    [[nodiscard]] RankMemoryUsage memory() const;
    [[nodiscard]] RankMemoryBudget budget() const;
    void reset_peaks();

    const RankParameters& parameters;
    DeviceContext& device;
    CudaExecutionView cuda;
    Resources resources;
    const ExecutionShape shape;
    const std::vector<ops::CausalAttentionExecutionEnvelope> envelopes;
    std::unique_ptr<GraphInputs> graph_inputs;
    std::unique_ptr<Graphs> graphs;
    GraphMemorySummary graph_memory;
    std::unique_ptr<ComputeFrame> compute;
    std::unique_ptr<ContextFrame> context;
    std::array<std::optional<runtime::ExecutionMember>, kMaximumConcurrency> bindings;
    std::deque<runtime::ExecutionEvent<ExecutionTypes>> events;
    runtime::ExecutionGpuTimer timer;
    std::optional<CudaEventTimer> prefill_timer;
    CudaCompletionEvent compute_done, context_done;
    std::size_t lifetime_peak = 0;
    bool drained              = false;
};

[[nodiscard]] ops::SamplingConfig sampling_config(const ResolvedSamplingParameters&);

} // namespace ninfer::models::spark2_5::detail
