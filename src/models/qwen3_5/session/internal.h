#pragma once

#include "models/qwen3_5/session/session.h"
#include "models/qwen3_5/state/continuation.h"
#include "models/common/context/snapshot.h"
#include "models/common/context/work.h"
#include "models/common/qwen/prefix_identity.h"
#include "models/common/qwen/recovery_store.h"
#include "models/common/qwen/vision/control.h"

#include "core/platform.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// TEMPDEBUG: opt-in maintenance lifecycle trace for serve-pressure diagnosis.
inline void maintenance_trace(const char* format, ...) noexcept {
    static const bool enabled = std::getenv("NINFER_DEBUG_MAINT") != nullptr;
    if (!enabled) return;
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

class SessionImpl;
struct PendingContext;

struct PreparedCaptureBacking {
    qwen::PrefixShortlistDigests digests;
    std::vector<TokenId> ledger;
    qwen::ResidentPrefixIdentity prefix_identity;
};

// A prompt frontier a flight may checkpoint.
struct CheckpointCandidate {
    std::shared_ptr<const PreparedCaptureBacking> identity;
    PrefixShortlistKey key;
    std::uint32_t frontier           = 0;
    SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
};

// Whether a packed or mixed traversal can carry a window that crosses a prefix execution
// boundary. The varlen Ops split such a window; a target without them runs it alone.
inline constexpr bool kSplitPacking = NINFER_TARGET_SPARK == 0;

struct RequestBasePlanImpl {
    std::shared_ptr<const qwen::PreparedPromptData> prompt;
    runtime::RequestPlanSummary summary;
    models::frontend::PreparedContextCache context_cache;
    ResolvedSamplingParameters sampling;
    std::shared_ptr<const qwen::VisionExecutionInput> vision;
    // The prompt frontiers a flight may checkpoint: SharedPrefix points that carry evidence and
    // lie strictly before the end of the request's prefill. A checkpoint inside a window is an
    // execution frontier of its flight: the window's traversal is cut there. Ascending.
    std::vector<CheckpointCandidate> checkpoint_candidates;
    // The frontiers of `checkpoint_candidates`, ascending: what the engine's demand tracking
    // observes a request crossing.
    std::vector<std::uint32_t> checkpoint_frontiers;
    std::shared_ptr<const PreparedCaptureBacking> capture_backing;
    qwen::PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    // The prompt frontier a sample fork splits at; zero when the request cannot fork.
    std::uint32_t fork_frontier = 0;
    bool allow_prefix_reuse     = false;

    // The frontier where the request's prefill ends: its fork frontier when it forks, else its
    // prompt.
    [[nodiscard]] std::uint32_t prefill_end() const noexcept {
        return fork_frontier ? fork_frontier : summary.prompt_tokens;
    }
};

// The metadata of a lane after `count` more tokens of forward progress.
[[nodiscard]] ContinuationMetadata forward_metadata(ContinuationMetadata value,
                                                    std::uint32_t count, SpeculativeBackend backend,
                                                    bool backend_pool);

// Most checkpoints one request places in flights: the system prompt and the last shared message.
inline constexpr std::size_t kMaxFlightCheckpoints = 2;
// How far past its frontier a decode unit's KV growth reaches when the pool must be reclaimed. A
// unit reserves only its own minimum, which its commit releases; a pool that cannot hold the
// minimum is reclaimed up to the horizon, so one reclaim serves many decode steps.
inline constexpr std::uint32_t kDecodeGrowthReserveTokens = 1024;

enum class Lifecycle : std::uint8_t {
    Empty,
    Binding,
    Prefilling,
    Replaying,
    Active,
    Pending,
    Finishable,
    Pausing
};

struct UnitDemand {
    ExecutionUnitKind kind         = ExecutionUnitKind::Decode;
    std::uint32_t tokens           = 0;
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
};

struct RecoveryPermit {
    // The active KV histories hold the physical reservation through this coverage.
    // These frontiers distinguish reconstructed work from committed new progress.
    UnitDemand coverage;
    std::uint32_t frontier    = 0;
    std::size_t ledger_tokens = 0;
};

struct KVHistoryView {
    context::HistoryId text;
    std::optional<context::HistoryId> backend;
    friend bool operator==(const KVHistoryView&, const KVHistoryView&) = default;
};

struct SequenceKVBundle {
    context::HistoryOwner text;
    std::optional<context::HistoryOwner> backend;

    [[nodiscard]] KVHistoryView view() const {
        return {text.handle(), backend ? std::optional(backend->handle()) : std::nullopt};
    }
};

struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Continuation continuation;
    context::StateId pending_write;
    std::uint32_t lane = 0;

    std::vector<TokenId> ledger;
    qwen::ResidentPrefixIdentity prefix_identity;
    qwen::PrefixShortlistDigests prefix_digests;
    SequenceHandle id;
    runtime::ResidencyId residency;

    [[nodiscard]] context::StateId write_state() const noexcept {
        return pending_write.valid() ? pending_write : continuation.image;
    }
};

using Snapshot = context::ContextSnapshot<ContinuationMetadata, PreparedCaptureBacking>;

struct CheckpointState {
    Snapshot snapshot;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    PrefixShortlistKey key;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;
    SharedCandidateEvidence evidence = SharedCandidateEvidence::None;

    [[nodiscard]] const Continuation& continuation() const noexcept {
        return snapshot.continuation();
    }

    [[nodiscard]] std::uint32_t frontier() const noexcept {
        return continuation().metadata.execution_frontier;
    }

    [[nodiscard]] qwen::RecoveryIdentityView recovery_identity() const {
        const auto& identity = snapshot.identity();
        return {frontier(),       key.identity_tag,
                identity.ledger,  identity.prefix_identity,
                identity.digests, snapshot.identity_owner().get()};
    }

    [[nodiscard]] std::uint32_t backend_frontier() const noexcept {
        return continuation().metadata.backend_kv_valid;
    }

    [[nodiscard]] const context::HistoryPrefix& text_prefix() const { return snapshot.prefix(0); }

    [[nodiscard]] const context::HistoryPrefix* backend_prefix() const {
        return snapshot.prefixes().size() == 2 ? &snapshot.prefix(1) : nullptr;
    }

    [[nodiscard]] KVHistoryView history() const {
        return {text_prefix().history(),
                backend_prefix() ? std::optional(backend_prefix()->history()) : std::nullopt};
    }
};

using Checkpoints = context::CheckpointStore<CheckpointState>;

// One checkpoint a flight of a lane writes. Its catalog point and its device State image are held
// from the reservation until the flight that carries it retires and publishes the point; a
// checkpoint that never publishes returns both.
struct FlightCheckpoint {
    SessionImpl* owner = nullptr;
    CheckpointCandidate candidate;
    Checkpoints::Reservation point;
    context::StateId image;

    FlightCheckpoint() = default;
    FlightCheckpoint(FlightCheckpoint&& other) noexcept
        : owner(std::exchange(other.owner, nullptr)), candidate(std::move(other.candidate)),
          point(std::move(other.point)), image(std::exchange(other.image, {})) {}
    FlightCheckpoint& operator=(FlightCheckpoint&& other) noexcept {
        if (this != &other) {
            release();
            owner     = std::exchange(other.owner, nullptr);
            candidate = std::move(other.candidate);
            point     = std::move(other.point);
            image     = std::exchange(other.image, {});
        }
        return *this;
    }
    FlightCheckpoint(const FlightCheckpoint&)            = delete;
    FlightCheckpoint& operator=(const FlightCheckpoint&) = delete;
    ~FlightCheckpoint() { release(); }

    // Returns the State image; the catalog point returns with the reservation.
    void release() noexcept;
};

struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    ResolvedSamplingParameters sampling_host;
    SpeculativeStats speculative_stats;
    std::shared_ptr<const RequestBasePlanImpl> base;
    std::optional<UnitDemand> permit;
    std::optional<RecoveryPermit> recovery;
    std::uint32_t replay_target = 0;
    std::uint32_t replay_cursor = 0;
    Lifecycle resume_lifecycle  = Lifecycle::Empty;
    bool publish_continuation   = true;
    // At most kMaxFlightCheckpoints candidates the request's prefill still passes, ascending. A
    // candidate leaves the plan when a flight carries it or when the Context cannot hold it.
    std::vector<CheckpointCandidate> checkpoint_plan;
    // Reserved for a window of this lane that no flight carries yet, ascending by frontier.
    std::vector<FlightCheckpoint> reserved_checkpoints;
    // Published by a retired flight and not yet taken through take_flight_checkpoints.
    std::vector<CheckpointHandle> unreported_checkpoints;
    std::uint32_t skipped_checkpoints = 0;

    struct Prefill {
        qwen::PreparedPromptData prompt;
        std::uint32_t base          = 0;
        std::uint32_t cursor        = 0;
        std::uint32_t prompt_tokens = 0;
        PrefixReusePath reuse       = PrefixReusePath::Root;
    };

    std::optional<Prefill> prefill;
};

struct PreparedCheckpoint {
    Checkpoints::Reservation record;
    std::shared_ptr<const PreparedCaptureBacking> identity;
};

struct ResumeStateImpl {
    SessionImpl* owner = nullptr;
    std::optional<CheckpointHandle> snapshot;
    SequenceState sequence;
    RequestControl control;
    std::uint32_t frontier = 0;
    ~ResumeStateImpl();
};

struct BindOperation {
    static constexpr auto kind = ContextOperationKind::Bind;
    std::uint32_t lane         = 0;
    std::optional<CheckpointHandle> source;
    Checkpoints::Lease source_lease;
    std::vector<CheckpointHandle> carried_points;
    std::vector<Checkpoints::Lease> carried_pins;
    std::vector<std::pair<CheckpointHandle, CheckpointHandle>> carried_clones;
    std::vector<CheckpointHandle> retired_points;
    std::shared_ptr<const RequestBasePlanImpl> base;
    ResumeState* resume            = nullptr;
    std::uint32_t reuse_frontier   = 0;
    std::uint32_t backend_frontier = 0;
    UnitDemand first_unit;
    UnitDemand reservation_demand;
    std::optional<RecoveryPermit> recovery;
    StateBindingAction state_action = StateBindingAction::Root;
    bool consume_source             = false;
    bool take_private               = false;
    // The source needs no Host restore: its State and KV replicas are on the Device.
    bool local_device               = false;
    bool resume_without_replay      = false;
    bool source_tail_hidden         = false;
};

struct BindingPlan {
    BindingEvaluation evaluation;
    BindOperation operation;
    ContextDemand demand;
    context::ContextDemandId state, main, backend, backup;
};

struct MaintenanceOperation {
    static constexpr auto kind = ContextOperationKind::Maintenance;
    std::optional<ContextDemotion> demotion;
    std::vector<Checkpoints::Lease> source_leases;
    context::HostDemandIds targets;
    std::optional<SequenceHandle> pause_sequence;
    runtime::PauseAction pause_action;
};

struct DemoteOperation {
    static constexpr auto kind = ContextOperationKind::Demote;
    // The checkpoints whose pages and State images this transfer reads.
    std::vector<CheckpointHandle> sources;
};

struct PauseOperation {
    static constexpr auto kind = ContextOperationKind::Pause;
    std::uint32_t lane;
    ResumeState paused;
    bool preserve_state_device = true;
};

struct PendingContext {
    using Action =
        std::variant<BindOperation, DemoteOperation, PauseOperation, MaintenanceOperation>;

    PendingContext(context::ContextDirectory& directory, Action value)
        : transaction(directory), action(std::move(value)) {}

    PendingContext(context::ContextTransaction&& prepared, Action value)
        : transaction(std::move(prepared)), action(std::move(value)) {}

    context::ContextTransaction transaction;
    Action action;
    std::optional<runtime::ExecutionReservation> reservation;
    std::optional<runtime::ExecutionTicket> ticket;
    std::optional<ContextResult> result;
    ContextWork input;
    context::ContextActionCursor cursor;
    bool submitted  = false;
    bool last_batch = false;
    std::vector<runtime::ContextTransferObservation> preceding_transfers;

    [[nodiscard]] std::optional<std::uint32_t> lane() const noexcept {
        return std::visit(
            [](const auto& value) -> std::optional<std::uint32_t> {
                if constexpr (requires { value.lane; }) return value.lane;
                return std::nullopt;
            },
            action);
    }

    [[nodiscard]] ContextOperationKind kind() const noexcept {
        return std::visit([](const auto& value) { return value.kind; }, action);
    }
};

class SessionImpl {
public:
    SessionImpl(std::shared_ptr<const ModelDefinition>, SessionPlan);
    ~SessionImpl();
    [[nodiscard]] RequestBasePlan plan_request(qwen::PreparedPromptData&& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                   bool consume_source                              = false,
                   std::span<const CheckpointHandle> private_points = {},
                   std::span<const CheckpointHandle> retired_points = {},
                   runtime::PlacementId target = {}, const ResumeState* resume = nullptr) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t frontier) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle checkpoint) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle checkpoint) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle checkpoint,
                                          const RequestBasePlan& base) const;
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle retained,
                                                             const RequestBasePlan& base,
                                                             std::uint32_t target) const;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle checkpoint) const noexcept;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle checkpoint) noexcept;
    [[nodiscard]] bool revoke_snapshot(ResumeState& paused) noexcept;
    // Exact Host bytes freed by deleting this fixed set, with physical sharing counted once.
    [[nodiscard]] std::size_t
    host_bytes_released(std::span<const CheckpointHandle> checkpoints) const;
    [[nodiscard]] std::optional<context::HostWriteRequirements>
        pause_host_requirements(SequenceHandle) const;
    [[nodiscard]] std::vector<PauseOption> pause_options(SequenceHandle) const;
    [[nodiscard]] bool pause_contributes(SequenceHandle, const PauseOption&,
                                         const ContextAssessment&) const;
    [[nodiscard]] bool prepare_pause(SequenceHandle, runtime::PauseAction,
                                     std::span<const CheckpointHandle> host_victims);
    [[nodiscard]] context::HostWriteRequirements
    demotion_host_requirements(const ContextDemotion&) const;
    [[nodiscard]] ContextDemotion combine_demotions(std::span<const ContextDemotion>) const;
    [[nodiscard]] std::size_t
    release_redundant_host(std::span<const CheckpointHandle> excluded,
                           std::optional<SequenceHandle> pending_backup = std::nullopt);
    // Reservations stay owned by their lane through commit. Failure leaves prior permits intact.
    [[nodiscard]] ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    [[nodiscard]] runtime::ContextStart start_maintenance(const ContextAssessment&);
    void release_units(std::span<const SequenceHandle> sequences) noexcept;
    [[nodiscard]] BindingPlan plan_binding(const RequestBasePlan&, runtime::LaneId,
                                           const SourceCandidate&, const ResumeState*,
                                           ExecutionUnitKind, std::uint32_t) const;
    // A binding whose source needs no transfer completes before this returns: its device work
    // rides the first execution of the lane. `allow_synchronous` false keeps the asynchronous
    // completion contract for a caller that waits on the binding's context operation.
    // `synchronous_only` declines, before anything starts, a binding that needs a context
    // operation (`BindingReservation::asynchronous`).
    [[nodiscard]] BindingReservation
    start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                  ResumeState* resume           = nullptr,
                  ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                  std::uint32_t resume_tokens   = 1, bool allow_synchronous = true,
                  bool synchronous_only = false);
    // Finite physical actions, each including every optional holder needed for its release.
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle> allowed,
                                                  std::span<const CheckpointHandle> excluded,
                                                  const ContextDemand&) const;
    [[nodiscard]] context::ReleasePlan<ContextRelease, ContextReleaseSet>
    plan_releases(std::span<const CheckpointHandle> allowed,
                  std::span<const CheckpointHandle> excluded, const ContextDemand&) const;
    [[nodiscard]] runtime::ContextStart
    start_demote(const ContextDemotion&, std::span<const CheckpointHandle> host_victims);
    [[nodiscard]] runtime::ContextStart submit_demotion(const ContextDemotion&,
                                                        context::HostDemandIds,
                                                        context::ContextReservation,
                                                        runtime::ExecutionReservation);
    [[nodiscard]] runtime::ContextStart
    defer_demotion(const ContextDemotion&, context::HostDemandIds, context::ContextReservation,
                   const ContextAssessment&, runtime::ExecutionReservation);
    [[nodiscard]] bool start_pause(SequenceHandle sequence, runtime::PauseAction);
    [[nodiscard]] bool has_context_operation() const noexcept;
    [[nodiscard]] std::uint64_t context_epoch() const noexcept;
    [[nodiscard]] bool context_pins(CheckpointHandle) const noexcept;

    // Called by Session entry points that change the retained set, its placements or pins, or
    // hand capacity back. Checkpoint publication and retirement advance the epoch through the
    // store; a context operation starting or ending advances it through Session's watch.
    void note_context_change() noexcept { ++context_mutations_; }

    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
    [[nodiscard]] bool execution_pending(SequenceHandle sequence) const noexcept;
    [[nodiscard]] bool terminal_blocked(SequenceHandle sequence) const noexcept;
    // A resumed binding retains its complete recovery capacity until committed new progress.
    [[nodiscard]] bool recovery_pending(SequenceHandle sequence) const noexcept;
    [[nodiscard]] std::uint32_t prefill_pack_limit() const noexcept;
    [[nodiscard]] std::uint32_t prefill_pack_tokens() const noexcept;
    [[nodiscard]] bool crosses_execution_boundary(const SequenceState&, std::uint32_t begin,
                                                  std::uint32_t end) const;
    [[nodiscard]] std::optional<PrefillStart> start_prefill(std::span<const SequenceHandle>,
                                                            std::span<const bool> constrained);
    [[nodiscard]] std::uint32_t sample_fork_limit() const noexcept;
    [[nodiscard]] std::vector<ForkGroupResult> fork_batch(std::span<const ForkGroupRequest> groups);
    [[nodiscard]] ForkGroupResult fork_group(const ForkGroupRequest& group,
                                             std::uint32_t parent_lane);
    [[nodiscard]] std::uint32_t mixed_decode_limit() const noexcept;
    [[nodiscard]] bool prefill_mixable(SequenceHandle) const;
    [[nodiscard]] std::optional<MixedStart>
    start_mixed(std::span<const SequenceHandle> prefill, std::span<const bool> prefill_constrained,
                std::span<const SequenceHandle> decode,
                std::span<const runtime::RoundBudget> budgets,
                std::span<const bool> decode_constrained);
    [[nodiscard]] std::optional<runtime::ExecutionTicket> start_replay(SequenceHandle);
    [[nodiscard]] std::optional<runtime::ExecutionTicket>
    start_decode(std::span<const SequenceHandle>, std::span<const runtime::RoundBudget>,
                 std::span<const bool> constrained = {});
    [[nodiscard]] std::optional<runtime::ExecutionTicket>
    start_control(std::span<const SequenceHandle>, std::span<const TokenId>, std::uint32_t stride,
                  std::span<const std::optional<std::uint32_t>> splits);
    [[nodiscard]] std::optional<runtime::ExecutionTicket> start_score(qwen::PreparedPromptData&&,
                                                                      std::uint32_t first_target);
    [[nodiscard]] std::optional<SessionEvent> poll();
    void supply(runtime::ExecutionTicket, runtime::DecisionInput&&);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const runtime::CommitDecision> decisions);
    [[nodiscard]] DiscardResult discard(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;

    std::shared_ptr<const ModelDefinition> definition;
    SessionPlan plan;
    vision::Window vision_window;
    [[nodiscard]] std::optional<qwen::VisionWork>
    prepare_visual(const SequenceState&, std::uint32_t begin, std::uint32_t end);
    runtime::Executor<ExecutionTypes>* executor_ = nullptr;

    // Binds the Session to the executor that runs its executions and sizes the execution table
    // to the executor's compute frames.
    void bind_executor(runtime::Executor<ExecutionTypes>& executor) {
        executor_ = &executor;
        executions.clear();
        executions.resize(executor.capacity().compute_frames);
    }

    [[nodiscard]] runtime::Executor<ExecutionTypes>& executor() const { return *executor_; }

    context::ContextDirectory directory;
    context::StateDirectory* state_store;
    context::HistoryDirectory* text_kv_histories;
    context::HistoryDirectory* backend_kv_histories;
    context::PageDirectory* text_kv_pages;
    context::PageDirectory* backend_kv_pages;
    const bool host_context_enabled;
    const std::uint32_t capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const ContextCacheOptions context_cache;
    const bool vision_enabled;
    const bool causal_scoring;
    // Lane tables, one entry per lane of the resolved concurrency; allocated at construction.
    std::vector<SequenceState> sequences;
    std::vector<RequestControl> requests;
    std::vector<std::uint64_t> lane_epochs;
    std::uint64_t next_sequence = 1;
    std::uint64_t context_mutations_ = 0;
    mutable context::RecoveryIndex<> recovery_index{directory.metadata_budget()};
    Checkpoints checkpoints;
    context::MetadataAccount count_storage{directory.metadata_budget()};
    TokenCountBuilder count_builder;
    std::unique_ptr<PendingContext> context_operation_;
    std::optional<ContextAssessment> pending_maintenance_;
    std::optional<SequenceHandle> normalizing_pause_;

    struct PauseReservation {
        SequenceHandle sequence;
        runtime::PauseAction action;
        ContextAssessment placement;
        context::HostDemandIds target_ids;
        context::ContextReservation targets;
        runtime::ExecutionReservation execution;
        std::optional<runtime::ExecutionReservation> normalization;
        std::vector<runtime::ContextTransferObservation> preceding_transfers;
        PreparedCheckpoint checkpoint;
    };

    std::optional<PauseReservation> pause_reservation_;
    bool pause_cancelled_ = false;
    std::deque<SessionEvent> events;
    std::vector<runtime::ResidencyId> releases;
    std::optional<runtime::ExecutionTicket> release_ticket;
    bool release_result = false;
    bool closing        = false;
    std::optional<SessionEvent> after_releases;
    void release_residency(runtime::ResidencyId);
    void submit_releases();

    enum class OperationKind { Prefill, Replay, Decode, Mixed, Control, Score, Normalize };

    struct Execution {
        OperationKind kind;
        runtime::ExecutionTicket ticket;
        // Prefill segments first, then decode rows. Every per-member vector follows this order; a
        // Mixed execution's pending batch and decisions cover the members from `decode_begin` on.
        std::vector<SequenceHandle> members;
        // The lane of every member, frozen at submission; the busy counters follow it.
        std::vector<std::uint32_t> lanes;
        // The checkpoints each member's window writes, parallel to `members` (a member without
        // one has an empty entry; the vector may be shorter than `members`). They publish when
        // the flight retires after its commit and return their resources otherwise.
        std::vector<std::vector<FlightCheckpoint>> checkpoints;
        // Whether each member carries a decision row of the execution's pending batch: decode
        // rows and final prefill segments produce; non-final segments do not.
        std::vector<bool> producing;
        // Member index of every decision row of the execution's pending batch, in row order.
        // Decode rows are the trailing members of a mixed execution; the final segments of a
        // packed prefill are their own members. Empty until the batch is generated.
        std::vector<std::uint32_t> pending_members;
        std::vector<Continuation> bases;
        std::vector<ContinuationMetadata> completed;
        std::vector<std::uint32_t> columns;
        std::vector<std::vector<TokenId>> inputs;
        std::vector<std::optional<std::uint32_t>> splits;
        std::unique_ptr<context::ContextTransaction> preparation;
        std::unique_ptr<context::ContextTransaction> settlement;
        std::optional<GeneratedResult> generated;
        ExecutionSample observed_result;
        std::optional<SessionEvent> completion;
        std::vector<runtime::CommitDecision> decisions;
        std::uint64_t vision_id = 0;
        std::size_t decode_begin = 0;
        bool result_ready       = false;
        bool committed          = false;
        bool discarded          = false;
    };

    // Executions in flight, one table entry per compute frame of the executor (sized when the
    // Session is bound to it). A ticket names its entry, so every result, decision, commit and
    // retirement is routed by `ticket.frame.index`.
    std::vector<std::optional<Execution>> executions;

    [[nodiscard]] std::optional<Execution>& execution_slot(runtime::FrameId);
    [[nodiscard]] Execution* find_execution(runtime::ExecutionTicket) noexcept;
    [[nodiscard]] const Execution* find_execution(runtime::ExecutionTicket) const noexcept;
    // Whether any execution is in flight. Context maintenance, pauses and scoring need the Session
    // without one.
    [[nodiscard]] bool executing() const noexcept;
    // Whether an execution that owns the whole Session (a vision window, a normalization or a
    // score) is in flight: nothing else starts beside it.
    [[nodiscard]] bool exclusive_execution() const noexcept;
    [[nodiscard]] static bool exclusive(const Execution&) noexcept;

    struct Scoring {
        runtime::ExecutionTicket public_ticket;
        RequestBasePlan base;
        SequenceHandle sequence;
        std::uint32_t first_target = 0;
        std::uint32_t cursor       = 0;
        std::vector<float> logprobs;
    };

    std::optional<Scoring> scoring;

    // Sequence identity to lane. An entry can outlive its lane's occupant; the lookup validates it
    // against the lane tables, and registering a lane drops the entry its previous occupant held.
    std::unordered_map<std::uint64_t, std::uint32_t> lane_index_;
    std::vector<std::uint64_t> lane_identity_;
    // Executions in flight that carry each lane.
    std::vector<std::uint32_t> lane_busy_;
    // Lanes whose residency release has not retired. A new residency of such a lane cannot
    // install before the release leaves, or the release would find the new installation.
    std::vector<std::uint8_t> lane_releasing_;
    std::vector<std::uint32_t> releasing_ticket_lanes_;
    void register_lane(std::uint32_t lane, SequenceHandle) noexcept;
    [[nodiscard]] std::uint32_t lane(SequenceHandle) const;
    [[nodiscard]] bool valid_sequence(SequenceHandle) const noexcept;
    [[nodiscard]] SequenceHandle sequence_handle(std::uint32_t) const noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t) const;
    void invalidate_lane(std::uint32_t) noexcept;
    [[nodiscard]] UnitDemand prefill_unit(std::uint32_t prompt, std::uint32_t cursor) const;
    [[nodiscard]] UnitDemand next_unit(const SequenceState&, const RequestControl&,
                                       ExecutionUnitKind, std::uint32_t tokens) const;
    void require_unit(std::uint32_t, ExecutionUnitKind, std::uint32_t tokens = 0) const;
    [[nodiscard]] bool prepare_settlement(std::uint32_t lane,
                                          const ContinuationMetadata& continuation,
                                          Lifecycle lifecycle, std::size_t ledger_tokens,
                                          context::ContextTransaction&);
    void adopt_unit(std::uint32_t lane, const ContinuationMetadata& continuation,
                    bool state_written, bool retain_recovery) noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState&) const noexcept;
    [[nodiscard]] bool has_backend_pool() const noexcept;
    [[nodiscard]] ExecutionRow execution_row(const SequenceState&) const;
    [[nodiscard]] InputWindow input_window(const SequenceState&, std::uint32_t begin,
                                           std::uint32_t end) const;
    [[nodiscard]] std::uint32_t window_end(const SequenceState&, const RequestControl&,
                                           std::uint32_t begin, std::uint32_t end,
                                           const vision::Window&) const;
    void ensure_sequence_kv_mapped(SequenceState&, std::uint32_t main, std::uint32_t backend,
                                   context::ContextTransaction&);
    void complete_state_write(SequenceState&);
    void release_sequence_state(SequenceState&) noexcept;
    void release_sequence_kv(SequenceState&) noexcept;
    void clear_lane(SequenceState&, RequestControl&) noexcept;

    void receive_result(runtime::ResultReady<ExecutionTypes>&&);
    void retire(runtime::DeviceRetired);
    void commit_completed_unit(Execution&);
    void advance_score();
    [[nodiscard]] PendingBatch generated_pending(const Execution&, const GeneratedResult&);
    [[nodiscard]] const Execution* matching_execution(const PendingBatch&) const noexcept;
    // The end of the window the sequence's next prefill execution would cover, when that window
    // can share an execution with decode rows.
    [[nodiscard]] std::optional<std::uint32_t> mixable_end(const SequenceState&,
                                                           const RequestControl&) const;
    void append_prefill_row(Execution&, ForwardWork&, SequenceHandle, std::uint32_t begin,
                            std::uint32_t end, bool constrained, bool produces_token);
    void append_decode_row(Execution&, DecodeWork&, SequenceHandle,
                           const runtime::RoundBudget&, bool constrained);
    // Settle and adopt the prefill segments of a mixed execution together with its decode rows.
    void prepare_segments(Execution&, std::span<bool> recovery);
    void adopt_segments(Execution&, std::span<const bool> recovery) noexcept;
    [[nodiscard]] std::optional<runtime::ExecutionTicket>
    submit_execution(Execution&&, runtime::ExecutionReservation, Work&&);

    // Holds a frozen State image that several lanes read as the source of their first write. The
    // reference returns when the last holder retires; the image is freed once its readers finish.
    struct StateAnchor {
        context::StateDirectory* states = nullptr;
        context::StateId image;
        StateAnchor(context::StateDirectory& owner, context::StateId frozen)
            : states(&owner), image(frozen) {
            states->retain_checkpoint_reference(image);
        }
        StateAnchor(const StateAnchor&)            = delete;
        StateAnchor& operator=(const StateAnchor&) = delete;
        ~StateAnchor() {
            if (!states->release_checkpoint_owner(image)) std::terminate();
        }
    };

    // Device work a lane installation deferred: its actions and its residency installation ride
    // the stream order of the lane's first execution instead of a context operation. The
    // transaction keeps every borrowed source (a checkpoint's tail page, a fork parent's tail page)
    // alive until that execution retires, where `publish` runs; `retained` holds the fork parent's
    // KV prefixes and `source_lease` the checkpoint the lane bound from for the same window.
    struct DeferredBinding {
        context::ContextTransaction transaction;
        context::ContextActions actions;
        std::vector<context::HistoryPrefix> retained;
        std::optional<ResidencyBinding> install;
        Checkpoints::Lease source_lease;
        std::shared_ptr<StateAnchor> anchor;
        // Set once a flight carries the actions and the installation.
        bool submitted = false;
        explicit DeferredBinding(context::ContextTransaction&& prepared)
            : transaction(std::move(prepared)) {}
        DeferredBinding(DeferredBinding&&) = default;
    };

    std::vector<std::optional<DeferredBinding>> deferred_bindings;
    // Merges the deferred device work of an execution's member lanes ahead of the flight's own
    // context actions, and completes those transactions with the flight.
    void adopt_deferred_bindings(Execution&, Work&);
    void publish_deferred_binding(DeferredBinding&) noexcept;
    void discard_deferred_binding(std::uint32_t lane) noexcept;

    [[nodiscard]] bool can_release_checkpoint(CheckpointHandle) const noexcept;
    [[nodiscard]] std::optional<ContextReleaseSet>
        checkpoint_release_set(std::span<const CheckpointHandle>) const;
    [[nodiscard]] CheckpointState& checkpoint(CheckpointHandle);
    [[nodiscard]] const CheckpointState& checkpoint(CheckpointHandle) const;
    [[nodiscard]] std::optional<CheckpointHandle> reserve_checkpoint();
    [[nodiscard]] PreparedCheckpoint prepare_checkpoint(SequenceState&);
    [[nodiscard]] std::optional<CheckpointHandle> detach_checkpoint(SequenceState&,
                                                                    PreparedCheckpoint = {});
    void initialize_prefill(RequestControl&, std::uint32_t base);

    // In-flight checkpoints. A Prefill unit's window may carry the candidates of its request's plan
    // that lie inside it: reserve_units reserves each one's catalog point and State image without
    // reclaiming (a shortage skips the candidate and is counted), the window carries it to the
    // flight, and the flight's retirement publishes it.
    void initialize_checkpoint_plan(RequestControl&, std::uint32_t base);
    void plan_flight_checkpoints(std::uint32_t lane, std::uint32_t window_end);
    [[nodiscard]] bool checkpoint_present(const PrefixShortlistKey&) const;
    void attach_checkpoints(Execution&, InputWindow&, SequenceState&, std::uint32_t begin,
                            std::uint32_t end);
    void publish_flight_checkpoints(Execution&, std::span<const std::uint8_t> clearing);
    void publish_flight_checkpoint(FlightCheckpoint&, SequenceState&, RequestControl&);
    [[nodiscard]] FlightCheckpoints take_flight_checkpoints(SequenceHandle);
    // Returns the lane's reserved checkpoints and releases the ones it published but nobody took.
    void drop_lane_checkpoints(RequestControl&) noexcept;
    // The prompt positions strictly inside (begin, end) where the window's traversal is cut: the
    // prefix execution frontiers and the columns of the checkpoints reserved for the lane.
    // Ascending and distinct.
    [[nodiscard]] std::vector<std::uint32_t> window_splits(const SequenceState&, std::uint32_t begin,
                                                           std::uint32_t end) const;
    [[nodiscard]] std::uint32_t split_count(const SequenceState&, std::uint32_t begin,
                                            std::uint32_t end) const;
    [[nodiscard]] PrefillRowProgress prefill_row(const Execution&, std::size_t member);
    void abort_context() noexcept;
    void cancel_context() noexcept;
    [[nodiscard]] bool prepare_backup(context::ContextTransaction&, CheckpointState&,
                                      bool shared_device, bool& preserve_state_device,
                                      context::ContextReservation&, const context::HostDemandIds&);
    void install_binding(BindOperation&);
    // `deferred` completes a binding whose device work has not run yet: the History bindings
    // publish their restores before the lane adopts them.
    void complete_binding(BindOperation&, context::ContextTransaction&, ContextProgress&,
                          bool deferred = false);
    [[nodiscard]] bool begin_pause(SequenceHandle, runtime::PauseAction);
    [[nodiscard]] bool continue_pause_ready(SequenceHandle, runtime::PauseAction);
    void continue_pause_after_normalize(SequenceHandle);
    [[nodiscard]] ResumeState complete_pause(PauseOperation&);
    [[nodiscard]] bool submit_context(PendingContext&, ContextWork&&);
    void submit_context_batch(PendingContext&);
    void receive_context_result(runtime::ExecutionTicket, ContextResult&&);
    void retire_context(runtime::DeviceRetired);
    [[nodiscard]] std::optional<runtime::ExecutionTicket>
        start_normalize(SequenceHandle, runtime::ExecutionReservation);
};

} // namespace ninfer::models::qwen3_5::detail
