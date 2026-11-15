#pragma once

#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/request_record.h"
#include "runtime/engine/context_cache/resource_manager.h"
#include "runtime/engine/context_cache/placement_policy.h"
#include "runtime/engine/scheduler.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/executor/contract.h"
#include "runtime/executor/measurement.h"
#include "runtime/observability/snapshot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class EngineCore {

public:
    static constexpr EnginePurpose purpose = EnginePurpose::Generation;

    using ModelContract      = typename Instance::ModelContract;
    using ContextAssessment  = typename ModelContract::Session::ContextAssessment;
    using PrefillStart       = typename ModelContract::PrefillStart;
    using MixedStart         = typename ModelContract::MixedStart;
    using ForkResult         = typename ModelContract::ForkResult;
    using ExecutionUnit      = typename ModelContract::ExecutionUnit;
    using UnitKind           = typename ModelContract::ExecutionUnitKind;
    using Checkpoint         = typename ModelContract::CheckpointHandle;
    using SequenceHandle     = typename ModelContract::SequenceHandle;
    using PendingBatch       = typename ModelContract::PendingBatch;
    using PreparedPrompt     = typename ModelContract::PreparedPrompt;
    using PublishedOutput    = typename ModelContract::PublishedOutput;
    using Request            = RequestRecord<ModelContract>;
    using Scheduling         = Scheduler<Request>;
    using RoundMembership    = typename Scheduling::RoundMembership;
    using ControlMembership  = typename Scheduling::ControlMembership;
    using ResourceManagement = ResourceManager<ModelContract>;
    using Clock              = std::chrono::steady_clock;

    EngineCore(Instance& instance, const EngineOptions& options,
               ContextMachineCostModel context_cost)
        : instance_(instance), max_context_(options.max_context),
          max_concurrency_(options.max_concurrency),
          prefill_chunk_(options.prefill_chunk),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          speculative_positions_(options.speculative.backend == SpeculativeBackend::None
                                     ? 0
                                     : options.speculative.draft_tokens),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          resources_(options.context_cache.enabled, std::move(context_cost)),
          slots_(options.max_concurrency),
          capture_decisions_(options.max_concurrency),
          scheduler_(options.max_concurrency),
          control_executed_(options.max_concurrency),
          reserved_units_(options.max_concurrency),
          current_decode_lanes_(options.max_concurrency) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("Engine core bounds are invalid");
        }
        // The resolved lane count sizes every lane table of the engine and its executor before
        // any execution can start.
        instance_.executor->configure_decode_statistics(max_concurrency_);
        materializing_.resize(max_concurrency_);
        lane_residency_epochs_.resize(max_concurrency_, 1);
        cumulative_stats_.context_transfers.kv.resize(instance_.session->kv_pools().size());
        for (const auto& pool : instance_.session->kv_pools()) {
            cumulative_stats_.kv_pools.push_back({pool.id.index, std::string(pool.label), 0});
        }
        cumulative_stats_.speculative_attempted_per_position.resize(speculative_positions_);
        cumulative_stats_.speculative_accepted_per_position.resize(speculative_positions_);
        for (const auto& placement : instance_.session->placement_descriptions())
            cumulative_stats_.placements.push_back({.id = placement.id});
        stats_snapshot_             = cumulative_stats_;
        published_snapshot_.runtime = cumulative_stats_;
        published_snapshot_.ranks.assign(instance_.executor->observations().begin(),
                                         instance_.executor->observations().end());
        paused_.reserve(max_outstanding_);
        request_observations_.reserve(max_outstanding_ + 1);
        published_requests_.reserve(max_outstanding_);
        published_snapshot_.decode_by_batch.resize(max_concurrency_);
        const auto frames = instance_.executor->capacity();
        // One flight table entry per compute frame of the executor, indexed by the frame of the
        // flight's ticket.
        compute_.resize(frames.compute_frames);
        retiring_.resize(frames.compute_frames);
        execution_observations_.reserve(frames.compute_frames + frames.context_frames);
        published_snapshot_.in_flight.reserve(frames.compute_frames + frames.context_frames);
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                instance_.executor->bind_to_current_thread();
                publish_runtime_stats();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
    }

    ~EngineCore() noexcept {
        request_stop();
        if (worker_.joinable()) { worker_.join(); }
    }

    // May be called while consumers are waiting; only the worker releases model resources.
    void request_stop() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
    }

    EngineCore(const EngineCore&)            = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    // The iteration bounds. `prefill_pack_tokens()` is the Session's own bound on the prefill
    // tokens one packed execution carries, so it is the token budget of one iteration; the
    // decode row limit is the Session's mixed-flight row bound (the lane capacity when the
    // Session never mixes, so every decode row still rides); and the floor keeps the head
    // prefill lane advancing beside the decode rows even when its own limit exceeds the budget.
    [[nodiscard]] IterationBudget iteration_budget() const noexcept {
        const std::uint32_t rows = instance_.session->mixed_decode_limit();
        return IterationBudget{
            .token_budget         = std::max(instance_.session->prefill_pack_tokens(), 1U),
            .decode_rows_limit    = rows != 0 ? rows : max_concurrency_,
            .prefill_floor_tokens = prefill_chunk_ != 0 ? prefill_chunk_ : 1U,
        };
    }

    // A sample group of one prepared prompt: `members` are the branch requests, each derived
    // from the same prompt with its own sampling seed. A forking group reserves one parent lane
    // (the hidden parent helper that prefills the shared prompt prefix) plus one child lane per
    // member; the members hold their child lanes in the Forking state until the parent rests at
    // the fork frontier and forks them in. A non-forking group admits its members as ordinary
    // requests that each prefill the whole prompt.
    struct SampleGroup {
        std::uint64_t id = 0;
        std::vector<std::shared_ptr<Request>> members;
        std::shared_ptr<Request> parent;
        std::vector<std::uint32_t> child_lanes;
        // `forking`: the group holds its parent helper and child lanes. `forked`: the fork
        // installed the children and only the parent's completion may still be pending.
        bool forking = false;
        bool forked  = false;
    };

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        [[nodiscard]] std::uint64_t request_id() const noexcept {
            return request_ ? request_->id : 0;
        }

        void wait_for_admission(const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            owner_->wait_for_admission(request_, cancellation);
        }

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
            if (streaming != (sink != nullptr)) {
                throw std::invalid_argument(
                    "GenerationHandle wait sink does not match its submitted consumer mode");
            }
            EngineCore* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(EngineCore& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        EngineCore* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class EngineCore;
    };

    Submission submit(PreparedPrompt prompt, PromptSummary prompt_summary, double prepare_seconds,
                      ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                      GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        auto request = create_request(std::move(prompt), std::move(options), prompt_summary,
                                      prepare_seconds, consumer_mode, std::move(observation),
                                      pending_deadline, submitted);
        request_admission_check();
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    // One sample group: `sample_count` branch requests of the same prepared prompt, in branch
    // order. Branch i derives its sampling seed from (the request seed, i). The branches are
    // queued consecutively and linked to one SampleGroup, so admission takes the group as a
    // unit: a forking group gains one parent lane plus one child lane per branch together, and
    // a non-forking group holds its lane count as its reservation floor.
    [[nodiscard]] std::vector<Submission>
    submit_samples(PreparedPrompt prompt, PromptSummary prompt_summary, double prepare_seconds,
                   ResolvedRequestOptions options, std::uint32_t sample_count,
                   OutputConsumerMode consumer_mode, GenerationObservationOptions observation,
                   Clock::time_point pending_deadline = {}) {
        if (sample_count == 0) {
            throw std::invalid_argument("sample count must be positive");
        }
        // A forking group needs its parent helper lane beside one child lane per branch.
        const std::uint32_t lane_capacity =
            instance_.session->sample_fork_limit() != 0 ? max_concurrency_ - 1 : max_concurrency_;
        if (sample_count > lane_capacity) {
            throw std::invalid_argument("sample group exceeds the engine lane capacity");
        }
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::vector<PreparedPrompt> prompts;
        prompts.reserve(sample_count - 1);
        for (std::uint32_t branch = 1; branch < sample_count; ++branch) {
            prompts.push_back(instance_.frontend.clone(prompt));
        }
        auto group = std::make_shared<SampleGroup>();
        group->members.reserve(sample_count);
        {
            std::lock_guard lock(queue_mutex_);
            group->id = next_group_id_++;
            groups_.push_back(group);
        }
        std::vector<std::shared_ptr<Request>> branches;
        branches.reserve(sample_count);
        try {
            for (std::uint32_t branch = 0; branch < sample_count; ++branch) {
                ResolvedRequestOptions branch_options = options;
                branch_options.execution.sampling.seed =
                    derive_sample_seed(options.execution.sampling.seed, branch);
                PreparedPrompt branch_prompt =
                    branch == 0 ? std::move(prompt) : std::move(prompts[branch - 1]);
                auto branch_request = create_request(
                    std::move(branch_prompt), std::move(branch_options), prompt_summary,
                    prepare_seconds, consumer_mode, observation, pending_deadline, submitted);
                branch_request->sample_group_id = group->id;
                branch_request->branch_index    = branch;
                group->members.push_back(branch_request);
                branches.push_back(std::move(branch_request));
            }
        } catch (...) {
            // A partially created group cancels its branches: every request that queued keeps
            // its completion path, and the caller observes only the submission failure.
            for (const auto& branch_request : branches) { abandon_request(branch_request); }
            throw;
        }
        request_admission_check();
        queue_cv_.notify_one();
        std::vector<Submission> submissions;
        submissions.reserve(branches.size());
        for (auto& branch_request : branches) {
            submissions.push_back(Submission(*this, std::move(branch_request)));
        }
        return submissions;
    }

    [[nodiscard]] EngineSnapshot snapshot(SnapshotOptions options = {}) const {
        std::lock_guard lock(stats_mutex_);
        auto result = published_snapshot_;
        if (options.include_requests) result.requests = published_requests_;
        return result;
    }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

    [[nodiscard]] EngineSnapshot measure(MeasurementCommand command) {
        auto request     = std::make_unique<MeasurementRequest>();
        request->command = command;
        auto result      = request->completion.get_future();
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) throw std::runtime_error("inference engine is unavailable");
            if (measurement_) throw std::logic_error("a measurement boundary is already pending");
            measurement_ = std::move(request);
        }
        queue_cv_.notify_all();
        return result.get();
    }

private:
    struct MeasurementRequest {
        MeasurementCommand command{};
        std::promise<EngineSnapshot> completion;
    };

    bool process_measurement();

    enum class HostWorkClass : std::uint8_t {
        Decode,
        Prefill,
        Control,
    };

    enum class FlightKind : std::uint8_t { Prefill, Decode, Control, Replay, Mixed };

    struct UnitPermit {
        ExecutionUnit unit;
        ResidencyId residency;
    };

    struct ComputeFlight {
        ExecutionTicket ticket;
        FlightKind kind = FlightKind::Prefill;
        std::vector<std::shared_ptr<Request>> requests;
        std::vector<SequenceHandle> sequences;
        std::vector<ResidencyId> residencies;
        std::vector<std::uint32_t> lanes;
        std::vector<bool> cancelled;
        // Whether each row of the flight produced committed progress.
        std::vector<bool> progressed;
        std::optional<std::uint64_t> retired_ns;
        bool service_accounted = false;
        std::vector<TokenId> control_tokens;
        std::uint32_t control_stride = 0;
        std::vector<std::size_t> generated_sizes;
        // Batch layout of a flight whose pending batch is not its membership in order. Empty
        // means the identity mapping (every batch row follows its member). `decode_rows` counts
        // the leading batch rows that are decode rows; the rows after them are the producing
        // prefill segments of a mixed or packed-final flight.
        std::vector<std::size_t> batch_rows;
        std::uint32_t decode_rows = 0;
        // The flight's prefill segments in membership order: whether each produced a token.
        std::vector<bool> produces;
        // The prefill segment count of a mixed flight's membership prefix.
        std::uint32_t prefill_members = 0;
        // The decode-row lanes of a mixed flight, in batch order (pure decode flights use
        // `lanes`).
        std::vector<std::uint32_t> decode_lanes;
        // The flight's submission index and statistics: the prefill tokens its segments
        // actually processed, and the host work of its iteration from planning to the Session
        // call returning.
        std::uint64_t index           = 0;
        std::uint32_t prefill_tokens  = 0;
        Clock::time_point started{};
        std::uint64_t host_submit_ns = 0;
    };

    using EngineHostPhase = RequestEngineHostPhase;

    [[nodiscard]] static nvtx::Name phase_range_name(EngineHostPhase phase) noexcept;

    struct ActiveExposure {
        std::shared_ptr<Request> request;
        std::uint32_t lane = 0;
    };

    // The resident requests of a cycle boundary, sized to the lane capacity.
    struct ActiveExposureSet {
        std::vector<ActiveExposure> entries;
        std::size_t size = 0;

        ActiveExposureSet() = default;

        explicit ActiveExposureSet(std::uint32_t lanes) : entries(lanes) {}
    };

    struct HostPhaseMeasurement {
        Clock::time_point started;
        std::uint64_t accounted_before = 0;
        ActiveExposureSet exposed;
    };

    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point started,
                                                  Clock::time_point finished) noexcept;

    [[nodiscard]] ActiveExposureSet active_exposure_set() const;

    [[nodiscard]] HostPhaseMeasurement begin_host_phase() const;

    void set_host_work_class(HostWorkClass work_class,
                             std::span<const std::uint32_t> decode_lanes = {}) noexcept;

    [[nodiscard]] bool current_decode_contains(std::uint32_t lane) const noexcept;

    void add_class_host_time(std::uint64_t host_ns, std::uint64_t scheduler_yield_ns) noexcept;

    void expose_engine_phase(const ActiveExposureSet& exposed, EngineHostPhase phase,
                             std::uint64_t elapsed) noexcept;

    void finish_engine_phase(const HostPhaseMeasurement& measurement,
                             EngineHostPhase phase) noexcept;

    void record_coordinator_call(CoordinatorCall call, std::uint64_t elapsed,
                                 const ActiveExposureSet& exposed) noexcept;

    void finish_execution_call(const HostPhaseMeasurement& measurement,
                               CoordinatorCall call) noexcept;

    void record_detail(std::uint64_t RuntimeHostWorkStats::*elapsed_member,
                       std::uint64_t RuntimeHostWorkStats::*invocation_member,
                       Clock::time_point started) noexcept;

    class EnginePhaseScope {
    public:
        EnginePhaseScope(EngineCore& owner, EngineHostPhase phase)
            : owner_(owner), phase_(phase), measurement_(owner.begin_host_phase()) {
            range_.emplace(phase_range_name(phase), nvtx::Category::Runtime);
        }

        ~EnginePhaseScope() { finish(); }

        EnginePhaseScope(const EnginePhaseScope&)            = delete;
        EnginePhaseScope& operator=(const EnginePhaseScope&) = delete;

        void pause_range() noexcept { range_.reset(); }

        void resume_range() noexcept {
            if (active_ && !range_) {
                range_.emplace(phase_range_name(phase_), nvtx::Category::Runtime);
            }
        }

        void checkpoint() noexcept {
            if (!active_) { return; }
            owner_.finish_engine_phase(measurement_, phase_);
            measurement_ = owner_.begin_host_phase();
        }

        void finish() noexcept {
            if (!active_) { return; }
            range_.reset();
            owner_.finish_engine_phase(measurement_, phase_);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        EngineHostPhase phase_;
        HostPhaseMeasurement measurement_;
        std::optional<nvtx::ScopedRange> range_;
        bool active_ = true;
    };

    class ExecutionCallScope {
    public:
        explicit ExecutionCallScope(EngineCore& owner,
                                    CoordinatorCall call = CoordinatorCall::Session)
            : owner_(owner), call_(call), measurement_(owner.begin_host_phase()) {}

        ~ExecutionCallScope() noexcept { finish(); }

        ExecutionCallScope(const ExecutionCallScope&)            = delete;
        ExecutionCallScope& operator=(const ExecutionCallScope&) = delete;

        void finish() noexcept {
            if (!active_) { return; }
            owner_.finish_execution_call(measurement_, call_);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        CoordinatorCall call_;
        HostPhaseMeasurement measurement_;
        bool active_ = true;
    };

    void publish_runtime_stats();
    [[nodiscard]] RequestSnapshot observe_request(const Request&) const;
    void record_completion(const std::shared_ptr<Request>&, FinishReason, bool failed);

    void wait_for_admission(const std::shared_ptr<Request>& request,
                            const CancellationView& cancellation);

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation);

    // Coalesces admission-visible queue/resource changes. Ordinary prefill/decode progress does not
    // re-arm a temporarily blocked inspection.
    void request_admission_check() noexcept {
        admission_check_pending_.store(true, std::memory_order_release);
        schedule_revision_.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] bool consume_admission_check() noexcept {
        return admission_check_pending_.exchange(false, std::memory_order_acq_rel);
    }

    // One lane's asynchronous binding in flight. The table is indexed by the binding's
    // destination lane; the Session's single context-operation slot keeps at most one entry
    // waiting for its ContextProgress at a time.
    struct MaterializingRequest {
        std::shared_ptr<Request> request;
        LaneId destination;
        BeginSummary summary;
        bool resumed = false;
        typename ResourceManagement::SourceChoice source;
    };

    struct AdmissionDecision {
        std::shared_ptr<Request> request;
        bool restoring = false;
        std::vector<typename ResourceManagement::SourceChoice> sources;
        std::size_t current = 0;
        typename ResourceManagement::ReclaimCursor reclaim;
    };

    struct CaptureDecision {
        SequenceHandle sequence;
        std::uint32_t frontier = 0;
        typename ResourceManagement::ReclaimCursor reclaim;
    };

    // ResourceManagement may release a checkpoint that a Capture or Demote transaction still
    // reads, so the Engine calls publish/finish/abandon/release_source only while the Session has
    // no context operation. A request that ends during one keeps its Session result and lane
    // release; only this resource tail is recorded here and replayed by begin_iteration.
    // `finished`: publish the continuation checkpoint, then finish its owner. Otherwise: release
    // the request's waiting source, then abandon its owner.
    struct DeferredSettlement {
        std::uint64_t request_id = 0;
        typename ResourceManagement::OwnerToken owner = 0;
        std::optional<Checkpoint> checkpoint;
        std::shared_ptr<const typename Request::BasePlan> base;
        std::uint64_t publication_order = 0;
        bool finished                   = false;
    };

    // Fixes, for one settle/cancel pass, whether resource tails must wait. The decision is taken
    // before the pass calls Session::finish/abort, because those calls queue their own residency
    // release and would otherwise make every tail look blocked.
    class SettlementPass {
    public:
        explicit SettlementPass(EngineCore& engine) : engine_(engine) {
            engine_.pass_blocked_ = engine_.resource_calls_blocked();
        }

        ~SettlementPass() { engine_.pass_blocked_.reset(); }

        SettlementPass(const SettlementPass&)            = delete;
        SettlementPass& operator=(const SettlementPass&) = delete;

    private:
        EngineCore& engine_;
    };

    // A context operation is in flight, or earlier tails are waiting and must stay ahead.
    [[nodiscard]] bool resource_calls_blocked() const noexcept {
        return !deferred_settlements_.empty() || instance_.session->has_context_operation();
    }

    // Runs the resource tail now, or records it when resource calls are blocked.
    void settle_resources(DeferredSettlement&& settlement);

    void apply_settlement(const DeferredSettlement& settlement);

    // Replays recorded tails in publication order while no context operation is in flight.
    void flush_deferred_settlements();

    // Whether the Session declines to end this lane's sequence now. A sequence already consumed
    // by commit has nothing left to end; only an execution that still names it defers settlement.
    [[nodiscard]] bool lane_terminal_blocked(const Request& request) const noexcept;

    [[nodiscard]] std::optional<GenerationTimingObservation>
    record_committed_output(const std::shared_ptr<Request>& request, std::uint32_t accepted_tokens);

    void record_first_output(const std::shared_ptr<Request>& request);

    void append_output(const std::shared_ptr<Request>& request, PublishedOutput output,
                       std::optional<GenerationTimingObservation> timing = std::nullopt,
                       EnginePhaseScope* phase                           = nullptr);

    // Model-format failures settle only this already-committed row. Native abort still owns the
    // sequence cleanup; decoder errors must not escape to the Engine-wide invariant handler.
    bool fail_output(const std::shared_ptr<Request>& request, std::uint32_t lane);

    void publish_prompt_progress(const std::shared_ptr<Request>& request);

    void publish_generation_start(const std::shared_ptr<Request>& request, BeginSummary begin);

    void release_reserved_capacity() noexcept;

    void release_consumer(const std::shared_ptr<Request>& request) noexcept;

    void abandon_request(std::shared_ptr<Request> request) noexcept;

    bool prepare_completion(const std::shared_ptr<Request>& request);

    void release_request(const std::shared_ptr<Request>& request) noexcept;

    // The worker publishes the matching scheduling snapshot before making the response visible.
    void publish_completion(const std::shared_ptr<Request>& request, GenerationResult result,
                            std::exception_ptr error = {});

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error);

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason);

    void complete_cancelled(const std::shared_ptr<Request>& request);

    [[nodiscard]] std::vector<bool> snapshot_cancellations() const noexcept;

    bool settle_terminal_requests(HostPhaseMeasurement& boundary);

    void cancel_active_requests(const std::vector<bool>& cancelled_at_boundary,
                                HostPhaseMeasurement& boundary);

    [[nodiscard]] bool expire_pending_requests();

    void commit_pending(PendingBatch&& pending, ComputeFlight flight);

    void reserve_active_capture(const std::shared_ptr<Request>& request, EngineRequestState next);

    bool prepare_semantic_capture(const std::shared_ptr<Request>& request);

    using PrefillRow = typename decltype(std::declval<typename ModelContract::PrefillProgress&>()
                                             .rows)::value_type;

    // Applies the prefill result of one flight: charges every segment to the request's prompt
    // progress, then resolves its rows in member order.
    void resolve_prefill_flight(typename ModelContract::PrefillProgress&& progress,
                                ComputeFlight flight);

    // Applies the result of one mixed flight: the prefill rows settle their segments, then the
    // shared pending batch commits through the flight's batch-row mapping.
    void resolve_mixed_flight(typename ModelContract::MixedProgress&& progress,
                              ComputeFlight flight);

    // Applies one prefill segment's prompt progress. `produced` names whether the segment's
    // row carried the flight's pending batch (it completes the prompt and produces the first
    // token).
    void resolve_prefill_progress(const std::shared_ptr<Request>& request, PrefillRow&& progress,
                                  bool produced);

    void ensure_base_plan(const std::shared_ptr<Request>& request);

    [[nodiscard]] std::optional<std::uint32_t> free_lane() const noexcept;

    [[nodiscard]] std::uint32_t free_lane_count() const noexcept;

    [[nodiscard]] bool resident_empty() const noexcept;

    void record_context_work(const typename ModelContract::ContextProgress& progress,
                             const std::shared_ptr<Request>& owner);

    void resolve_context_progress(typename ModelContract::ContextProgress&& progress,
                                  HostPhaseMeasurement& boundary);

    ReclaimProgress reclaim(const ContextAssessment& assessment, bool allow_pause,
                            std::span<const Checkpoint> excluded               = {},
                            typename ResourceManagement::ReclaimCursor* cursor = nullptr);

    bool pause_reclaim_victim(const ContextAssessment&);

    enum class PauseGoal { YieldLane, ReleaseCapacity, RelocateOnly };
    bool pause_resident(std::uint32_t lane, PauseGoal goal = PauseGoal::YieldLane,
                        const ContextAssessment* pressure = nullptr);

    void order_binding_sources(const std::shared_ptr<Request>&, LaneId,
                               std::vector<typename ResourceManagement::SourceChoice>&);

    void finish_source_wait(const std::shared_ptr<Request>& request,
                            Clock::time_point now) noexcept;

    void retain_admission_source(const std::shared_ptr<Request>& request,
                                 const typename ResourceManagement::SourceChoice& choice,
                                 bool restoring);

    // The outcome of one admission attempt.
    enum class AdmissionStatus : std::uint8_t { Admitted, Deferred, Blocked };

    // The outcome of engaging a forking sample group. `Failed` ends the group: its lanes or its
    // shared plan cannot carry the fork, and the members fail on their own passes.
    enum class Engagement : std::uint8_t { Engaged, Waiting, Failed };

    // One request's ordinary admission path (root or reused source): the per-request state
    // machine over `admission_decision_`, with source selection, reclamation and binding.
    // While flights compute, only a predicted-synchronous root binding proceeds; every path
    // that would start or reclaim through an asynchronous context operation defers to a
    // quiescent iteration.
    [[nodiscard]] AdmissionStatus admit_request(const std::shared_ptr<Request>& request,
                                                 bool restoring);

    // One admission pass over the queue: groups engage atomically, single requests admit in
    // scan order, and each started binding retires before the next one begins.
    [[nodiscard]] bool admit_pending(HostPhaseMeasurement& boundary);

    // Engages a forking sample group: one parent helper lane plus one child lane per member,
    // taken together or not at all. `first` names the member whose plan carried the fork
    // frontier; its stashed prompt clone seeds the parent helper.
    [[nodiscard]] Engagement engage_fork_group(const std::shared_ptr<SampleGroup>& group,
                                               const std::shared_ptr<Request>& first);

    // Forks every engaged group whose parent rests at its fork frontier, adopts the children
    // as their members' sequences, and ends the parent helper.
    void advance_sample_groups(HostPhaseMeasurement& boundary);

    // Cancels and settles groups whose parent or members ended (cancelled, failed, helper
    // error) before the fork completed.
    void settle_sample_groups(HostPhaseMeasurement& boundary);

    [[nodiscard]] std::shared_ptr<SampleGroup> find_group(std::uint64_t id) const noexcept;

    // Ends a forking parent helper: aborts its sequence, settles its resources, releases its
    // lane and capacity, and counts no request statistic.
    void complete_helper(const std::shared_ptr<Request>& parent);

    // Creates one queued request: queue accounting, the output session, the Request record and
    // its pending entry. `queued` false keeps an engine-internal helper out of the pending
    // queue.
    [[nodiscard]] std::shared_ptr<Request>
    create_request(PreparedPrompt prompt, ResolvedRequestOptions options,
                   PromptSummary prompt_summary, double prepare_seconds,
                   OutputConsumerMode consumer_mode, GenerationObservationOptions observation,
                   Clock::time_point pending_deadline, Clock::time_point submitted,
                   bool queued = true);

    // Adopts a synchronously completed root binding: the lane, sequence, residency and
    // resources become the request's without a ContextProgress.
    void adopt_bound_request(const std::shared_ptr<Request>& request,
                             const typename ResourceManagement::SourceChoice& source,
                             std::uint32_t lane, std::optional<SequenceHandle> sequence,
                             std::optional<ResidencyId> residency,
                             const BeginSummary& summary);

    // Longest an admission that waits for capacity keeps the Engine in its iteration to chain its
    // reclaim steps (see begin_iteration).
    static constexpr std::chrono::milliseconds kAdmissionBurst{50};

    // Polls until the running context operation finishes or `deadline` passes. True when no
    // context operation remains.
    bool await_context_operation(HostPhaseMeasurement& boundary, Clock::time_point deadline);

    void observe_scheduling(const std::shared_ptr<Request>& request,
                            GenerationSchedulingTransition transition, Clock::time_point now = {});

    void update_recovery(const std::shared_ptr<Request>& request, bool released = false);

    enum class ReservationScope { Round, Prefill };

    [[nodiscard]] ExecutionUnit execution_unit(const Request& request) const;

    using LaneSlots = std::vector<std::shared_ptr<Request>>;

    // The lanes one iteration may take work from: resident lanes without a pending capture,
    // cancellation, Session execution or context block.
    [[nodiscard]] LaneSlots runnable_lanes() const;

    // Mirrors the resident lanes into the Scheduler's planning table: decode- and control-ready
    // lanes flag their readiness, prefilling lanes carry the most tokens their next unit could
    // process (0 for a lane that takes none, including a fork parent resting at its frontier),
    // and the priority lane names the Session's prefill continuation.
    [[nodiscard]] LaneTable build_lane_table(const LaneSlots& slots) const;

    // The planned membership of one iteration, with the token caps the plan admitted. The
    // replay lane runs alone ahead of the mixed flight.
    struct IterationMembership {
        std::vector<std::uint32_t> control_lanes;
        std::vector<std::uint32_t> decode_lanes;
        std::vector<std::uint32_t> prefill_lanes;
        std::vector<std::uint32_t> prefill_caps;
        std::optional<std::uint32_t> replay_lane;
    };

    // Reserves the units of the planned membership in order. Reclamation, pressure handoffs and
    // pauses run only while nothing computes; otherwise a lane that cannot reserve keeps its
    // work for a quiescent iteration.
    void reserve_planned_units(const IterationMembership& membership);

    [[nodiscard]] bool unit_runnable(std::uint32_t lane) const;

    [[nodiscard]] ComputeFlight freeze_members(std::span<const std::uint32_t> lanes,
                                               FlightKind kind) const;

    // Starts one prefill execution for a non-empty prefix of `lanes`, which are all prefilling
    // or a single replaying lane. Lanes the Session does not admit keep their sequences and lose
    // their Engine unit; the next iteration reserves them again.
    [[nodiscard]] bool start_prefill(std::span<const std::uint32_t> lanes,
                                     std::uint64_t host_submit_ns = 0);

    [[nodiscard]] bool start_decode(std::span<const std::uint32_t> lanes,
                                    std::uint64_t host_submit_ns = 0);

    // Starts one mixed flight: the planned prefill segments followed by the planned decode rows.
    // The flight's membership carries both; its pending batch maps back through `batch_rows`.
    // Returns the decode rows the flight admitted (zero when it fell back to a pure packed
    // prefill), or nullopt when no flight started.
    [[nodiscard]] std::optional<std::uint32_t>
    start_mixed(std::span<const std::uint32_t> prefill_lanes,
                std::span<const std::uint32_t> prefill_caps,
                std::span<const std::uint32_t> decode_lanes, std::uint64_t host_submit_ns);

    void discard_control_preview(const ComputeFlight& flight) noexcept;

    [[nodiscard]] bool start_control(const ControlMembership& membership);

    void resolve_control(ComputeFlight& flight);

    void supply_decision(ExecutionTicket ticket, const DecisionRequest& decision);

    void record_decision_upload(const DecisionConsumed& consumed);

    [[nodiscard]] bool poll_execution(HostPhaseMeasurement& boundary);

    [[nodiscard]] bool execution_outstanding() const noexcept;


    // The flight a ticket names in `compute_` / `retiring_`; null when the ticket's frame holds
    // another flight or none.
    [[nodiscard]] ComputeFlight* find_compute(const ExecutionTicket& ticket) noexcept;
    [[nodiscard]] ComputeFlight* find_retiring(const ExecutionTicket& ticket) noexcept;

    // Flights in either table.
    [[nodiscard]] std::size_t compute_flights() const noexcept;

    // The oldest flight still computing: it names the host work class of a wait.
    [[nodiscard]] const ComputeFlight* oldest_compute() const noexcept;

    // Whether an executor compute frame can take another flight.
    [[nodiscard]] bool compute_frame_available() const noexcept;

    // Records a started flight in the table entry of its ticket's frame.
    void admit_compute(ComputeFlight&& flight);

    // Records one fully retired flight in the runtime statistics.
    void record_flight(const ComputeFlight& flight, const ExecutionSample& sample) noexcept;

    void record_service(ComputeFlight& flight) noexcept;
    void log_worker_failure(std::exception_ptr error) const noexcept;
    void fail_all(std::exception_ptr error) noexcept;

    // One iteration's boundary: settles finished work, starts due captures, hands the service
    // turn over, restores the pressure owner, advances forking sample groups, and admits new
    // work. Admission of a synchronous root binding and the fork step run while flights
    // compute; reclamation and asynchronous bindings keep waiting for a quiescent iteration.
    void begin_iteration(HostPhaseMeasurement& boundary);

    // Plans and submits one iteration's flights: the control rows, one replaying lane, and one
    // mixed flight of every planned decode row plus the prefill segments the token budget
    // admits. A Session that cannot mix, or a first prefill segment that is not mixable,
    // falls back to separate decode and prefill flights. True when any flight was submitted.
    [[nodiscard]] bool submit_iteration(HostPhaseMeasurement& boundary);

    void worker_loop() noexcept;

    Instance& instance_;
    const std::uint32_t max_context_;
    const std::uint32_t max_concurrency_;
    const std::uint32_t prefill_chunk_;
    const std::size_t max_outstanding_;
    const std::uint32_t speculative_positions_;
    const std::chrono::milliseconds pending_timeout_;
    ResourceManagement resources_;
    PlacementId next_placement_tie_;

    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::unique_ptr<MeasurementRequest> measurement_;
    std::size_t preparing_requests_       = 0;
    std::size_t outstanding_              = 0;
    std::uint64_t next_request_id_        = 1;
    std::uint64_t submitted_requests_     = 0;
    std::uint64_t next_publication_order_ = 1;
    std::vector<std::shared_ptr<Request>> slots_;
    // Asynchronous bindings in flight, indexed by destination lane; the Session's single
    // context-operation slot keeps at most one entry waiting at a time.
    std::vector<std::optional<MaterializingRequest>> materializing_;
    std::vector<std::shared_ptr<SampleGroup>> groups_;
    std::vector<std::uint64_t> lane_residency_epochs_;
    std::uint64_t next_group_id_ = 1;
    std::optional<AdmissionDecision> admission_decision_;
    std::uint64_t admission_generation_ = 1;
    std::vector<std::optional<CaptureDecision>> capture_decisions_;
    std::shared_ptr<Request> context_owner_;
    std::deque<DeferredSettlement> deferred_settlements_;
    std::optional<bool> pass_blocked_;
    std::vector<std::shared_ptr<Request>> paused_;
    Scheduling scheduler_;
    // Flights of the executor's compute frames, indexed by ticket.frame.index. A flight is in
    // `compute_` from submission until its result is resolved; one whose model commit was issued
    // waits for device retirement in `retiring_`. A frame holds at most one of the two.
    std::vector<std::optional<ComputeFlight>> compute_;
    std::vector<std::optional<ComputeFlight>> retiring_;
    std::uint64_t flight_index_     = 0;
    std::vector<bool> control_executed_{};
    std::vector<std::optional<UnitPermit>> reserved_units_{};
    std::atomic<bool> admission_check_pending_{false};
    std::atomic<std::uint64_t> schedule_revision_{0};
    std::uint64_t published_revision_          = 0;
    Clock::time_point pending_deadline_        = Clock::time_point::max();
    std::uint64_t worker_accounted_elapsed_ns_ = 0;
    HostWorkClass current_host_work_class_     = HostWorkClass::Control;
    std::vector<std::uint32_t> current_decode_lanes_{};
    std::size_t current_decode_lane_count_ = 0;
    RuntimeStats cumulative_stats_;
    RuntimeStats stats_snapshot_;
    EngineSnapshot published_snapshot_;
    std::vector<RequestSnapshot> request_observations_, published_requests_;
    std::vector<ExecutionProgress> execution_observations_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime

#include "runtime/engine/engine_metrics.inl"

#include "runtime/engine/engine_requests.inl"
#include "runtime/engine/engine_schedule.inl"
#include "runtime/engine/engine_execution.inl"
