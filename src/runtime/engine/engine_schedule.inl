#pragma once

namespace ninfer::runtime {

template <class Instance>
void EngineCore<Instance>::reserve_active_capture(const std::shared_ptr<Request>& request,
                                                  EngineRequestState next) {
    request->model_state        = next;
    request->capture_pending    = true;
    request->post_capture_state = next;
    // A capture is a context operation: it starts only without any execution in flight. While
    // another flight runs the request keeps its pending capture and the iteration boundary
    // starts it at the next quiescent boundary.
    if (instance_.session->has_context_operation() ||
        instance_.session->execution_pending(*request->sequence) || compute_flights() != 0) {
        return;
    }
    if (scheduler_.progress_owner()) {
        instance_.session->skip_capture(*request->sequence);
        request->capture_pending = false;
        return;
    }
    if (!instance_.session->start_capture(*request->sequence)) {
        instance_.session->skip_capture(*request->sequence);
        request->capture_pending = false;
        return;
    }
    context_owner_ = request;
}

template <class Instance>
bool EngineCore<Instance>::prepare_semantic_capture(const std::shared_ptr<Request>& request) {
    auto& decision = capture_decisions_[request->lane->value];
    if (decision && decision->sequence != *request->sequence) { decision.reset(); }
    if (instance_.session->capture_is_input(*request->sequence) && !decision) {
        (void)resources_.recycle_input(*instance_.session, request->continuation_owner);
    }
    // A Retry re-entry changes no engine state and a Changed re-entry frees capacity the
    // preparation may still not place (its shortage can be another domain). Without a bound a
    // preparation whose reservation keeps failing spins this iteration and starves every other
    // engine service, whatever reclaim reports. After a few attempts the capture defers to a
    // later iteration: committed releases persist, so a later pass retries from better state.
    constexpr int kMaxCaptureReclaimAttempts = 8;
    int attempts = 0;
    while (const auto capture = instance_.session->prepare_capture(*request->sequence)) {
        if (scheduler_.progress_owner()) {
            instance_.session->skip_capture(*request->sequence);
            decision.reset();
            continue;
        }
        if (instance_.session->capture_is_shared(*request->sequence) &&
            resources_.retain_shared_checkpoint(
                *instance_.session, *request->base_plan, capture->frontier,
                instance_.session->placement(*request->sequence), capture->evidence)) {
            instance_.session->skip_capture(*request->sequence, CaptureSkip::Retained);
            decision.reset();
            continue;
        }
        if (capture->reserved) {
            decision.reset();
            return true;
        }
        if (instance_.session->has_context_operation()) { return false; }
        const auto admission = resources_.capture_admission(
            *instance_.session, request->continuation_owner, *request->base_plan, capture->frontier,
            instance_.session->capture_is_input(*request->sequence));
        if (!decision || decision->frontier != capture->frontier) {
            decision.emplace(
                CaptureDecision{*request->sequence, capture->frontier,
                                resources_.begin_reclaim(*instance_.session, admission)});
        }
        if (!capture->host.empty()) {
            if (!instance_.session->can_place_host(capture->host)) {
                (void)instance_.session->release_redundant_host({}, request->sequence);
            }
            if (const auto victims =
                    resources_.host_victims(*instance_.session, capture->host, std::nullopt, {}, {},
                                            admission, &decision->reclaim)) {
                const auto loss  = resources_.release_cost(*instance_.session, *victims);
                const auto retry = instance_.session->prepare_capture(*request->sequence, *victims);
                if (retry && !retry->retired_points.empty()) {
                    if (retry->retired_points != *victims)
                        throw std::logic_error("capture consumed only part of its victim quote");
                    resources_.adopt_release(*instance_.session, retry->retired_points,
                                             decision->reclaim, loss);
                    request_admission_check();
                }
                if (retry && retry->reserved) {
                    decision.reset();
                    return true;
                }
            }
        }
        const auto progress =
            resources_.reclaim(*instance_.session, capture->assessment, {}, decision->reclaim);
        if (progress == ReclaimProgress::Transferring || progress == ReclaimProgress::Deferred) {
            return false;
        }
        if (progress == ReclaimProgress::Retry) {
            if (++attempts >= kMaxCaptureReclaimAttempts) return false;
            continue;
        }
        if (progress == ReclaimProgress::Changed) {
            if (++attempts >= kMaxCaptureReclaimAttempts) return false;
            continue;
        }
        instance_.session->skip_capture(*request->sequence);
        decision.reset();
    }
    decision.reset();
    return true;
}

template <class Instance>
void EngineCore<Instance>::ensure_base_plan(const std::shared_ptr<Request>& request) {
    if (!request->base_plan) {
        request->base_plan.emplace(instance_.session->plan_request(std::move(request->prompt),
                                                                   request->options.execution));
        const auto& summary = request->base_plan->summary();
        request->budget.emplace(summary.effective_output_tokens, summary.effective_limit_reason);
        request->generated.reserve(summary.effective_output_tokens);
    }
}

template <class Instance>
[[nodiscard]] std::optional<std::uint32_t> EngineCore<Instance>::free_lane() const noexcept {
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (!slots_[lane] && !materializing_[lane]) { return lane; }
    }
    return std::nullopt;
}

template <class Instance>
[[nodiscard]] std::uint32_t EngineCore<Instance>::free_lane_count() const noexcept {
    std::uint32_t count = 0;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (!slots_[lane] && !materializing_[lane]) { ++count; }
    }
    return count;
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::resident_empty() const noexcept {
    return std::all_of(materializing_.begin(), materializing_.end(),
                       [](const auto& entry) { return !entry; }) &&
           std::all_of(slots_.begin(), slots_.end(), [](const auto& p) { return !p; });
}

template <class Instance>
[[nodiscard]] std::shared_ptr<typename EngineCore<Instance>::SampleGroup>
EngineCore<Instance>::find_group(std::uint64_t id) const noexcept {
    std::shared_ptr<SampleGroup> found;
    {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find_if(groups_.begin(), groups_.end(),
                                     [id](const auto& group) { return group->id == id; });
        if (it != groups_.end() && (*it)->sealed) { found = *it; }
    }
    return found;
}

template <class Instance>
void EngineCore<Instance>::resolve_context_progress(
    typename ModelContract::ContextProgress&& progress, HostPhaseMeasurement& boundary) {
    if (!progress.complete) {
        throw std::logic_error("Session emitted an incomplete context result");
    }
    const auto owner = context_owner_;
    record_context_work(progress, owner);
    if (progress.kind == decltype(progress.kind)::Maintenance) {
        request_admission_check();
        return;
    }
    if (owner && progress.kind == decltype(progress.kind)::Pause) {
        owner->speculative_stats = std::move(progress.request_speculative);
    }
    std::optional<MaterializingRequest> control;
    if (owner && owner->is_materializing()) {
        const auto entry =
            std::find_if(materializing_.begin(), materializing_.end(),
                         [&owner](const auto& candidate) {
                             return candidate && candidate->request == owner;
                         });
        if (entry == materializing_.end()) {
            throw std::logic_error("materializing request lost its lane entry");
        }
        control = std::move(*entry);
        entry->reset();
    }
    if (control) {
        const auto request = control->request;
        context_owner_.reset();
        if (!progress.published) {
            complete_cancelled(request);
            request_admission_check();
            return;
        }
        auto carried = progress.private_points;
        if (control->resumed) {
            for (const auto point : resources_.points(request->continuation_owner)) {
                if (instance_.session->valid_checkpoint(point) &&
                    std::find(carried.begin(), carried.end(), point) == carried.end()) {
                    carried.push_back(point);
                }
            }
        }
        request->continuation_owner = resources_.adopt(
            *instance_.session, control->source, *request->base_plan, request->publication_order,
            control->resumed, carried, progress.retired_checkpoints);
        resources_.release_source(*instance_.session, request->id);
        request->lane      = control->destination;
        request->sequence  = progress.sequence;
        request->residency = progress.residency;
        if (!request->residency || request->residency->lane != control->destination) {
            throw std::logic_error("binding completed without the expected residency");
        }
        if (!request->sequence) {
            throw std::logic_error("binding completed without a native sequence");
        }
        request->placement = instance_.session->placement(*request->sequence);
        request->placement_tokens =
            std::max(request->placement_tokens, instance_.session->load_tokens(*request->sequence));
        request->model_state = progress.recovery.route == GenerationRecoveryRoute::Replay
                                   ? EngineRequestState::Replay
                               : control->resumed ? request->resume_phase
                                                  : EngineRequestState::Prefill;
        if (control->resumed) {
            request->recovery = progress.recovery;
            if (progress.recovery.route == GenerationRecoveryRoute::Snapshot) {
                ++cumulative_stats_.snapshot_restores;
                ++request->snapshot_restores;
            } else if (progress.recovery.route == GenerationRecoveryRoute::Direct) {
                ++cumulative_stats_.direct_restores;
                ++request->direct_restores;
            }
            request->suspended.reset();
            request->recovery_pending = instance_.session->recovery_pending(*request->sequence);
            if (request->paused_at) {
                request->paused_ns += elapsed_ns(*request->paused_at, Clock::now());
            }
            request->paused_at.reset();
            observe_scheduling(request, GenerationSchedulingTransition::Restored);
        } else {
            request->initial_binding_ns = elapsed_ns(*request->admitted_at, Clock::now());
            if (control->summary.reused_prompt_tokens) {
                ++cumulative_stats_.checkpoint_selections;
            } else {
                ++cumulative_stats_.root_selections;
            }
            cumulative_stats_.reused_prompt_tokens += control->summary.reused_prompt_tokens;
            cumulative_stats_.prompt_tokens += control->summary.prompt_tokens;
            cumulative_stats_.last_selected_frontier_tokens = control->summary.reused_prompt_tokens;
            // Generation start was published when the first binding obtained its resources.
        }
        finish_engine_phase(boundary, EngineHostPhase::Boundary);
        slots_[control->destination.value] = request;
        ++slots_epoch_;
        boundary = begin_host_phase();
    } else if (context_owner_) {
        const auto request = context_owner_;
        if (request->model_state == EngineRequestState::Pausing) {
            const auto lane = request->lane->value;
            if (!progress.published) {
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
                context_owner_.reset();
                complete_cancelled(request);
                boundary = begin_host_phase();
                return;
            }
            request->placement.reset();
            request->placement_tokens = 0;
            request->suspended        = std::move(progress.paused);
            request->model_state      = EngineRequestState::Paused;
            observe_scheduling(request, GenerationSchedulingTransition::Paused);
            paused_.push_back(request);
            std::sort(paused_.begin(), paused_.end(),
                      [](const auto& a, const auto& b) { return a->id < b->id; });
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            slots_[lane].reset();
            ++slots_epoch_;
            request->lane.reset();
            request->residency.reset();
            request->sequence.reset();
            request_admission_check();
            boundary = begin_host_phase();
        } else if constexpr (kInFlightCapture) {
            throw std::logic_error("context operation completed for a request that is not pausing");
        } else {
            if (progress.published) {
                ++cumulative_stats_.active_captures_completed;
            } else {
                ++cumulative_stats_.active_captures_aborted;
            }
            request->capture_pending = false;
            request->model_state     = request->post_capture_state;
            for (const auto point : progress.captured_checkpoints) {
                resources_.publish(*instance_.session, request->continuation_owner, point);
            }
        }
        context_owner_.reset();
    }
    request_admission_check();
    return;
}

template <class Instance>
ReclaimProgress EngineCore<Instance>::reclaim(const ContextAssessment& assessment, bool allow_pause,
                                              std::span<const Checkpoint> excluded,
                                              typename ResourceManagement::ReclaimCursor* cursor) {
    if (assessment.ready()) return ReclaimProgress::Retry;
    if (assessment.waiting()) return ReclaimProgress::Deferred;
    if (assessment.impossible()) return ReclaimProgress::Blocked;
    if constexpr (!kInFlightCapture) {
        if (instance_.session->reclaim_capture_reservation(assessment)) {
            request_admission_check();
            return ReclaimProgress::Changed;
        }
    }
    const auto progress =
        cursor ? resources_.reclaim(*instance_.session, assessment, excluded, *cursor)
               : resources_.reclaim(*instance_.session, assessment, excluded);
    if (progress != ReclaimProgress::Blocked) {
        if (progress == ReclaimProgress::Changed) {
            request_admission_check();
        }
        return progress;
    }
    if (cursor && cursor->rights.purpose != ReclaimPurpose::Execution)
        return ReclaimProgress::Blocked;
    std::vector<Checkpoint> paused_sources;
    for (auto it = paused_.rbegin(); it != paused_.rend(); ++it) {
        const auto& request = **it;
        if (!request.suspended || !request.suspended->has_snapshot()) { continue; }
        const auto handle = *request.suspended->snapshot_handle();
        if (std::find(excluded.begin(), excluded.end(), handle) == excluded.end()) {
            paused_sources.push_back(handle);
        }
    }
    const auto releases =
        instance_.session->plan_releases(paused_sources, excluded, assessment.demand);
    for (const auto& release : releases.releases) {
        if (instance_.session->useful_release_bytes(release.released, assessment) == 0 &&
            !instance_.session->releases_metadata(release.released, assessment)) {
            continue;
        }
        bool changed = false;
        for (const auto handle : release.sources) {
            const auto owner =
                std::find_if(paused_.begin(), paused_.end(), [&](const auto& request) {
                    return request->suspended && request->suspended->snapshot_handle() == handle;
                });
            if (owner == paused_.end() ||
                !instance_.session->revoke_snapshot(*(*owner)->suspended)) {
                continue;
            }
            (*owner)->recovery.reset();
            observe_scheduling(*owner, GenerationSchedulingTransition::SnapshotRevoked);
            changed = true;
        }
        if (changed) {
            request_admission_check();
            return ReclaimProgress::Changed;
        }
    }
    if (allow_pause && pause_reclaim_victim(assessment))
        return instance_.session->has_context_operation() ? ReclaimProgress::Transferring
                                                          : ReclaimProgress::Changed;
    return ReclaimProgress::Blocked;
}

template <class Instance>
bool EngineCore<Instance>::pause_reclaim_victim(const ContextAssessment& pressure) {
    std::vector<std::uint32_t> skipped;
    while (const auto victim = scheduler_.preemption_victim(slots_, max_concurrency_, skipped)) {
        if (pause_resident(*victim, PauseGoal::ReleaseCapacity, &pressure)) { return true; }
        skipped.push_back(*victim);
    }
    return false;
}

template <class Instance>
bool EngineCore<Instance>::pause_resident(std::uint32_t lane, PauseGoal goal,
                                          const ContextAssessment* pressure) {
    if (instance_.session->has_context_operation()) { return false; }
    auto request = slots_[lane];
    if (!request || !request->sequence || instance_.session->execution_pending(*request->sequence))
        return false;
    std::optional<PauseAction> selected;
    auto options = instance_.session->pause_options(*request->sequence);
    for (const auto& option : options) {
        if (goal == PauseGoal::RelocateOnly && option.action.kind != PauseKind::Relocate) {
            continue;
        }
        if (pressure &&
            !(goal == PauseGoal::RelocateOnly && option.action.kind == PauseKind::Relocate) &&
            !instance_.session->pause_contributes(*request->sequence, option, *pressure)) {
            continue;
        }
        if (option.action.kind == PauseKind::Relocate) {
            const auto target = instance_.session->assess(option.demand);
            if (!target.ready()) {
                auto cursor =
                    resources_.begin_reclaim(*instance_.session, std::nullopt,
                                             ReclaimRights{ReclaimPurpose::Execution, request->id});
                const auto progress =
                    resources_.reclaim(*instance_.session, target, option.sources, cursor);
                if (progress == ReclaimProgress::Transferring) { return true; }
                if (progress == ReclaimProgress::Changed) {
                    request_admission_check();
                    return true;
                }
                if (progress != ReclaimProgress::Retry ||
                    !instance_.session->assess(option.demand).ready()) {
                    continue;
                }
            }
        }
        std::vector<Checkpoint> victims;
        if (option.host_write) {
            if (!instance_.session->can_place_host(*option.host_write)) {
                (void)instance_.session->release_redundant_host({}, request->sequence);
            }
            std::vector<Checkpoint> later;
            for (auto it = paused_.rbegin(); it != paused_.rend(); ++it) {
                if ((*it)->id > request->id && (*it)->suspended &&
                    (*it)->suspended->has_snapshot()) {
                    later.push_back(*(*it)->suspended->snapshot_handle());
                }
            }
            auto candidate = resources_.host_victims(*instance_.session, *option.host_write,
                                                     std::nullopt, later);
            if (!candidate) { continue; }
            victims = std::move(*candidate);
        }
        if (!instance_.session->prepare_pause(*request->sequence, option.action, victims)) {
            continue;
        }
        for (const auto handle : victims) {
            if (resources_.erase(*instance_.session, handle)) { continue; }
            const auto owner = std::find_if(paused_.begin(), paused_.end(), [&](const auto& other) {
                return other->suspended && other->suspended->snapshot_handle() == handle;
            });
            if (owner == paused_.end() ||
                !instance_.session->revoke_snapshot(*(*owner)->suspended)) {
                throw std::logic_error("pause victim changed after preflight");
            }
            (*owner)->recovery.reset();
            observe_scheduling(*owner, GenerationSchedulingTransition::SnapshotRevoked);
        }
        selected = option.action;
        break;
    }
    if (!selected) { return false; }
    ExecutionCallScope call(*this);
    const auto pause_started = Clock::now();
    if (!instance_.session->start_pause(*request->sequence, *selected)) { return false; }
    call.finish();
    if (!request->is_replaying()) { request->resume_phase = request->model_state; }
    request->model_state = EngineRequestState::Pausing;
    request->paused_at   = pause_started;
    ++request->preemption_count;
    ++cumulative_stats_.preemptions;
    request->recovery.reset();
    observe_scheduling(request, GenerationSchedulingTransition::PauseStarted, pause_started);
    context_owner_ = request;
    capture_decisions_[lane].reset();
    return true;
}

template <class Instance>
void EngineCore<Instance>::finish_source_wait(const std::shared_ptr<Request>& request,
                                              Clock::time_point now) noexcept {
    if (!request->source_wait_started) { return; }
    request->admission.source_wait_seconds +=
        static_cast<double>(elapsed_ns(*request->source_wait_started, now)) * 1.0e-9;
    request->source_wait_started.reset();
}

template <class Instance>
void EngineCore<Instance>::retain_admission_source(
    const std::shared_ptr<Request>& request,
    const typename ResourceManagement::SourceChoice& choice, bool restoring) {
    resources_.retain_source(*instance_.session, request->id, choice);
    if (!restoring && choice.source.reused_tokens && !request->source_wait_started) {
        request->source_wait_started = Clock::now();
    } else if (!restoring && !choice.source.reused_tokens) {
        finish_source_wait(request, Clock::now());
    }
}

template <class Instance>
void EngineCore<Instance>::order_binding_sources(
    const std::shared_ptr<Request>& request, LaneId lane,
    std::vector<typename ResourceManagement::SourceChoice>& sources) {
    const auto placements = instance_.session->placements(*request->base_plan);
    if (placements.size() <= 1) { return; }
    std::vector<PlacementLoad> loads;
    std::vector<PlacementCandidate> targets;
    for (const auto placement : placements) {
        loads.push_back({placement});
        targets.push_back({placement});
    }
    const auto account = [&](const std::shared_ptr<Request>& active) {
        if (!active || !active->placement) { return; }
        auto found = std::find_if(loads.begin(), loads.end(), [&](const auto& load) {
            return load.placement == *active->placement;
        });
        if (found == loads.end()) {
            loads.push_back({*active->placement});
            found = std::prev(loads.end());
        }
        if (active->placement_tokens > UINT64_MAX - found->tokens) {
            throw std::overflow_error("placement token load overflow");
        }
        found->tokens += active->placement_tokens;
        ++found->requests;
    };
    for (const auto& active : slots_) { account(active); }
    for (const auto& entry : materializing_) {
        if (entry) { account(entry->request); }
    }
    const auto* resume = request->suspended ? &*request->suspended : nullptr;
    const auto kind    = request->resume_phase == EngineRequestState::Prefill ? UnitKind::Prefill
                         : request->resume_phase == EngineRequestState::ControlReady
                             ? UnitKind::Control
                             : UnitKind::Decode;
    const auto tokens =
        kind == UnitKind::Prefill ? 0U
        : kind == UnitKind::Control
            ? static_cast<std::uint32_t>(request->output.pending_control_tokens().size())
            : 1U;
    std::vector<typename ResourceManagement::SourceChoice> waiting;
    std::erase_if(sources, [&](const auto& choice) {
        auto assessment = instance_.session->assess_binding(*request->base_plan, lane,
                                                            choice.source, resume, kind, tokens);
        if (!assessment.source_valid) { return true; }
        if (!assessment.capacity_possible) {
            auto transferable = choice;
            const ReclaimRights rights{
                resume ? ReclaimPurpose::Execution : ReclaimPurpose::FreshAdmission, request->id};
            if (resources_.prepare_source(*instance_.session, *request->base_plan, transferable,
                                          request->id, rights, resume)) {
                assessment = instance_.session->assess_binding(
                    *request->base_plan, lane, transferable.source, resume, kind, tokens);
            }
        }
        if (!assessment.source_valid || !assessment.capacity_possible) { return true; }
        if (assessment.assessment.waiting()) {
            waiting.push_back(choice);
            return true;
        }
        auto target = std::find_if(targets.begin(), targets.end(), [&](const auto& target) {
            return target.placement == choice.source.placement;
        });
        if (target == targets.end()) {
            throw std::logic_error("binding candidate placement is absent from the model plan");
        }
        target->feasible = true;
        if (choice.source.local_device) {
            target->local_prefix_tokens =
                std::max<std::uint64_t>(target->local_prefix_tokens, choice.source.reused_tokens);
        }
        return false;
    });
    const auto weight  = std::max<std::uint64_t>(request->base_plan->summary().prompt_tokens,
                                                resume ? resume->frontier() : 0);
    const auto ordered = order_placements(loads, targets, weight, next_placement_tie_);
    std::vector<typename ResourceManagement::SourceChoice> result;
    result.reserve(sources.size() + waiting.size());
    for (const auto index : ordered) {
        for (auto& source : sources) {
            if (source.source.placement == targets[index].placement) {
                result.push_back(std::move(source));
            }
        }
    }
    result.insert(result.end(), std::make_move_iterator(waiting.begin()),
                  std::make_move_iterator(waiting.end()));
    sources = std::move(result);
}

// Creates one queued request: the queue accounting and identity, the output session, the
// Request record and its pending entry.
template <class Instance>
[[nodiscard]] std::shared_ptr<typename EngineCore<Instance>::Request>
EngineCore<Instance>::create_request(PreparedPrompt prompt, ResolvedRequestOptions options,
                                     PromptSummary prompt_summary, double prepare_seconds,
                                     OutputConsumerMode consumer_mode,
                                     GenerationObservationOptions observation,
                                     Clock::time_point pending_deadline,
                                     Clock::time_point submitted, SampleGroup* group,
                                     std::uint32_t branch) {
    // Only the branches of a sample group share their prompt with sibling requests.
    if (group == nullptr) { options.execution.sample_count = 1; }
    std::uint64_t request_id        = 0;
    std::uint64_t publication_order = 0;
    {
        std::unique_lock lock(queue_mutex_);
        if (!queue_cv_.wait_until(lock, pending_deadline,
                                  [&] { return stopping_ || failed_ || !measurement_; }))
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired at a measurement boundary");
        if (stopping_ || failed_) {
            throw RequestError(RequestErrorKind::Unavailable,
                               "inference engine is unavailable");
        }
        if (outstanding_ >= max_outstanding_) {
            throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
        }
        if (next_request_id_ == 0 || next_publication_order_ == 0) {
            throw std::overflow_error("request identity space exhausted");
        }
        ++outstanding_;
        ++preparing_requests_;
        request_id        = next_request_id_++;
        publication_order = next_publication_order_++;
    }

    std::shared_ptr<Request> request;
    try {
        const auto constraint_started = observation.constraint_timings ? Clock::now() : submitted;
        auto output = instance_.frontend.make_output_session(
            prompt, options.stop, options.output, options.execution.thinking, options.constraint,
            options.tool_choice);
        if (Clock::now() >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired during grammar preparation");
        }
        const std::uint32_t capacity_output =
            max_context_ - prompt_summary.prompt_tokens + static_cast<std::uint32_t>(1);
        try {
            output.validate_generation_capacity(
                std::min(options.execution.requested_output_tokens, capacity_output));
        } catch (const std::invalid_argument& error) {
            throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                               error.what());
        }
        const auto ready = Clock::now();
        output.observe_constraint(
            observation.constraint_timings,
            std::chrono::duration<double>(ready - constraint_started).count());
        prepare_seconds += std::chrono::duration<double>(ready - submitted).count();
        request = std::make_shared<Request>(request_id, publication_order, std::move(prompt),
                                            std::move(output), prompt_summary, prepare_seconds,
                                            std::move(options), consumer_mode,
                                            std::move(observation), pending_deadline, ready);
        request->decode_by_batch.resize(max_concurrency_);
        request->speculative_stats.attempted_per_position.resize(speculative_positions_);
        request->speculative_stats.accepted_per_position.resize(speculative_positions_);
        request->context_transfers.resize(instance_.session->kv_pools().size() + 1);
    } catch (...) {
        {
            std::lock_guard lock(queue_mutex_);
            --preparing_requests_;
            --outstanding_;
        }
        queue_cv_.notify_all();
        throw;
    }

    {
        std::lock_guard lock(queue_mutex_);
        --preparing_requests_;
        if (stopping_ || failed_) {
            --outstanding_;
            throw RequestError(RequestErrorKind::Unavailable,
                               "inference engine is unavailable");
        }
        pending_.push_back(request);
        ++submitted_requests_;
        if (group != nullptr) {
            request->sample_group_id = group->id;
            request->branch_index    = branch;
            group->members.push_back(request);
        }
    }
    queue_cv_.notify_all();
    return request;
}

// Adopts a synchronously completed root binding: the Session installed the sequence and its
// residency before start_binding returned, so the lane is adopted without a ContextProgress.
template <class Instance>
void EngineCore<Instance>::adopt_bound_request(
    const std::shared_ptr<Request>& request,
    const typename ResourceManagement::SourceChoice& source, std::uint32_t lane,
    std::optional<SequenceHandle> sequence, std::optional<ResidencyId> residency,
    const BeginSummary& summary) {
    request->continuation_owner = resources_.adopt(
        *instance_.session, source, *request->base_plan, request->publication_order, false, {},
        source.source.retired_points);
    resources_.release_source(*instance_.session, request->id);
    request->lane      = LaneId{lane};
    request->sequence  = sequence;
    request->residency = residency;
    if (!request->residency || request->residency->lane != LaneId{lane}) {
        throw std::logic_error("binding completed without the expected residency");
    }
    if (!request->sequence) {
        throw std::logic_error("binding completed without a native sequence");
    }
    request->placement = instance_.session->placement(*request->sequence);
    request->placement_tokens =
        std::max(request->placement_tokens, instance_.session->load_tokens(*request->sequence));
    request->model_state = EngineRequestState::Prefill;
    request->initial_binding_ns = elapsed_ns(*request->admitted_at, Clock::now());
    if (summary.reused_prompt_tokens) {
        ++cumulative_stats_.checkpoint_selections;
    } else {
        ++cumulative_stats_.root_selections;
    }
    cumulative_stats_.reused_prompt_tokens += summary.reused_prompt_tokens;
    cumulative_stats_.prompt_tokens += summary.prompt_tokens;
    cumulative_stats_.last_selected_frontier_tokens = summary.reused_prompt_tokens;
    slots_[lane] = request;
    ++slots_epoch_;
}

template <class Instance>
typename EngineCore<Instance>::AdmissionStatus
EngineCore<Instance>::admit_request(const std::shared_ptr<Request>& request, bool restoring) {
    if (instance_.session->has_context_operation()) { return AdmissionStatus::Deferred; }
    const auto lane  = free_lane();
    const auto owner = scheduler_.progress_owner();
    const auto suspended =
        restoring ? std::find_if(paused_.begin(), paused_.end(),
                                 [&](const auto& r) { return r->id == owner; })
                  : paused_.end();
    if (restoring && suspended == paused_.end()) return AdmissionStatus::Blocked;
    if (!lane && restoring) {
        if (const auto victim = scheduler_.preemption_victim(slots_, max_concurrency_))
            return pause_resident(*victim) ? AdmissionStatus::Admitted : AdmissionStatus::Blocked;
        return AdmissionStatus::Blocked;
    }
    if (admission_decision_ && (admission_decision_->restoring != restoring ||
                                (restoring && *suspended != admission_decision_->request))) {
        admission_decision_.reset();
    }
    if (!admission_decision_) {
        if (!restoring) {
            scheduler_.admission_candidate_checked(request->id);
            if (request->admission_generation == admission_generation_) {
                return AdmissionStatus::Deferred;
            }
            request->admission_generation = admission_generation_;
        }
        try {
            ensure_base_plan(request);
        } catch (const RequestError&) {
            // Planning has not bound a sequence or submitted device work. A native
            // request rejection therefore ends only this pending request.
            if (restoring) throw;
            const auto error = std::current_exception();
            {
                std::lock_guard lock(queue_mutex_);
                std::erase(pending_, request);
            }
            complete_error(request, error);
            request_admission_check();
            return AdmissionStatus::Admitted;
        }
        std::shared_ptr<SampleGroup> group;
        if (!restoring && request->sample_group_id) {
            group = find_group(request->sample_group_id);
            // A group is admitted once all of its branches exist.
            if (!group) { return AdmissionStatus::Deferred; }
        }
        if constexpr (kInFlightCapture) {
            // A forkable request belongs to a sample group: its prefill rests at the fork frontier
            // and every branch decodes the last prompt token.
            if (!restoring && request->base_plan->fork_frontier() != 0) {
                if (!group) { throw std::logic_error("a forkable request has no sample group"); }
                const auto outcome = engage_fork_group(*group, request);
                if (outcome == Engagement::Failed) {
                    const auto error = std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "sample group exceeds the fork limit"));
                    {
                        std::lock_guard lock(queue_mutex_);
                        std::erase(pending_, request);
                    }
                    complete_error(request, error);
                    request_admission_check();
                    return AdmissionStatus::Admitted;
                }
                if (outcome == Engagement::Waiting) { return AdmissionStatus::Deferred; }
                // The group's lanes are free: the request binds as its parent.
                group.reset();
            }
        }
        if (group) {
            // A group that does not fork gains all of its lanes together: a member waits until the
            // lanes its still queued branches need are free.
            std::uint32_t remaining = 0;
            for (const auto& member : group->members) {
                if (member->is_queued()) { ++remaining; }
            }
            if (remaining > free_lane_count()) { return AdmissionStatus::Deferred; }
        }
        const auto retained = resources_.retained_source(request->id);
        const auto revoked  = resources_.source_revocations(request->id);
        if (!restoring && revoked > request->admission.revoked_checkpoints) {
            request->admission.revoked_checkpoints = revoked;
            if (!retained) {
                request->admission.fallback_reason = AdmissionFallbackReason::SourceRevoked;
            }
        }
        auto sources =
            resources_.candidates(*instance_.session, *request->base_plan,
                                  restoring ? request->suspended->frontier() : UINT32_MAX,
                                  restoring && request->continuation_owner
                                      ? std::optional(request->continuation_owner)
                                      : std::nullopt,
                                  request->id, request->publication_order,
                                  restoring ? &*request->suspended : nullptr);
        if (lane) { order_binding_sources(request, LaneId{*lane}, sources); }
        if (!restoring && !sources.empty()) {
            const auto preferred = sources.front().source.reused_tokens;
            if (!request->admission_observed) {
                request->admission.preferred_reused_tokens = preferred;
                request->admission_observed                = true;
            } else if (retained && preferred < retained->source.reused_tokens) {
                request->admission.fallback_reason = AdmissionFallbackReason::CostChanged;
            }
        }
        admission_decision_.emplace(AdmissionDecision{
            request, restoring, std::move(sources), 0,
            resources_.begin_reclaim(*instance_.session, std::nullopt,
                                     ReclaimRights{
                                         restoring ? ReclaimPurpose::Execution
                                                   : ReclaimPurpose::FreshAdmission,
                                         request->id})});
    }
    const auto& decision_request = admission_decision_->request;
    auto& decision               = *admission_decision_;
    ContextAssessment assessment;
    bool deferred = false;
    std::optional<typename ResourceManagement::SourceChoice> waiting_choice;
    // A Retry re-entry changes no engine state and a Changed re-entry frees capacity the bind
    // may still not place (see prepare_semantic_capture): a source whose bind keeps failing
    // must not spin the admission pass, whatever reclaim reports.
    constexpr int kMaxAdmissionReclaimAttempts = 8;
    int attempts = 0;
    while (decision.current < decision.sources.size()) {
        auto& choice = decision.sources[decision.current];
        if (!resources_.prepare_source(*instance_.session, *decision_request->base_plan, choice,
                                       decision_request->id, std::nullopt,
                                       restoring ? &*decision_request->suspended : nullptr)) {
            if (!restoring) {
                decision_request->admission.fallback_reason =
                    resources_.source_revocations(decision_request->id) &&
                            !resources_.retained_source(decision_request->id)
                        ? AdmissionFallbackReason::SourceRevoked
                        : AdmissionFallbackReason::SourceInvalid;
            }
            ++decision.current;
            continue;
        }
        if (!lane) {
            retain_admission_source(decision_request, choice, restoring);
            deferred = true;
            break;
        }
        // While flights compute, a Session with in-flight capture starts only the bindings it
        // completes synchronously, and reclaims capacity without waiting. Any other Session starts
        // only a fresh penalty-free root binding; every other binding, and the reclamation a
        // not-ready capacity assessment needs, belongs to a quiescent iteration.
        const bool quiescent = compute_flights() == 0;
        if constexpr (!kInFlightCapture) {
            const auto& sampling = decision_request->options.execution.sampling;
            const bool predicted_synchronous =
                !restoring && !decision_request->suspended && !choice.source.checkpoint &&
                sampling.presence_penalty == 0.0F && sampling.frequency_penalty == 0.0F;
            if (!quiescent && !predicted_synchronous) { return AdmissionStatus::Deferred; }
        }
        // A new selection replaces its old waiting reference before capacity is
        // checked. The replacement is retained before any old point is released.
        if (resources_.has_source_record(decision_request->id)) {
            retain_admission_source(decision_request, choice, restoring);
        }
        auto binding_started = Clock::now();
        const auto resume_state = restoring ? &*decision_request->suspended : nullptr;
        const auto resume_kind =
            decision_request->resume_phase == EngineRequestState::Prefill ? UnitKind::Prefill
            : decision_request->resume_phase == EngineRequestState::ControlReady ? UnitKind::Control
                                                                                  : UnitKind::Decode;
        const auto resume_tokens =
            decision_request->resume_phase == EngineRequestState::Prefill ? 0U
            : decision_request->resume_phase == EngineRequestState::ControlReady
                ? static_cast<std::uint32_t>(
                      decision_request->output.pending_control_tokens().size())
                : 1U;
        const auto bind = [&](const auto& source) {
            if constexpr (kInFlightCapture) {
                return instance_.session->start_binding(*decision_request->base_plan,
                                                        LaneId{*lane}, source, resume_state,
                                                        resume_kind, resume_tokens, !quiescent);
            } else {
                return instance_.session->start_binding(*decision_request->base_plan,
                                                        LaneId{*lane}, source, resume_state,
                                                        resume_kind, resume_tokens);
            }
        };
        auto reservation = bind(choice.source);
        if constexpr (kInFlightCapture) {
            // The binding needs a context operation; it waits for a quiescent iteration.
            if (reservation.asynchronous) { return AdmissionStatus::Deferred; }
        }
        if (!reservation && !reservation.source_valid) {
            if (!restoring) {
                decision_request->admission.fallback_reason =
                    resources_.source_revocations(decision_request->id) &&
                            !resources_.retained_source(decision_request->id)
                        ? AdmissionFallbackReason::SourceRevoked
                        : AdmissionFallbackReason::SourceInvalid;
            }
            ++decision.current;
            continue;
        }
        if (!reservation) {
            if constexpr (!kInFlightCapture) {
                if (!quiescent) {
                    // Capacity reclamation is a quiescent-iteration path; this source waits.
                    retain_admission_source(decision_request, choice, restoring);
                    if (!waiting_choice) { waiting_choice = choice; }
                    deferred = true;
                    ++decision.current;
                    continue;
                }
            }
            assessment = reservation.assessment;
            retain_admission_source(decision_request, choice, restoring);
            const auto protected_source = choice.source.checkpoint;
            const auto excluded         = protected_source
                                              ? std::span<const Checkpoint>(&*protected_source, 1)
                                              : std::span<const Checkpoint>{};
            const auto reclaimed = reclaim(assessment, restoring, excluded, &decision.reclaim);
            if (reclaimed == ReclaimProgress::Transferring) return AdmissionStatus::Admitted;
            if (reclaimed == ReclaimProgress::Deferred) {
                if (!waiting_choice) { waiting_choice = choice; }
                deferred = true;
                ++decision.current;
                continue;
            }
            if (reclaimed == ReclaimProgress::Changed || reclaimed == ReclaimProgress::Retry) {
                if (++attempts >= kMaxAdmissionReclaimAttempts) {
                    // The bind keeps failing although reclaim keeps reporting progress: leave
                    // this source like a deferred one instead of spinning the admission pass.
                    if (!waiting_choice) { waiting_choice = choice; }
                    deferred = true;
                    ++decision.current;
                    continue;
                }
                continue;
            }
            // Another waiting reference may be the only obstacle to a Move.
            // Grant its revocation only when the resulting complete bind succeeds.
            {
                auto transferable = choice;
                if (resources_.prepare_source(*instance_.session, *decision_request->base_plan,
                                              transferable, decision_request->id,
                                              decision.reclaim.rights,
                                              restoring ? &*decision_request->suspended : nullptr) &&
                    (transferable.source.consume_source != choice.source.consume_source ||
                     transferable.source.retired_points != choice.source.retired_points)) {
                    const auto transfer_started = Clock::now();
                    auto attempt                = bind(transferable.source);
                    if (attempt.source_valid) {
                        reservation.capacity_possible |= attempt.capacity_possible;
                    }
                    if (attempt) {
                        binding_started = transfer_started;
                        reservation     = std::move(attempt);
                        choice          = std::move(transferable);
                    }
                }
            }
            if (!reservation) {
                // Resident progress can release/unlock capacity. A root chunk's
                // smaller initial allocation is not a reason to discard this source.
                if (reservation.capacity_possible &&
                    (!resident_empty() || (!restoring && !paused_.empty()))) {
                    if (!waiting_choice) { waiting_choice = choice; }
                    deferred = true;
                    ++decision.current;
                    continue;
                }
                if (!restoring) {
                    decision_request->admission.fallback_reason =
                        reservation.capacity_possible
                            ? AdmissionFallbackReason::IsolatedCapacity
                            : AdmissionFallbackReason::CapacityLimit;
                }
                ++decision.current;
                continue;
            }
        }
        const auto bound_request  = decision_request;
        const auto bound_choice   = choice;
        decision_request->placement = bound_choice.source.placement;
        decision_request->placement_tokens =
            std::max<std::uint64_t>(decision_request->base_plan->summary().prompt_tokens,
                                    restoring ? decision_request->suspended->frontier() : 0);
        next_placement_tie_.value = bound_choice.source.placement.value + 1;
        resources_.binding_started(decision_request->id, reservation.retired_points,
                                   reservation.consumed_source);
        const auto& source = bound_choice.source;
        const BeginSummary begin{.prompt_tokens = decision_request->base_plan->summary().prompt_tokens,
                                 .reused_prompt_tokens = source.reused_tokens,
                                 .prefix_reuse_path    = source.reused_tokens
                                                             ? PrefixReusePath::Checkpoint
                                                             : PrefixReusePath::Root};
        if (restoring) {
            std::erase(paused_, decision_request);
            observe_scheduling(decision_request, GenerationSchedulingTransition::RestoreStarted,
                               binding_started);
        } else {
            std::lock_guard lock(queue_mutex_);
            scheduler_.admitted(pending_, *decision_request, max_concurrency_);
            std::erase(pending_, decision_request);
        }
        admission_decision_.reset();
        if (!restoring) {
            finish_source_wait(bound_request, binding_started);
            bound_request->admission.revoked_checkpoints =
                resources_.source_revocations(bound_request->id);
            bound_request->admitted_at = binding_started;
            bound_request->host_timing.queue_wait_ns =
                elapsed_ns(bound_request->submitted, *bound_request->admitted_at);
            bound_request->queue_wait_recorded = true;
            publish_generation_start(bound_request, begin);
        }
        if (reservation.synchronous) {
            adopt_bound_request(bound_request, bound_choice, *lane, std::move(reservation.sequence),
                                std::move(reservation.residency), begin);
        } else {
            bound_request->model_state = EngineRequestState::Materializing;
            materializing_[*lane].emplace(
                MaterializingRequest{bound_request, LaneId{*lane}, begin, restoring,
                                     bound_choice});
            context_owner_ = bound_request;
        }
        if constexpr (kInFlightCapture) {
            if (!restoring) { claim_fork_children(bound_request); }
        }
        return AdmissionStatus::Admitted;
    }
    if (waiting_choice &&
        resources_.prepare_source(*instance_.session, *decision_request->base_plan,
                                  *waiting_choice, decision_request->id, std::nullopt,
                                  restoring ? &*decision_request->suspended : nullptr)) {
        retain_admission_source(decision_request, *waiting_choice, restoring);
    }
    admission_decision_.reset();
    if (deferred) {
        if (restoring) {
            return AdmissionStatus::Blocked;
        }
        return AdmissionStatus::Deferred;
    }
    if (admission_check_pending_.load(std::memory_order_acquire)) {
        return AdmissionStatus::Admitted;
    }
    if (resident_empty()) {
        throw std::logic_error("isolated request cannot reserve its first legal unit after "
                               "context reclamation");
    }
    return restoring ? AdmissionStatus::Blocked : AdmissionStatus::Deferred;
}

// Decides whether `parent` can head a fork group now. The parent is the group's first queued
// branch: the other queued branches wait for their turn and then take their lanes as children.
template <class Instance>
typename EngineCore<Instance>::Engagement
EngineCore<Instance>::engage_fork_group(const SampleGroup& group,
                                        const std::shared_ptr<Request>& parent) {
    std::uint32_t queued = 0;
    for (const auto& member : group.members) {
        if (!member->is_queued()) { continue; }
        if (queued++ == 0 && member != parent) { return Engagement::Waiting; }
    }
    if (queued != 0 && queued - 1 > instance_.session->sample_fork_limit()) {
        return Engagement::Failed;
    }
    return free_lane_count() >= queued ? Engagement::Ready : Engagement::Waiting;
}

// The parent is bound: the group's other queued branches leave the queue and take one free lane
// each in the Forking state. They hold no sequence until the fork installs them, and their
// prompt starts at the fork frontier the parent shares.
template <class Instance>
void EngineCore<Instance>::claim_fork_children(const std::shared_ptr<Request>& parent) {
    const auto frontier = parent->base_plan->fork_frontier();
    // A parent that starts at or past its fork frontier prefills the whole prompt itself.
    if (frontier == 0 || !parent->admitted_begin ||
        parent->admitted_begin->reused_prompt_tokens >= frontier) {
        return;
    }
    const auto group = find_group(parent->sample_group_id);
    if (!group) { throw std::logic_error("fork parent lost its sample group"); }
    ForkGroup fork{.parent = parent};
    const auto now      = Clock::now();
    std::uint32_t lane = 0;
    for (const auto& member : group->members) {
        if (member == parent || !member->is_queued()) { continue; }
        try {
            ensure_base_plan(member);
        } catch (const RequestError&) {
            // The branch's own plan was rejected: it ends alone and the group goes on.
            const auto error = std::current_exception();
            {
                std::lock_guard lock(queue_mutex_);
                std::erase(pending_, member);
            }
            complete_error(member, error);
            continue;
        }
        while (lane < max_concurrency_ && (slots_[lane] || materializing_[lane])) { ++lane; }
        // The branches without a lane stay queued and head another fork group later.
        if (lane == max_concurrency_) { break; }
        {
            std::lock_guard lock(queue_mutex_);
            scheduler_.admitted_with(*member);
            std::erase(pending_, member);
        }
        member->admitted_at                = now;
        member->host_timing.queue_wait_ns  = elapsed_ns(member->submitted, now);
        member->queue_wait_recorded        = true;
        member->model_state                = EngineRequestState::Forking;
        member->lane                       = LaneId{lane};
        slots_[lane]                       = member;
        ++slots_epoch_;
        publish_generation_start(
            member, BeginSummary{.prompt_tokens = member->base_plan->summary().prompt_tokens,
                                 .reused_prompt_tokens = frontier,
                                 .prefix_reuse_path    = PrefixReusePath::Root});
        fork.children.push_back(member);
        fork.child_lanes.push_back(lane);
    }
    fork_groups_.push_back(std::move(fork));
}

// Forks, with one Session call, every engaged group whose parent rests at its fork frontier
// with nothing pending, and adopts the installed children. The parent and its children decode
// the last prompt token as ordinary sequences from the next flight on. A group the Context
// cannot hold yet is retried by the next iteration.
template <class Instance>
void EngineCore<Instance>::fork_ready_groups() {
    if constexpr (kInFlightCapture) {
        using GroupRequest = typename ModelContract::ForkGroupRequest;
        std::vector<GroupRequest> requests;
        std::vector<std::size_t> ready;
        for (std::size_t index = 0; index < fork_groups_.size(); ++index) {
            const auto& group  = fork_groups_[index];
            const auto& parent = group.parent;
            if (!parent->is_prefilling() || !parent->sequence ||
                parent->cancelled.load(std::memory_order_acquire) ||
                instance_.session->context_blocks(*parent->sequence) ||
                instance_.session->execution_pending(*parent->sequence) ||
                parent->admitted_begin->reused_prompt_tokens + parent->computed_prompt_tokens !=
                    parent->base_plan->fork_frontier()) {
                continue;
            }
            GroupRequest request{.parent = *parent->sequence};
            for (std::size_t branch = 0; branch < group.children.size(); ++branch) {
                request.children.push_back(LaneId{group.child_lanes[branch]});
                request.sampling.push_back(group.children[branch]->options.execution.sampling);
            }
            requests.push_back(std::move(request));
            ready.push_back(index);
        }
        if (ready.empty()) { return; }
        std::vector<typename ModelContract::ForkGroupResult> results;
        {
            ExecutionCallScope call(*this);
            results = instance_.session->fork_batch(requests);
            call.finish();
        }
        ++cumulative_stats_.fork_batches;
        if (results.size() != ready.size()) {
            throw std::logic_error("fork_batch changed its group membership");
        }
        std::vector<std::size_t> forked;
        bool reclaimed = false;
        for (std::size_t i = 0; i < ready.size(); ++i) {
            if (!results[i].reserved) {
                // The Context cannot hold this group yet: reclaim what its assessment lacks
                // (one group per iteration) and let the next iteration retry the fork.
                if (!reclaimed) {
                    reclaimed = true;
                    (void)reclaim(results[i].assessment, false);
                }
                continue;
            }
            const auto& group  = fork_groups_[ready[i]];
            const auto& parent = group.parent;
            if (results[i].children.size() != group.children.size()) {
                throw std::logic_error("fork changed the requested child membership");
            }
            for (std::size_t branch = 0; branch < group.children.size(); ++branch) {
                const auto& child = group.children[branch];
                const auto lane   = group.child_lanes[branch];
                child->sequence   = results[i].children[branch];
                child->residency  = ResidencyId{LaneId{lane}, ++lane_residency_epochs_[lane]};
                child->placement  = instance_.session->placement(*child->sequence);
                child->placement_tokens =
                    std::max<std::uint64_t>(child->placement_tokens,
                                            instance_.session->load_tokens(*child->sequence));
                child->model_state = EngineRequestState::DecodeReady;
            }
            parent->begin       = parent->admitted_begin;
            parent->model_state = EngineRequestState::DecodeReady;
            forked.push_back(ready[i]);
            cumulative_stats_.forked_branches += 1 + group.children.size();
        }
        for (auto it = forked.rbegin(); it != forked.rend(); ++it) {
            fork_groups_.erase(fork_groups_.begin() + static_cast<std::ptrdiff_t>(*it));
        }
        if (!forked.empty()) { request_admission_check(); }
    }
}

// Ends the branches an engaged group lost before its fork: a cancelled branch leaves alone, and
// a parent that is cancelled or ended takes the branches that wait for its fork with it. Also
// forgets submitted groups that have no queued branch left.
template <class Instance>
void EngineCore<Instance>::settle_sample_groups(HostPhaseMeasurement& boundary) {
    {
        std::lock_guard lock(queue_mutex_);
        std::erase_if(groups_, [](const auto& group) {
            return group->sealed &&
                   std::none_of(group->members.begin(), group->members.end(),
                                [](const auto& member) { return member->is_queued(); });
        });
    }
    for (std::size_t index = 0; index < fork_groups_.size();) {
        auto& group = fork_groups_[index];
        const bool parent_gone = group.parent->response_done || group.parent->is_model_finished() ||
                                 group.parent->cancelled.load(std::memory_order_acquire);
        std::vector<std::shared_ptr<Request>> ended;
        std::vector<std::uint32_t> kept_lanes;
        std::vector<std::shared_ptr<Request>> kept;
        for (std::size_t branch = 0; branch < group.children.size(); ++branch) {
            if (parent_gone || group.children[branch]->cancelled.load(std::memory_order_acquire)) {
                ended.push_back(group.children[branch]);
            } else {
                kept.push_back(group.children[branch]);
                kept_lanes.push_back(group.child_lanes[branch]);
            }
        }
        if (ended.empty()) {
            ++index;
            continue;
        }
        group.children    = std::move(kept);
        group.child_lanes = std::move(kept_lanes);
        if (parent_gone) {
            fork_groups_.erase(fork_groups_.begin() + static_cast<std::ptrdiff_t>(index));
        } else {
            ++index;
        }
        const SettlementPass pass(*this);
        for (const auto& child : ended) {
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            complete_cancelled(child);
            boundary = begin_host_phase();
        }
        request_admission_check();
    }
}

template <class Instance>
void EngineCore<Instance>::observe_scheduling(const std::shared_ptr<Request>& request,
                                              GenerationSchedulingTransition transition,
                                              Clock::time_point now) {
    if (!request->observation.scheduling) { return; }
    if (now == Clock::time_point{}) { now = Clock::now(); }
    const GenerationSchedulingObservation observation{
        .transition = transition,
        .route      = request->recovery ? request->recovery->route : GenerationRecoveryRoute::None,
        .phase      = observe_request(*request).phase,
        .engine_request_id       = request->id,
        .steady_ns               = elapsed_ns(Clock::time_point{}, now),
        .elapsed_ns              = elapsed_ns(request->submitted, now),
        .preemption_index        = request->preemption_count,
        .global_prefill_tokens   = cumulative_stats_.computed_prefill_tokens,
        .global_decode_tokens    = cumulative_stats_.committed_decode_tokens,
        .global_replayed_tokens  = cumulative_stats_.replayed_tokens,
        .request_prefill_tokens  = request->computed_prompt_tokens,
        .request_decode_tokens   = request->committed_decode_tokens,
        .request_replayed_tokens = request->replayed_tokens,
    };
    {
        std::lock_guard lock(request->mutex);
        request->events.emplace_back(
            std::make_unique<GenerationSchedulingObservation>(observation));
    }
    request->cv.notify_one();
}

template <class Instance>
void EngineCore<Instance>::update_recovery(const std::shared_ptr<Request>& request, bool released) {
    if (!request->recovery_pending) { return; }
    request->recovery_pending =
        !released && request->sequence && instance_.session->recovery_pending(*request->sequence);
    if (!request->recovery_pending) {
        if (!released) {
            observe_scheduling(request, GenerationSchedulingTransition::RecoveryComplete);
        }
        request_admission_check();
    }
}

template <class Instance>
typename EngineCore<Instance>::ExecutionUnit
EngineCore<Instance>::execution_unit(const Request& request) const {
    return {*request.sequence,
            request.is_control_ready()  ? UnitKind::Control
            : request.is_decode_ready() ? UnitKind::Decode
            : request.is_replaying()    ? UnitKind::Replay
                                        : UnitKind::Prefill,
            request.is_control_ready()
                ? static_cast<std::uint32_t>(request.output.pending_control_tokens().size())
            : request.is_decode_ready()
                ? (request.recovery_pending
                       ? 1U
                       : request.output.model_token_budget_remaining(request.budget->remaining()))
                : 0U};
}

template <class Instance>
typename EngineCore<Instance>::LaneSlots EngineCore<Instance>::runnable_lanes() const {
    LaneSlots lanes = slots_;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        const auto& request = lanes[lane];
        if (!request) continue;
        if (!request->sequence || request->capture_pending ||
            request->cancelled.load(std::memory_order_acquire) ||
            instance_.session->context_blocks(*request->sequence) ||
            instance_.session->execution_pending(*request->sequence))
            lanes[lane].reset();
    }
    return lanes;
}

template <class Instance>
LaneTable EngineCore<Instance>::build_lane_table(const LaneSlots& lanes) const {
    LaneTable table;
    table.lanes.resize(max_concurrency_);
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        const auto& request = lanes[lane];
        if (!request) continue;
        auto& entry            = table.lanes[lane];
        entry.decode_ready     = request->is_decode_ready();
        entry.control_ready    = request->is_control_ready();
        entry.prefilling       = request->is_prefilling();
        entry.replaying        = request->is_replaying();
        if (!entry.prefilling) continue;
        // The most tokens the lane's next prefill unit could process: the public chunk bounded
        // by the prompt suffix it still has to compute, and for a forking request by its fork
        // frontier (a parent resting there takes no unit; the fork step advances it).
        std::uint32_t suffix = request->admitted_begin
                                   ? request->admitted_begin->prompt_tokens -
                                         request->admitted_begin->reused_prompt_tokens
                                   : 0;
        bool forking         = false;
        if (const auto fork = request->base_plan ? request->base_plan->fork_frontier() : 0) {
            const std::uint32_t base =
                request->admitted_begin ? request->admitted_begin->reused_prompt_tokens : 0;
            suffix  = std::min(suffix, fork > base ? fork - base : 0);
            forking = true;
        }
        const std::uint32_t computed = std::min(request->computed_prompt_tokens, suffix);
        // A source covering the whole prompt leaves no suffix, yet a Session with in-flight
        // capture still needs one unit to sample the first token from the source's retained
        // hidden state.
        if ((forking || !kInFlightCapture) && suffix <= computed) continue;
        const std::uint32_t chunk =
            prefill_chunk_ == 0 ? std::numeric_limits<std::uint32_t>::max() : prefill_chunk_;
        entry.prefill_limit = std::max(std::min(chunk, suffix - computed), 1U);
    }
    if (const auto preferred = instance_.session->prefill_continuation()) {
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = lanes[lane];
            if (request && request->sequence == preferred &&
                (request->is_prefilling() || request->is_replaying())) {
                table.priority_lane = lane;
                break;
            }
        }
    }
    return table;
}

template <class Instance>
void EngineCore<Instance>::reserve_planned_units(const IterationMembership& membership) {
    const auto owner     = scheduler_.progress_owner();
    const bool quiescent = compute_flights() == 0;
    // One iteration's reservation shortfall work is bounded as a whole: every failing lane
    // spends reclaim selections against the inventory, and enough failing lanes together
    // would outlast the report cadence and starve every other engine service. Past the
    // budget the remaining lanes keep no unit this iteration; committed releases persist
    // and a later iteration retries from the better state.
    constexpr auto kReserveBudget = std::chrono::steady_clock::duration{
        std::chrono::milliseconds{250}};
    const auto reserve_started = Clock::now();
    std::vector<std::uint32_t> order;
    order.reserve(membership.control_lanes.size() + membership.decode_lanes.size() +
                  membership.prefill_lanes.size());
    std::optional<std::uint32_t> priority;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        const auto& request = slots_[lane];
        if (request && request->sequence && request->id == owner) priority = lane;
    }
    if (priority) order.push_back(*priority);
    // A replaying lane runs alone and takes no part in the prefill list, so it is ordered apart.
    if (membership.replay_lane && (!priority || *priority != *membership.replay_lane))
        order.push_back(*membership.replay_lane);
    const auto append = [&](const std::vector<std::uint32_t>& lanes) {
        for (const auto lane : lanes) {
            if (!priority || lane != *priority) order.push_back(lane);
        }
    };
    append(membership.control_lanes);
    append(membership.decode_lanes);
    append(membership.prefill_lanes);

    // Decode lanes that fit together reserve in one Session call. Any shortfall leaves them to
    // the lane loop below, which owns reclaim and the per-lane outcome; the order of the
    // reservations only matters then.
    std::vector<std::uint8_t> reserved_together(max_concurrency_, 0);
    if (!priority && membership.control_lanes.empty()) {
        std::vector<ExecutionUnit> units;
        std::vector<std::uint32_t> lanes;
        units.reserve(membership.decode_lanes.size());
        lanes.reserve(membership.decode_lanes.size());
        for (const auto lane : membership.decode_lanes) {
            const auto& request = slots_[lane];
            if (!request || !request->sequence || request->capture_pending ||
                request->cancelled.load(std::memory_order_acquire) ||
                instance_.session->context_blocks(*request->sequence) ||
                instance_.session->execution_pending(*request->sequence) ||
                !request->is_decode_ready())
                continue;
            units.push_back(execution_unit(*request));
            lanes.push_back(lane);
        }
        if (units.size() > 1 && instance_.session->reserve_units(units)) {
            for (std::size_t i = 0; i < lanes.size(); ++i) {
                reserved_units_[lanes[i]] = UnitPermit{units[i], *slots_[lanes[i]]->residency};
                reserved_together[lanes[i]] = 1;
            }
        }
    }

    std::vector<std::uint32_t> granted_prefill;
    for (const auto lane : order) {
        if (reserved_together[lane]) continue;
        const auto request = slots_[lane];
        if (!request || !request->sequence) continue;
        if (request->capture_pending || request->cancelled.load(std::memory_order_acquire) ||
            instance_.session->context_blocks(*request->sequence) ||
            instance_.session->execution_pending(*request->sequence) ||
            !(request->is_control_ready() || request->is_decode_ready() ||
              request->is_prefilling() || request->is_replaying())) {
            if (priority && *priority == lane) return;
            continue;
        }
        const bool replay_lane  = membership.replay_lane && *membership.replay_lane == lane;
        const bool prefill_lane = request->is_prefilling() || request->is_replaying();
        if (prefill_lane && !replay_lane &&
            std::find(membership.prefill_lanes.begin(), membership.prefill_lanes.end(), lane) ==
                membership.prefill_lanes.end()) {
            continue;
        }
        auto unit = execution_unit(*request);
        if (replay_lane) {
            // A replay window is bounded by the public chunk and spends none of the iteration's
            // prefill token budget: the replay runs alone.
            unit.tokens = prefill_chunk_ == 0 ? std::numeric_limits<std::uint32_t>::max()
                                              : prefill_chunk_;
        } else if (prefill_lane) {
            const auto cap = std::find(membership.prefill_lanes.begin(),
                                       membership.prefill_lanes.end(), lane) -
                             membership.prefill_lanes.begin();
            // Nothing is left of the token budget for this lane.
            if (membership.prefill_caps[cap] == 0) {
                continue;
            }
            unit.tokens = membership.prefill_caps[cap];
        }
        std::optional<typename ResourceManagement::ReclaimCursor> cursor;
        // A Retry re-entry changes no engine state and a Changed re-entry frees capacity the
        // reservation may still not place (see prepare_semantic_capture). Without a bound a
        // reservation whose shortage reclaim cannot move spins this iteration and starves
        // every other engine service, whatever reclaim reports. After a few attempts the
        // lane keeps no unit this iteration; committed releases persist for a later pass.
        constexpr int kMaxReserveReclaimAttempts = 8;
        int attempts = 0;
        for (;;) {
            if (Clock::now() - reserve_started >= kReserveBudget) break;
            const auto result = instance_.session->reserve_units({&unit, 1});
            if (result) {
                reserved_units_[lane] = UnitPermit{unit, *request->residency};
                if (request->is_prefilling()) granted_prefill.push_back(lane);
                break;
            }
            // A context operation in flight invalidates the reservation pass: the reclaim
            // it may start is quiescent-iteration work.
            if (result.status == UnitReservationStatus::ExecutionUnavailable) {
                if (priority && *priority == lane) return;
                break;
            }
            // A Session with in-flight capture releases checkpoints beside the flights; any
            // other Session reclaims only while nothing computes.
            if ((!kInFlightCapture && !quiescent) || instance_.session->has_context_operation()) {
                if (priority && *priority == lane) return;
                break;
            }
            // Only the turn owner may change capacity ownership under pressure.
            if (scheduler_.progress_owner() && scheduler_.progress_owner() != request->id) {
                scheduler_.require_progress(request->id);
                break;
            }
            if (!cursor) cursor.emplace(resources_.begin_reclaim(*instance_.session));
            const auto reclaimed = reclaim(result.assessment, false, {}, &*cursor);
            if (reclaimed == ReclaimProgress::Transferring ||
                reclaimed == ReclaimProgress::Deferred) {
                // A running context operation holds back only the lane whose shortage it could
                // not cover with releases; the other lanes keep their units.
                if (instance_.session->has_context_operation() &&
                    (!priority || *priority != lane)) {
                    break;
                }
                return;
            }
            if (reclaimed == ReclaimProgress::Changed || reclaimed == ReclaimProgress::Retry) {
                if (++attempts >= kMaxReserveReclaimAttempts) {
                    break;
                }
                continue;
            }
            // Pausing a resident is quiescent-iteration work.
            if (!quiescent) {
                if (priority && *priority == lane) return;
                break;
            }
            if (request->recovery_pending)
                throw std::logic_error("protected recovery exceeded its Native reservation");
            scheduler_.require_progress(request->id);
            if (pause_resident(lane, PauseGoal::RelocateOnly, &result.assessment) ||
                pause_reclaim_victim(result.assessment))
                return;
            throw std::logic_error("progress owner cannot obtain its legal unit");
        }
    }
    // A lane whose capture cannot be prepared loses its unit and leaves the execution; the
    // reclamation a capture preparation may start is quiescent-iteration work. Each attempt
    // runs a full admission and reclaim evaluation, so preparing every granted lane inside
    // one iteration stalls the engine for minutes under capture pressure (many lanes, large
    // checkpoint inventory). Preparations are optional: a bounded number per iteration keeps
    // the engine servicing, and deferred lanes retry once committed releases shrank the
    // pressure.
    if constexpr (!kInFlightCapture) {
        if (quiescent && !instance_.session->has_context_operation()) {
            constexpr std::size_t kMaxCapturePreparations = 8;
            std::size_t preparations = 0;
            for (const auto lane : granted_prefill) {
                if (preparations >= kMaxCapturePreparations) {
                    reserved_units_[lane].reset();
                    continue;
                }
                ++preparations;
                if (!prepare_semantic_capture(slots_[lane])) reserved_units_[lane].reset();
            }
        }
    }
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::unit_runnable(std::uint32_t lane) const {
    const auto& request = slots_[lane];
    const auto& permit  = reserved_units_[lane];
    if (!permit || !request || !request->sequence || !request->residency ||
        request->model_state == EngineRequestState::Pausing ||
        request->cancelled.load(std::memory_order_acquire) ||
        instance_.session->context_blocks(*request->sequence) ||
        instance_.session->execution_pending(*request->sequence))
        return false;
    const auto current = execution_unit(*request);
    // The tokens of a prefill/replay permit are the cap fixed when it was reserved, not a
    // function of the request, so only the other kinds compare them.
    const bool windowed = current.kind == UnitKind::Prefill || current.kind == UnitKind::Replay;
    return permit->residency == *request->residency && permit->unit.sequence == current.sequence &&
           permit->unit.kind == current.kind &&
           (windowed ? permit->unit.tokens != 0 : permit->unit.tokens == current.tokens);
}

// One admission pass over the queue. Each started binding retires before the next one begins
// (the Session's single context-operation slot), bounded by one admission burst.
template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::admit_pending(HostPhaseMeasurement& boundary) {
    const bool recovering =
        std::any_of(slots_.begin(), slots_.end(), [](const auto& r) { return r && r->recovery_pending; }) ||
        std::any_of(materializing_.begin(), materializing_.end(),
                    [](const auto& entry) { return entry && entry->resumed; });
    if (scheduler_.progress_owner() || !paused_.empty() || context_owner_ || recovering) {
        return false;
    }
    if (!admission_decision_ && consume_admission_check()) {
        ++admission_generation_;
        scheduler_.begin_admission_scan();
    }
    if (!admission_decision_ && !scheduler_.admission_scan_pending()) return false;
    bool admitted_any = false;
    const auto burst_end = Clock::now() + kAdmissionBurst;
    for (;;) {
        std::vector<std::shared_ptr<Request>> candidates;
        if (admission_decision_) {
            candidates.push_back(admission_decision_->request);
        } else {
            std::lock_guard lock(queue_mutex_);
            candidates = scheduler_.fresh_candidates(pending_, max_concurrency_);
        }
        if (candidates.empty()) break;
        bool admitted = false;
        for (const auto& candidate : candidates) {
            const auto status = admit_request(candidate, false);
            admitted = admitted || status == AdmissionStatus::Admitted;
            // One admission decision is live at a time; its request is the next candidate.
            if (admission_decision_) break;
            if (status == AdmissionStatus::Admitted) break;
        }
        admitted_any = admitted_any || admitted;
        // A binding that started a context operation (silent or asynchronous) must retire
        // before the next one can begin; settle whatever it produced on the way. Synchronous
        // bindings chain until the queue, the lanes or the burst budget ends the pass.
        if (!instance_.session->has_context_operation()) {
            if (!admitted || Clock::now() >= burst_end) break;
            continue;
        }
        if (!await_context_operation(boundary, burst_end)) break;
    }
    return admitted_any;
}

template <class Instance>
void EngineCore<Instance>::balance_retention() {
    const auto points = resources_.retained_points();
    if (points == 0) return;
    const auto usage = instance_.session->physical_usage();
    std::size_t keep = points;
    for (const auto& pool : usage.pools) {
        if (pool.domain != ContextMemoryDomain::Device || pool.cached_bytes == 0) continue;
        const std::uint64_t free = pool.assigned_free_bytes + pool.unassigned_free_bytes +
                                   pool.pending_retirement_bytes;
        const std::uint64_t wanted = pool.capacity_bytes / kPoolHeadroomDivisor;
        if (free >= wanted) continue;
        const std::uint64_t per_point = pool.cached_bytes / points;
        if (per_point == 0) continue;
        const std::uint64_t excess = (wanted - free + per_point - 1) / per_point;
        keep = std::min<std::size_t>(keep, excess >= points ? 0 : points - excess);
    }
    if (keep < points) resources_.trim_retained(*instance_.session, keep);
}

template <class Instance>
void EngineCore<Instance>::begin_iteration(HostPhaseMeasurement& boundary) {
    std::fill(control_executed_.begin(), control_executed_.end(), false);
    // Tails recorded while a context operation was in flight run before anything else in this
    // iteration, and in particular before admission can select a source.
    flush_deferred_settlements();
    (void)expire_pending_requests();
    if constexpr (!kInFlightCapture) {
        if (!instance_.session->has_context_operation() && compute_flights() == 0) {
            for (const auto& request : slots_) {
                if (request && request->capture_pending &&
                    !instance_.session->execution_pending(*request->sequence)) {
                    reserve_active_capture(request, request->post_capture_state);
                    break;
                }
            }
        }
    }
    (void)settle_terminal_requests(boundary);
    cancel_active_requests(snapshot_cancellations(), boundary);
    {
        const SettlementPass paused_pass(*this);
        for (auto it = paused_.begin(); it != paused_.end();) {
            const auto request = *it;
            // Dropping a suspended snapshot erases a checkpoint that an in-flight context operation
            // may still hold; such a request waits. Replay-route requests own no checkpoint.
            if (!request->cancelled.load(std::memory_order_acquire) ||
                (request->suspended && request->suspended->has_snapshot() &&
                 instance_.session->has_context_operation())) {
                ++it;
                continue;
            }
            request->suspended.reset();
            it = paused_.erase(it);
            complete_cancelled(request);
        }
    }
    settle_sample_groups(boundary);
    const bool recovering =
        std::any_of(slots_.begin(), slots_.end(), [](const auto& r) { return r && r->recovery_pending; }) ||
        std::any_of(materializing_.begin(), materializing_.end(),
                    [](const auto& entry) { return entry && entry->resumed; });
    if (!instance_.session->has_context_operation() && compute_flights() == 0 &&
        scheduler_.advance_service(!paused_.empty(), recovering)) {
        // An unused permit belongs to the previous turn. Retained recovery capacity is
        // released only by Native after actual progress, never by this handoff.
        for (const auto& request : slots_)
            if (request && request->sequence)
                instance_.session->release_units({&*request->sequence, 1});
        admission_decision_.reset();
        request_admission_check();
    }
    if (!recovering && scheduler_.progress_owner() && compute_flights() == 0) {
        const auto suspended = std::find_if(
            paused_.begin(), paused_.end(),
            [&](const auto& request) { return request->id == scheduler_.progress_owner(); });
        if (suspended != paused_.end()) {
            (void)admit_request(*suspended, true);
            if (instance_.session->has_context_operation()) (void)poll_execution(boundary);
        }
    }
    balance_retention();
    // Groups at their fork frontier fork beside whatever computes: the fork opens no context
    // operation and needs only its own parent at rest.
    fork_ready_groups();
    // Admission runs beside the flights in flight: a Session with in-flight capture binds
    // synchronously and reclaims without waiting; any other Session starts only fresh root
    // bindings, and its reclamation and asynchronous bindings wait for a quiescent iteration.
    (void)admit_pending(boundary);
}

template <class Instance>
bool EngineCore<Instance>::await_context_operation(HostPhaseMeasurement& boundary,
                                                   Clock::time_point deadline) {
    while (instance_.session->has_context_operation()) {
        (void)poll_execution(boundary);
        if (!instance_.session->has_context_operation()) break;
        if (Clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

// Plans and submits one iteration's flights.
template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::submit_iteration(HostPhaseMeasurement& boundary) {
    bool submitted = false;
    // Reservation passes wait for a free context slot: a reclamation they may start is a
    // context operation.
    if (instance_.session->has_context_operation()) {
        (void)poll_execution(boundary);
        if (instance_.session->has_context_operation()) return submitted;
    }

    // The membership the plan admits, before units are reserved for it.
    IterationMembership membership;
    {
        const auto control = scheduler_.build_control_membership(slots_, max_concurrency_);
        membership.control_lanes.reserve(control.size);
        for (std::size_t row = 0; row < control.size; ++row) {
            membership.control_lanes.push_back(control.lanes[row]);
        }
        const auto lanes = runnable_lanes();
        const auto table = build_lane_table(lanes);
        const auto plan  = scheduler_.plan_iteration(table, iteration_budget());
        membership.decode_lanes  = plan.decode_lanes;
        membership.prefill_lanes = plan.prefill_lanes;
        membership.prefill_caps  = plan.prefill_tokens;
        // A replaying lane runs alone, ahead of the mixed flight; the Session's continuation
        // or the pressure owner leads.
        std::optional<std::uint32_t> preferred;
        if (table.priority_lane) { preferred = table.priority_lane; }
        if (const auto owner = scheduler_.progress_owner()) {
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (const auto& request = lanes[lane];
                    request && request->id == owner && request->is_replaying()) {
                    preferred = lane;
                }
            }
        }
        const auto pick_replay = [&](std::uint32_t lane) {
            if (membership.replay_lane) return;
            if (const auto& request = lanes[lane];
                request && request->is_replaying()) {
                membership.replay_lane = lane;
            }
        };
        if (preferred) { pick_replay(*preferred); }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) { pick_replay(lane); }
        if (membership.replay_lane) {
            // A replay lane takes no part in the mixed flight's prefill list.
            const auto found = std::find(membership.prefill_lanes.begin(),
                                         membership.prefill_lanes.end(), *membership.replay_lane);
            if (found != membership.prefill_lanes.end()) {
                const auto index = static_cast<std::size_t>(
                    std::distance(membership.prefill_lanes.begin(), found));
                membership.prefill_lanes.erase(found);
                membership.prefill_caps.erase(membership.prefill_caps.begin() +
                                              static_cast<std::ptrdiff_t>(index));
            }
        }
        // The pack segment bound of the Session ends the prefill list; the lanes beyond it
        // keep their work for the next iteration.
        const auto segments = instance_.session->prefill_pack_limit();
        if (membership.prefill_lanes.size() > segments) {
            membership.prefill_lanes.resize(segments);
            membership.prefill_caps.resize(segments);
        }
    }

    reserve_planned_units(membership);
    // A capture preparation may have started a context operation; its reclamation settles
    // before any flight of this iteration starts.
    if (instance_.session->has_context_operation()) {
        (void)poll_execution(boundary);
        if (instance_.session->has_context_operation()) return submitted;
    }

    // Control rows fly first; a lane that ran control does not decode in the same iteration.
    if (!membership.control_lanes.empty()) {
        ControlMembership all = scheduler_.build_control_membership(slots_, max_concurrency_);
        ControlMembership selected(max_concurrency_);
        selected.row_stride = all.row_stride;
        for (std::size_t row = 0; row < all.size; ++row) {
            if (!unit_runnable(all.lanes[row])) continue;
            selected.lanes[selected.size]       = all.lanes[row];
            selected.sequences[selected.size++] = all.sequences[row];
            const auto begin                    = all.tokens.begin() +
                               static_cast<std::ptrdiff_t>(row * all.row_stride);
            selected.tokens.insert(selected.tokens.end(), begin,
                                   begin + static_cast<std::ptrdiff_t>(all.row_stride));
        }
        if (!selected.empty()) {
            set_host_work_class(HostWorkClass::Control);
            if (!start_control(selected)) return submitted;
            submitted = true;
        }
    }

    // The replaying lane runs alone.
    if (membership.replay_lane && unit_runnable(*membership.replay_lane)) {
        const std::uint32_t lane = *membership.replay_lane;
        set_host_work_class(HostWorkClass::Prefill);
        if (!start_prefill({&lane, 1})) return submitted;
        submitted = true;
    }

    // The mixed flight: every runnable planned decode row plus the prefill segments the token
    // budget admitted. Lanes that lost their reservation wait for the next iteration.
    std::vector<std::uint32_t> decode;
    for (const auto lane : membership.decode_lanes) {
        if (!unit_runnable(lane) || control_executed_[lane]) continue;
        decode.push_back(lane);
    }
    std::vector<std::uint32_t> prefill;
    std::vector<std::uint32_t> prefill_caps;
    for (std::size_t i = 0; i < membership.prefill_lanes.size(); ++i) {
        const auto lane = membership.prefill_lanes[i];
        if (!unit_runnable(lane)) continue;
        prefill.push_back(lane);
        prefill_caps.push_back(membership.prefill_caps[i]);
    }

    const auto plan_started = Clock::now();
    const auto host_submit  = [&] { return elapsed_ns(plan_started, Clock::now()); };
    if (!decode.empty() && !prefill.empty()) {
        if (instance_.session->prefill_mixable(*slots_[prefill.front()]->sequence)) {
            const auto mixed_rows = start_mixed(prefill, prefill_caps, decode, host_submit());
            if (mixed_rows) {
                submitted = true;
                // The decode rows the mixed flight did not admit (its row limit, or a
                // constrained row) ride their own decode flight while a frame remains.
                decode.erase(decode.begin(),
                             decode.begin() + static_cast<std::ptrdiff_t>(*mixed_rows));
                if (!decode.empty() && compute_frame_available()) {
                    set_host_work_class(HostWorkClass::Decode, decode);
                    if (start_decode(decode, host_submit())) submitted = true;
                }
                return submitted;
            }
            if (!compute_frame_available()) return submitted;
        }
        // The Session cannot mix this membership: decode and prefill fly separately.
        set_host_work_class(HostWorkClass::Decode, decode);
        if (start_decode(decode, host_submit())) submitted = true;
        if (!compute_frame_available()) return submitted;
        set_host_work_class(HostWorkClass::Prefill);
        if (start_prefill(prefill, host_submit())) submitted = true;
        return submitted;
    }
    if (!decode.empty()) {
        set_host_work_class(HostWorkClass::Decode, decode);
        if (start_decode(decode, host_submit())) submitted = true;
        return submitted;
    }
    if (!prefill.empty()) {
        set_host_work_class(HostWorkClass::Prefill);
        if (start_prefill(prefill, host_submit())) submitted = true;
    }
    return submitted;
}

template <class Instance>
bool EngineCore<Instance>::process_measurement() {
    if (!resident_empty() || !paused_.empty() ||
        std::any_of(materializing_.begin(), materializing_.end(),
                    [](const auto& entry) { return entry.has_value(); }) ||
        context_owner_ || admission_decision_ || !deferred_settlements_.empty() ||
        execution_outstanding())
        return false;
    MeasurementCommand command;
    {
        std::lock_guard lock(queue_mutex_);
        if (!measurement_ || !pending_.empty() || preparing_requests_ || !groups_.empty()) {
            return false;
        }
        command = measurement_->command;
    }
    EngineSnapshot result;
    std::exception_ptr error;
    try {
        instance_.executor->measure(command);
        publish_runtime_stats();
        result = snapshot();
    } catch (...) {
        error = std::current_exception();
        if (!instance_.executor->failed()) {
            try {
                instance_.executor->measure({MeasurementAction::Cancel});
            } catch (...) {
                instance_.executor->fail("failed to close measurement after a boundary error");
            }
        }
        if (instance_.executor->failed()) std::rethrow_exception(error);
    }
    std::unique_ptr<MeasurementRequest> completed;
    {
        std::lock_guard lock(queue_mutex_);
        completed = std::move(measurement_);
    }
    if (error)
        completed->completion.set_exception(error);
    else
        completed->completion.set_value(std::move(result));
    queue_cv_.notify_all();
    return true;
}

template <class Instance>
void EngineCore<Instance>::worker_loop() noexcept {
    std::uint64_t scheduled_revision = 0;
    for (;;) {
        {
            std::unique_lock lock(queue_mutex_);
            if (!stopping_ && pending_.empty() && resident_empty() && paused_.empty() &&
                deferred_settlements_.empty() && !execution_outstanding()) {
                queue_cv_.wait(lock,
                               [&] { return stopping_ || !pending_.empty() || measurement_; });
            }
            if (stopping_) {
                lock.unlock();
                fail_all(std::make_exception_ptr(RequestError(
                    RequestErrorKind::Unavailable, "inference engine is shutting down")));
                return;
            }
        }
        bool progressed  = false;
        bool outstanding = false;
        try {
            if (const auto* oldest = oldest_compute()) {
                set_host_work_class(oldest->kind == FlightKind::Decode ||
                                            (oldest->kind == FlightKind::Mixed &&
                                             oldest->decode_rows != 0)
                                        ? HostWorkClass::Decode
                                        : oldest->kind == FlightKind::Control
                                            ? HostWorkClass::Control
                                            : HostWorkClass::Prefill,
                                    oldest->kind == FlightKind::Decode
                                        ? std::span<const std::uint32_t>(oldest->lanes)
                                        : std::span<const std::uint32_t>{});
            }
            auto boundary = begin_host_phase();
            progressed    = poll_execution(boundary);
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            const auto revision = schedule_revision_.load(std::memory_order_acquire);
            const bool schedule = progressed || revision != scheduled_revision ||
                                  !execution_outstanding() || Clock::now() >= pending_deadline_;
            // The snapshot is published once the iteration's flights are submitted, so the
            // device does not wait for it.
            bool published = false;
            if (schedule) {
                scheduled_revision = revision;
                set_host_work_class(HostWorkClass::Control);
                boundary = begin_host_phase();
                begin_iteration(boundary);
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
                if (submit_iteration(boundary)) {
                    progressed = true;
                    publish_runtime_stats();
                    published = true;
                }
            }
            if (!published && revision != published_revision_) publish_runtime_stats();
            {
                ExecutionCallScope call(*this, CoordinatorCall::Flush);
                instance_.executor->flush();
                call.finish();
            }
            flush_completions();
            outstanding = execution_outstanding();
            if (!outstanding) progressed |= process_measurement();
        } catch (...) {
            log_worker_failure(std::current_exception());
            fail_all(std::current_exception());
            return;
        }
        if (outstanding) {
            const auto exposed = active_exposure_set();
            const auto started = Clock::now();
            std::this_thread::yield();
            record_coordinator_call(CoordinatorCall::Yield, elapsed_ns(started, Clock::now()),
                                    *exposed);
        }
        if (!outstanding && !progressed) {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(1));
        }
    }
}

} // namespace ninfer::runtime
