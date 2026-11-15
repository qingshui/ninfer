#pragma once

namespace ninfer::runtime {

template <class Instance>
void EngineCore<Instance>::commit_pending(PendingBatch&& pending, ComputeFlight flight) {
    const std::uint32_t frame_index = flight.ticket.frame.index;
    if (frame_index >= retiring_.size() || retiring_[frame_index])
        throw std::logic_error("another model commit is still retiring on this frame");
    EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
    // The batch rows of a mixed or packed-final flight map back to their flight members; an
    // empty mapping is the identity (every batch row follows its member).
    const auto member_of = [&flight](std::size_t row) -> std::size_t {
        return flight.batch_rows.empty() ? row : flight.batch_rows[row];
    };
    const std::size_t batch_rows = flight.batch_rows.empty() ? flight.requests.size()
                                                            : flight.batch_rows.size();
    const bool decode_flight = flight.kind == FlightKind::Decode;
    const std::size_t row_count = batch_rows;
    for (const auto& request : flight.requests)
        request->pending_execution = pending.execution_sample();
    if (row_count == 0 || row_count != pending.row_count() || pending.row_stride() == 0 ||
        (!pending.row_counts().empty() && pending.row_counts().size() != row_count) ||
        pending.tokens().size() < static_cast<std::size_t>(pending.row_stride()) * row_count) {
        (void)instance_.session->discard(std::move(pending));
        throw std::logic_error("pending batch returned an invalid ragged layout");
    }

    std::vector<CommitDecision> decisions(row_count);
    std::vector<FinishReason> finish_reasons(row_count);
    std::vector<ContinuationAction> continuations(row_count);
    std::vector<std::size_t> generated_sizes(row_count, 0);
    std::vector<bool> cancelled(row_count, false);
    std::vector<bool> failed(row_count, false);
    bool generated_staged         = false;
    const auto rollback_generated = [&]() {
        for (std::size_t row = 0; row < row_count; ++row) {
            const auto& request = flight.requests[member_of(row)];
            if (request != nullptr) { request->output.discard_preview(); }
            if (generated_staged && request != nullptr &&
                request->generated.size() >= generated_sizes[row]) {
                request->generated.resize(generated_sizes[row]);
            }
        }
        generated_staged = false;
    };
    try {
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t lane = flight.lanes[member_of(row)];
            const auto& request      = flight.requests[member_of(row)];
            // A decode row of a mixed batch counts as a decode round; a producing prefill
            // segment row is the branch's begin row instead.
            const bool decode_row =
                decode_flight || (flight.kind == FlightKind::Mixed && row < flight.decode_rows);
            if (request == nullptr || !request->sequence || !request->lane ||
                request->lane->value != lane || !request->budget ||
                request->sequence != flight.sequences[member_of(row)] ||
                request->residency != flight.residencies[member_of(row)]) {
                throw std::logic_error("pending row has no active Engine request");
            }
            if (decode_row) { ++request->host_timing.decode_rounds; }
            cancelled[row] = flight.cancelled[member_of(row)];
            failed[row]    = !cancelled[row] && pending.constraint_failed(row);
            const std::int32_t raw_count =
                pending.row_counts().empty() ? 1 : pending.row_counts()[row];
            if (raw_count <= 0 || raw_count > static_cast<std::int32_t>(pending.row_stride())) {
                throw std::logic_error("pending row has an invalid licensed extent");
            }
            const std::uint32_t count = static_cast<std::uint32_t>(raw_count);
            const auto row_tokens     = pending.tokens().subspan(row * pending.row_stride(),
                                                                 static_cast<std::size_t>(count));
            generated_sizes[row]      = request->generated.size();
            if (failed[row]) {
                decisions[row] = CommitDecision{.terminal = true, .failed = true};
                continue;
            }
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                decisions[row] = CommitDecision{
                    .accepted_tokens = 0,
                    .terminal        = true,
                    .cancelled       = true,
                };
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            const OutputDecision decision = request->output.preview_model(
                row_tokens, request->budget->remaining(), request->budget->limit_reason());
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                (!decision.finished() && decision.accepted_tokens != count) ||
                (decision.finished() && decision.continuation != ContinuationAction::Decode) ||
                (decision.prefix_execution_split_after &&
                 (*decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("output policy returned an invalid licensed prefix");
            }
            decisions[row] = CommitDecision{
                .accepted_tokens              = decision.accepted_tokens,
                .terminal                     = decision.finished(),
                .cancelled                    = false,
                .prefix_execution_split_after = decision.prefix_execution_split_after,
            };
            finish_reasons[row] = decision.finish_reason;
            continuations[row]  = decision.continuation;
        }
        generated_staged = true;
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t accepted = decisions[row].accepted_tokens;
            if (accepted == 0) { continue; }
            const auto& request = flight.requests[member_of(row)];
            if (request->generated.size() > request->generated.capacity() ||
                accepted > request->generated.capacity() - request->generated.size()) {
                throw std::logic_error("admission did not reserve generated-token capacity");
            }
            const auto first =
                pending.tokens().begin() + static_cast<std::ptrdiff_t>(row * pending.row_stride());
            request->generated.insert(request->generated.end(), first,
                                      first + static_cast<std::ptrdiff_t>(accepted));
        }
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        rollback_generated();
        const auto discarded = instance_.session->discard(std::move(pending));
        if (discarded.status != ConsumeStatus::Consumed) {
            throw std::logic_error("failed pending batch could not be discarded");
        }
        std::rethrow_exception(error);
    }

    std::optional<typename ModelContract::CommitResult> committed_storage;
    try {
        phase.pause_range();
        ExecutionCallScope program_call(*this);
        auto committed = instance_.session->commit(
            std::move(pending), std::span<const CommitDecision>(decisions.data(), row_count));
        program_call.finish();
        committed_storage.emplace(std::move(committed));
        phase.resume_range();
    } catch (...) {
        rollback_generated();
        throw;
    }
    generated_staged = false;
    auto& committed  = *committed_storage;
    if (committed.row_count != row_count) {
        throw std::logic_error("Runtime commit result is not row aligned");
    }
    for (std::size_t row = 0; row < row_count; ++row) {
        const auto member = member_of(row);
        const bool decode_row =
            decode_flight || (flight.kind == FlightKind::Mixed && row < flight.decode_rows);
        const CommitDisposition expected = failed[row]      ? CommitDisposition::FailedReleased
                                           : cancelled[row] ? CommitDisposition::CancelledReleased
                                           : decisions[row].terminal ? CommitDisposition::Finishable
                                                                     : CommitDisposition::Active;
        if (committed.rows[row].disposition != expected) {
            throw std::logic_error("Runtime commit row disposition is invalid");
        }
        if (committed.capture_ready[row] && (decode_row || expected != CommitDisposition::Active)) {
            throw std::logic_error("Runtime exposed a capture outside a committed Begin row");
        }
    }
    const bool terminal_in_batch =
        std::any_of(decisions.begin(), decisions.begin() + static_cast<std::ptrdiff_t>(row_count),
                    [](const CommitDecision& decision) { return decision.terminal; });

    for (std::size_t row = 0; row < row_count; ++row) {
        const auto member  = member_of(row);
        const auto& request = flight.requests[member];
        auto& observed       = request->speculative_stats;
        const auto& counters = committed.rows[row].speculative;
        cumulative_stats_.speculative_rounds += counters.rounds - observed.rounds;
        cumulative_stats_.speculative_draft_tokens +=
            counters.drafted_tokens - observed.drafted_tokens;
        cumulative_stats_.speculative_accepted_tokens +=
            counters.accepted_tokens - observed.accepted_tokens;
        cumulative_stats_.speculative_fallback_steps +=
            counters.fallback_steps - observed.fallback_steps;
        cumulative_stats_.speculative_committed_draft_tokens +=
            counters.committed_draft_tokens - observed.committed_draft_tokens;
        for (std::size_t i = 0; i < counters.attempted_per_position.size(); ++i) {
            cumulative_stats_.speculative_attempted_per_position.at(i) +=
                counters.attempted_per_position[i] - observed.attempted_per_position.at(i);
            cumulative_stats_.speculative_accepted_per_position.at(i) +=
                counters.accepted_per_position[i] - observed.accepted_per_position.at(i);
            observed.attempted_per_position[i] = counters.attempted_per_position[i];
            observed.accepted_per_position[i]  = counters.accepted_per_position[i];
        }
        observed.rounds                 = counters.rounds;
        observed.drafted_tokens         = counters.drafted_tokens;
        observed.accepted_tokens        = counters.accepted_tokens;
        observed.fallback_steps          = counters.fallback_steps;
        observed.committed_draft_tokens = counters.committed_draft_tokens;
        if (cancelled[row] || failed[row]) {
            request->speculative_stats = std::move(committed.rows[row].speculative);
        }
    }

    if (decode_flight) {
        for (std::size_t row = 0; row < row_count; ++row) {
            if (!cancelled[row]) {
                cumulative_stats_.committed_decode_tokens += decisions[row].accepted_tokens;
                flight.requests[member_of(row)]->committed_decode_tokens +=
                    decisions[row].accepted_tokens;
            }
        }
    }

    for (std::size_t row = 0; row < row_count; ++row) {
        const auto member           = member_of(row);
        const std::uint32_t lane     = flight.lanes[member];
        const auto& request          = flight.requests[member];
        const std::uint32_t accepted = decisions[row].accepted_tokens;
        if (failed[row]) {
            request->terminal_error     = std::make_exception_ptr(RequestError(
                RequestErrorKind::ConstraintDeadEnd, "grammar has no valid next token"));
            request->consumed_by_commit = true;
            request->model_state        = EngineRequestState::ModelFinished;
            continue;
        }
        if (!cancelled[row]) {
            flight.progressed[member] = accepted != 0;
            request->budget->commit(accepted);
            request->placement_tokens = instance_.session->load_tokens(*request->sequence);
        }
        request->consumed_by_commit = cancelled[row];
        update_recovery(request, cancelled[row]);
        auto published = request->output.commit_preview();
        auto timing    = record_committed_output(request, accepted);
        append_output(request, std::move(published), std::move(timing), &phase);
        if (request->output.output_error()) {
            if (committed.capture_ready[row]) {
                instance_.session->skip_capture(*request->sequence);
                committed.capture_ready[row] = false;
            }
            if (fail_output(request, lane)) continue;
        }
        if (decisions[row].terminal) {
            request->model_state     = EngineRequestState::ModelFinished;
            request->terminal_reason = finish_reasons[row];
        } else if (committed.capture_ready[row]) {
            if (!request->is_prefilling()) {
                throw std::logic_error("prompt-frontier capture lost its prefill owner");
            }
            const EngineRequestState post_capture_state =
                continuations[row] == ContinuationAction::ApplyTargetControl
                    ? EngineRequestState::ControlReady
                    : EngineRequestState::DecodeReady;
            if (terminal_in_batch) {
                instance_.session->skip_capture(*request->sequence);
                request->model_state = post_capture_state;
            } else {
                reserve_active_capture(request, post_capture_state);
            }
            committed.capture_ready[row] = false;
            if (!request->capture_pending) {
                request->model_state = continuations[row] == ContinuationAction::ApplyTargetControl
                                           ? EngineRequestState::ControlReady
                                           : EngineRequestState::DecodeReady;
            }
        } else {
            request->model_state = continuations[row] == ContinuationAction::ApplyTargetControl
                                       ? EngineRequestState::ControlReady
                                       : EngineRequestState::DecodeReady;
        }
    }
    phase.finish();
    record_service(flight);
    retiring_[frame_index].emplace(std::move(flight));
}

template <class Instance>
void EngineCore<Instance>::resolve_prefill_flight(
    typename ModelContract::PrefillProgress&& progress, ComputeFlight flight) {
    if (progress.rows.size() != flight.requests.size() || progress.rows.empty()) {
        throw std::logic_error("prefill result changed its frozen membership");
    }
    std::optional<PendingBatch> batch;
    std::vector<bool> produced(progress.rows.size(), false);
    for (std::size_t row = 0; row < progress.rows.size(); ++row) {
        const std::uint32_t tokens = progress.rows[row].processed_prompt_tokens;
        flight.prefill_tokens += tokens;
        flight.progressed[row] = tokens != 0;
        if (progress.rows[row].pending) {
            if (batch) { throw std::logic_error("prefill result carried two pending batches"); }
            batch.emplace(std::move(*progress.rows[row].pending));
            produced[row] = true;
        }
    }
    record_service(flight);
    for (std::size_t row = 0; row < progress.rows.size(); ++row) {
        const auto request = flight.requests[row];
        resolve_prefill_progress(request, std::move(progress.rows[row]), produced[row]);
        update_recovery(request);
    }
    if (!batch) { return; }
    // The batch of a packed final carries one decision row per producing segment, in member
    // order; it rides the first producing row. The producing members therefore come from the
    // flags the flight froze at its start when the Session reports them, and from the pending
    // row alone otherwise.
    for (std::size_t member = 0; member < flight.requests.size(); ++member) {
        const bool producing = member < flight.produces.size() ? flight.produces[member]
                                                              : produced[member];
        if (producing) { flight.batch_rows.push_back(member); }
    }
    if (flight.batch_rows.empty()) {
        throw std::logic_error("a packed prefill produced a batch without a producing segment");
    }
    flight.decode_rows = 0;
    commit_pending(std::move(*batch), std::move(flight));
}

template <class Instance>
void EngineCore<Instance>::resolve_prefill_progress(const std::shared_ptr<Request>& request,
                                                    PrefillRow&& progress, bool produced) {
    EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
    ++request->host_timing.prefill_units;
    cumulative_stats_.computed_prefill_tokens += progress.processed_prompt_tokens;
    if (!request->admitted_begin) {
        throw std::logic_error("prefill progress has no committed admission summary");
    }
    const BeginSummary& begin = *request->admitted_begin;
    if (begin.reused_prompt_tokens > begin.prompt_tokens) {
        throw std::logic_error("admitted prefix exceeds its prompt");
    }
    const std::uint32_t suffix_tokens = begin.prompt_tokens - begin.reused_prompt_tokens;
    if (request->computed_prompt_tokens > suffix_tokens ||
        progress.processed_prompt_tokens > suffix_tokens - request->computed_prompt_tokens) {
        throw std::logic_error("prefill unit exceeded the admitted prompt suffix");
    }
    const auto computed_begin = begin.reused_prompt_tokens + request->computed_prompt_tokens;
    request->computed_prompt_tokens += progress.processed_prompt_tokens;
    if (!request->cancelled.load(std::memory_order_acquire)) {
        resources_.observe_committed(*request->base_plan, request->id, computed_begin,
                                     computed_begin + progress.processed_prompt_tokens,
                                     instance_.session->capture_frontiers(*request->sequence));
    }
    if (progress.complete && request->computed_prompt_tokens != suffix_tokens) {
        throw std::logic_error("completed prefill did not reach the admitted prompt frontier");
    }
    if (progress.processed_prompt_tokens != 0) {
        request->last_progress = Clock::now();
        publish_prompt_progress(request);
    }
    if (progress.capture_ready) {
        if (progress.complete || progress.pending) {
            throw std::logic_error("prefill capture offer overlaps prompt completion");
        }
        reserve_active_capture(request, EngineRequestState::Prefill);
        return;
    }
    if (!progress.complete) { return; }
    if (!request->lane || produced != progress.pending.has_value()) {
        throw std::logic_error("completed prefill has no lane or pending token");
    }
    if (!request->admitted_begin || progress.summary != *request->admitted_begin) {
        throw std::logic_error("runtime Begin summary differs from committed admission");
    }
    request->begin = progress.summary;
}

template <class Instance>
void EngineCore<Instance>::resolve_mixed_flight(typename ModelContract::MixedProgress&& progress,
                                                 ComputeFlight flight) {
    const std::size_t segments = flight.prefill_members;
    const std::size_t producing =
        std::count(flight.produces.begin(), flight.produces.end(), true);
    if (flight.kind != FlightKind::Mixed || flight.decode_rows == 0 ||
        segments == 0 || segments > flight.requests.size() ||
        progress.prefill.rows.size() != segments ||
        progress.decode.row_count() != flight.decode_rows + producing) {
        throw std::logic_error("mixed result changed its frozen membership");
    }
    // The prefill segments settle first: their prompt progress is committed by the shared
    // batch, so none of their rows carries its own pending batch.
    for (std::size_t row = 0; row < progress.prefill.rows.size(); ++row) {
        const std::uint32_t tokens = progress.prefill.rows[row].processed_prompt_tokens;
        flight.prefill_tokens += tokens;
        flight.progressed[row] = tokens != 0;
        if (progress.prefill.rows[row].pending) {
            throw std::logic_error("mixed prefill row carried its own pending batch");
        }
        resolve_prefill_progress(flight.requests[row], std::move(progress.prefill.rows[row]),
                                 false);
        update_recovery(flight.requests[row]);
    }
    // The shared batch: the decode rows first, then the producing segments, in member order.
    flight.batch_rows.clear();
    for (std::size_t row = 0; row < flight.decode_rows; ++row) {
        flight.batch_rows.push_back(segments + row);
    }
    for (std::size_t member = 0; member < segments; ++member) {
        if (flight.produces[member]) { flight.batch_rows.push_back(member); }
    }
    record_service(flight);
    commit_pending(std::move(progress.decode), std::move(flight));
}

template <class Instance>
[[nodiscard]] typename EngineCore<Instance>::ComputeFlight
EngineCore<Instance>::freeze_members(std::span<const std::uint32_t> lanes,
                                     FlightKind kind) const {
    ComputeFlight flight;
    flight.kind = kind;
    flight.requests.reserve(lanes.size());
    flight.sequences.reserve(lanes.size());
    flight.residencies.reserve(lanes.size());
    flight.lanes.assign(lanes.begin(), lanes.end());
    flight.cancelled.reserve(lanes.size());
    flight.progressed.assign(lanes.size(), false);
    for (const auto lane : lanes) {
        const auto& request = slots_.at(lane);
        if (!request || !request->sequence || !request->residency || !request->lane ||
            request->lane->value != lane) {
            throw std::logic_error("execution membership has no resident request");
        }
        flight.requests.push_back(request);
        flight.sequences.push_back(*request->sequence);
        flight.residencies.push_back(*request->residency);
        flight.cancelled.push_back(request->cancelled.load(std::memory_order_acquire));
    }
    return flight;
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::start_prefill(std::span<const std::uint32_t> lanes,
                                                       std::uint64_t host_submit_ns) {
    if (lanes.empty() || lanes.size() > kMaximumConcurrency) {
        throw std::logic_error("prefill membership is invalid");
    }
    const bool replay = slots_[lanes.front()]->is_replaying();
    if (replay && lanes.size() != 1) {
        throw std::logic_error("a replaying lane cannot share an execution");
    }
    auto flight = freeze_members(lanes, replay ? FlightKind::Replay : FlightKind::Prefill);
    // The Session reads a contiguous constrained flag per segment.
    auto constrained = std::make_unique<bool[]>(flight.requests.size());
    for (std::size_t row = 0; row < flight.requests.size(); ++row) {
        constrained[row] = flight.requests[row]->output.constrained();
    }
    std::optional<ExecutionTicket> ticket;
    std::optional<PrefillStart> admitted;
    std::size_t segments = 1;
    flight.started        = Clock::now();
    ExecutionCallScope call(*this);
    if (replay) {
        ticket = instance_.session->start_replay(flight.sequences.front());
    } else {
        admitted = instance_.session->start_prefill(
            flight.sequences, {constrained.get(), flight.requests.size()});
        if (admitted) {
            ticket   = admitted->ticket;
            segments = admitted->segments;
        }
    }
    call.finish();
    if (!ticket) return false;
    if (segments == 0 || segments > lanes.size()) {
        throw std::logic_error("Session admitted an invalid number of prefill segments");
    }
    // Freeze only the admitted prefix. The remaining lanes lose their unit for this execution.
    flight.requests.resize(segments);
    flight.sequences.resize(segments);
    flight.residencies.resize(segments);
    flight.lanes.resize(segments);
    flight.cancelled.resize(segments);
    flight.progressed.resize(segments);
    flight.ticket        = *ticket;
    flight.host_submit_ns = host_submit_ns;
    if (!replay) {
        // The producing flags of the admitted segments, for the packed-final batch mapping. A
        // Session that reports only the segment count freezes none; its pending row names its
        // producing member on its own.
        flight.produces.assign(segments, false);
        if constexpr (requires { admitted->segments[std::size_t{0}].produces_token; }) {
            for (std::size_t segment = 0; segment < segments; ++segment) {
                flight.produces[segment] = admitted->segments[segment].produces_token;
            }
        }
    } else {
        flight.prefill_members = 1;
    }
    admit_compute(std::move(flight));
    for (const auto lane : lanes) reserved_units_[lane].reset();
    return true;
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::start_decode(std::span<const std::uint32_t> lanes,
                                                      std::uint64_t host_submit_ns) {
    if (lanes.empty() || lanes.size() > kMaximumConcurrency) {
        throw std::logic_error("decode membership is invalid");
    }
    auto flight = freeze_members(lanes, FlightKind::Decode);
    flight.decode_rows = static_cast<std::uint32_t>(flight.requests.size());
    std::vector<RoundBudget> budgets;
    budgets.reserve(flight.requests.size());
    // The Session reads a contiguous constrained flag per row.
    auto constrained = std::make_unique<bool[]>(flight.requests.size());
    for (std::size_t row = 0; row < flight.requests.size(); ++row) {
        const auto& request = flight.requests[row];
        if (!request->budget) {
            throw std::logic_error("decode-ready request has no generation budget");
        }
        budgets.push_back(RoundBudget{
            .generated_tokens_remaining =
                request->recovery_pending
                    ? 1U
                    : request->output.model_token_budget_remaining(request->budget->remaining())});
        if (budgets.back().generated_tokens_remaining == 0) {
            throw std::logic_error("decode-ready request has no licensed model tokens");
        }
        constrained[row] = request->output.constrained();
    }
    flight.started        = Clock::now();
    flight.host_submit_ns = host_submit_ns;
    ExecutionCallScope call(*this);
    const auto ticket = instance_.session->start_decode(
        flight.sequences, budgets, {constrained.get(), flight.requests.size()});
    call.finish();
    if (!ticket) return false;
    flight.ticket = *ticket;
    for (const auto lane : flight.lanes) reserved_units_[lane].reset();
    admit_compute(std::move(flight));
    return true;
}

// Starts one mixed flight: the planned prefill segments in order, followed by the planned
// decode rows. Returns the decode rows the Session admitted.
template <class Instance>
[[nodiscard]] std::optional<std::uint32_t>
EngineCore<Instance>::start_mixed(std::span<const std::uint32_t> prefill_lanes,
                                  std::span<const std::uint32_t> prefill_caps,
                                  std::span<const std::uint32_t> decode_lanes,
                                  std::uint64_t host_submit_ns) {
    if (prefill_lanes.empty() || decode_lanes.empty() || prefill_caps.size() != prefill_lanes.size() ||
        prefill_lanes.size() + decode_lanes.size() > kMaximumConcurrency) {
        throw std::logic_error("mixed membership is invalid");
    }
    std::vector<std::uint32_t> lanes(prefill_lanes.begin(), prefill_lanes.end());
    lanes.insert(lanes.end(), decode_lanes.begin(), decode_lanes.end());
    auto flight = freeze_members(lanes, FlightKind::Mixed);
    std::vector<RoundBudget> budgets;
    budgets.reserve(decode_lanes.size());
    auto prefill_constrained = std::make_unique<bool[]>(prefill_lanes.size());
    auto decode_constrained  = std::make_unique<bool[]>(decode_lanes.size());
    for (std::size_t row = 0; row < prefill_lanes.size(); ++row) {
        prefill_constrained[row] = flight.requests[row]->output.constrained();
    }
    for (std::size_t row = 0; row < decode_lanes.size(); ++row) {
        const auto& request = flight.requests[prefill_lanes.size() + row];
        if (!request->budget) {
            throw std::logic_error("decode-ready request has no generation budget");
        }
        budgets.push_back(RoundBudget{
            .generated_tokens_remaining =
                request->recovery_pending
                    ? 1U
                    : request->output.model_token_budget_remaining(request->budget->remaining())});
        if (budgets.back().generated_tokens_remaining == 0) {
            throw std::logic_error("decode-ready request has no licensed model tokens");
        }
        decode_constrained[row] = request->output.constrained();
    }
    flight.started        = Clock::now();
    flight.host_submit_ns = host_submit_ns;
    std::optional<MixedStart> started;
    {
        ExecutionCallScope call(*this);
        started = instance_.session->start_mixed(
            {flight.sequences.data(), prefill_lanes.size()},
            {prefill_constrained.get(), prefill_lanes.size()},
            {flight.sequences.data() + prefill_lanes.size(), decode_lanes.size()}, budgets,
            {decode_constrained.get(), decode_lanes.size()});
        call.finish();
    }
    if (!started) { return std::nullopt; }
    // A Session that reports only the segment count names no producing flag; a mixed flight
    // of such a Session cannot exist, so its produces stay empty.
    const auto fill_produces = [&flight, &started](std::size_t count) {
        flight.produces.assign(count, false);
        if constexpr (requires { started->segments[std::size_t{0}].produces_token; }) {
            for (std::size_t segment = 0; segment < count; ++segment) {
                flight.produces[segment] = started->segments[segment].produces_token;
            }
        }
    };
    const std::size_t segments = [&] {
        if constexpr (requires { started->segments.size(); }) {
            return started->segments.size();
        } else {
            return static_cast<std::size_t>(started->segments);
        }
    }();
    const std::uint32_t rows    = started->decode_rows;
    if (segments == 0 || segments > prefill_lanes.size() || rows > decode_lanes.size()) {
        throw std::logic_error("Session admitted an invalid mixed membership");
    }
    if (rows == 0) {
        // The Session fell back to a pure packed prefill: no decode row rides, and the
        // membership is the admitted prefill prefix alone.
        flight.requests.resize(segments);
        flight.sequences.resize(segments);
        flight.residencies.resize(segments);
        flight.lanes.resize(segments);
        flight.cancelled.resize(segments);
        flight.progressed.resize(segments);
        flight.kind        = FlightKind::Prefill;
        flight.decode_rows = 0;
        fill_produces(segments);
        flight.ticket = started->ticket;
        admit_compute(std::move(flight));
        for (const auto lane : lanes) reserved_units_[lane].reset();
        return 0U;
    }
    // The membership is the admitted prefill prefix plus the admitted decode rows.
    flight.requests.resize(segments + rows);
    flight.sequences.resize(segments + rows);
    flight.residencies.resize(segments + rows);
    flight.lanes.resize(segments + rows);
    flight.cancelled.resize(segments + rows);
    flight.progressed.resize(segments + rows);
    flight.decode_rows   = rows;
    flight.prefill_members = segments;
    fill_produces(segments);
    flight.decode_lanes.assign(flight.lanes.begin() + static_cast<std::ptrdiff_t>(segments),
                               flight.lanes.end());
    flight.ticket = started->ticket;
    admit_compute(std::move(flight));
    for (const auto lane : lanes) reserved_units_[lane].reset();
    return rows;
}

template <class Instance>
void EngineCore<Instance>::discard_control_preview(const ComputeFlight& flight) noexcept {
    for (std::size_t row = 0; row < flight.requests.size(); ++row) {
        const auto& request = flight.requests[row];
        request->output.discard_preview();
        if (row < flight.generated_sizes.size()) {
            request->generated.resize(flight.generated_sizes[row]);
        }
    }
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::start_control(const ControlMembership& membership) {
    if (membership.empty() || membership.row_stride == 0 ||
        membership.tokens.size() != membership.size * membership.row_stride) {
        throw std::logic_error("thinking control membership is invalid");
    }
    auto flight           = freeze_members(membership.lane_span(), FlightKind::Control);
    flight.control_tokens = membership.tokens;
    flight.control_stride = membership.row_stride;
    flight.generated_sizes.reserve(flight.requests.size());
    for (const auto& request : flight.requests) {
        flight.generated_sizes.push_back(request->generated.size());
    }
    flight.started        = Clock::now();
    flight.host_submit_ns = 0;
    std::vector<std::optional<std::uint32_t>> splits(flight.requests.size());
    try {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        for (std::size_t row = 0; row < flight.requests.size(); ++row) {
            const auto& request = flight.requests[row];
            const auto tokens   = std::span<const TokenId>(flight.control_tokens)
                                    .subspan(row * flight.control_stride, flight.control_stride);
            const auto decision =
                request->output.preview_control(tokens, request->budget->remaining());
            if (decision.accepted_tokens != flight.control_stride || decision.finished() ||
                decision.continuation != ContinuationAction::Decode ||
                (decision.prefix_execution_split_after &&
                 (*decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("target control preview returned an invalid decision");
            }
            splits[row] = decision.prefix_execution_split_after;
            if (tokens.size() > request->generated.capacity() - request->generated.size()) {
                throw std::logic_error("admission did not reserve control token capacity");
            }
            request->generated.insert(request->generated.end(), tokens.begin(), tokens.end());
        }
        phase.finish();
        ExecutionCallScope call(*this);
        const auto ticket = instance_.session->start_control(
            flight.sequences, flight.control_tokens, flight.control_stride,
            {splits.data(), flight.requests.size()});
        call.finish();
        if (!ticket) {
            discard_control_preview(flight);
            return false;
        }
        flight.ticket = *ticket;
    } catch (...) {
        discard_control_preview(flight);
        throw;
    }
    for (const auto lane : flight.lanes) {
        reserved_units_[lane].reset();
        control_executed_[lane] = true;
    }
    admit_compute(std::move(flight));
    return true;
}

template <class Instance>
void EngineCore<Instance>::resolve_control(ComputeFlight& flight) {
    EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
    cumulative_stats_.committed_decode_tokens += flight.requests.size() * flight.control_stride;
    cumulative_stats_.injected_control_tokens += flight.requests.size() * flight.control_stride;
    for (std::size_t row = 0; row < flight.requests.size(); ++row) {
        const auto& request = flight.requests[row];
        ++request->host_timing.control_units;
        request->committed_decode_tokens += flight.control_stride;
        request->injected_control_tokens += flight.control_stride;
        request->budget->commit(flight.control_stride);
        flight.progressed[row]    = flight.control_stride != 0;
        request->placement_tokens = instance_.session->load_tokens(*request->sequence);
        auto timing               = record_committed_output(request, flight.control_stride);
        append_output(request, request->output.commit_preview(), std::move(timing), &phase);
        if (fail_output(request, flight.lanes[row])) continue;
        update_recovery(request);
        request->model_state = EngineRequestState::DecodeReady;
    }
    record_service(flight);
}

template <class Instance>
void EngineCore<Instance>::supply_decision(ExecutionTicket ticket,
                                           const DecisionRequest& decision) {
    const auto* flight = find_compute(ticket);
    if (flight == nullptr || decision.vocabulary == 0) {
        throw std::logic_error("decision input does not belong to an active execution");
    }
    DecisionInput input;
    input.stage = decision.stage;
    input.rows.reserve(decision.rows.size());
    std::vector<bool> seen(flight->requests.size(), false);
    const std::size_t words_per_position = (std::size_t(decision.vocabulary) + 31) / 32;
    for (const auto& row : decision.rows) {
        if (row.member >= flight->requests.size() || seen[row.member]) {
            throw std::logic_error("decision request has invalid compact membership");
        }
        seen[row.member]    = true;
        const auto& request = flight->requests[row.member];
        if (!request->output.constrained()) {
            throw std::logic_error("decision requested a mask for an unconstrained row");
        }
        auto& result     = input.rows.emplace_back();
        result.member    = row.member;
        result.positions = static_cast<std::uint32_t>(row.drafts.size() + 1);
        result.words.resize(words_per_position * result.positions);
        result.dead_positions = request->output.grammar_masks(row.drafts, result.words);
    }
    ExecutionCallScope call(*this);
    instance_.session->supply(ticket, std::move(input));
    call.finish();
}

template <class Instance>
void EngineCore<Instance>::record_decision_upload(const DecisionConsumed& consumed) {
    auto* flight = find_compute(consumed.ticket);
    if (flight == nullptr) {
        throw std::logic_error("decision upload does not belong to an active execution");
    }
    for (const auto& row : consumed.rows) {
        if (row.member >= flight->requests.size()) {
            throw std::logic_error("decision upload has invalid compact membership");
        }
        flight->requests[row.member]->output.constraint_uploaded(row.bytes);
    }
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::poll_execution(HostPhaseMeasurement& boundary) {
    const bool context_was_pending = instance_.session->has_context_operation();
    const auto owner               = context_owner_;
    if (owner && owner->cancelled.load(std::memory_order_acquire) &&
        instance_.session->has_context_operation()) {
        instance_.session->cancel_context();
    }
    bool progressed = false;
    for (;;) {
        ExecutionCallScope call(*this, CoordinatorCall::Poll);
        auto event = instance_.session->poll();
        call.finish();
        if (!event) break;
        progressed = true;
        if (auto* decision = std::get_if<DecisionRequest>(&event->result)) {
            supply_decision(event->ticket, *decision);
            continue;
        }
        if (const auto* consumed = std::get_if<DecisionConsumed>(&event->result)) {
            record_decision_upload(*consumed);
            continue;
        }
        if (auto* context = std::get_if<typename ModelContract::ContextProgress>(&event->result)) {
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            const auto work_class   = current_host_work_class_;
            const auto decode_lanes = current_decode_lanes_;
            const auto decode_count = current_decode_lane_count_;
            set_host_work_class(HostWorkClass::Control);
            boundary = begin_host_phase();
            resolve_context_progress(std::move(*context), boundary);
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            set_host_work_class(work_class, {decode_lanes.data(), decode_count});
            boundary = begin_host_phase();
            continue;
        }
        if (const auto* complete = std::get_if<ExecutionComplete>(&event->result)) {
            auto* retiring = find_retiring(event->ticket);
            auto* flight   = retiring ? retiring : find_compute(event->ticket);
            if (flight == nullptr || flight->sequences != event->members ||
                complete->row_count != flight->requests.size())
                throw std::logic_error("execution completion changed its frozen membership");
            const auto kind = static_cast<std::size_t>(complete->sample.work.kind);
            const std::size_t batch_rows =
                flight->batch_rows.empty() ? flight->requests.size() : flight->batch_rows.size();
            for (const auto& request : flight->requests) {
                request->executions[kind].observe(complete->sample.latency);
                if (complete->sample.work.kind == ExecutionKind::Decode)
                    request->decode_by_batch[batch_rows - 1].observe(complete->sample.latency);
                accumulate_gpu(request->max_rank_gpu, complete->sample.max_rank_gpu);
                request->pending_execution.reset();
            }
            flight->retired_ns = complete->sample.latency.total_ns;
            record_service(*flight);
            record_flight(*flight, complete->sample);
            if (retiring) retiring_[event->ticket.frame.index].reset();
            continue;
        }
        auto* computing = find_compute(event->ticket);
        if (computing == nullptr || computing->sequences != event->members) {
            throw std::logic_error("execution result changed its frozen membership");
        }
        auto flight = std::move(*computing);
        compute_[event->ticket.frame.index].reset();
        const bool decode_flight =
            flight.kind == FlightKind::Decode ||
            (flight.kind == FlightKind::Mixed && flight.decode_rows != 0);
        set_host_work_class(decode_flight ? HostWorkClass::Decode
                            : flight.kind == FlightKind::Control ? HostWorkClass::Control
                                                                  : HostWorkClass::Prefill,
                            flight.kind == FlightKind::Decode
                                ? std::span<const std::uint32_t>(flight.lanes)
                                : flight.kind == FlightKind::Mixed
                                    ? std::span<const std::uint32_t>(flight.decode_lanes)
                                    : std::span<const std::uint32_t>{});
        if (auto* pending = std::get_if<PendingBatch>(&event->result)) {
            commit_pending(std::move(*pending), std::move(flight));
        } else if (auto* prefill =
                       std::get_if<typename ModelContract::PrefillProgress>(&event->result)) {
            resolve_prefill_flight(std::move(*prefill), std::move(flight));
        } else if (auto* mixed =
                       std::get_if<typename ModelContract::MixedProgress>(&event->result)) {
            resolve_mixed_flight(std::move(*mixed), std::move(flight));
        } else if (auto* replay =
                       std::get_if<typename ModelContract::ReplayProgress>(&event->result)) {
            const auto& request = flight.requests.front();
            if (!request->recovery) throw std::logic_error("replay progress has no restoration");
            request->recovery->frontier = replay->frontier;
            if (replay->processed_tokens) request->last_progress = Clock::now();
            flight.prefill_tokens = replay->processed_tokens;
            request->replayed_tokens += replay->processed_tokens;
            cumulative_stats_.replayed_tokens += replay->processed_tokens;
            if (replay->capture_ready) reserve_active_capture(request, EngineRequestState::Replay);
            if (replay->complete) {
                ++cumulative_stats_.replay_restores;
                ++request->replay_restores;
                request->model_state = request->resume_phase;
                observe_scheduling(request, GenerationSchedulingTransition::ReplayComplete);
            }
            update_recovery(request);
            record_service(flight);
        } else if (std::holds_alternative<typename ModelContract::ControlComplete>(event->result)) {
            resolve_control(flight);
        } else {
            throw std::logic_error("generation Session emitted a scoring result");
        }
    }
    if (context_was_pending && !instance_.session->has_context_operation()) {
        request_admission_check();
        progressed = true;
    }
    return progressed;
}

template <class Instance>
void EngineCore<Instance>::record_flight(const ComputeFlight& flight,
                                         const ExecutionSample& sample) noexcept {
    auto& stats = cumulative_stats_;
    stats.flight_prefill_tokens += flight.prefill_tokens;
    stats.flight_prefill_segments +=
        flight.kind == FlightKind::Mixed
            ? flight.prefill_members
            : (flight.kind == FlightKind::Prefill || flight.kind == FlightKind::Replay
                   ? static_cast<std::uint32_t>(flight.requests.size())
                   : 0U);
    stats.flight_decode_rows += flight.kind == FlightKind::Decode
                                    ? static_cast<std::uint32_t>(flight.requests.size())
                                    : flight.decode_rows;
    stats.flight_host_submit_ns += flight.host_submit_ns;
    stats.flight_gpu_ns += sample.latency.total_ns;
    constexpr std::size_t kRecentFlights = 16;
    stats.recent_flights.push_back(FlightRecord{
        .index           = flight.index,
        .prefill_tokens  = flight.prefill_tokens,
        .prefill_segments = flight.kind == FlightKind::Mixed
                                ? flight.prefill_members
                                : (flight.kind == FlightKind::Prefill ||
                                               flight.kind == FlightKind::Replay
                                           ? static_cast<std::uint32_t>(flight.requests.size())
                                           : 0U),
        .decode_rows     = flight.kind == FlightKind::Decode
                               ? static_cast<std::uint32_t>(flight.requests.size())
                               : flight.decode_rows,
        .host_submit_ns  = flight.host_submit_ns,
        .gpu_ns          = sample.latency.total_ns,
        .gpu_timing      = sample,
    });
    if (stats.recent_flights.size() > kRecentFlights) {
        stats.recent_flights.erase(stats.recent_flights.begin());
    }
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::execution_outstanding() const noexcept {
    return compute_flights() != 0 || instance_.session->has_context_operation() ||
           instance_.executor->has_pending_work();
}

template <class Instance>
typename EngineCore<Instance>::ComputeFlight*
EngineCore<Instance>::find_compute(const ExecutionTicket& ticket) noexcept {
    if (ticket.frame.kind != FrameKind::Compute || ticket.frame.index >= compute_.size())
        return nullptr;
    auto& flight = compute_[ticket.frame.index];
    return flight && flight->ticket == ticket ? &*flight : nullptr;
}

template <class Instance>
typename EngineCore<Instance>::ComputeFlight*
EngineCore<Instance>::find_retiring(const ExecutionTicket& ticket) noexcept {
    if (ticket.frame.kind != FrameKind::Compute || ticket.frame.index >= retiring_.size())
        return nullptr;
    auto& flight = retiring_[ticket.frame.index];
    return flight && flight->ticket == ticket ? &*flight : nullptr;
}

template <class Instance>
std::size_t EngineCore<Instance>::compute_flights() const noexcept {
    const auto occupied = [](const auto& flight) { return flight.has_value(); };
    return static_cast<std::size_t>(std::count_if(compute_.begin(), compute_.end(), occupied) +
                                    std::count_if(retiring_.begin(), retiring_.end(), occupied));
}

template <class Instance>
const typename EngineCore<Instance>::ComputeFlight*
EngineCore<Instance>::oldest_compute() const noexcept {
    const ComputeFlight* oldest = nullptr;
    for (const auto& flight : compute_)
        if (flight && (oldest == nullptr || flight->ticket.id < oldest->ticket.id))
            oldest = &*flight;
    return oldest;
}

template <class Instance>
bool EngineCore<Instance>::compute_frame_available() const noexcept {
    // The executor owns frame admission; the tables only mirror its occupied compute frames.
    return compute_flights() < compute_.size();
}

template <class Instance>
void EngineCore<Instance>::admit_compute(ComputeFlight&& flight) {
    const auto index = flight.ticket.frame.index;
    if (flight.ticket.frame.kind != FrameKind::Compute || index >= compute_.size() ||
        compute_[index] || retiring_[index])
        throw std::logic_error("execution ticket names an occupied compute frame");
    flight.index = flight_index_++;
    ++cumulative_stats_.flights;
    compute_[index].emplace(std::move(flight));
}

template <class Instance>
void EngineCore<Instance>::record_service(ComputeFlight& flight) noexcept {
    if (!flight.retired_ns || flight.service_accounted) return;
    if (flight.kind == FlightKind::Replay) {
        for (const auto& request : flight.requests)
            scheduler_.recovery_work(request->id, *flight.retired_ns);
        flight.service_accounted = true;
        return;
    }
    bool progressed = false;
    for (std::size_t row = 0; row < flight.requests.size(); ++row) {
        if (!flight.progressed[row]) continue;
        scheduler_.served(flight.requests[row]->id, *flight.retired_ns);
        progressed = true;
    }
    flight.service_accounted = progressed;
}

template <class Instance>
void EngineCore<Instance>::log_worker_failure(std::exception_ptr error) const noexcept {
    // Preserve the first cause before cleanup mutates membership or drains the executor.
    // Do not depend on a product-specific logger, or throw from the failure handler.
    const auto report = [&](const char* reason) {
        std::fprintf(stderr,
                     "NInfer worker failed: %s | progress_owner=%llu paused=%zu "
                     "context=%d compute=%d retiring=%d\n",
                     reason,
                     static_cast<unsigned long long>(scheduler_.progress_owner().value_or(0)),
                     paused_.size(), instance_.session->has_context_operation(),
                     static_cast<int>(std::count_if(compute_.begin(), compute_.end(),
                                                    [](const auto& f) { return f.has_value(); })),
                     static_cast<int>(std::count_if(retiring_.begin(), retiring_.end(),
                                                    [](const auto& f) { return f.has_value(); })));
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request)
                std::fprintf(stderr, "  request=%llu lane=%u state=%u recovery=%d\n",
                             static_cast<unsigned long long>(request->id), lane,
                             static_cast<unsigned>(request->model_state),
                             request->recovery_pending);
        }
    };
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& cause) { report(cause.what()); } catch (...) {
        report("non-standard exception");
    }
}

template <class Instance>
void EngineCore<Instance>::fail_all(std::exception_ptr error) noexcept {
    std::deque<std::shared_ptr<Request>> pending;
    std::unique_ptr<MeasurementRequest> measurement;
    {
        std::lock_guard lock(queue_mutex_);
        failed_ = true;
        pending.swap(pending_);
        measurement = std::move(measurement_);
    }
    scheduler_ = Scheduling{max_concurrency_};
    std::vector<std::shared_ptr<Request>> materializing;
    for (auto& entry : materializing_) {
        if (entry) { materializing.push_back(entry->request); }
        entry.reset();
    }
    // Native retires all transfer readers before any external checkpoint owner is destroyed.
    instance_.session->fail_all_cleanup();
    for (auto& flight : compute_) {
        if (flight && flight->kind == FlightKind::Control) discard_control_preview(*flight);
        flight.reset();
    }
    for (auto& flight : retiring_) flight.reset();
    {
        std::lock_guard lock(queue_mutex_);
        groups_.clear();
    }
    admission_decision_.reset();
    for (auto& decision : capture_decisions_) { decision.reset(); }
    for (auto& unit : reserved_units_) { unit.reset(); }
    context_owner_.reset();
    std::vector<std::shared_ptr<Request>> paused;
    paused.swap(paused_);
    const auto residents = slots_;
    const auto release   = [&](const std::shared_ptr<Request>& request) {
        if (request) record_completion(request, FinishReason::None, true);
        if (request && prepare_completion(request)) { release_request(request); }
    };
    for (const auto& request : paused) { release(request); }
    for (const auto& request : residents) { release(request); }
    for (const auto& request : materializing) { release(request); }
    for (const auto& request : pending) { release(request); }
    resources_.release_all(*instance_.session);
    publish_runtime_stats();
    const auto publish = [&](const std::shared_ptr<Request>& request) {
        if (request) { publish_completion(request, {}, error); }
    };
    for (const auto& request : paused) { publish(request); }
    for (const auto& request : residents) { publish(request); }
    for (const auto& request : materializing) { publish(request); }
    for (const auto& request : pending) { publish(request); }
    if (measurement) measurement->completion.set_exception(error);
    queue_cv_.notify_all();
}

} // namespace ninfer::runtime
