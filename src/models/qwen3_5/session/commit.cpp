#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

// The pending batch covers every decision row of the execution: all members of a decode
// execution, the decode rows of a mixed one followed by its final segments, and the final
// segments of a packed prefill.
PendingBatch SessionImpl::generated_pending(const Execution& current,
                                            const GeneratedResult& generated) {
    const auto batch = current.pending_members.size();
    if (!generated.row_stride || generated.tokens.size() != batch * generated.row_stride ||
        generated.counts.size() != batch || generated.dead_positions.size() != batch)
        throw std::logic_error("generated result has an invalid shape");
    if ((current.kind == OperationKind::Decode || current.kind == OperationKind::Mixed) &&
        (generated.draft_extents.size() != batch || generated.accepted_drafts.size() != batch))
        throw std::logic_error("decode result omitted its proposal extents");
    PendingBatch out;
    out.ticket_ = current.ticket;
    out.members_.reserve(batch);
    for (const auto member : current.pending_members)
        out.members_.push_back(current.members[member]);
    out.tokens_    = generated.tokens;
    out.stride_    = generated.row_stride;
    out.execution_ = current.observed_result;
    for (std::size_t b = 0; b < batch; ++b) {
        const auto count = generated.counts[b];
        if (!count || count > generated.row_stride || count > 16)
            throw std::logic_error("generated result has an invalid licensed prefix");
        out.counts_.push_back(static_cast<std::int32_t>(count));
        out.failed_.push_back((generated.dead_positions[b] & ((1U << count) - 1)) != 0);
        auto& state       = sequences[lane(out.members_[b])];
        auto& request     = requests[state.lane];
        request.lifecycle = Lifecycle::Pending;
    }
    return out;
}

// The execution a pending batch belongs to, by its ticket, when the batch still names the
// decision rows of that execution.
const SessionImpl::Execution*
SessionImpl::matching_execution(const PendingBatch& pending) const noexcept {
    const auto* current = find_execution(pending.ticket_);
    if (!current || current->pending_members.size() != pending.members_.size()) return nullptr;
    for (std::size_t b = 0; b < pending.members_.size(); ++b)
        if (current->members[current->pending_members[b]] != pending.members_[b]) return nullptr;
    return current;
}

namespace {
ContinuationMetadata forward_metadata(ContinuationMetadata value, std::uint32_t count,
                                      SpeculativeBackend backend, bool backend_pool) {
    if (!count) return value;
    value.execution_frontier += count;
    value.text_kv_valid           = value.execution_frontier;
    value.tail_hidden_valid       = true;
    value.dflash_context_frontier = value.execution_frontier;
    value.backend_kv_valid = backend == SpeculativeBackend::Mtp ? value.execution_frontier - 1
                             : backend_pool                     ? value.execution_frontier
                                                                : 0;
    return value;
}

// One prefill segment of a completed execution, reported before its settlement is adopted.
PrefillRowProgress prefill_row(const SessionImpl& session, const SessionImpl::Execution& current,
                               std::size_t b) {
    const auto& request = session.requests[session.lane(current.members[b])];
    const auto end      = current.completed[b].execution_frontier;
    PrefillRowProgress row;
    row.summary = {request.base->summary.prompt_tokens, request.prefill->base,
                   request.prefill->reuse};
    row.processed_prompt_tokens = current.columns[b];
    row.capture_ready =
        request.capture_reservation && request.capture_reservation->frontier == end;
    row.complete = end == request.base->summary.prompt_tokens;
    return row;
}

void prepare_identity(SequenceState& state, std::size_t count, std::optional<std::uint32_t> split) {
    if (state.prefix_identity.size() != state.ledger.size() ||
        state.prefix_digests.size() != state.ledger.size())
        throw std::logic_error("generated identity does not match its ledger");
    state.ledger.reserve(state.ledger.size() + count);
    state.prefix_identity.prepare_generated(count, state.continuation.metadata.rope_delta, split);
    state.prefix_digests.prepare_generated(count, state.continuation.metadata.rope_delta, split);
}

void append_identity(SequenceState& state, std::span<const TokenId> tokens,
                     std::optional<std::uint32_t> split) noexcept {
    state.ledger.insert(state.ledger.end(), tokens.begin(), tokens.end());
    state.prefix_identity.append_generated(tokens.size(), state.continuation.metadata.rope_delta,
                                           split);
    state.prefix_digests.append_generated(tokens, state.continuation.metadata.rope_delta, split);
}
} // namespace

void SessionImpl::receive_result(runtime::ResultReady<ExecutionTypes>&& event) {
    if (auto* context = std::get_if<ContextResult>(&event.result.output)) {
        if (release_ticket && *release_ticket == event.ticket) {
            if (release_result) throw std::logic_error("duplicate residency release result");
            release_result = true;
            return;
        }
        receive_context_result(event.ticket, std::move(*context));
        return;
    }
    auto* owner = find_execution(event.ticket);
    if (!owner || owner->result_ready)
        throw std::logic_error("execution result does not match its transaction");
    auto& current           = *owner;
    current.result_ready    = true;
    current.observed_result = event.execution;

    if (auto* generated = std::get_if<GeneratedResult>(&event.result.output)) {
        if (current.kind != OperationKind::Decode)
            throw std::logic_error("decode result has the wrong execution kind");
        current.generated.emplace(std::move(*generated));
        current.pending_members.resize(current.members.size());
        for (std::size_t b = 0; b < current.members.size(); ++b) current.pending_members[b] = b;
        auto pending = generated_pending(current, *current.generated);
        events.push_back({current.ticket, current.members, std::move(pending)});
        return;
    }
    if (auto* mixed = std::get_if<MixedResult>(&event.result.output)) {
        const auto first = current.decode_begin;
        if (current.kind != OperationKind::Mixed || !first || mixed->prefill.generated ||
            mixed->prefill.processed_tokens.size() != first ||
            !std::equal(mixed->prefill.processed_tokens.begin(),
                        mixed->prefill.processed_tokens.end(), current.columns.begin()))
            throw std::logic_error("mixed result changed its reserved input windows");
        // The segments settle with the decision rows, so their metadata waits in `completed`.
        MixedProgress progress;
        progress.prefill.rows.reserve(first);
        for (std::size_t b = 0; b < first; ++b) {
            const bool produces = b < current.producing.size() && current.producing[b];
            current.completed.push_back(forward_metadata(current.bases[b].metadata,
                                                         current.columns[b], speculative_backend,
                                                         has_backend_pool()));
            progress.prefill.rows.push_back(prefill_row(*this, current, b));
            // Only a non-final segment without its own decision row must not have finished the
            // prompt; a final one reports complete and samples below.
            if (progress.prefill.rows.back().complete && !produces)
                throw std::logic_error("completed prefill omitted its first token");
        }
        current.generated.emplace(std::move(mixed->decode));
        // The batch carries the decode rows first, then the final segments in segment order.
        current.pending_members.clear();
        for (std::size_t b = first; b < current.members.size(); ++b)
            current.pending_members.push_back(static_cast<std::uint32_t>(b));
        for (std::size_t b = 0; b < first; ++b)
            if (current.producing[b]) current.pending_members.push_back(static_cast<std::uint32_t>(b));
        progress.decode = generated_pending(current, *current.generated);
        events.push_back({current.ticket, current.members, std::move(progress)});
        return;
    }
    if (auto* forward = std::get_if<ForwardResult>(&event.result.output)) {

        if (forward->processed_tokens != current.columns)
            throw std::logic_error("forward result changed its reserved input window");
        for (std::size_t b = 0; b < current.members.size(); ++b)
            current.completed.push_back(forward_metadata(current.bases[b].metadata,
                                                         current.columns[b], speculative_backend,
                                                         has_backend_pool()));
        if (current.kind == OperationKind::Prefill) {
            PrefillProgress progress;
            progress.rows.reserve(current.members.size());
            for (std::size_t b = 0; b < current.members.size(); ++b) {
                progress.rows.push_back(prefill_row(*this, current, b));
                // A member that carries a decision row re-derives its capture readiness when the
                // commit adopts its settlement; a resting member settles in the same commit but
                // reports no decision row, so its row offers the capture here exactly as a mixed
                // execution's segment does.
                const bool produces = b < current.producing.size() && current.producing[b];
                if (forward->generated && produces)
                    progress.rows.back().capture_ready = false;
            }
            if (forward->generated) {
                // The batch carries one decision row per final segment, in segment order.
                current.pending_members.clear();
                for (std::size_t b = 0; b < current.members.size(); ++b) {
                    if (!(b < current.producing.size() && current.producing[b])) continue;
                    if (!progress.rows[b].complete)
                        throw std::logic_error("incomplete prefill produced a token");
                    current.pending_members.push_back(static_cast<std::uint32_t>(b));
                }
                if (current.pending_members.empty())
                    throw std::logic_error("sampled prefill has no producing segment");
                current.generated.emplace(std::move(*forward->generated));
                // The whole batch rides the first producing row; its members name every
                // producing segment of the execution.
                auto pending = generated_pending(current, *current.generated);
                progress.rows[current.pending_members.front()].pending.emplace(std::move(pending));
                events.push_back({current.ticket, current.members, std::move(progress)});
                return;
            }
            if (std::any_of(progress.rows.begin(), progress.rows.end(),
                            [](const auto& row) { return row.complete; }))
                throw std::logic_error("completed prefill omitted its first token");
            current.completion.emplace(
                SessionEvent{current.ticket, current.members, std::move(progress)});
        } else if (current.kind == OperationKind::Replay) {
            const auto& request = requests[lane(current.members[0])];
            const auto end      = current.completed[0].execution_frontier;
            const bool capture =
                request.capture_reservation && request.capture_reservation->frontier == end;
            current.completion.emplace(
                SessionEvent{current.ticket, current.members,
                             ReplayProgress{capture, current.columns[0],
                                            end == request.replay_target && !capture, end}});
        } else if (current.kind == OperationKind::Control) {
            current.completion.emplace(
                SessionEvent{current.ticket, current.members, ControlComplete{}});
        } else
            throw std::logic_error("unexpected forward result");
        commit_completed_unit(current);
        return;
    }
    if (std::holds_alternative<NormalizeResult>(event.result.output)) {
        if (current.kind != OperationKind::Normalize)
            throw std::logic_error("normalize result has the wrong execution kind");
        auto metadata                    = current.bases[0].metadata;
        metadata.dflash_context_frontier = metadata.execution_frontier;
        metadata.backend_kv_valid        = has_backend_pool() ? metadata.execution_frontier : 0;
        current.completed.push_back(metadata);
        commit_completed_unit(current);
        return;
    }
    if (auto* score = std::get_if<ScoreResult>(&event.result.output)) {
        if (!scoring || current.kind != OperationKind::Score ||
            score->processed_tokens != current.columns[0])
            throw std::logic_error("score result has the wrong execution window");
        current.completed.push_back(forward_metadata(
            current.bases[0].metadata, score->processed_tokens, SpeculativeBackend::None, false));
        if (score->logprobs.size() > scoring->logprobs.capacity() - scoring->logprobs.size())
            throw std::logic_error("score result exceeds its prepared target capacity");
        commit_completed_unit(current);
        scoring->cursor += score->processed_tokens;
        scoring->logprobs.insert(scoring->logprobs.end(), score->logprobs.begin(),
                                 score->logprobs.end());

        return;
    }
    throw std::logic_error("unknown native result");
}

void SessionImpl::commit_completed_unit(Execution& current) {
    current.settlement = std::make_unique<context::ContextTransaction>(directory);
    const auto recovery = std::make_unique<bool[]>(max_concurrency);
    Commit physical;
    physical.rows.reserve(current.members.size());
    for (std::size_t b = 0; b < current.members.size(); ++b) {
        const auto l        = lane(current.members[b]);
        auto& state         = sequences[l];
        const auto appended = current.kind == OperationKind::Control ? current.inputs[b].size() : 0;
        if (appended) prepare_identity(state, appended, current.splits[b]);
        recovery[b] = prepare_settlement(l, current.completed[b], requests[l].lifecycle,
                                         state.ledger.size() + appended, *current.settlement);
        physical.rows.push_back({current.columns[b], false, false});
    }
    current.settlement->prepare_actions();
    physical.context = current.settlement->actions();
    (void)executor().commit(current.ticket, std::move(physical));
    current.settlement->mark_submitted();
    current.settlement->adopt();
    current.committed = true;
    for (std::size_t b = 0; b < current.members.size(); ++b) {
        const auto l  = lane(current.members[b]);
        auto& state   = sequences[l];
        auto& request = requests[l];
        adopt_unit(l, current.completed[b],
                   current.columns[b] != 0 || current.kind == OperationKind::Normalize,
                   recovery[b]);
        const auto end = current.completed[b].execution_frontier;
        request.capture_pending =
            request.capture_reservation && request.capture_reservation->frontier == end;
        if (current.kind == OperationKind::Prefill) request.prefill->cursor = end;
        if (current.kind == OperationKind::Replay) request.replay_cursor = end;
        if (current.kind == OperationKind::Control)
            append_identity(state, current.inputs[b], current.splits[b]);
    }
}

CommitResult SessionImpl::commit(PendingBatch&& pending,
                                 std::span<const runtime::CommitDecision> decisions) {
    auto* owner = find_execution(pending.ticket_);
    if (!owner || !matching_execution(pending) || owner->committed || !owner->generated ||
        decisions.size() != pending.row_count())
        throw std::logic_error("generated commit has no matching transaction");
    auto& current     = *owner;
    const auto first  = current.decode_begin;
    for (std::size_t b = 0; b < decisions.size(); ++b) {
        const auto& d = decisions[b];
        if ((d.cancelled && d.failed) || (d.failed && !pending.failed_[b]) ||
            ((d.cancelled || d.failed)
                 ? (!d.terminal || d.accepted_tokens != 0)
                 : (!d.accepted_tokens ||
                    d.accepted_tokens > static_cast<std::uint32_t>(pending.counts_[b]) ||
                    (!d.terminal &&
                     d.accepted_tokens != static_cast<std::uint32_t>(pending.counts_[b])) ||
                    pending.failed_[b])) ||
            (d.prefix_execution_split_after &&
             (!*d.prefix_execution_split_after ||
              *d.prefix_execution_split_after > d.accepted_tokens)))
            throw std::logic_error("invalid accepted-prefix decision");
    }
    CommitResult result;
    result.row_count = decisions.size();
    result.rows.resize(decisions.size());
    result.capture_ready.resize(decisions.size());
    // Lane-indexed recovery marks; one bool per lane of the resolved concurrency.
    const auto recovery        = std::make_unique<bool[]>(max_concurrency);
    const auto segment_recovery = std::make_unique<bool[]>(max_concurrency);
    const auto recovery_span        = std::span<bool>(recovery.get(), max_concurrency);
    const auto segment_recovery_span = std::span<bool>(segment_recovery.get(), max_concurrency);
    const bool resting_members       = current.kind == OperationKind::Prefill &&
                                 current.pending_members.size() < current.members.size();
    try {
        current.decisions.assign(decisions.begin(), decisions.end());
        current.settlement = std::make_unique<context::ContextTransaction>(directory);
        if (current.kind == OperationKind::Decode || current.kind == OperationKind::Mixed) {
            current.completed.reserve(current.members.size());
            for (std::size_t b = 0; b < decisions.size(); ++b) {
                const auto m = current.pending_members[b];
                // A mixed execution's final segments settled their metadata when the result
                // arrived; its decode rows and every decode row settle here.
                if (current.kind == OperationKind::Mixed && m < first) continue;
                auto metadata = forward_metadata(current.bases[m].metadata,
                                                 decisions[b].accepted_tokens, speculative_backend,
                                                 has_backend_pool());
                if (is_masked_draft_backend(speculative_backend)) {
                    metadata.dflash_context_frontier =
                        decisions[b].terminal ? metadata.execution_frontier
                                              : current.bases[m].metadata.execution_frontier;
                    metadata.backend_kv_valid =
                        has_backend_pool() ? metadata.dflash_context_frontier : 0;
                }
                current.completed.push_back(metadata);
            }
        }
        Commit physical;
        physical.rows.reserve(decisions.size());
        if (current.kind == OperationKind::Mixed) prepare_segments(current, segment_recovery_span);
        // The members of a batch-carrying packed prefill that produced no decision row settled
        // no window yet: their rows ride this commit exactly as a mixed execution's segments
        // do, or their lane would keep its pre-flight frontier and re-process the window.
        if (resting_members) {
            for (std::size_t b = 0; b < current.members.size(); ++b) {
                if (std::find(current.pending_members.begin(), current.pending_members.end(),
                              static_cast<std::uint32_t>(b)) != current.pending_members.end())
                    continue;
                const auto l = lane(current.members[b]);
                segment_recovery[b] =
                    prepare_settlement(l, current.completed[b], requests[l].lifecycle,
                                       sequences[l].ledger.size(), *current.settlement);
            }
        }
        for (std::size_t b = 0; b < decisions.size(); ++b) {
            const auto& d = decisions[b];
            const auto m  = current.pending_members[b];
            const auto l  = lane(current.members[m]);
            auto& state   = sequences[l];
            result.rows[b].speculative.attempted_per_position.reserve(
                requests[l].speculative_stats.attempted_per_position.size());
            result.rows[b].speculative.accepted_per_position.reserve(
                requests[l].speculative_stats.accepted_per_position.size());
            physical.rows.push_back({d.accepted_tokens, d.terminal, d.cancelled || d.failed});
            if (d.cancelled || d.failed) continue;
            prepare_identity(state, d.accepted_tokens, d.prefix_execution_split_after);
            recovery[b] = prepare_settlement(
                l, current.completed[m],
                d.terminal ? Lifecycle::Finishable : Lifecycle::Active,
                state.ledger.size() + d.accepted_tokens, *current.settlement);
        }
        current.settlement->prepare_actions();
        physical.context = current.settlement->actions();
        (void)executor().commit(current.ticket, std::move(physical));
        current.settlement->mark_submitted();
        current.settlement->adopt();
        current.committed = true;
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
    if (current.kind == OperationKind::Mixed) adopt_segments(current, segment_recovery_span);
    if (resting_members) {
        for (std::size_t b = 0; b < current.members.size(); ++b) {
            if (std::find(current.pending_members.begin(), current.pending_members.end(),
                          static_cast<std::uint32_t>(b)) != current.pending_members.end())
                continue;
            const auto l  = lane(current.members[b]);
            auto& request = requests[l];
            const auto end = current.completed[b].execution_frontier;
            adopt_unit(l, current.completed[b], current.columns[b] != 0, segment_recovery[b]);
            request.capture_pending =
                request.capture_reservation && request.capture_reservation->frontier == end;
            request.prefill->cursor = end;
        }
    }
    for (std::size_t b = 0; b < decisions.size(); ++b) {
        const auto& d = decisions[b];
        const auto m  = current.pending_members[b];
        const auto l  = lane(current.members[m]);
        auto& state   = sequences[l];
        auto& request = requests[l];
        if (!d.cancelled && !d.failed) {
            adopt_unit(l, current.completed[m],
                       current.kind == OperationKind::Decode ||
                           current.kind == OperationKind::Mixed || current.columns[m] != 0,
                       recovery[b]);
            if (current.kind == OperationKind::Prefill)
                request.prefill->cursor = current.completed[m].execution_frontier;
            const auto accepted = std::span<const TokenId>(pending.tokens_)
                                      .subspan(b * pending.stride_, d.accepted_tokens);
            append_identity(state, accepted, d.prefix_execution_split_after);
            request.lifecycle = d.terminal ? Lifecycle::Finishable : Lifecycle::Active;
            request.capture_pending =
                request.capture_reservation &&
                request.capture_reservation->frontier ==
                    current.completed[m].execution_frontier;
        }
        if (d.terminal) {
            request.capture_pending = false;
            request.capture_reservation.reset();
            request.prefill.reset();
        } else if (current.kind == OperationKind::Prefill && !request.capture_pending) {
            request.prefill.reset();
        }
        auto& stats = request.speculative_stats;
        if ((current.kind == OperationKind::Decode || current.kind == OperationKind::Mixed) &&
            speculative_backend != SpeculativeBackend::None) {
            const auto drafted  = current.generated->draft_extents.at(b);
            const auto accepted = current.generated->accepted_drafts.at(b);
            stats.committed_draft_tokens += std::min(d.accepted_tokens, accepted);
            if (drafted)
                ++stats.rounds;
            else
                ++stats.fallback_steps;
            stats.drafted_tokens += drafted;
            for (std::uint32_t i = 0; i < drafted; ++i) ++stats.attempted_per_position[i];
            stats.accepted_tokens += accepted;
            for (std::uint32_t j = 0; j < accepted; ++j) ++stats.accepted_per_position[j];
        }
        auto& row               = result.rows[b];
        row.disposition         = d.cancelled  ? runtime::CommitDisposition::CancelledReleased
                                  : d.failed   ? runtime::CommitDisposition::FailedReleased
                                  : d.terminal ? runtime::CommitDisposition::Finishable
                                               : runtime::CommitDisposition::Active;
        row.speculative         = stats;
        result.capture_ready[b] = !d.terminal && request.capture_pending;
    }
    pending.ticket_ = {};
    return result;
}

DiscardResult SessionImpl::discard(PendingBatch&& pending) noexcept {
    auto* owner = find_execution(pending.ticket_);
    if (!owner || !matching_execution(pending) || owner->committed) return {};
    try {
        auto& current    = *owner;
        const auto first = current.decode_begin;
        if (current.kind == OperationKind::Mixed) {
            // The decision rows are dropped but their segments completed: commit every row
            // as discarded so the segments' State and KV settle and keep their progress.
            const auto recovery = std::make_unique<bool[]>(max_concurrency);
            const auto recovery_span = std::span<bool>(recovery.get(), max_concurrency);
            current.settlement = std::make_unique<context::ContextTransaction>(directory);
            prepare_segments(current, recovery_span);
            current.settlement->prepare_actions();
            Commit physical;
            physical.rows.assign(current.pending_members.size(), CommitRow{0, true, true});
            physical.context = current.settlement->actions();
            (void)executor().commit(current.ticket, std::move(physical));
            current.settlement->mark_submitted();
            current.settlement->adopt();
            adopt_segments(current, recovery_span);
        } else {
            executor().discard(current.ticket);
        }
        current.discarded = true;
        current.committed = true;
        pending.ticket_   = {};
        return {runtime::ConsumeStatus::Consumed, current.pending_members.size()};
    } catch (...) { return {}; }
}

// A mixed execution settles its prefill segments with its decode rows, because both share one
// device commit. The segments advance exactly as a completed prefill would.
void SessionImpl::prepare_segments(Execution& current, std::span<bool> recovery) {
    for (std::size_t b = 0; b < current.decode_begin; ++b) {
        const auto l = lane(current.members[b]);
        recovery[b]  = prepare_settlement(l, current.completed[b], requests[l].lifecycle,
                                          sequences[l].ledger.size(), *current.settlement);
    }
}

void SessionImpl::adopt_segments(Execution& current,
                                 std::span<const bool> recovery) noexcept {
    for (std::size_t b = 0; b < current.decode_begin; ++b) {
        const auto l   = lane(current.members[b]);
        auto& request  = requests[l];
        const auto end = current.completed[b].execution_frontier;
        adopt_unit(l, current.completed[b], current.columns[b] != 0, recovery[b]);
        request.capture_pending =
            request.capture_reservation && request.capture_reservation->frontier == end;
        request.prefill->cursor = end;
    }
}

void SessionImpl::retire(runtime::DeviceRetired event) {
    if (release_ticket && *release_ticket == event.ticket) {
        if (!release_result) throw std::logic_error("residency retired before release result");
        release_ticket.reset();
        release_result = false;
        if (after_releases && releases.empty()) {
            events.push_back(std::move(*after_releases));
            after_releases.reset();
        }
        return;
    }
    if (event.ticket.frame.kind == runtime::FrameKind::Context) {
        retire_context(event);
        return;
    }
    auto* owner = find_execution(event.ticket);
    if (!owner || !owner->committed)
        throw std::logic_error("execution retired before its resolution");
    if (owner->vision_id) {
        const auto* window = vision_window.current();
        bool complete      = owner->discarded;
        if (window && valid_sequence(window->owner.sequence)) {
            const auto& state = sequences[lane(window->owner.sequence)];
            complete |=
                state.continuation.metadata.execution_frontier >= window->plan->consume_end &&
                (speculative_backend != SpeculativeBackend::Mtp ||
                 backend_kv_valid(state) >= window->plan->consume_end - 1);
        } else
            complete = true;
        vision_window.consumed(owner->vision_id, complete);
        vision_window.retired(event.ticket);
    }
    auto current = std::move(*owner);
    execution_slot(event.ticket.frame).reset();
    current.preparation->mark_completed();
    current.preparation->publish();
    if (current.settlement) {
        current.settlement->mark_completed();
        current.settlement->publish();
    }
    // Deferred lane installations completed with this flight's stream order; their borrowed
    // sources publish here. A lane cleared before this retirement already discarded its
    // deferred work.
    for (const auto handle : current.members) {
        if (!valid_sequence(handle)) continue;
        auto& deferred = deferred_bindings[lane(handle)];
        if (deferred) publish_deferred_binding(*deferred);
        deferred.reset();
    }
    for (std::size_t b = 0; b < current.members.size(); ++b) {
        if (!valid_sequence(current.members[b])) continue;
        const auto l  = lane(current.members[b]);
        auto& request = requests[l];
        if (current.kind == OperationKind::Replay &&
            request.replay_cursor == request.replay_target && !request.capture_pending) {
            request.lifecycle = request.resume_lifecycle;
            if (request.lifecycle == Lifecycle::Prefilling)
                initialize_captures(request, request.replay_target,
                                    request.base->summary.prompt_tokens);
        }
    }
    if (current.kind != OperationKind::Normalize && current.kind != OperationKind::Score)
        events.push_back({current.ticket, current.members,
                          runtime::ExecutionComplete{event.execution, current.members.size()}});
    for (std::size_t row = 0; row < current.pending_members.size(); ++row) {
        const auto b = current.pending_members[row];
        if (!valid_sequence(current.members[b])) continue;
        const auto l  = lane(current.members[b]);
        auto& state   = sequences[l];
        auto& request = requests[l];
        if (current.discarded ||
            (!current.decisions.empty() &&
             (current.decisions[row].cancelled || current.decisions[row].failed)))
            clear_lane(state, request);
    }
    if (current.completion) events.push_back(std::move(*current.completion));
    if (current.kind == OperationKind::Normalize && normalizing_pause_)
        continue_pause_after_normalize(*normalizing_pause_);
    if (current.kind == OperationKind::Score) advance_score();
}

} // namespace ninfer::models::qwen3_5::detail
