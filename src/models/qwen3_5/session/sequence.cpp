#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

std::uint32_t SessionImpl::lane(SequenceHandle handle) const {
    for (std::uint32_t i = 0; i < max_concurrency; ++i)
        if (requests[i].lifecycle != Lifecycle::Empty && sequences[i].id == handle) return i;
    throw std::logic_error("sequence identity is not resident");
}

bool SessionImpl::valid_sequence(SequenceHandle handle) const noexcept {
    if (!handle.value || !handle.generation) return false;
    for (std::uint32_t i = 0; i < max_concurrency; ++i)
        if (requests[i].lifecycle != Lifecycle::Empty && sequences[i].id == handle) return true;
    return false;
}

SequenceHandle SessionImpl::sequence_handle(std::uint32_t lane) const noexcept {
    return sequences[lane].id;
}

SequenceState& SessionImpl::active_sequence(std::uint32_t lane) {
    if (lane >= max_concurrency || requests[lane].lifecycle == Lifecycle::Empty)
        throw std::logic_error("execution lane is empty");
    return sequences[lane];
}

const SequenceState& SessionImpl::active_sequence(std::uint32_t lane) const {
    if (lane >= max_concurrency || requests[lane].lifecycle == Lifecycle::Empty)
        throw std::logic_error("execution lane is empty");
    return sequences[lane];
}

void SessionImpl::invalidate_lane(std::uint32_t lane) noexcept {
    if (!++lane_epochs[lane]) ++lane_epochs[lane];
}

bool SessionImpl::execution_pending(SequenceHandle handle) const noexcept {
    return std::any_of(executions.begin(), executions.end(), [&](const auto& slot) {
        return slot && std::find(slot->members.begin(), slot->members.end(), handle) !=
                           slot->members.end();
    });
}

bool SessionImpl::recovery_pending(SequenceHandle handle) const noexcept {
    if (!valid_sequence(handle)) return false;
    return requests[lane(handle)].recovery.has_value();
}

bool SessionImpl::has_backend_pool() const noexcept { return backend_kv_histories != nullptr; }

std::uint32_t SessionImpl::backend_kv_valid(const SequenceState& state) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp)
        return state.continuation.metadata.backend_kv_valid;
    return speculative_backend == SpeculativeBackend::DFlash
               ? state.continuation.metadata.dflash_context_frontier
               : 0;
}

ExecutionRow SessionImpl::execution_row(const SequenceState& state) const {
    ExecutionRow row;
    row.member            = {state.id, state.residency};
    row.source_state      = state.continuation.image;
    row.destination_state = state.write_state();
    row.state_offsets     = state_store->selectors(row.source_state, row.destination_state);
    row.main_table        = text_kv_histories->execution_row(state.kv->text.handle()).handle();
    if (state.kv->backend)
        row.backend_table =
            backend_kv_histories->execution_row(state.kv->backend->handle()).handle();
    row.main_frontier    = state.continuation.metadata.execution_frontier;
    row.backend_frontier = is_masked_draft_backend(speculative_backend)
                               ? state.continuation.metadata.dflash_context_frontier
                               : backend_kv_valid(state);
    row.rope_delta       = state.continuation.metadata.rope_delta;
    if (row.main_frontier) {
        const auto p       = row.main_frontier - 1;
        const auto& prompt = *requests[state.lane].base->prompt;
        for (int axis = 0; axis < 3; ++axis)
            row.tail_rope_position[axis] = p < prompt.token_ids.size()
                                               ? prompt.position_axis(axis)[p]
                                               : static_cast<std::int32_t>(p) + row.rope_delta;
    }
    return row;
}

InputWindow SessionImpl::input_window(const SequenceState& state, std::uint32_t begin,
                                      std::uint32_t end) const {
    if (begin > end || end > state.ledger.size())
        throw std::logic_error("input window exceeds its committed ledger");
    InputWindow out;
    out.tokens.assign(state.ledger.begin() + begin, state.ledger.begin() + end);
    out.rope_positions.resize(3ULL * (end - begin));
    const auto& request = requests[state.lane];
    const auto& prompt  = *request.base->prompt;
    for (int axis = 0; axis < 3; ++axis)
        for (auto p = begin; p < end; ++p)
            out.rope_positions[axis * (end - begin) + p - begin] =
                p < prompt.token_ids.size()
                    ? prompt.position_axis(axis)[p]
                    : static_cast<std::int32_t>(p) + state.continuation.metadata.rope_delta;
    out.prompt_begin = begin;
    if (!causal_scoring) {
        const auto frontiers = state.prefix_identity.execution_frontiers();
        for (auto it = std::upper_bound(frontiers.begin(), frontiers.end(), begin);
             it != frontiers.end() && *it < end; ++it)
            out.execution_splits.push_back(*it - begin);
        std::sort(out.execution_splits.begin(), out.execution_splits.end());
        out.execution_splits.erase(
            std::unique(out.execution_splits.begin(), out.execution_splits.end()),
            out.execution_splits.end());
    }
    return out;
}

std::uint32_t SessionImpl::window_end(const SequenceState& state, const RequestControl& request,
                                      std::uint32_t begin, std::uint32_t end,
                                      const vision::Window& window) const {
    end = std::min(end, begin + prefill_chunk);
    if (request.capture_reservation) end = std::min(end, request.capture_reservation->frontier);
    if (begin < request.base->summary.prompt_tokens)
        end = std::min(end, request.base->summary.prompt_tokens);
    if (request.lifecycle == Lifecycle::Replaying) {
        const auto frontiers = state.prefix_identity.execution_frontiers();
        const auto next      = std::upper_bound(frontiers.begin(), frontiers.end(), begin);
        if (next != frontiers.end()) end = std::min(end, *next);
    }
    if (request.base->vision && begin < request.base->summary.prompt_tokens && begin < end)
        end = window.limit({state.id, state.residency}, request.base->vision->units, begin, end,
                           plan.vision_max_features);
    return end;
}

void SessionImpl::ensure_sequence_kv_mapped(SequenceState& state, std::uint32_t main,
                                            std::uint32_t backend,
                                            context::ContextTransaction& transaction) {
    if (state.pending_write.valid()) transaction.retain(*state_store, state.continuation.image);
    const auto main_begin = text_kv_histories->committed_frontier(state.kv->text.handle());
    if (main > main_begin)
        text_kv_histories->prepare_write(state.kv->text, main_begin, main, transaction);
    if (state.kv->backend) {
        const auto start = backend_kv_histories->committed_frontier(state.kv->backend->handle());
        if (backend > start)
            backend_kv_histories->prepare_write(*state.kv->backend, start, backend, transaction);
    }
}

void SessionImpl::complete_state_write(SequenceState& state) {
    const auto destination = state.write_state();
    state_store->complete_write(state.continuation.image, destination);
    state.continuation.image = destination;
    state.pending_write      = {};
}

void SessionImpl::release_sequence_state(SequenceState& state) noexcept {
    try {
        if (state.pending_write.valid())
            state_store->abort_write(state.continuation.image, state.write_state());
        if (state_store->valid(state.write_state()) && !state_store->release(state.write_state()))
            std::terminate();
        state.continuation.image = {};
        state.pending_write      = {};
    } catch (...) { std::terminate(); }
}

void SessionImpl::release_sequence_kv(SequenceState& state) noexcept { state.kv.reset(); }

void SessionImpl::clear_lane(SequenceState& state, RequestControl& request) noexcept {
    vision_window.revoke({state.id, state.residency});
    const auto lane = state.lane;
    // A lane cleared before its first flight discards its deferred installation; the actions
    // were never submitted to a stream, so nothing on the device references the lane.
    discard_deferred_binding(lane);
    if (!closing && state.residency.generation) release_residency(state.residency);
    request.prefill.reset();
    release_sequence_kv(state);
    release_sequence_state(state);
    state      = {};
    state.lane = lane;
    request    = {};
    invalidate_lane(lane);
}

void SessionImpl::release_residency(runtime::ResidencyId residency) {
    const auto owner = vision_window.owner();
    if (owner && owner->residency == residency) vision_window.revoke(*owner);
    if (std::find(releases.begin(), releases.end(), residency) == releases.end())
        releases.push_back(residency);
}

// Installs one child sequence in each of `children`'s lanes from the rested `parent`: every child
// shares the parent's whole-page KV through the fork frontier, copies the tail page, the GDN/conv
// State image and the continuation metadata, and carries the full prompt ledger, so it decodes the
// last prompt token like an ordinary active sequence. The copies ride each child's first
// execution in its stream order; the deferred transaction keeps the parent's images and pages
// alive until that execution retires.
ForkResult SessionImpl::fork(SequenceHandle parent, std::span<const runtime::LaneId> children) {
    ForkResult result;
    if (!valid_sequence(parent))
        throw std::logic_error("fork requires a resident parent sequence");
    const auto parent_lane = lane(parent);
    auto& state   = sequences[parent_lane];
    auto& request = requests[parent_lane];
    const auto& base = *request.base;
    const auto frontier = base.fork_frontier;
    if (!frontier || request.lifecycle != Lifecycle::Prefilling || !request.prefill ||
        request.prefill->cursor != frontier ||
        state.continuation.metadata.execution_frontier != frontier || !state.kv ||
        request.capture_pending || execution_pending(parent) || children.empty() ||
        children.size() > sample_fork_limit())
        throw std::logic_error("fork requires a rested forking parent and its children");
    for (std::size_t i = 0; i < children.size(); ++i) {
        const auto lane = children[i].value;
        if (std::find(children.begin(), children.begin() + i, children[i]) != children.begin() + i ||
            lane >= max_concurrency || lane == parent_lane ||
            requests[lane].lifecycle != Lifecycle::Empty || deferred_bindings[lane])
            throw std::logic_error("fork requires distinct free child lanes");
    }
    if (has_context_operation())
        return {.assessment = {.status = context::ContextAvailability::Waiting}};
    auto frame = executor().try_reserve(runtime::FrameKind::Context);
    if (!frame) return {.assessment = {.status = context::ContextAvailability::Waiting}};

    // One child decodes one token past the frontier on its own tail; later growth reserves its
    // pages when the engine admits the decode unit.
    const auto main_peak = frontier + 1;
    const auto backend_frontier = backend_kv_valid(state);
    const auto backend_peak =
        speculative_backend == SpeculativeBackend::Mtp
            ? (main_peak ? main_peak - 1 : 0)
            : has_backend_pool() ? main_peak : 0;
    auto text_source = text_kv_histories->retain_prefix(state.kv->text.handle(), frontier);
    std::optional<context::HistoryPrefix> backend_source;
    if (state.kv->backend)
        backend_source = backend_kv_histories->retain_prefix(state.kv->backend->handle(),
                                                             backend_frontier);
    ContextDemand demand;
    std::vector<context::ContextDemandId> state_ids, main_ids, backend_ids;
    const auto state_variant = state_store->variant(state.continuation.image);
    for (std::size_t i = 0; i < children.size(); ++i) {
        state_ids.push_back(demand.add_state(
            {.count = 1, .variant = state_variant, .logical_count = 1}));
        main_ids.push_back(demand.add_history(
            kTextKvPool.index,
            text_kv_histories->binding_demand(&text_source, frontier, main_peak, {})));
        if (state.kv->backend)
            backend_ids.push_back(demand.add_history(
                kBackendKvPool.index,
                backend_kv_histories->binding_demand(&*backend_source, backend_frontier,
                                                     backend_peak, {})));
    }
    const auto assessment = directory.assess(demand);
    if (!assessment.ready()) return {.assessment = assessment};
    auto reserved = directory.reserve(demand);
    if (!reserved) return {.assessment = reserved.assessment};
    auto& destinations = *reserved.reservation;

    std::vector<SequenceHandle> installed;
    try {
        const auto image_bytes = state_store->image_bytes(0, state_variant);
        for (std::size_t i = 0; i < children.size(); ++i) {
            const auto lane = children[i].value;
            context::ContextTransaction migration(directory);
            migration.add_binding(
                kTextKvPool.index,
                text_kv_histories->prepare_binding(&text_source, frontier, main_peak, lane,
                                                   destinations.take(main_ids[i]), false));
            if (state.kv->backend)
                migration.add_binding(
                    kBackendKvPool.index,
                    backend_kv_histories->prepare_binding(&*backend_source, backend_frontier,
                                                          backend_peak, lane,
                                                          destinations.take(backend_ids[i]),
                                                          false));
            const auto image =
                state_store->reserve_destination(destinations.take(state_ids[i]));
            migration.own_state(*image);
            migration.add_state_group(state.continuation.image, 0, image_bytes);
            migration.prepare_actions();
            auto& control = requests[lane];
            auto& child   = sequences[lane];
            control       = RequestControl{};
            control.base  = request.base;
            control.lifecycle           = Lifecycle::Active;
            control.sampling_host       = base.sampling;
            control.publish_continuation = request.publish_continuation;
            control.speculative_stats   = request.speculative_stats;
            initialize_captures(control, frontier, base.summary.prompt_tokens);
            child           = SequenceState{};
            child.lane      = lane;
            child.id        = SequenceHandle{next_sequence++, executor().epoch().value};
            if (++lane_epochs[lane] == 0) ++lane_epochs[lane];
            child.residency    = {{lane}, lane_epochs[lane]};
            child.continuation = state.continuation;
            child.continuation.image = *image;
            child.pending_write      = {};
            child.ledger = base.prompt->token_ids;
            child.prefix_identity.assign(*base.prompt);
            child.prefix_digests = base.prefix_digests;
            auto& text = *migration.binding(kTextKvPool.index);
            text.publish_restores();
            SequenceKVBundle kv{.text = text.commit()};
            if (auto* backend = migration.binding(kBackendKvPool.index)) {
                backend->publish_restores();
                kv.backend.emplace(backend->commit());
            }
            child.kv.emplace(std::move(kv));
            (void)migration.state_destination().release();
            deferred_bindings[lane].emplace(std::move(migration));
            deferred_bindings[lane]->actions = deferred_bindings[lane]->transaction.actions();
            deferred_bindings[lane]->retained.push_back(text_source);
            if (backend_source) deferred_bindings[lane]->retained.push_back(*backend_source);
            installed.push_back(child.id);
        }
        context_operation_ = std::make_unique<PendingContext>(directory, ResidencyOperation{});
        context_operation_->silent = true;
        context_operation_->reservation.emplace(std::move(*frame));
        ContextWork work;
        for (const auto handle : installed)
            work.bindings.push_back(
                {{handle, sequences[lane(handle)].residency}, base.sampling});
        (void)submit_context(*context_operation_, std::move(work));
    } catch (...) {
        abort_context();
        // Every child lane that is no longer empty carries a partial or complete installation;
        // no child has a flight, so discarding its deferred work rolls the whole group back.
        for (const auto id : children) {
            const auto lane = id.value;
            if (requests[lane].lifecycle == Lifecycle::Empty) continue;
            if (deferred_bindings[lane]) discard_deferred_binding(lane);
            clear_lane(sequences[lane], requests[lane]);
        }
        throw;
    }
    result.children = std::move(installed);
    result.reserved = true;
    return result;
}

void SessionImpl::submit_releases() {
    if (closing || releases.empty() || release_ticket) return;
    auto reservation = executor().try_reserve(runtime::FrameKind::Context);
    if (!reservation) return;
    ContextWork context;
    context.releases = releases;
    release_result   = false;
    release_ticket =
        executor().submit(std::move(*reservation), Work{.operation = std::move(context)});
    releases.clear();
}

std::optional<qwen::VisionWork>
SessionImpl::prepare_visual(const SequenceState& state, std::uint32_t begin, std::uint32_t end) {
    const auto& request = requests[state.lane];
    const auto& source  = request.base->vision;
    if (!source || begin >= end) return {};
    const auto* active = vision_window.current();
    if (active && active->owner.sequence == state.id && active->owner.residency == state.residency)
        return qwen::prepare_vision_work(vision_window.reserve(active->owner, {}), source, 1);
    auto planned = vision::plan_window(source->units, begin, end, plan.vision_max_features, 1);
    if (!planned) return {};
    return qwen::prepare_vision_work(
        vision_window.reserve({state.id, state.residency}, std::move(planned)), source, 1);
}

} // namespace ninfer::models::qwen3_5::detail
