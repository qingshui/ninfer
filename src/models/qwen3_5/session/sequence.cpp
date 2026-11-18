#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

void SessionImpl::register_lane(std::uint32_t lane, SequenceHandle handle) noexcept {
    // A lane names one identity at a time: the entry of its previous occupant leaves with it, and
    // an identity that moves lanes (a resume) leaves its previous lane.
    if (const auto previous = lane_identity_[lane]; previous) {
        const auto it = lane_index_.find(previous);
        if (it != lane_index_.end() && it->second == lane) lane_index_.erase(it);
    }
    if (const auto it = lane_index_.find(handle.value);
        it != lane_index_.end() && it->second != lane)
        lane_identity_[it->second] = 0;
    lane_identity_[lane]      = handle.value;
    lane_index_[handle.value] = lane;
}

std::uint32_t SessionImpl::lane(SequenceHandle handle) const {
    if (handle.value && handle.generation) {
        const auto it = lane_index_.find(handle.value);
        if (it != lane_index_.end()) {
            const auto index = it->second;
            if (requests[index].lifecycle != Lifecycle::Empty && sequences[index].id == handle)
                return index;
        }
    }
    throw std::logic_error("sequence identity is not resident");
}

bool SessionImpl::valid_sequence(SequenceHandle handle) const noexcept {
    if (!handle.value || !handle.generation) return false;
    const auto it = lane_index_.find(handle.value);
    if (it == lane_index_.end()) return false;
    const auto index = it->second;
    return requests[index].lifecycle != Lifecycle::Empty && sequences[index].id == handle;
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
    if (!valid_sequence(handle)) return false;
    return lane_busy_[lane_index_.find(handle.value)->second] != 0;
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
    for (const auto split : window_splits(state, begin, end))
        out.execution_splits.push_back(split - begin);
    return out;
}

std::uint32_t SessionImpl::window_end(const SequenceState& state, const RequestControl& request,
                                      std::uint32_t begin, std::uint32_t end,
                                      const vision::Window& window) const {
    end = std::min(end, begin + prefill_chunk);
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
    // were never submitted to a stream, so nothing on the device references the lane. Its
    // residency was never installed either, so there is nothing to release.
    const bool uninstalled = deferred_bindings[lane] && deferred_bindings[lane]->install;
    discard_deferred_binding(lane);
    if (!closing && state.residency.generation && !uninstalled) release_residency(state.residency);
    drop_lane_checkpoints(request);
    request.prefill.reset();
    release_sequence_kv(state);
    release_sequence_state(state);
    if (const auto identity = lane_identity_[lane]; identity) {
        const auto it = lane_index_.find(identity);
        if (it != lane_index_.end() && it->second == lane) lane_index_.erase(it);
        lane_identity_[lane] = 0;
    }
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
    lane_releasing_[residency.lane.value] = 1;
}

// Installs, for every group of `groups` that the Context can hold, the sampling branches of one
// prompt: the parent is branch 0 and keeps its lane, and each child takes one of the requested
// lanes. Every branch decodes the last prompt token as an ordinary active sequence. The branches
// share the parent's whole-page KV and read the parent's State image at the fork frontier as the
// source of their first write; each branch writes a State image of its own, so no branch copies
// State and none waits for another. A child copies the parent's tail page and installs its
// residency through the stream order of its first execution, so no context frame is reserved.
std::vector<ForkGroupResult> SessionImpl::fork_batch(std::span<const ForkGroupRequest> groups) {
    // Validation covers every group before any group changes.
    std::vector<std::uint32_t> parent_lanes;
    parent_lanes.reserve(groups.size());
    std::vector<std::uint8_t> claimed(max_concurrency, 0);
    for (const auto& group : groups) {
        if (!valid_sequence(group.parent))
            throw std::logic_error("fork requires a resident parent sequence");
        if (!group.sampling.empty() && group.sampling.size() != group.children.size())
            throw std::invalid_argument("fork sampling must be empty or parallel to its children");
        const auto parent_lane = lane(group.parent);
        const auto& state      = sequences[parent_lane];
        const auto& request    = requests[parent_lane];
        const auto frontier    = request.base ? request.base->fork_frontier : 0U;
        if (!frontier || request.lifecycle != Lifecycle::Prefilling || !request.prefill ||
            request.prefill->cursor != frontier ||
            state.continuation.metadata.execution_frontier != frontier || !state.kv ||
            state.pending_write.valid() || execution_pending(group.parent) ||
            deferred_bindings[parent_lane] || group.children.size() > sample_fork_limit() ||
            claimed[parent_lane])
            throw std::logic_error("fork requires a rested forking parent");
        claimed[parent_lane] = 1;
        parent_lanes.push_back(parent_lane);
        for (const auto child : group.children) {
            if (child.value >= max_concurrency || claimed[child.value] ||
                requests[child.value].lifecycle != Lifecycle::Empty ||
                deferred_bindings[child.value])
                throw std::logic_error("fork requires distinct free child lanes");
            claimed[child.value] = 1;
        }
    }
    std::vector<ForkGroupResult> results(groups.size());
    for (std::size_t g = 0; g < groups.size(); ++g)
        results[g] = fork_group(groups[g], parent_lanes[g]);
    return results;
}

ForkGroupResult SessionImpl::fork_group(const ForkGroupRequest& group, std::uint32_t parent_lane) {
    auto& state         = sequences[parent_lane];
    auto& request       = requests[parent_lane];
    const auto& base    = *request.base;
    const auto frontier = base.fork_frontier;
    const auto& children = group.children;
    ForkGroupResult result;
    const auto activate = [&] {
        request.lifecycle = Lifecycle::Active;
        request.prefill.reset();
        request.reserved_checkpoints.clear();
        request.checkpoint_plan.clear();
    };
    if (children.empty()) {
        activate();
        result.reserved = true;
        return result;
    }
    // A child cannot install its residency while a release of its lane is still on its way.
    for (const auto child : children)
        if (lane_releasing_[child.value])
            return {.assessment = {.status = context::ContextAvailability::Waiting}};

    // A branch decodes one token past the frontier on its own tail; later growth reserves its
    // pages when the engine admits the decode unit.
    const auto main_peak        = frontier + 1;
    const auto backend_frontier = backend_kv_valid(state);
    const auto backend_peak     = speculative_backend == SpeculativeBackend::Mtp
                                      ? (main_peak ? main_peak - 1 : 0)
                                  : has_backend_pool() ? main_peak
                                                       : 0;
    const auto text_source = text_kv_histories->retain_prefix(state.kv->text.handle(), frontier);
    std::optional<context::HistoryPrefix> backend_source;
    if (state.kv->backend)
        backend_source =
            backend_kv_histories->retain_prefix(state.kv->backend->handle(), backend_frontier);
    const auto source        = state.continuation.image;
    const auto state_variant = state_store->variant(source);
    ContextDemand demand;
    // One State image for the parent's own writes, then one image and one KV binding per child.
    const auto parent_state = demand.add_state({.count = 1, .variant = state_variant,
                                                .logical_count = 1});
    std::vector<context::ContextDemandId> state_ids, main_ids, backend_ids;
    for (std::size_t i = 0; i < children.size(); ++i) {
        state_ids.push_back(
            demand.add_state({.count = 1, .variant = state_variant, .logical_count = 1}));
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
    if (!reserved.reservation->ready())
        throw std::logic_error("fork placement changed after its accepted assessment");
    auto& destinations = *reserved.reservation;

    std::vector<SequenceHandle> installed;
    std::vector<std::uint32_t> installed_lanes;
    context::StateId parent_write;
    std::shared_ptr<StateAnchor> anchor;
    try {
        // The parent's image becomes the shared immutable source of every branch's first write.
        state_store->freeze(source);
        anchor = std::make_shared<StateAnchor>(*state_store, source);
        parent_write = *state_store->reserve_destination(destinations.take(parent_state),
                                                         state_variant);
        (void)state_store->begin_write(source, parent_write);
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
            const auto image = state_store->reserve_destination(destinations.take(state_ids[i]),
                                                                state_variant);
            if (!image) throw std::logic_error("fork lost its reserved State record");
            try {
                (void)state_store->begin_write(source, *image);
            } catch (...) {
                (void)state_store->release(*image);
                throw;
            }
            migration.prepare_actions();
            auto& control = requests[lane];
            auto& child   = sequences[lane];
            control       = RequestControl{};
            control.base  = request.base;
            control.lifecycle            = Lifecycle::Active;
            // Each child decodes with its own sampling when the caller names one; an empty span
            // repeats the parent's, which makes every child sample identically.
            control.sampling_host        = group.sampling.empty() ? base.sampling : group.sampling[i];
            control.publish_continuation = request.publish_continuation;
            control.speculative_stats    = request.speculative_stats;
            child                        = SequenceState{};
            child.lane                   = lane;
            child.id = SequenceHandle{next_sequence++, executor().epoch().value};
            register_lane(lane, child.id);
            if (++lane_epochs[lane] == 0) ++lane_epochs[lane];
            child.residency          = {{lane}, lane_epochs[lane]};
            child.continuation       = state.continuation;
            child.continuation.image = source;
            child.pending_write      = *image;
            child.ledger             = base.prompt->token_ids;
            child.prefix_identity.assign(*base.prompt);
            child.prefix_digests = base.prefix_digests;
            installed.push_back(child.id);
            installed_lanes.push_back(lane);
            auto& text = *migration.binding(kTextKvPool.index);
            text.publish_restores();
            SequenceKVBundle kv{.text = text.commit()};
            if (auto* backend = migration.binding(kBackendKvPool.index)) {
                backend->publish_restores();
                kv.backend.emplace(backend->commit());
            }
            child.kv.emplace(std::move(kv));
            auto& deferred = deferred_bindings[lane].emplace(std::move(migration));
            deferred.actions = deferred.transaction.actions();
            deferred.retained.push_back(text_source);
            if (backend_source) deferred.retained.push_back(*backend_source);
            deferred.install = ResidencyBinding{{child.id, child.residency},
                                                group.sampling.empty() ? base.sampling
                                                                       : group.sampling[i]};
            deferred.anchor  = anchor;
        }
        // The parent decodes beside its children: its first flight holds the same source.
        context::ContextTransaction resting(directory);
        resting.prepare_actions();
        auto& deferred   = deferred_bindings[parent_lane].emplace(std::move(resting));
        deferred.actions = deferred.transaction.actions();
        deferred.anchor  = anchor;
        state.pending_write = parent_write;
        parent_write        = {};
        activate();
    } catch (...) {
        // No branch has a flight, so discarding its deferred work rolls the whole group back.
        for (const auto lane : installed_lanes) {
            if (deferred_bindings[lane]) discard_deferred_binding(lane);
            clear_lane(sequences[lane], requests[lane]);
        }
        if (deferred_bindings[parent_lane]) discard_deferred_binding(parent_lane);
        if (parent_write.valid()) {
            if (state_store->can_abort_write(source, parent_write))
                state_store->abort_write(source, parent_write);
            (void)state_store->release(parent_write);
        }
        anchor.reset();
        if (state_store->valid(source) &&
            state_store->role(source) == context::StateRole::CheckpointImmutable &&
            !state_store->checkpoint_references(source) && !state_store->source_pins(source))
            state_store->make_mutable(source);
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
    releasing_ticket_lanes_.clear();
    for (const auto& residency : releases) releasing_ticket_lanes_.push_back(residency.lane.value);
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
