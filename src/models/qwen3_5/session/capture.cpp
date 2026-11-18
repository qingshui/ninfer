#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <stdexcept>

// In-flight checkpoints. A checkpoint is a State image that a flight writes beside the lane's own
// image, at a column of the lane's window that is an execution boundary of the prefix. The
// reservation of its catalog point and State image happens while the unit is reserved, the window
// hands both to the flight, and the retirement of the flight publishes the point together with a
// retained KV prefix. Nothing here waits for a context operation or a quiet Session.
namespace ninfer::models::qwen3_5::detail {

void FlightCheckpoint::release() noexcept {
    if (!owner) return;
    auto* session = std::exchange(owner, nullptr);
    if (image.valid() && session->state_store->valid(image) && !session->state_store->release(image))
        std::terminate();
    image = {};
    // The catalog point returns with the reservation.
    point = {};
}

void SessionImpl::initialize_checkpoint_plan(RequestControl& request, std::uint32_t from) {
    request.checkpoint_plan.clear();
    request.reserved_checkpoints.clear();
    request.skipped_checkpoints = 0;
    std::vector<const CheckpointCandidate*> eligible;
    for (const auto& candidate : request.base->checkpoint_candidates)
        if (candidate.frontier > from) eligible.push_back(&candidate);
    // The first candidate is the system prompt, the last one the last shared message.
    static_assert(kMaxFlightCheckpoints == 2);
    if (eligible.size() > kMaxFlightCheckpoints) eligible = {eligible.front(), eligible.back()};
    for (const auto* candidate : eligible) request.checkpoint_plan.push_back(*candidate);
}

bool SessionImpl::checkpoint_present(const PrefixShortlistKey& key) const {
    for (const auto& slot : checkpoints)
        if (slot.value && slot.value->key == key) return true;
    for (const auto& request : requests)
        for (const auto& reserved : request.reserved_checkpoints)
            if (reserved.candidate.key == key) return true;
    for (const auto& slot : executions) {
        if (!slot) continue;
        for (const auto& list : slot->checkpoints)
            for (const auto& flight : list)
                if (flight.candidate.key == key) return true;
    }
    return false;
}

// Reserves the checkpoints of the lane's plan that lie inside the window that ends at `window_end`.
// The reservation never reclaims: a Context that cannot hold a candidate skips it, and the skip is
// counted for the next row of the request. Calling it again for the same window changes nothing.
void SessionImpl::plan_flight_checkpoints(std::uint32_t lane, std::uint32_t window_end) {
    auto& request = requests[lane];
    auto& state   = sequences[lane];
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill || request.recovery ||
        request.checkpoint_plan.empty())
        return;
    const auto begin = state.continuation.metadata.execution_frontier;
    auto& plan       = request.checkpoint_plan;
    std::erase_if(plan, [&](const auto& candidate) { return candidate.frontier <= begin; });
    std::erase_if(request.reserved_checkpoints,
                  [&](const auto& flight) { return flight.candidate.frontier <= begin; });
    auto& reserved = request.reserved_checkpoints;
    for (std::size_t i = 0; i < plan.size();) {
        const auto& candidate = plan[i];
        if (candidate.frontier > window_end) break;
        if (std::any_of(reserved.begin(), reserved.end(), [&](const auto& flight) {
                return flight.candidate.frontier == candidate.frontier;
            })) {
            ++i;
            continue;
        }
        // Another lane or the catalog already holds the point; there is nothing to write.
        if (checkpoint_present(candidate.key)) {
            plan.erase(plan.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        FlightCheckpoint flight;
        flight.candidate = candidate;
        flight.point     = checkpoints.reserve_owned();
        if (flight.point.valid()) {
            ContextDemand demand;
            const auto image_id = demand.add_state({.count = 1});
            if (directory.assess(demand).ready()) {
                auto placed = directory.reserve(demand);
                if (placed && placed.reservation->ready()) {
                    if (const auto image =
                            state_store->reserve_destination(placed.reservation->take(image_id))) {
                        flight.image = *image;
                        flight.owner = this;
                    }
                }
            }
        }
        if (!flight.owner) {
            // The point returns with `flight`; the candidate leaves the plan.
            ++request.skipped_checkpoints;
            plan.erase(plan.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        const auto position =
            std::find_if(reserved.begin(), reserved.end(), [&](const auto& other) {
                return other.candidate.frontier > flight.candidate.frontier;
            });
        reserved.insert(position, std::move(flight));
        ++i;
    }
}

// Hands the reserved checkpoints that lie inside [begin, end] to the flight that carries the
// window. Their columns are execution boundaries of the window or its end, in ascending order.
void SessionImpl::attach_checkpoints(Execution& pending, InputWindow& window, SequenceState& state,
                                     std::uint32_t begin, std::uint32_t end) {
    auto& request  = requests[state.lane];
    auto& reserved = request.reserved_checkpoints;
    pending.checkpoints.resize(pending.members.size());
    auto& carried = pending.checkpoints[pending.members.size() - 1];
    for (auto it = reserved.begin(); it != reserved.end();) {
        const auto frontier = it->candidate.frontier;
        if (frontier <= begin || frontier > end) {
            ++it;
            continue;
        }
        window.checkpoints.push_back({.column       = frontier - begin,
                                      .state        = it->image,
                                      .state_offset = state_store->physical_offset(it->image)});
        std::erase_if(request.checkpoint_plan,
                      [&](const auto& candidate) { return candidate.frontier == frontier; });
        carried.push_back(std::move(*it));
        it = reserved.erase(it);
    }
}

// Publishes, when its flight retires, the checkpoints the flight wrote for each member. The flight
// has committed and its State image is complete: the image becomes an immutable snapshot with the
// KV prefix through its frontier. A checkpoint of a lane that is being cleared, or whose KV the
// commit did not reach, publishes nothing and returns its resources.
void SessionImpl::publish_flight_checkpoints(Execution& current,
                                             std::span<const std::uint8_t> clearing) {
    for (std::size_t b = 0; b < current.checkpoints.size(); ++b) {
        auto& list = current.checkpoints[b];
        if (list.empty()) continue;
        if ((b < clearing.size() && clearing[b]) || !valid_sequence(current.members[b])) {
            list.clear();
            continue;
        }
        const auto l = lane(current.members[b]);
        for (auto& flight : list) publish_flight_checkpoint(flight, sequences[l], requests[l]);
        list.clear();
    }
}

void SessionImpl::publish_flight_checkpoint(FlightCheckpoint& flight, SequenceState& state,
                                            RequestControl& request) {
    const auto frontier = flight.candidate.frontier;
    const auto backend_frontier = speculative_backend == SpeculativeBackend::Mtp ? frontier - 1
                                  : has_backend_pool()                           ? frontier
                                                                                 : 0;
    if (!flight.owner || !state.kv || state.continuation.metadata.execution_frontier < frontier ||
        text_kv_histories->committed_frontier(state.kv->text.handle()) < frontier ||
        (state.kv->backend &&
         backend_kv_histories->committed_frontier(state.kv->backend->handle()) <
             backend_frontier)) {
        ++request.skipped_checkpoints;
        return;
    }
    state_store->complete_initialization(flight.image);
    state_store->freeze(flight.image);
    auto captured                        = state.continuation;
    captured.image                       = flight.image;
    captured.metadata.execution_frontier = frontier;
    captured.metadata.text_kv_valid      = frontier;
    captured.metadata.backend_kv_valid   = backend_frontier;
    captured.metadata.dflash_context_frontier = frontier;
    captured.metadata.tail_hidden_valid  = true;
    std::vector<context::HistoryPrefix> prefixes;
    prefixes.push_back(text_kv_histories->retain_prefix(state.kv->text.handle(), frontier));
    if (state.kv->backend)
        prefixes.push_back(
            backend_kv_histories->retain_prefix(state.kv->backend->handle(), backend_frontier));
    CheckpointState record{.snapshot = Snapshot(*state_store, captured, std::move(prefixes),
                                                flight.candidate.identity),
                           .role     = runtime::CheckpointRole::SharedPrefix,
                           .key      = flight.candidate.key,
                           .origin   = runtime::CheckpointOrigin::Semantic,
                           .evidence = flight.candidate.evidence};
    // The snapshot owns the image now; the reservation of the catalog point is consumed.
    flight.image = {};
    flight.owner = nullptr;
    request.unreported_checkpoints.push_back(flight.point.publish(std::move(record)));
}

void SessionImpl::drop_lane_checkpoints(RequestControl& request) noexcept {
    request.reserved_checkpoints.clear();
    for (const auto handle : request.unreported_checkpoints) (void)release_checkpoint(handle);
    request.unreported_checkpoints.clear();
}

FlightCheckpoints SessionImpl::take_flight_checkpoints(SequenceHandle handle) {
    if (!valid_sequence(handle)) return {};
    auto& request = requests[lane(handle)];
    FlightCheckpoints taken{std::move(request.unreported_checkpoints),
                            std::exchange(request.skipped_checkpoints, 0U)};
    request.unreported_checkpoints.clear();
    return taken;
}

std::vector<std::uint32_t> SessionImpl::window_splits(const SequenceState& state,
                                                      std::uint32_t begin,
                                                      std::uint32_t end) const {
    std::vector<std::uint32_t> splits;
    if (causal_scoring) return splits;
    const auto frontiers = state.prefix_identity.execution_frontiers();
    for (auto it = std::upper_bound(frontiers.begin(), frontiers.end(), begin);
         it != frontiers.end() && *it < end; ++it)
        splits.push_back(*it);
    for (const auto& reserved : requests[state.lane].reserved_checkpoints)
        if (reserved.candidate.frontier > begin && reserved.candidate.frontier < end)
            splits.push_back(reserved.candidate.frontier);
    std::sort(splits.begin(), splits.end());
    splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
    return splits;
}

std::uint32_t SessionImpl::split_count(const SequenceState& state, std::uint32_t begin,
                                       std::uint32_t end) const {
    return static_cast<std::uint32_t>(window_splits(state, begin, end).size());
}

// One prefill segment of a completed execution, reported before its settlement is adopted.
PrefillRowProgress SessionImpl::prefill_row(const Execution& current, std::size_t member) {
    const auto& request = requests[lane(current.members[member])];
    PrefillRowProgress row;
    row.summary = {request.base->summary.prompt_tokens, request.prefill->base,
                   request.prefill->reuse};
    row.processed_prompt_tokens = current.columns[member];
    row.complete = current.completed[member].execution_frontier ==
                   request.base->summary.prompt_tokens;
    return row;
}

} // namespace ninfer::models::qwen3_5::detail
