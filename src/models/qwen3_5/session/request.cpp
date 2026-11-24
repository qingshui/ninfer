#include "models/qwen3_5/session/internal.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace ninfer::models::qwen3_5::detail {
namespace {

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

} // namespace

RequestBasePlan SessionImpl::plan_request(qwen::PreparedPromptData&& prompt,
                                          const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= static_cast<std::int32_t>(plan.profile->public_tokens)) {
            throw std::invalid_argument("prompt contains token outside the public token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.media_payloads.empty() ||
        prompt.media_payloads.size() != prompt.vision_items.size()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    for (std::size_t i = 0; i < prompt.media_payloads.size(); ++i) {
        if (!prompt.media_payloads[i] ||
            prompt.media_payloads[i]->features_per_patch != qwen::kPreparedVisionPatchFeatures ||
            prompt.media_payloads[i]->patch_elements !=
                prompt.vision_items[i].patch_count * qwen::kPreparedVisionPatchFeatures) {
            throw std::invalid_argument("prepared prompt media item payload has an invalid shape");
        }
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);
    if (const auto& rewrite = prompt.identity.rewrite_checkpoint;
        rewrite &&
        (rewrite->recovery_frontier == 0 || rewrite->recovery_frontier > rewrite->frontier ||
         rewrite->frontier > prompt.token_ids.size())) {
        throw std::invalid_argument("input recovery is outside its typed rewrite prefix");
    }

    auto base                             = std::make_shared<RequestBasePlanImpl>();
    base->context_cache                   = prompt.context_cache;
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->sampling                       = options.sampling;
    base->allow_prefix_reuse             = options.allow_prefix_reuse && prompt.identity.reusable;
    base->summary.publish_continuation = options.allow_prefix_reuse &&
                                         options.publish_prefix_continuation &&
                                         prompt.identity.reusable && context_cache.enabled;
    if (prompt.has_media()) {
        if (!plan.vision_max_features) {
            throw std::logic_error("Vision prompt has no startup workspace plan");
        }
        base->vision = qwen::prepare_vision_input(prompt, *plan.profile->config.vision,
                                                  plan.vision_max_features);
    }

    base->prefix_digests.assign(prompt);
    base->prefix_identity_tag = static_cast<std::uint32_t>(speculative_backend) |
                                (static_cast<std::uint32_t>(plan.proposal_head) << 8U) |
                                (static_cast<std::uint32_t>(plan.kv_storage) << 16U);
    std::uint32_t previous = 0;
    for (const auto frontier : prompt.identity.rewrite_execution_frontiers) {
        if (frontier <= previous || frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "rewrite execution frontiers must be ordered prompt positions");
        }
        previous = frontier;
    }
    // The branches of a sample group fork a text-only prompt with no masked draft state: the
    // parent prefills through prompt_tokens - 1, each child decodes the last prompt token with its
    // own sampling, and the parent's whole-page KV is shared. Penalty counts do not block the fork:
    // they count generated tokens only, so every branch installs its own empty row. A rewrite
    // checkpoint does not either: no flight writes an input-replay point, so the prompt's recovery
    // frontier needs no private state of a branch that forks past it. Vision and masked draft
    // windows keep the ordinary single-sequence path, and so does a request nobody shares.
    if (options.sample_count > 1 && !base->vision && base->summary.prompt_tokens >= 2 &&
        !is_masked_draft_backend(speculative_backend) && !causal_scoring) {
        base->fork_frontier = base->summary.prompt_tokens - 1;
    }
    base->prompt = std::make_shared<const qwen::PreparedPromptData>(std::move(prompt));
    if (base->summary.publish_continuation) {
        const auto& prepared = *base->prompt;
        auto backing         = std::make_shared<PreparedCaptureBacking>();
        backing->digests     = base->prefix_digests;
        backing->ledger      = prepared.token_ids;
        backing->prefix_identity.assign(prepared);
        base->capture_backing = backing;
        // A flight can write a checkpoint at a SharedPrefix point that carries evidence, strictly
        // before the end of the request's prefill; the flight cuts its traversal there. The end of
        // the prefill is left to the continuation checkpoint a finished request publishes.
        // Vision windows and draft windows keep state that only a single-row execution owns.
        if (!base->vision && !is_masked_draft_backend(speculative_backend) && !causal_scoring) {
            const auto end = base->prefill_end();
            for (const auto& opportunity : prepared.context_cache.opportunities) {
                const auto frontier = opportunity.frontier;
                if (opportunity.kind == PromptCacheMarkerKind::PrivateLongAnchor ||
                    opportunity.evidence == SharedCandidateEvidence::None || frontier == 0 ||
                    frontier >= end)
                    continue;
                auto it = std::find_if(base->checkpoint_candidates.begin(),
                                       base->checkpoint_candidates.end(), [&](const auto& other) {
                                           return other.frontier == frontier;
                                       });
                if (it == base->checkpoint_candidates.end()) {
                    base->checkpoint_candidates.push_back(
                        {.identity = backing,
                         .key      = {base->prefix_digests.at(frontier), frontier,
                                      base->prefix_identity_tag},
                         .frontier = frontier});
                    it = std::prev(base->checkpoint_candidates.end());
                }
                it->evidence |= opportunity.evidence;
            }
            std::sort(base->checkpoint_candidates.begin(), base->checkpoint_candidates.end(),
                      [](const auto& a, const auto& b) { return a.frontier < b.frontier; });
            for (const auto& candidate : base->checkpoint_candidates)
                base->checkpoint_frontiers.push_back(candidate.frontier);
        }
    }
    return RequestBasePlan(std::move(base));
}

namespace {
// End of a prefill/replay window that processes at most `tokens` prompt positions from `begin`;
// zero leaves the window to the Session.
std::uint32_t bounded_end(std::uint32_t begin, std::uint32_t end, std::uint32_t tokens) noexcept {
    return tokens == 0 ? end
                       : static_cast<std::uint32_t>(
                             std::min<std::uint64_t>(end, std::uint64_t{begin} + tokens));
}

bool is_window_unit(ExecutionUnitKind kind) noexcept {
    return kind == ExecutionUnitKind::Prefill || kind == ExecutionUnitKind::Replay;
}
// The prompt frontier a forking request's prefill stops at: the fork point when the request
// still has prompt work left to share, otherwise the whole prompt.
std::uint32_t prefill_target(const RequestControl& request,
                             const runtime::RequestPlanSummary& summary) {
    const auto fork = request.base->fork_frontier;
    return fork && request.prefill->base < fork ? fork : summary.prompt_tokens;
}
} // namespace

UnitDemand SessionImpl::prefill_unit(std::uint32_t prompt, std::uint32_t cursor) const {
    UnitDemand out{.kind = ExecutionUnitKind::Prefill};
    out.main_frontier    = std::min(prompt, cursor + prefill_chunk);
    out.backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                               ? (out.main_frontier ? out.main_frontier - 1 : 0)
                           : has_backend_pool() ? out.main_frontier
                                                : 0;
    return out;
}

UnitDemand SessionImpl::next_unit(const SequenceState& state, const RequestControl& request,
                                  ExecutionUnitKind kind, std::uint32_t tokens) const {
    UnitDemand out{.kind = kind, .tokens = tokens};
    const auto f = state.continuation.metadata.execution_frontier;
    switch (kind) {
    case ExecutionUnitKind::Prefill:
        if (request.lifecycle != Lifecycle::Prefilling || !request.prefill)
            throw std::logic_error("prefill demand requires a prefill cursor");
        out.main_frontier =
            window_end(state, request, request.prefill->cursor,
                       bounded_end(request.prefill->cursor,
                                   prefill_target(request, request.base->summary), tokens),
                       vision::Window{});
        break;
    case ExecutionUnitKind::Replay:
        if (request.lifecycle != Lifecycle::Replaying)
            throw std::logic_error("request is not replaying");
        out.main_frontier = window_end(
            state, request, request.replay_cursor,
            bounded_end(request.replay_cursor, request.replay_target, tokens), vision::Window{});
        break;
    case ExecutionUnitKind::Decode:
        if (request.lifecycle != Lifecycle::Active || !tokens || f >= capacity)
            throw std::logic_error("decode demand requires an active token and output budget");
        out.main_frontier = f + 1 +
                            (speculative_backend == SpeculativeBackend::None
                                 ? 0
                                 : std::min({tokens - 1, capacity - f - 1, draft_window}));
        break;
    case ExecutionUnitKind::Control:
        if (request.lifecycle != Lifecycle::Active || !tokens || tokens > capacity - f)
            throw std::logic_error("control demand is outside the available context");
        out.main_frontier = f + tokens;
        break;
    case ExecutionUnitKind::Normalize:
        out.main_frontier = f;
        break;
    }
    out.backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                               ? (out.main_frontier ? out.main_frontier - 1 : 0)
                           : has_backend_pool() ? out.main_frontier
                                                : 0;
    return out;
}

ResourceReservation SessionImpl::reserve_units(std::span<const ExecutionUnit> units) {
    // Lane-indexed tables, one entry per lane of the resolved concurrency.
    const auto lanes = static_cast<std::size_t>(max_concurrency);
    std::vector<UnitDemand> demands(lanes);
    std::vector<UnitDemand> coverage(lanes);
    std::vector<std::uint32_t> old_main(lanes, 0), old_backend(lanes, 0);
    std::vector<std::uint32_t> add_main(lanes, 0), add_backend(lanes, 0);
    // Growth of a decode unit out to the reserve horizon; only lanes in `horizon` use it.
    std::vector<std::uint32_t> horizon_main(lanes, 0), horizon_backend(lanes, 0);
    std::vector<std::uint8_t> horizon(lanes, 0);
    std::vector<std::uint8_t> seen(lanes, 0);
    for (std::size_t index = 0; index < units.size(); ++index) {
        const auto& unit = units[index];
        if (!valid_sequence(unit.sequence)) {
            throw std::logic_error("execution unit has a stale lane");
        }
        const auto lane = this->lane(unit.sequence);
        if (seen[lane]) { throw std::logic_error("duplicate lane in unit reservation"); }
        seen[lane]        = true;
        auto& request     = requests[lane];
        const auto& state = active_sequence(lane);
        demands[lane]     = next_unit(state, request, unit.kind, unit.tokens);
        if (is_window_unit(unit.kind)) {
            auto& demand = demands[lane];
            demand.main_frontier =
                window_end(state, request, state.continuation.metadata.execution_frontier,
                           bounded_end(state.continuation.metadata.execution_frontier,
                                       demand.main_frontier, unit.tokens),
                           vision_window);
            demand.backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                                          ? (demand.main_frontier ? demand.main_frontier - 1 : 0)
                                      : has_backend_pool() ? demand.main_frontier
                                                           : 0;
        }
        if ((unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) &&
            demands[lane].main_frontier == state.continuation.metadata.execution_frontier &&
            state.continuation.metadata.execution_frontier < request.base->summary.prompt_tokens)
            return {.status = runtime::UnitReservationStatus::ExecutionUnavailable};
        if (request.permit) {
            auto& permit = *request.permit;
            if (permit.kind != unit.kind ||
                (!is_window_unit(unit.kind) && permit.tokens != unit.tokens))
                throw std::logic_error("an existing execution permit cannot change its operation");
            if (is_window_unit(unit.kind)) {
                // A permit retains capacity through its frontier. A cap, a checkpoint or an
                // execution window may shorten the next prefix, but cannot enlarge that
                // retained grant. The permit keeps the shorter frontier, which bounds the window
                // the execution starts with.
                demands[lane].main_frontier =
                    std::min(demands[lane].main_frontier, permit.main_frontier);
                demands[lane].backend_frontier =
                    std::min(demands[lane].backend_frontier, permit.backend_frontier);
                permit.main_frontier    = demands[lane].main_frontier;
                permit.backend_frontier = demands[lane].backend_frontier;
                permit.tokens           = unit.tokens;
            } else if (permit.main_frontier != demands[lane].main_frontier ||
                       permit.backend_frontier != demands[lane].backend_frontier)
                throw std::logic_error("an existing execution permit cannot change its frontier");
        }
        coverage[lane] = request.recovery ? request.recovery->coverage : demands[lane];
        if (demands[lane].main_frontier > coverage[lane].main_frontier ||
            demands[lane].backend_frontier > coverage[lane].backend_frontier)
            throw std::logic_error("execution exceeds the retained recovery capacity");
        if (request.permit) continue;
        const auto main = text_kv_histories->write_pages(state.kv->text.handle(),
                                                         state.continuation.metadata.text_kv_valid,
                                                         coverage[lane].main_frontier);
        old_main[lane]  = text_kv_histories->reserved_growth_pages(state.kv->text.handle());
        add_main[lane]  = main > old_main[lane] ? main - old_main[lane] : 0;
        if (state.kv->backend) {
            const auto backend = backend_kv_histories->write_pages(state.kv->backend->handle(),
                                                                   backend_kv_valid(state),
                                                                   coverage[lane].backend_frontier);
            old_backend[lane] =
                backend_kv_histories->reserved_growth_pages(state.kv->backend->handle());
            add_backend[lane] = backend > old_backend[lane] ? backend - old_backend[lane] : 0;
        }
        // A decode unit also sizes the reclaim of a short pool for the pages of the next
        // kDecodeGrowthReserveTokens positions, so one reclaim serves many steps. A recovering
        // lane already holds its full coverage.
        const auto current = state.continuation.metadata.execution_frontier;
        const auto target = std::min<std::uint64_t>(capacity, std::uint64_t{current} +
                                                                  kDecodeGrowthReserveTokens);
        if (unit.kind == ExecutionUnitKind::Decode && !request.permit && !request.recovery &&
            target > demands[lane].main_frontier) {
            const auto main_target = static_cast<std::uint32_t>(target);
            const auto backend_target = speculative_backend == SpeculativeBackend::Mtp
                                            ? main_target - 1
                                        : has_backend_pool() ? main_target
                                                             : 0;
            const auto main = text_kv_histories->write_pages(
                state.kv->text.handle(), state.continuation.metadata.text_kv_valid, main_target);
            horizon_main[lane] = main > old_main[lane] ? main - old_main[lane] : 0;
            horizon_backend[lane] = add_backend[lane];
            if (state.kv->backend) {
                const auto backend = backend_kv_histories->write_pages(
                    state.kv->backend->handle(), backend_kv_valid(state), backend_target);
                horizon_backend[lane] = backend > old_backend[lane] ? backend - old_backend[lane] : 0;
            }
            horizon[lane] = true;
        }
    }
    struct Bundle {
        ContextDemand demand;
        std::vector<context::ContextDemandId> main, backend;
    };

    const auto build = [&](bool with_horizon) {
        Bundle bundle;
        bundle.main.resize(lanes);
        bundle.backend.resize(lanes);
        for (const auto& unit : units) {
            const auto lane = this->lane(unit.sequence);
            if (requests[lane].permit) continue;
            const auto& state = active_sequence(lane);
            const bool wide   = with_horizon && horizon[lane];
            bundle.main[lane] = bundle.demand.add_growth(
                *text_kv_histories, state.kv->text.handle(),
                wide ? horizon_main[lane] : add_main[lane]);
            if (state.kv->backend)
                bundle.backend[lane] = bundle.demand.add_growth(
                    *backend_kv_histories, state.kv->backend->handle(),
                    wide ? horizon_backend[lane] : add_backend[lane]);
        }
        return bundle;
    };
    // The minimum is the requirement and the only reservation: the unit's commit releases its
    // growth, so a reserved horizon would be taken and returned on every step. The horizon sizes
    // the reclaim of a short pool instead, so one reclaim serves many decode steps.
    auto chosen     = build(false);
    auto assessment = directory.assess(chosen.demand);
    if (!assessment.ready()) {
        // When the minimum is short, let one reclaim cover the horizon as well, unless that
        // larger demand needs something the minimum does not.
        if (std::any_of(horizon.begin(), horizon.end(), [](bool v) { return v; })) {
            auto wanted = directory.assess(build(true).demand);
            if (wanted.status == context::ContextAvailability::NeedsReclaim)
                return {.assessment = std::move(wanted)};
        }
        return {.assessment = std::move(assessment)};
    }
    auto& bundle   = chosen;
    auto& main_ids = bundle.main;
    auto& backend_ids = bundle.backend;
    auto reserved = directory.reserve(bundle.demand);
    if (!reserved) return {.assessment = std::move(reserved.assessment)};
    std::vector<std::uint8_t> added(lanes, 0);
    try {
        for (const auto& unit : units) {
            const auto lane = this->lane(unit.sequence);
            if (requests[lane].permit) { continue; }
            auto& state = active_sequence(lane);
            added[lane] = true;
            auto main   = reserved.reservation->take(main_ids[lane]);
            text_kv_histories->prepare_growth(state.kv->text.handle(), main);
            text_kv_histories->adopt_growth(state.kv->text.handle(), std::move(main));
            if (state.kv->backend) {
                auto backend = reserved.reservation->take(backend_ids[lane]);
                backend_kv_histories->prepare_growth(state.kv->backend->handle(), backend);
                backend_kv_histories->adopt_growth(state.kv->backend->handle(), std::move(backend));
            }
            requests[lane].permit = demands[lane];
        }
    } catch (...) {
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            if (!added[lane]) { continue; }
            auto& state = active_sequence(lane);
            text_kv_histories->shrink_growth(state.kv->text.handle(), old_main[lane]);
            if (state.kv->backend) {
                backend_kv_histories->shrink_growth(state.kv->backend->handle(), old_backend[lane]);
            }
            requests[lane].permit.reset();
        }
        throw;
    }
    // Checkpoints are reserved once the units hold their KV, so they never compete with a unit's
    // own growth.
    for (const auto& unit : units) {
        if (unit.kind != ExecutionUnitKind::Prefill) continue;
        const auto lane = this->lane(unit.sequence);
        auto& request   = requests[lane];
        if (!request.permit || request.recovery) continue;
        plan_flight_checkpoints(lane, request.permit->main_frontier);
    }
    return {.status = runtime::UnitReservationStatus::Granted};
}

void SessionImpl::release_units(std::span<const SequenceHandle> handles) noexcept {
    try {
        for (const auto handle : handles) {
            if (!valid_sequence(handle) || execution_pending(handle)) continue;
            const auto l  = lane(handle);
            auto& state   = sequences[l];
            auto& request = requests[l];
            if (state.kv) {
                text_kv_histories->shrink_growth(
                    state.kv->text.handle(),
                    request.recovery
                        ? text_kv_histories->write_pages(state.kv->text.handle(),
                                                         state.continuation.metadata.text_kv_valid,
                                                         request.recovery->coverage.main_frontier)
                        : 0);
                if (state.kv->backend)
                    backend_kv_histories->shrink_growth(
                        state.kv->backend->handle(),
                        request.recovery ? backend_kv_histories->write_pages(
                                               state.kv->backend->handle(), backend_kv_valid(state),
                                               request.recovery->coverage.backend_frontier)
                                         : 0);
            }
            request.permit.reset();
            // A checkpoint is reserved for the window of a permit; both leave together.
            request.reserved_checkpoints.clear();
        }
    } catch (...) { std::terminate(); }
}

void SessionImpl::require_unit(std::uint32_t lane, ExecutionUnitKind kind,
                               std::uint32_t tokens) const {
    // The tokens of a prefill/replay permit are an upper bound that the permit frontier already
    // carries, not part of the unit's identity.
    if (!requests[lane].permit || requests[lane].permit->kind != kind ||
        (!is_window_unit(kind) && requests[lane].permit->tokens != tokens))
        throw std::logic_error("execution does not match its reserved unit");
}

bool SessionImpl::prepare_settlement(std::uint32_t lane, const ContinuationMetadata& continuation,
                                     Lifecycle lifecycle, std::size_t ledger_tokens,
                                     context::ContextTransaction& transaction) {
    auto& state   = sequences[lane];
    auto& request = requests[lane];
    const bool retain =
        request.recovery && (lifecycle == Lifecycle::Replaying || lifecycle == Lifecycle::Pending ||
                             (lifecycle != Lifecycle::Finishable &&
                              continuation.execution_frontier <= request.recovery->frontier &&
                              ledger_tokens <= request.recovery->ledger_tokens));
    if (state.kv) {
        text_kv_histories->prepare_commit(state.kv->text, continuation.text_kv_valid,
                                          retain ? request.recovery->coverage.main_frontier : 0,
                                          transaction);
        if (state.kv->backend)
            backend_kv_histories->prepare_commit(
                *state.kv->backend, continuation.backend_kv_valid,
                retain ? request.recovery->coverage.backend_frontier : 0, transaction);
    }
    return retain;
}

void SessionImpl::adopt_unit(std::uint32_t lane, const ContinuationMetadata& continuation,
                             bool state_written, bool retain_recovery) noexcept {
    auto& state   = sequences[lane];
    auto& request = requests[lane];
    if (state_written) complete_state_write(state);
    state.continuation.metadata = continuation;
    if (!retain_recovery) request.recovery.reset();
    request.permit.reset();
}

} // namespace ninfer::models::qwen3_5::detail
