#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Per-lane scheduling inputs of one iteration. The Engine mirrors its resident lanes into the
// table; the Scheduler owns every planning decision derived from it.
struct LaneTableEntry {
    bool decode_ready   = false;
    bool control_ready  = false;
    bool prefilling     = false;
    bool replaying      = false;
    // Most prompt tokens the lane's next prefill unit could process; 0 when the lane takes no
    // prefill work this iteration. Replaying lanes run alone and are never prefill candidates.
    std::uint32_t prefill_limit = 0;
};

// The lanes one iteration plans over, indexed by lane. `priority_lane` names the lane the plan
// visits first: the Session's prefill continuation, else the pressure owner's prefill lane.
// `decode_in_flight` counts the decode lanes whose current unit is still executing: they are in
// no entry, and join the flight of the iteration planned once that unit completes.
struct LaneTable {
    std::vector<LaneTableEntry> lanes;
    std::optional<std::uint32_t> priority_lane;
    std::uint32_t decode_in_flight = 0;

    [[nodiscard]] std::uint32_t size() const noexcept {
        return static_cast<std::uint32_t>(lanes.size());
    }
};

// The bounds of one iteration's mixed flight. `token_budget` bounds the prefill tokens the
// iteration carries, `decode_rows_limit` the decode rows, and `prefill_floor_tokens` the tokens
// the first prefill lane is guaranteed even when the budget left is smaller: decode rows always
// ride, so the floor is what keeps prefill work from starving behind them.
//
// A flight that carries prefill costs more than the decode step it rides with, by a fixed amount
// that does not depend on how many prefill tokens it holds. Prefill batching spreads that cost
// over more tokens: while `batch_decode_rows` or more decode lanes are resident, prefill lanes
// that together hold fewer than `batch_prefill_tokens` wait, for at most `batch_max_deferrals`
// decode flights in a row, and those flights carry the decode rows alone. While those decode
// lanes are resident no prefill flies by itself, however many tokens it holds: it joins the decode
// rows' next flight. Zero `batch_decode_rows` plans prefill in every iteration.
struct IterationBudget {
    std::uint32_t token_budget         = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t decode_rows_limit    = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t prefill_floor_tokens = 0;
    std::uint32_t batch_decode_rows    = 0;
    std::uint32_t batch_prefill_tokens = 0;
    std::uint32_t batch_max_deferrals  = 0;
};

// The membership of one iteration. `decode_lanes` are decode rows, in lane order; every planned
// decode row rides the iteration's flight, bounded only by `decode_rows_limit`. `prefill_lanes`
// with `prefill_tokens` (parallel) are the prefill segments in admission order, each entry the
// token cap of its segment.
struct IterationPlan {
    std::vector<std::uint32_t> decode_lanes;
    std::vector<std::uint32_t> prefill_lanes;
    std::vector<std::uint32_t> prefill_tokens;

    [[nodiscard]] bool empty() const noexcept {
        return decode_lanes.empty() && prefill_lanes.empty();
    }

    [[nodiscard]] bool has_prefill() const noexcept { return !prefill_lanes.empty(); }

    [[nodiscard]] std::uint64_t prefill_token_total() const noexcept {
        std::uint64_t total = 0;
        for (const auto tokens : prefill_tokens) total += tokens;
        return total;
    }
};

// Membership and fairness only. Native stores own every physical reservation.
template <class Request>
class Scheduler {
public:
    using RequestPtr     = std::shared_ptr<Request>;
    using SequenceHandle = typename Request::SequenceHandle;

    // `lanes` is the resolved engine concurrency; every lane-indexed table of the Scheduler
    // is allocated for it.
    explicit Scheduler(std::uint32_t lanes) : lanes_(lanes) {}

    [[nodiscard]] std::uint32_t lanes() const noexcept { return lanes_; }

    struct RoundMembership {
        std::vector<std::uint32_t> lanes;
        std::vector<SequenceHandle> sequences;
        std::vector<RoundBudget> budgets;
        std::size_t size = 0;

        RoundMembership() = default;

        explicit RoundMembership(std::uint32_t capacity)
            : lanes(capacity, 0), sequences(capacity), budgets(capacity) {}

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ControlMembership {
        std::vector<std::uint32_t> lanes;
        std::vector<SequenceHandle> sequences;
        std::vector<TokenId> tokens;
        std::uint32_t row_stride = 0;
        std::size_t size         = 0;

        ControlMembership() = default;

        explicit ControlMembership(std::uint32_t capacity)
            : lanes(capacity, 0), sequences(capacity) {}

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }
    };

    template <class Slots>
    [[nodiscard]] RoundMembership build_round_membership(const Slots& slots,
                                                         std::uint32_t max_concurrency) const {
        RoundMembership membership(max_concurrency);
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_decode_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            if (!request->sequence) {
                throw std::logic_error("decode-ready request has no sequence handle");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.budgets[membership.size]   = RoundBudget{
                  .generated_tokens_remaining = request->recovery_pending
                                                    ? 1U
                                                    : request->output.model_token_budget_remaining(
                                                        request->budget->remaining()),
            };
            if (membership.budgets[membership.size].generated_tokens_remaining == 0) {
                throw std::logic_error("decode-ready request has no licensed model tokens");
            }
            ++membership.size;
        }
        return membership;
    }

    template <class Slots>
    [[nodiscard]] ControlMembership build_control_membership(const Slots& slots,
                                                             std::uint32_t max_concurrency) const {
        ControlMembership membership(max_concurrency);
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_control_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget || !request->sequence) {
                throw std::logic_error("control-ready request has no generation state");
            }
            const std::span<const TokenId> control = request->output.pending_control_tokens();
            if (control.empty() || control.size() > request->budget->remaining()) {
                throw std::logic_error("control-ready request has no admissible control span");
            }
            if (membership.row_stride == 0) {
                membership.row_stride = static_cast<std::uint32_t>(control.size());
                membership.tokens.reserve(static_cast<std::size_t>(membership.row_stride) *
                                          max_concurrency);
            } else if (control.size() != membership.row_stride) {
                throw std::logic_error("compact control membership has ragged target spans");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.tokens.insert(membership.tokens.end(), control.begin(), control.end());
            ++membership.size;
        }
        return membership;
    }

    // Iteration planning contract. The Engine runs one mixed flight per iteration: every planned
    // decode row plus the prefill segments the token budget admits. The plan is the only
    // authority for which lanes take which work:
    //  - Decode rows are admitted in lane order up to `decode_rows_limit`. Prefill never
    //    displaces a decode row; a decode row waits only when the row limit or the Session's
    //    own mixed-flight bounds exclude it, never because prefill work exists.
    //  - Prefill lanes are visited round-robin from the cursor, `priority_lane` first. The
    //    first visited lane receives at least `prefill_floor_tokens` of its own limit when the
    //    budget left is smaller, so an iteration with decode rows still advances prefill; a
    //    lane whose cap does not fit the budget left ends the prefill list and keeps the cursor,
    //    so the next iteration starts with it (no skipping). Every later lane is bounded by the
    //    budget left exactly.
    //  - Replaying lanes never join the prefill list: a replay unit runs alone.
    //  - The cursor moves past the last admitted prefill lane, so k prefilling lanes each
    //    advance within k iterations.
    //  - Prefill batching (see IterationBudget) holds the whole prefill list back for a bounded
    //    number of iterations. It never applies to the priority lane, so a continuation or a
    //    pressure owner keeps advancing, and it ends as soon as the waiting lanes hold enough
    //    tokens or the bound is reached.
    [[nodiscard]] IterationPlan plan_iteration(const LaneTable& table, IterationBudget budget) {
        IterationPlan plan;
        const std::uint32_t lanes = table.size();
        if (lanes_ != 0 && lanes != 0 && lanes != lanes_) {
            throw std::logic_error("iteration lane table does not match the scheduler lanes");
        }
        for (std::uint32_t lane = 0; lane < lanes && plan.decode_lanes.size() < budget.decode_rows_limit;
             ++lane) {
            if (table.lanes[lane].decode_ready) plan.decode_lanes.push_back(lane);
        }
        if (defers_prefill(table, budget, plan.decode_lanes.size())) return plan;
        std::int64_t remaining = budget.token_budget;
        bool floored           = false;
        const auto visit = [&](std::uint32_t lane) {
            const auto& entry = table.lanes[lane];
            if (!entry.prefilling || entry.prefill_limit == 0) return;
            std::int64_t cap = std::min<std::int64_t>(entry.prefill_limit, remaining);
            if (!floored && budget.prefill_floor_tokens > 0) {
                // The first prefill lane of the iteration keeps its floor even when the budget
                // left cannot cover it.
                cap = std::max<std::int64_t>(cap,
                                             std::min<std::int64_t>(entry.prefill_limit,
                                                                    budget.prefill_floor_tokens));
            }
            if (cap <= 0) return;
            plan.prefill_lanes.push_back(lane);
            plan.prefill_tokens.push_back(static_cast<std::uint32_t>(cap));
            remaining -= cap;
            floored          = true;
            prefill_cursor_  = (lane + 1U) % lanes;
        };
        if (table.priority_lane && *table.priority_lane < lanes) visit(*table.priority_lane);
        // The visit pass reads its starting cursor once: `visit` advances the cursor as it
        // admits, so the remaining offsets must not re-read it and revisit a lane.
        const std::uint32_t start = prefill_cursor_;
        for (std::uint32_t offset = 0; offset < lanes; ++offset) {
            const std::uint32_t lane = (start + offset) % lanes;
            if (table.priority_lane && *table.priority_lane < lanes && lane == *table.priority_lane)
                continue;
            visit(lane);
        }
        return plan;
    }

    // Whether prefill batching holds back every prefill lane of this iteration's plan: the ones
    // waiting now and any an admission before the plan would add while the waiting tokens stay
    // below the target. `decode_rows` decode lanes fly now, enough decode lanes are resident, and
    // the deferral bound is not reached. Reads the batching state without advancing it.
    [[nodiscard]] bool batching_holds(const LaneTable& table, const IterationBudget& budget,
                                      std::size_t decode_rows) const noexcept {
        return decode_rows != 0 && batching_resident(table, budget, decode_rows) &&
               waiting_prefill(table) < budget.batch_prefill_tokens &&
               deferred_iterations_ < budget.batch_max_deferrals;
    }

    // Queue/capacity changes start a bounded pass. Ordinary decode cycles only continue
    // an unfinished pass; they do not restart failed admission checks.
    void begin_admission_scan() noexcept { admission_scan_requested_ = true; }

    [[nodiscard]] bool admission_scan_pending() const noexcept {
        return admission_scan_requested_ || !admission_scan_remaining_.empty();
    }

    void admission_candidate_checked(std::uint64_t request_id) noexcept {
        // The tickets stay sorted (fresh_candidates), so a pass over hundreds of queued requests
        // checks each one off in logarithmic time.
        const auto found = std::lower_bound(admission_scan_remaining_.begin(),
                                            admission_scan_remaining_.end(), request_id);
        if (found != admission_scan_remaining_.end() && *found == request_id)
            admission_scan_remaining_.erase(found);
        if (request_id != admission_scan_head_) { scan_after_ = request_id; }
    }

    // Only a successful younger admission consumes the older request's bypass allowance.
    [[nodiscard]] std::vector<RequestPtr> fresh_candidates(const std::deque<RequestPtr>& fresh,
                                                           std::uint32_t concurrency) {
        std::vector<RequestPtr> candidates;
        if (fresh.empty()) {
            admission_scan_requested_ = false;
            admission_scan_remaining_.clear();
            return candidates;
        }
        admission_scan_head_ = fresh.front()->id;
        // One read of each queued request: the requests behind one whose bypass allowance is
        // spent are not eligible. The queue can hold hundreds of requests, so everything after
        // this pass works on the copied ids.
        auto& ids = scan_ids_;
        ids.clear();
        for (const auto& request : fresh) {
            ids.push_back(request->id);
            if (request->admission_bypasses >= concurrency) break;
        }
        if (admission_scan_requested_) {
            admission_scan_remaining_.assign(ids.begin(), ids.end());
            std::sort(admission_scan_remaining_.begin(), admission_scan_remaining_.end());
            admission_scan_requested_ = false;
        } else {
            // Cancellation and a newly exhausted bypass allowance can remove eligibility
            // while the pass is in progress. Keep tickets, not queue positions or iterators.
            auto& eligible = scan_sorted_;
            eligible.assign(ids.begin(), ids.end());
            std::sort(eligible.begin(), eligible.end());
            std::erase_if(admission_scan_remaining_, [&](std::uint64_t id) {
                return !std::binary_search(eligible.begin(), eligible.end(), id);
            });
        }
        if (admission_scan_remaining_.empty()) { return candidates; }
        candidates.push_back(fresh.front());
        const auto& pending = admission_scan_remaining_;
        // The younger requests still to check, as (id, queue position), in id order.
        auto& younger = scan_younger_;
        younger.clear();
        for (std::size_t i = 1; i < ids.size(); ++i) {
            if (std::binary_search(pending.begin(), pending.end(), ids[i])) {
                younger.emplace_back(ids[i], i);
            }
        }
        std::sort(younger.begin(), younger.end());
        const auto next = std::find_if(younger.begin(), younger.end(),
                                       [&](const auto& entry) { return entry.first > scan_after_; });
        std::rotate(younger.begin(), next, younger.end());
        const auto count = std::min<std::size_t>(concurrency, younger.size());
        candidates.reserve(1 + count);
        for (std::size_t i = 0; i < count; ++i) candidates.push_back(fresh[younger[i].second]);
        return candidates;
    }

    // A branch the Engine takes together with the request it just admitted (a sample group's
    // children) joins the admitted membership without bypassing anyone: the group was one
    // admission, and the FIFO allowance of the requests ahead of it counted that admission once.
    void admitted_with(const Request& sibling) { service_order_.push_back(sibling.id); }

    void admitted(const std::deque<RequestPtr>& fresh, const Request& admitted,
                  std::uint32_t concurrency) {
        for (const auto& skipped : fresh) {
            if (skipped->id == admitted.id) {
                service_order_.push_back(admitted.id);
                return;
            }
            if (skipped->admission_bypasses >= concurrency) {
                throw std::logic_error("fresh admission exceeded the FIFO bypass allowance");
            }
            ++skipped->admission_bypasses;
        }
        throw std::logic_error("admitted request is absent from the fresh queue");
    }

    // A pressure turn owns the next capacity handoff, including asynchronous reclamation,
    // binding and replay. Request identity remains independent of service order.
    [[nodiscard]] std::optional<std::uint64_t> progress_owner() const noexcept {
        if (!pressure_ || service_order_.empty()) return std::nullopt;
        return service_order_.front();
    }

    void require_progress(std::uint64_t request) {
        contended_ = true;
        if (pressure_) return;
        const auto owner = std::find(service_order_.begin(), service_order_.end(), request);
        if (owner == service_order_.end())
            throw std::logic_error("capacity waiter is absent from admitted membership");
        std::rotate(service_order_.begin(), owner, service_order_.end());
        pressure_ = true;
        reset_service();
    }

    // The Engine charges an execution once both new progress and retirement are known.
    // Replay and time spent waiting for a grant do not consume this service budget.
    void served(std::uint64_t request, std::uint64_t elapsed_ns) noexcept {
        if (progress_owner() != request) return;
        served_ = true;
        service_ns_ += std::min(elapsed_ns, maximum_service_ns - service_ns_);
    }

    void recovery_work(std::uint64_t request, std::uint64_t elapsed_ns) noexcept {
        if (progress_owner() == request)
            recovery_ns_ += std::min(elapsed_ns, maximum_service_ns - recovery_ns_);
    }

    // Called at a safe Engine boundary. Waiting and recovery never expire a turn.
    bool advance_service(bool have_paused, bool recovering) noexcept {
        if (!pressure_ || recovering || !served_) return false;
        if (service_ns_ < std::max(service_quantum_ns, recovery_ns_) && service_order_.size() > 1)
            return false;
        if (!have_paused && (!contended_ || service_order_.size() <= 1)) {
            pressure_ = false;
            begin_admission_scan();
        } else {
            std::rotate(service_order_.begin(), service_order_.begin() + 1, service_order_.end());
        }
        reset_service();
        contended_ = false;
        return true;
    }

    void finished(std::uint64_t request) noexcept {
        const bool owner = progress_owner() == request;
        std::erase(service_order_, request);
        if (owner) reset_service();
        if (service_order_.empty()) pressure_ = false;
        begin_admission_scan();
    }

    template <class Slots>
    [[nodiscard]] std::optional<std::uint32_t>
    preemption_victim(const Slots& slots, std::uint32_t concurrency,
                      std::span<const std::uint32_t> excluded = {}) const noexcept {
        const auto owner = progress_owner();
        if (!owner) return std::nullopt;
        std::optional<std::uint32_t> youngest;
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            const auto& request = slots[lane];
            if (!request || request->id == owner || request->is_materializing() ||
                request->is_model_finished() || request->recovery_pending ||
                std::find(excluded.begin(), excluded.end(), lane) != excluded.end())
                continue;
            if (!youngest || request->id > slots[*youngest]->id) youngest = lane;
        }
        return youngest;
    }

    // Grant at least one second of useful execution. An expensive restore earns up to
    // four seconds to amortize replay/transfer, while retaining a finite service bound.
    static constexpr std::uint64_t service_quantum_ns = 1'000'000'000;
    static constexpr std::uint64_t maximum_service_ns = 4 * service_quantum_ns;

private:
    // The configured lane capacity; fixed after construction except for the failure reset.
    std::uint32_t lanes_;
    // Round-robin cursor over prefill lanes, advanced past the last admitted prefill lane.
    std::uint32_t prefill_cursor_ = 0;
    // Consecutive iterations that held prefill back under prefill batching.
    std::uint32_t deferred_iterations_ = 0;

    std::uint64_t scan_after_          = 0;
    std::uint64_t admission_scan_head_ = 0;
    std::vector<std::uint64_t> admission_scan_remaining_;
    // Working storage of fresh_candidates, kept to avoid an allocation per call.
    std::vector<std::uint64_t> scan_ids_, scan_sorted_;
    std::vector<std::pair<std::uint64_t, std::size_t>> scan_younger_;
    bool admission_scan_requested_ = false;
    std::deque<std::uint64_t> service_order_;
    bool pressure_             = false;
    bool contended_            = false;
    bool served_               = false;
    std::uint64_t service_ns_  = 0;
    std::uint64_t recovery_ns_ = 0;

    // The prefill tokens the lanes of a plan's table wait to process.
    [[nodiscard]] static std::uint64_t waiting_prefill(const LaneTable& table) noexcept {
        std::uint64_t waiting = 0;
        for (const auto& entry : table.lanes) {
            if (entry.prefilling) waiting += entry.prefill_limit;
        }
        return waiting;
    }

    // Enough decode lanes are resident for prefill batching, and no priority lane exempts it.
    [[nodiscard]] static bool batching_resident(const LaneTable& table,
                                                const IterationBudget& budget,
                                                std::size_t decode_rows) noexcept {
        return budget.batch_decode_rows != 0 &&
               decode_rows + table.decode_in_flight >= budget.batch_decode_rows &&
               !table.priority_lane;
    }

    // Whether prefill batching holds the prefill list back. It engages only while enough decode
    // lanes are resident and the waiting prefill is small; large prefill work flies with the
    // decode rows at once. While every resident decode lane executes, the waiting lanes join the
    // decode rows' next flight whatever they hold, and the iteration neither spends nor resets the
    // count, which advances once per iteration that flies decode rows.
    [[nodiscard]] bool defers_prefill(const LaneTable& table, const IterationBudget& budget,
                                      std::size_t decode_rows) noexcept {
        const std::uint64_t waiting = waiting_prefill(table);
        const bool resident         = batching_resident(table, budget, decode_rows);
        if (resident && decode_rows == 0 && waiting != 0) return true;
        const bool engaged = resident && waiting != 0 && waiting < budget.batch_prefill_tokens;
        if (!engaged) {
            deferred_iterations_ = 0;
            return false;
        }
        const bool defer     = deferred_iterations_ < budget.batch_max_deferrals;
        deferred_iterations_ = defer ? deferred_iterations_ + 1 : 0;
        return defer;
    }

    void reset_service() noexcept {
        served_      = false;
        service_ns_  = 0;
        recovery_ns_ = 0;
    }
};

} // namespace ninfer::runtime
