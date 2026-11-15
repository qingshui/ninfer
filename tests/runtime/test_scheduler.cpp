#include "runtime/engine/scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

struct Request {
    using SequenceHandle = std::uint64_t;
    enum class Phase { Waiting, Materializing, Prefill, Replay, Decode, Control, Finished };

    struct Budget {
        std::uint32_t tokens = 8;

        [[nodiscard]] std::uint32_t remaining() const noexcept { return tokens; }
    };

    struct Output {
        std::uint32_t model_tokens = 8;
        std::vector<ninfer::TokenId> control;

        [[nodiscard]] std::uint32_t model_token_budget_remaining(std::uint32_t total) const {
            return std::min(total, model_tokens);
        }

        [[nodiscard]] std::span<const ninfer::TokenId> pending_control_tokens() const {
            return control;
        }
    };

    [[nodiscard]] bool is_decode_ready() const { return phase == Phase::Decode; }

    [[nodiscard]] bool is_control_ready() const { return phase == Phase::Control; }

    [[nodiscard]] bool is_prefilling() const { return phase == Phase::Prefill; }

    [[nodiscard]] bool is_replaying() const { return phase == Phase::Replay; }

    [[nodiscard]] bool is_materializing() const { return phase == Phase::Materializing; }

    [[nodiscard]] bool is_model_finished() const { return phase == Phase::Finished; }

    std::uint64_t id                 = 0;
    std::uint32_t admission_bypasses = 0;
    Phase phase                      = Phase::Waiting;
    bool capture_pending             = false;
    bool recovery_pending            = false;
    std::optional<Budget> budget     = Budget{};
    std::optional<SequenceHandle> sequence;
    Output output;
};

using Scheduler  = ninfer::runtime::Scheduler<Request>;
using RequestPtr = std::shared_ptr<Request>;
// Lane slot tables are runtime-sized; tests carry their own capacity.
constexpr std::uint32_t kSlots = 6;
using Slots                     = std::array<RequestPtr, kSlots>;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

RequestPtr request(std::uint64_t id, Request::Phase phase = Request::Phase::Waiting) {
    auto value      = std::make_shared<Request>();
    value->id       = id;
    value->phase    = phase;
    value->sequence = id;
    return value;
}

void test_fifo_bypass_survives_pressure() {
    Scheduler scheduler(kSlots);
    const auto head = request(10);
    std::deque<RequestPtr> fresh{head, request(11), request(12), request(13)};

    // Merely inspecting requests that cannot obtain resources never spends bypass credit.
    for (int attempt = 0; attempt < 6; ++attempt) {
        scheduler.begin_admission_scan();
        const auto candidates = scheduler.fresh_candidates(fresh, 2);
        for (const auto& candidate : candidates) {
            scheduler.admission_candidate_checked(candidate->id);
        }
        require(!candidates.empty() && candidates.front() == head && candidates.size() <= 3,
                "fresh scan did not preserve the head and bounded work");
    }
    require(head->admission_bypasses == 0 && fresh[1]->admission_bypasses == 0,
            "failed admission attempts consumed FIFO bypass credit");

    scheduler.admitted(fresh, *fresh[1], 2);
    fresh.erase(fresh.begin() + 1);
    require(head->admission_bypasses == 1, "successful younger admission was not charged");

    scheduler.require_progress(11);
    scheduler.served(11, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(false, false) && !scheduler.progress_owner(),
            "resolved pressure retained the service owner");
    require(!scheduler.fresh_candidates(fresh, 2).empty(),
            "resolved pressure did not reopen fresh candidate scanning");
    require(head->admission_bypasses == 1, "pressure recovery reset a waiting request's credit");

    scheduler.begin_admission_scan();
    const auto candidates = scheduler.fresh_candidates(fresh, 2);
    const auto second     = std::find(candidates.begin(), candidates.end(), fresh[1]);
    require(second != candidates.end(), "second permitted younger request was not considered");
    scheduler.admitted(fresh, **second, 2);
    fresh.erase(fresh.begin() + 1);
    for (std::uint64_t id = 14; id < 20; ++id) {
        fresh.push_back(request(id));
        scheduler.begin_admission_scan();
        const auto blocked = scheduler.fresh_candidates(fresh, 2);
        require(blocked.size() == 1 && blocked.front() == head,
                "new arrivals or completion events bypassed an exhausted FIFO head");
    }

    // Cancelling the blocked head exposes the next request with its own existing credit.
    fresh.pop_front();
    scheduler.begin_admission_scan();
    const auto after_cancel = scheduler.fresh_candidates(fresh, 2);
    require(after_cancel.size() > 1 && after_cancel.size() <= 3 && after_cancel.front()->id == 13 &&
                after_cancel.front()->admission_bypasses == 0,
            "head removal transferred its exhausted credit to the next request");
}

void test_scan_progress_and_waiter_barrier() {
    Scheduler scheduler(kSlots);
    std::deque<RequestPtr> fresh{request(1), request(2), request(3),
                                 request(4), request(5), request(6)};
    std::vector<std::uint64_t> seen;
    const auto observe = [&](const std::vector<RequestPtr>& candidates) {
        require(!candidates.empty() && candidates.front() == fresh.front() &&
                    candidates.size() <= 3,
                "candidate scan exceeded the head plus bounded younger requests");
        for (const auto& candidate : candidates) {
            require(std::find(fresh.begin(), fresh.end(), candidate) != fresh.end(),
                    "removed request remained a candidate");
            scheduler.admission_candidate_checked(candidate->id);
            if (std::find(seen.begin(), seen.end(), candidate->id) == seen.end()) {
                seen.push_back(candidate->id);
            }
        }
    };
    scheduler.begin_admission_scan();
    const auto first = scheduler.fresh_candidates(fresh, 2);
    observe(first);
    require(scheduler.admission_scan_pending(),
            "failed bounded batch abandoned later requests that could fit");
    const auto next = scheduler.fresh_candidates(fresh, 2);
    require(
        next.size() == 3 && next[1]->id == 4 && next[2]->id == 5,
        "a later fitting request required another arrival or resident completion to be checked");
    observe(next);

    // Remove the last examined ticket and add an arrival. The next event must preserve
    // cursor progress rather than restart indefinitely at the first failed requests.
    std::erase(fresh, next.back());
    fresh.push_back(request(7));
    scheduler.begin_admission_scan();
    const auto after_change = scheduler.fresh_candidates(fresh, 2);
    require(after_change.size() == 3 && after_change[1]->id == 6 && after_change[2]->id == 7,
            "queue changes reset the stable-ticket admission cursor");
    observe(after_change);
    for (std::size_t attempt = 0; scheduler.admission_scan_pending() && attempt < fresh.size();
         ++attempt) {
        observe(scheduler.fresh_candidates(fresh, 2));
    }
    require(!scheduler.admission_scan_pending(), "a complete admission pass kept retrying");
    require(scheduler.fresh_candidates(fresh, 2).empty(),
            "ordinary decode restarted a failed admission pass");
    for (const auto& pending : fresh) {
        require(std::find(seen.begin(), seen.end(), pending->id) != seen.end(),
                "failed leading candidates starved a later admission attempt");
        require(pending->admission_bypasses == 0, "candidate scanning spent admission credit");
    }

    // A skipped waiter retains its own limit even while an older request is still ahead of it.
    fresh[1]->admission_bypasses = 2;
    for (int attempt = 0; attempt < 3; ++attempt) {
        scheduler.begin_admission_scan();
        const auto candidates = scheduler.fresh_candidates(fresh, 2);
        require(candidates.size() == 2 && candidates[1] == fresh[1],
                "rotating scan crossed an exhausted non-head waiter");
        observe(candidates);
        require(!scheduler.admission_scan_pending(),
                "blocked younger requests kept the scan awake behind a FIFO barrier");
    }
}

void test_scan_keeps_unexamined_candidates() {
    Scheduler scheduler(kSlots);
    std::deque<RequestPtr> fresh{request(1), request(2), request(3), request(4)};
    scheduler.begin_admission_scan();
    const auto first = scheduler.fresh_candidates(fresh, 2);
    require(first.size() == 3, "admission did not expose a bounded candidate batch");
    scheduler.admission_candidate_checked(first[0]->id);
    scheduler.admission_candidate_checked(first[1]->id);
    scheduler.admitted(fresh, *first[1], 2);
    std::erase(fresh, first[1]);
    // Admission returns early, so the last returned candidate has not actually been inspected.
    const auto continued = scheduler.fresh_candidates(fresh, 2);
    require(continued.size() == 3 && continued[1]->id == 3 && continued[2]->id == 4,
            "successful admission discarded candidates that had never been checked");
    for (const auto& candidate : continued) {
        scheduler.admission_candidate_checked(candidate->id);
    }
    require(!scheduler.admission_scan_pending(),
            "completed candidate checks did not retire the pass");
    scheduler.begin_admission_scan();
    require(!scheduler.fresh_candidates(fresh, 2).empty(),
            "a later capacity event did not retry a previously failed request");
}

void admit(Scheduler& scheduler, const RequestPtr& value) {
    scheduler.admitted(std::deque<RequestPtr>{value}, *value, 4);
}

void test_pressure_handoff_and_fair_service() {
    Scheduler scheduler(kSlots);
    Slots slots{};
    auto a = slots[0] = request(10, Request::Phase::Decode);
    auto b = slots[1] = request(20, Request::Phase::Decode);
    auto c = slots[2] = request(30, Request::Phase::Decode);
    for (const auto& value : {a, b, c}) admit(scheduler, value);

    scheduler.require_progress(a->id);
    require(scheduler.preemption_victim(slots, 4) == 2,
            "capacity handoff did not exclude its beneficiary");
    const std::array excluded{2U};
    require(scheduler.preemption_victim(slots, 4, excluded) == 1,
            "noncontributing victim prevented another release candidate");
    slots[2].reset(); // c paused; the same capacity is still owed to a.
    scheduler.require_progress(c->id);
    require(scheduler.progress_owner() == a->id && !scheduler.advance_service(true, false),
            "a paused victim stole an unfinished capacity handoff");
    scheduler.served(b->id, Scheduler::service_quantum_ns);
    require(!scheduler.advance_service(true, false), "another row's work expired the owner");
    scheduler.served(a->id, Scheduler::service_quantum_ns - 1);
    require(!scheduler.advance_service(true, false), "an incomplete service slice expired");
    scheduler.served(a->id, 1);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == b->id,
            "waiting request did not receive the next service turn");
    scheduler.served(b->id, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == c->id,
            "suspended request was skipped by service rotation");
    require(scheduler.preemption_victim(slots, 4) == 1,
            "a waiting request could not preempt an older served resident");
    b->recovery_pending = true;
    require(scheduler.preemption_victim(slots, 4) == 0,
            "protected Native recovery was selected as a victim");
    b->recovery_pending = false;
    // Arbitrarily many resource polls/replay boundaries cannot consume c's service turn.
    for (int i = 0; i < 100; ++i)
        require(!scheduler.advance_service(true, true) && scheduler.progress_owner() == c->id,
                "replay exhausted a turn without new progress");
    scheduler.served(c->id, Scheduler::service_quantum_ns);
    require(!scheduler.advance_service(true, true), "Native recovery protection was bypassed");
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == a->id,
            "fair rotation failed to return to the earlier request");
    scheduler.finished(a->id);
    require(scheduler.progress_owner() == b->id, "cancelled owner retained the handoff");
    scheduler.finished(b->id);
    require(scheduler.progress_owner() == c->id, "terminal owner blocked remaining work");
    scheduler.served(c->id, 1);
    require(scheduler.advance_service(false, false) && !scheduler.progress_owner(),
            "resolved pressure did not reopen fresh admission");
    scheduler.finished(c->id);
}

void test_recovery_cost_has_a_bounded_service_allowance() {
    Scheduler scheduler(kSlots);
    auto a = request(1), b = request(2);
    admit(scheduler, a);
    admit(scheduler, b);
    scheduler.require_progress(a->id);
    scheduler.recovery_work(a->id, 10 * Scheduler::maximum_service_ns);
    require(!scheduler.advance_service(true, false), "replay was counted as new progress");
    scheduler.served(a->id, Scheduler::service_quantum_ns);
    require(!scheduler.advance_service(true, false), "expensive replay earned no useful service");
    scheduler.served(a->id, Scheduler::maximum_service_ns - Scheduler::service_quantum_ns);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == b->id,
            "recovery cost allowed unbounded service");
    scheduler.served(b->id, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(true, false), "recovery cost leaked into another turn");
}

void test_bounded_service_under_continuous_contention() {
    Scheduler scheduler(kSlots);
    std::array<RequestPtr, 4> requests{request(1), request(2), request(3), request(4)};
    for (const auto& value : requests) admit(scheduler, value);
    scheduler.require_progress(1);
    std::array<unsigned, 4> grants{};
    for (unsigned turn = 0; turn < 40; ++turn) {
        const auto owner = scheduler.progress_owner();
        require(owner && *owner == turn % 4 + 1, "an admitted waiter was bypassed");
        ++grants[*owner - 1];
        for (const auto& value : requests) scheduler.require_progress(value->id);
        scheduler.served(*owner, Scheduler::service_quantum_ns);
        require(scheduler.advance_service(true, false), "contended service failed to rotate");
    }
    require(std::all_of(grants.begin(), grants.end(), [](auto n) { return n == 10; }),
            "continuous contention starved a request");
}

// One iteration plans every decode row in lane order up to the row limit, and prefill work
// never displaces a decode row: a decode row waits only when the row limit excludes it.
void test_plan_iteration_decode_rows_ride_within_limit() {
    Scheduler scheduler(kSlots);
    ninfer::runtime::LaneTable table;
    table.lanes.resize(kSlots);
    table.lanes[0].decode_ready = true;
    table.lanes[1].prefilling   = true;
    table.lanes[1].prefill_limit = 128;
    table.lanes[2].decode_ready = true;
    table.lanes[3].decode_ready = true;
    table.lanes[4].prefilling   = true;
    table.lanes[4].prefill_limit = 64;

    const auto plan = scheduler.plan_iteration(table, ninfer::runtime::IterationBudget{
                                                             .token_budget      = 256,
                                                             .decode_rows_limit = 2,
                                                             .prefill_floor_tokens = 0});
    require(plan.decode_lanes == std::vector<std::uint32_t>({0, 2}),
            "decode rows were not admitted in lane order up to the limit");
    require(!plan.prefill_lanes.empty() && plan.prefill_token_total() <= 256,
            "prefill displaced or out-budgeted the decode rows");

    // Every decode row rides when the row limit covers them, even with a tiny token budget.
    const auto all_decodes = scheduler.plan_iteration(table, ninfer::runtime::IterationBudget{
                                                                    .token_budget = 1,
                                                                    .decode_rows_limit = kSlots,
                                                                    .prefill_floor_tokens = 0});
    require(all_decodes.decode_lanes == std::vector<std::uint32_t>({0, 2, 3}),
            "a decode row waited behind prefill work");
}

// The token budget bounds the prefill segments one iteration carries: a lane whose limit does
// not fit is admitted at the budget left, and the floor keeps the first visited prefill lane
// advancing beside the decode rows even when the budget left is smaller than the floor.
void test_plan_iteration_prefill_budget_and_floor() {
    Scheduler scheduler(kSlots);
    ninfer::runtime::LaneTable table;
    table.lanes.resize(kSlots);
    table.lanes[0].prefilling    = true;
    table.lanes[0].prefill_limit = 100;
    table.lanes[1].prefilling    = true;
    table.lanes[1].prefill_limit = 100;
    table.lanes[2].prefilling    = true;
    table.lanes[2].prefill_limit = 64;

    const auto plan = scheduler.plan_iteration(table, ninfer::runtime::IterationBudget{
                                                             .token_budget      = 256,
                                                             .decode_rows_limit = kSlots,
                                                             .prefill_floor_tokens = 0});
    require(plan.prefill_lanes == std::vector<std::uint32_t>({0, 1, 2}) &&
                plan.prefill_tokens == std::vector<std::uint32_t>({100, 100, 56}),
            "the last lane was not admitted at the budget left");

    // A floor larger than the budget still admits the first lane's own limit, and nothing
    // after it.
    const auto floored = scheduler.plan_iteration(table, ninfer::runtime::IterationBudget{
                                                                 .token_budget      = 32,
                                                                 .decode_rows_limit = kSlots,
                                                                 .prefill_floor_tokens = 128});
    require(floored.prefill_lanes == std::vector<std::uint32_t>({0}) &&
                floored.prefill_tokens[0] == 100,
            "the floor did not keep the head prefill lane advancing");
}

// The plan visits prefill lanes round-robin from its cursor: the priority lane leads, a lane
// that exhausted the budget keeps the cursor so the next iteration starts with it, and k
// prefilling lanes each advance within k iterations behind decode rows.
void test_plan_iteration_round_robin_and_priority() {
    Scheduler scheduler(kSlots);
    ninfer::runtime::LaneTable table;
    table.lanes.resize(kSlots);
    for (std::uint32_t lane = 0; lane < 4; ++lane) {
        table.lanes[lane].prefilling    = true;
        table.lanes[lane].prefill_limit = 64;
    }
    const auto budget = ninfer::runtime::IterationBudget{
        .token_budget = 64, .decode_rows_limit = kSlots, .prefill_floor_tokens = 64};

    const auto first = scheduler.plan_iteration(table, budget);
    require(first.prefill_lanes == std::vector<std::uint32_t>({0}),
            "the token budget admitted more than the first lane");
    const auto second = scheduler.plan_iteration(table, budget);
    require(second.prefill_lanes == std::vector<std::uint32_t>({1}),
            "the cursor did not advance past the admitted lane");

    // The priority lane leads its iteration and takes the floor.
    table.priority_lane = 3;
    const auto led = scheduler.plan_iteration(table, budget);
    require(!led.prefill_lanes.empty() && led.prefill_lanes.front() == 3,
            "the priority lane did not lead the prefill list");
    table.priority_lane.reset();

    // Replaying lanes are never prefill candidates; a lane with no prefill work keeps the order.
    table.lanes[2].prefilling = false;
    table.lanes[2].replaying  = true;
    const auto skip = scheduler.plan_iteration(table, budget);
    require(std::find(skip.prefill_lanes.begin(), skip.prefill_lanes.end(), 2) ==
                skip.prefill_lanes.end(),
            "a replaying lane joined the prefill list");
}

// A table whose size does not match the configured lanes is rejected.
void test_plan_iteration_rejects_foreign_lane_table() {
    Scheduler scheduler(kSlots);
    ninfer::runtime::LaneTable table;
    table.lanes.resize(kSlots + 1);
    bool threw = false;
    try {
        (void)scheduler.plan_iteration(table, ninfer::runtime::IterationBudget{});
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "a foreign lane table was planned");
}

// k prefilling lanes with more work than the budget admits per iteration: each lane advances
// within k iterations, none is skipped, and every iteration respects the token budget.
void test_plan_iteration_does_not_starve_prefill_lanes() {
    for (std::uint32_t lanes = 2; lanes <= 5; ++lanes) {
        for (const std::uint32_t budget_tokens : {32U, 100U, 256U}) {
            Scheduler scheduler(kSlots);
            ninfer::runtime::LaneTable table;
            table.lanes.resize(kSlots);
            std::array<std::uint32_t, kSlots> left{900, 128, 40, 513, 300, 1024};
            std::array<std::uint32_t, kSlots> last{};
            for (std::uint32_t lane = 0; lane < lanes; ++lane) {
                table.lanes[lane].prefilling = true;
            }
            const auto budget = ninfer::runtime::IterationBudget{
                .token_budget = budget_tokens, .decode_rows_limit = kSlots,
                .prefill_floor_tokens = budget_tokens};
            for (std::uint32_t iteration = 1; iteration <= 400; ++iteration) {
                for (std::uint32_t lane = 0; lane < lanes; ++lane) {
                    table.lanes[lane].prefill_limit = left[lane] > 0 ? std::min(left[lane], 128U) : 0;
                }
                const auto plan = scheduler.plan_iteration(table, budget);
                require(plan.prefill_token_total() <= budget_tokens,
                        "an iteration exceeded its token budget");
                for (std::size_t i = 0; i < plan.prefill_lanes.size(); ++i) {
                    const auto lane = plan.prefill_lanes[i];
                    require(plan.prefill_tokens[i] <= table.lanes[lane].prefill_limit,
                            "a segment cap exceeded its lane limit");
                    left[lane] -= plan.prefill_tokens[i];
                    if (left[lane] == 0) { table.lanes[lane].prefilling = false; }
                    last[lane] = iteration;
                }
                bool live = false;
                for (std::uint32_t lane = 0; lane < lanes; ++lane) {
                    if (left[lane] == 0) continue;
                    live = true;
                    require(iteration - last[lane] <= lanes, "a prefilling lane was starved");
                }
                if (!live) break;
            }
            for (std::uint32_t lane = 0; lane < lanes; ++lane) {
                require(left[lane] == 0, "a prefilling lane did not finish");
            }
        }
    }
}

void test_decode_and_control_membership() {
    Scheduler scheduler(kSlots);
    Slots slots{};
    slots[0]                      = request(1, Request::Phase::Decode);
    slots[0]->budget->tokens      = 11;
    slots[0]->output.model_tokens = 3;
    slots[2]                      = request(2, Request::Phase::Decode);
    slots[2]->recovery_pending    = true;
    slots[3]                      = request(3, Request::Phase::Decode);
    slots[3]->capture_pending     = true;
    const auto round              = scheduler.build_round_membership(slots, 4);
    require(round.size == 2 && round.lanes[0] == 0 && round.lanes[1] == 2 &&
                round.sequences[0] == 1 && round.sequences[1] == 2 &&
                round.budgets[0].generated_tokens_remaining == 3 &&
                round.budgets[1].generated_tokens_remaining == 1,
            "compact decode lost output limits, first-unit limits or capture dependencies");

    slots[0]->phase          = Request::Phase::Control;
    slots[0]->output.control = {7, 8, 9};
    slots[2]->phase          = Request::Phase::Control;
    slots[2]->output.control = {10, 11, 12};
    const auto control       = scheduler.build_control_membership(slots, 4);
    require(control.size == 2 && control.row_stride == 3 &&
                control.tokens == std::vector<ninfer::TokenId>({7, 8, 9, 10, 11, 12}) &&
                control.sequences[0] == 1 && control.sequences[1] == 2,
            "compact control membership changed committed control spans or request order");
}

// Lane tables are sized by the configured concurrency, not by any fixed upper bound: a
// scheduler with more than 16 lanes plans, caps and charges lanes beyond the old ceiling
// exactly like the lanes below it.
void test_lane_tables_follow_the_configured_capacity() {
    constexpr std::uint32_t lanes = 20;
    Scheduler scheduler(lanes);
    ninfer::runtime::LaneTable table;
    table.lanes.resize(lanes);
    for (std::uint32_t lane = 16; lane < lanes; ++lane) {
        table.lanes[lane].prefilling    = true;
        table.lanes[lane].prefill_limit = 16;
    }
    const auto plan = scheduler.plan_iteration(table, ninfer::runtime::IterationBudget{
                                                             .token_budget      = 64,
                                                             .decode_rows_limit = lanes,
                                                             .prefill_floor_tokens = 0});
    require(!plan.prefill_lanes.empty() && plan.prefill_lanes.front() == 16 &&
                plan.prefill_tokens.front() == 16,
            "a lane beyond the old ceiling was not planned");

    std::vector<RequestPtr> slots(lanes);
    slots[19] = request(100, Request::Phase::Decode);
    const auto round = scheduler.build_round_membership(slots, lanes);
    require(round.size == 1 && round.lanes[0] == 19 &&
                round.budgets[0].generated_tokens_remaining == 8,
            "decode membership lost a lane beyond the old ceiling");
}

} // namespace

int main() {
    try {
        test_fifo_bypass_survives_pressure();
        test_scan_progress_and_waiter_barrier();
        test_scan_keeps_unexamined_candidates();
        test_pressure_handoff_and_fair_service();
        test_bounded_service_under_continuous_contention();
        test_recovery_cost_has_a_bounded_service_allowance();
        test_plan_iteration_decode_rows_ride_within_limit();
        test_plan_iteration_prefill_budget_and_floor();
        test_plan_iteration_round_robin_and_priority();
        test_plan_iteration_rejects_foreign_lane_table();
        test_plan_iteration_does_not_starve_prefill_lanes();
        test_decode_and_control_membership();
        test_lane_tables_follow_the_configured_capacity();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
