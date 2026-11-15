// T6 / mixed-flight scheduling, through Scheduler's public interface only. One iteration plans
// one flight: every decode row rides it, and the prefill segments the token budget admits ride
// beside them. The Engine submits the planned membership as one Session call; these tests check
// the plan the Engine turns into that flight.
#include "runtime/engine/scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

struct Request {
    using SequenceHandle = std::uint64_t;
    enum class Phase { Waiting, Prefill, Replay, Decode };
    enum class Flag { DecodeReady, ControlReady, Prefilling, Replaying };

    struct Budget {
        [[nodiscard]] std::uint32_t remaining() const noexcept { return 8; }
    };
    struct Output {
        [[nodiscard]] std::uint32_t model_token_budget_remaining(std::uint32_t total) const {
            return total;
        }
        [[nodiscard]] std::span<const ninfer::TokenId> pending_control_tokens() const {
            return {};
        }
    };

    [[nodiscard]] bool is_decode_ready() const { return phase == Phase::Decode; }
    [[nodiscard]] bool is_control_ready() const { return false; }
    [[nodiscard]] bool is_prefilling() const { return phase == Phase::Prefill; }
    [[nodiscard]] bool is_replaying() const { return phase == Phase::Replay; }

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
constexpr std::uint32_t kSlots = 6;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// One iteration's flight membership: every decode row plus the prefill segments the token
// budget admits, with each prefill cap bounded by its lane's own limit.
void test_one_iteration_is_one_mixed_flight() {
    Scheduler scheduler(kSlots);
    ninfer::runtime::LaneTable table;
    table.lanes.resize(kSlots);
    table.lanes[0].decode_ready    = true;
    table.lanes[1].prefilling      = true;
    table.lanes[1].prefill_limit   = 64;
    table.lanes[2].decode_ready    = true;
    table.lanes[3].prefilling      = true;
    table.lanes[3].prefill_limit   = 256;
    table.lanes[4].decode_ready    = true;
    table.lanes[5].prefilling      = true;
    table.lanes[5].prefill_limit   = 8;

    const auto plan = scheduler.plan_iteration(table, ninfer::runtime::IterationBudget{
                                                             .token_budget      = 200,
                                                             .decode_rows_limit = kSlots,
                                                             .prefill_floor_tokens = 0});
    require(plan.decode_lanes == std::vector<std::uint32_t>({0, 2, 4}),
            "a decode row was left out of the iteration's flight");
    require(plan.prefill_token_total() <= 200,
            "the flight's prefill segments exceeded the token budget");
    for (std::size_t i = 0; i < plan.prefill_lanes.size(); ++i) {
        require(plan.prefill_tokens[i] <= table.lanes[plan.prefill_lanes[i]].prefill_limit,
                "a flight segment exceeded its lane's own limit");
    }
    // The budget ran out inside lane 3, so the trailing lane 5 waits for a later iteration.
    require(std::find(plan.prefill_lanes.begin(), plan.prefill_lanes.end(), 5) ==
                plan.prefill_lanes.end(),
            "a lane was admitted past the exhausted budget");
}

// k = 2..6 prefilling lanes behind a permanent decode load: every lane completes, none waits
// more than k iterations behind the decode rows, and each flight respects the token budget.
void test_prefill_lanes_do_not_starve_behind_decode_rows() {
    constexpr std::uint32_t chunk = 128;
    for (std::uint32_t lanes = 2; lanes <= 6; ++lanes) {
        for (const std::uint32_t budget : {32U, 100U, 128U, 256U}) {
            Scheduler scheduler(kSlots);
            ninfer::runtime::LaneTable table;
            table.lanes.resize(kSlots);
            table.lanes[0].decode_ready = true;
            std::array<std::uint32_t, kSlots> left{900, 128, 40, 513, 300, 1024};
            std::array<std::uint32_t, kSlots> last{};
            const auto bounds = ninfer::runtime::IterationBudget{
                .token_budget = budget, .decode_rows_limit = kSlots,
                .prefill_floor_tokens = budget};
            for (std::uint32_t iteration = 1; iteration <= 400; ++iteration) {
                for (std::uint32_t lane = 1; lane < lanes; ++lane) {
                    table.lanes[lane].prefilling    = left[lane] != 0;
                    table.lanes[lane].prefill_limit = std::min(chunk, left[lane]);
                }
                const auto plan = scheduler.plan_iteration(table, bounds);
                require(plan.decode_lanes == std::vector<std::uint32_t>({0}),
                        "a decode row was starved by prefill work");
                require(plan.prefill_token_total() <= budget,
                        "a flight exceeded the token budget between decode rounds");
                for (std::size_t i = 0; i < plan.prefill_lanes.size(); ++i) {
                    const auto lane = plan.prefill_lanes[i];
                    left[lane] -= plan.prefill_tokens[i];
                    if (left[lane] == 0) { table.lanes[lane].prefilling = false; }
                    last[lane] = iteration;
                }
                bool live = false;
                for (std::uint32_t lane = 1; lane < lanes; ++lane) {
                    if (left[lane] == 0) continue;
                    live = true;
                    require(iteration - last[lane] <= lanes, "a prefilling lane was starved");
                }
                if (!live) break;
            }
            for (std::uint32_t lane = 1; lane < lanes; ++lane) {
                require(left[lane] == 0, "a lane did not finish prefill behind decode rows");
            }
        }
    }
}

// Ample budget lets one flight carry several prefill lanes beside the decode rows.
void test_ample_budget_packs_several_lanes_into_one_flight() {
    Scheduler scheduler(4);
    ninfer::runtime::LaneTable table;
    table.lanes.resize(4);
    table.lanes[0].decode_ready  = true;
    table.lanes[1].prefilling    = true;
    table.lanes[1].prefill_limit = 128;
    table.lanes[2].prefilling    = true;
    table.lanes[2].prefill_limit = 128;
    table.lanes[3].prefilling    = true;
    table.lanes[3].prefill_limit = 128;
    const auto plan = scheduler.plan_iteration(table, ninfer::runtime::IterationBudget{
                                                             .token_budget      = 512,
                                                             .decode_rows_limit = 4,
                                                             .prefill_floor_tokens = 0});
    require(plan.prefill_lanes == std::vector<std::uint32_t>({1, 2, 3}) &&
                plan.prefill_token_total() == 384,
            "ample budget still limited the flight to fewer prefill lanes");
}

} // namespace

int main() {
    try {
        test_one_iteration_is_one_mixed_flight();
        test_prefill_lanes_do_not_starve_behind_decode_rows();
        test_ample_budget_packs_several_lanes_into_one_flight();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
