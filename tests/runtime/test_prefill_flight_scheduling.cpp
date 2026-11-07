// T5 / F3: multi-request prefill scheduling, through Scheduler's public interface only.
// A "flight" is the set of lanes returned by prefill_candidates(); the Engine commits it once
// and reports each segment to prefill_executed() as one unit.
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
    enum class Phase { Waiting, Prefill, Replay, Decode };

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
using Slots      = std::array<RequestPtr, ninfer::kMaximumConcurrency>;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

RequestPtr request(std::uint64_t id, Request::Phase phase = Request::Phase::Waiting) {
    auto value      = std::make_shared<Request>();
    value->id       = id;
    value->phase    = phase;
    value->sequence = id;
    return value;
}

using ninfer::runtime::PrefillPacing;

// Runs one Throughput cycle as flights: caps for the lanes, one candidate list per flight, one
// prefill_executed per segment. The wall clock of a flight is shared by its segments in
// proportion to their tokens. Returns tokens executed in the cycle. `left` is each lane's
// remaining prompt.
std::uint32_t run_cycle_as_flights(Scheduler& scheduler, Slots& slots, std::uint32_t lanes,
                                   std::uint32_t chunk, std::array<std::uint32_t, 6>& left,
                                   std::array<std::uint32_t, 6>& last_progress,
                                   std::uint32_t cycle, std::uint32_t& flights) {
    scheduler.begin_prefill_cycle(false);
    std::uint32_t executed = 0;
    for (;;) {
        ninfer::runtime::PrefillUnitTokens limits{};
        for (std::uint32_t lane = 0; lane < lanes; ++lane)
            if (slots[lane]) limits[lane] = std::min(chunk, left[lane]);
        const auto caps   = scheduler.prefill_unit_caps(limits);
        const auto flight = scheduler.prefill_candidates(slots, lanes, caps);
        if (flight.empty()) break;
        ++flights;
        std::uint32_t flight_tokens = 0;
        for (const std::uint32_t lane : flight.lane_span()) flight_tokens += caps[lane];
        for (const std::uint32_t lane : flight.lane_span()) {
            const std::uint32_t done = caps[lane];
            left[lane] -= done;
            executed += done;
            last_progress[lane] = cycle;
            scheduler.prefill_executed(lane, lanes, done,
                                       std::uint64_t{flight_tokens} * 1000 * done / flight_tokens);
            if (left[lane] == 0) slots[lane].reset();
        }
    }
    return executed;
}

// Each segment of a flight is charged as one unit against both budgets.

// k = 2..6 prefilling lanes, flights per cycle: every lane completes, none waits > k cycles,
// and prefill work between two decode rounds respects the Throughput token allowance.
void test_flights_do_not_starve_and_gap_is_bounded() {
    constexpr std::uint32_t chunk = 128;
    for (std::uint32_t lanes = 2; lanes <= 6; ++lanes) {
        for (const std::uint32_t budget : {32U, 100U, 128U, 256U, 512U}) {
            Scheduler scheduler;
            scheduler.configure_prefill(PrefillPacing{.throughput_tokens_per_cycle = budget});
            Slots slots{};
            std::array<std::uint32_t, 6> left{900, 128, 40, 513, 300, 1024};
            std::array<std::uint32_t, 6> last{};
            for (std::uint32_t lane = 0; lane < lanes; ++lane)
                slots[lane] = request(lane + 1, Request::Phase::Prefill);
            std::uint32_t flights = 0;
            for (std::uint32_t cycle = 1; cycle <= 400; ++cycle) {
                const auto executed =
                    run_cycle_as_flights(scheduler, slots, lanes, chunk, left, last, cycle, flights);
                require(executed <= budget,
                        "prefill work between decode rounds exceeded the contract bound");
                bool live = false;
                for (std::uint32_t lane = 0; lane < lanes; ++lane) {
                    if (!slots[lane]) continue;
                    live = true;
                    require(cycle - last[lane] <= lanes, "a prefilling lane was starved");
                }
                if (!live) break;
            }
            for (std::uint32_t lane = 0; lane < lanes; ++lane)
                require(!slots[lane] && left[lane] == 0, "a lane did not finish prefill");
        }
    }
}

// Concurrent lanes advance together in a cycle when the budget allows it.
void test_budget_lets_several_lanes_advance_per_cycle() {
    Scheduler scheduler;
    scheduler.configure_prefill(PrefillPacing{.throughput_tokens_per_cycle = 512});
    Slots slots{};
    for (std::uint32_t lane = 0; lane < 4; ++lane)
        slots[lane] = request(lane + 1, Request::Phase::Prefill);
    ninfer::runtime::PrefillUnitTokens limits{};
    limits.fill(128);
    const auto caps   = scheduler.prefill_unit_caps(limits);
    const auto flight = scheduler.prefill_candidates(slots, 4, caps);
    require(flight.size == 4, "ample budget still limited the cycle to fewer lanes");
}

// Several waiting requests are offered in one scan; FIFO head leads and bypass credit holds.

} // namespace

int main() {
    try {
        test_flights_do_not_starve_and_gap_is_bounded();
        test_budget_lets_several_lanes_advance_per_cycle();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
