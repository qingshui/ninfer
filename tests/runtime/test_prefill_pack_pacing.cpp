// K1 / F5: the scheduling side of the packed-prefill contract, using only the public Scheduler
// and prefill-chunk API. Segments of one packed execution are the lanes the scheduler admits for
// a cycle; their summed token caps must respect the pack capacity, and a lane that does not fit
// falls back to fewer segments (or to a single, truncated one).
#include "models/qwen3_5/prefill_chunk.h"
#include "runtime/engine/scheduler.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>

namespace {

struct Request {
    using SequenceHandle = std::uint64_t;
    enum class Phase { Waiting, Materializing, Prefill, Replay, Decode, Control, Finished };

    [[nodiscard]] bool is_prefilling() const { return phase == Phase::Prefill; }
    [[nodiscard]] bool is_replaying() const { return phase == Phase::Replay; }

    std::uint64_t id     = 0;
    bool capture_pending = false;
    Phase phase      = Phase::Prefill;
    std::optional<SequenceHandle> sequence;
};

using Scheduler = ninfer::runtime::Scheduler<Request>;
using Slots     = std::array<std::shared_ptr<Request>, ninfer::kMaximumConcurrency>;
using ninfer::runtime::PrefillPacing;
using ninfer::runtime::PrefillRegime;
using ninfer::runtime::PrefillUnitTokens;

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

Slots prefilling_lanes(std::uint32_t n) {
    Slots slots{};
    for (std::uint32_t i = 0; i < n; ++i) {
        slots[i]           = std::make_shared<Request>();
        slots[i]->id       = i + 1;
        slots[i]->sequence = i + 1;
    }
    return slots;
}

constexpr std::uint32_t kChunk = 2048;
constexpr std::uint32_t kPack  = 4096;

PrefillPacing pacing() {
    PrefillPacing p;
    p.throughput_tokens_per_cycle = kPack;
    p.throughput_units_per_cycle  = 6;
    p.interleave_ns               = 100'000'000;
    p.interleave_tokens_per_cycle = kChunk;
    p.interleave_cold_tokens      = 512;
    p.unit_granularity            = 128;
    return p;
}

// Segments never sum past the pack capacity; capacity exhaustion drops trailing lanes.
void test_throughput_membership_fits_pack() {
    Scheduler scheduler;
    scheduler.configure_prefill(pacing());
    scheduler.begin_prefill_cycle(false);
    require(scheduler.prefill_regime() == PrefillRegime::Throughput, "regime");
    require(scheduler.prefill_tokens_remaining() == kPack, "throughput token allowance");

    const auto slots = prefilling_lanes(3);
    PrefillUnitTokens limits{};
    limits[0] = limits[1] = limits[2] = kChunk;
    const auto caps                   = scheduler.prefill_unit_caps(limits);
    const auto members                = scheduler.prefill_candidates(slots, 3, caps);
    require(!members.empty() && members.size < 3, "capacity must reduce the segment count");
    std::uint64_t sum = 0;
    for (auto lane : members.lane_span()) {
        require(caps[lane] > 0 && caps[lane] <= limits[lane], "segment cap out of range");
        sum += caps[lane];
    }
    require(sum <= kPack, "segments exceed the pack capacity");
    const auto next = scheduler.next_prefill(slots, 3, caps);
    require(next && *next == members.lane_span().front(), "next_prefill != first member");
}

// A request longer than the pack still gets a nonzero, truncated unit: single-segment fallback.

// Executed units are charged to the cycle allowance; a drained allowance admits nothing more.
void test_charging_and_regimes() {
    Scheduler scheduler;
    scheduler.configure_prefill(pacing());
    scheduler.begin_prefill_cycle(false);
    scheduler.prefill_executed(0, 3, kChunk, 1'000'000);
    require(scheduler.prefill_tokens_remaining() == kPack - kChunk, "tokens not charged");
    scheduler.prefill_executed(1, 3, kChunk, 1'000'000);
    require(scheduler.prefill_tokens_remaining() == 0, "tokens not drained");
    PrefillUnitTokens limits{};
    limits[2]       = kChunk;
    const auto caps = scheduler.prefill_unit_caps(limits);
    require(caps[2] == 0, "drained allowance still admits a unit");

    scheduler.begin_prefill_cycle(true);
    require(scheduler.prefill_regime() == PrefillRegime::Interleave, "interleave regime");
    require(scheduler.prefill_tokens_remaining() == kChunk, "interleave token allowance");
    require(scheduler.prefill_ns_remaining() == 100'000'000, "interleave ns allowance");
    scheduler.prefill_executed(0, 3, 256, 10'000'000);
    require(scheduler.prefill_ns_remaining() == 90'000'000, "ns not charged");
}


// Pack capacity is >= one window, 128-aligned, and never beyond the context.


} // namespace

int main() {
    try {
        test_throughput_membership_fits_pack();
        test_charging_and_regimes();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
