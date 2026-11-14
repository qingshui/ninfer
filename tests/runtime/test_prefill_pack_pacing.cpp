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
// Lane slot tables are runtime-sized; tests carry their own capacity.
constexpr std::uint32_t kSlots = 3;
using Slots     = std::array<std::shared_ptr<Request>, kSlots>;
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
    Scheduler scheduler(kSlots);
    scheduler.configure_prefill(pacing());
    scheduler.begin_prefill_cycle(false);
    require(scheduler.prefill_regime() == PrefillRegime::Throughput, "regime");
    require(scheduler.prefill_tokens_remaining() == kPack, "throughput token allowance");

    const auto slots = prefilling_lanes(3);
    PrefillUnitTokens limits(kSlots, 0);
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
void test_long_request_truncated_unit() {
    Scheduler scheduler(kSlots);
    scheduler.configure_prefill(pacing());
    scheduler.begin_prefill_cycle(false);
    PrefillUnitTokens limits(kSlots, 0);
    limits[0]         = 4 * kPack;
    const auto caps   = scheduler.prefill_unit_caps(limits);
    require(caps[0] == kPack, "a window longer than the pack is truncated to the pack");
    const auto slots = prefilling_lanes(1);
    const auto members = scheduler.prefill_candidates(slots, kSlots, caps);
    require(!members.empty() && members.size == 1, "a truncated window runs as a single segment");
    require(caps[members.lane_span().front()] > 0, "the truncated unit is nonzero");
}

// Executed units are charged to the cycle allowance; a drained allowance admits nothing more.
void test_charging_and_regimes() {
    Scheduler scheduler(kSlots);
    scheduler.configure_prefill(pacing());
    scheduler.begin_prefill_cycle(false);
    scheduler.prefill_executed(0, 3, kChunk, 1'000'000);
    require(scheduler.prefill_tokens_remaining() == kPack - kChunk, "tokens not charged");
    scheduler.prefill_executed(1, 3, kChunk, 1'000'000);
    require(scheduler.prefill_tokens_remaining() == 0, "tokens not drained");
    PrefillUnitTokens limits(kSlots, 0);
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


// Pack capacity is >= one window, 128-aligned, and never beyond the context. The pack budget is
// decoupled from the lane count: the segment bound comes from kPrefillPackSegmentLimit alone, and
// the token bound makes one merged flight of the resident chunk reach 2048-8192 tokens.
void test_pack_capacity_bounds() {
    namespace q = ninfer::models::qwen3_5;
    require(q::kPrefillPackSegmentLimit >= 2, "segment budget fits a mixed flight");
    const auto flight = q::kResidentPrefillChunk * q::kPrefillPackChunks;
    require(flight >= 2048 && flight <= 8192, "one flight carries 2048-8192 resident tokens");
    for (const std::uint32_t chunk : {128u, 512u, 2048u, 4096u}) {
        for (const std::uint32_t ctx : {chunk, 262144u}) {
            const auto p = q::resolve_prefill_pack_tokens(chunk, ctx);
            require(p % 128 == 0, "pack tokens is a multiple of 128");
            require(p >= chunk, "pack tokens holds at least one chunk");
            if (ctx >= chunk)
                require(p <= (ctx + 127) / 128 * 128, "pack tokens bounded by the context");
        }
    }
    require(q::resolve_prefill_pack_tokens(128, 128) == 128, "pack == chunk when context == chunk");
    require(q::resolve_prefill_pack_tokens(2048, 262144) ==
                q::kPrefillPackChunks * 2048,
            "an unbounded context takes the full chunk budget");
}

} // namespace

int main() {
    try {
        test_throughput_membership_fits_pack();
        test_long_request_truncated_unit();
        test_charging_and_regimes();
        test_pack_capacity_bounds();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
