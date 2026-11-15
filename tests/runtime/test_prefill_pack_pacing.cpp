// The packed-prefill capacity contract, through the public prefill-chunk API alone. The pacing
// half of this file moved into the iteration budget: the token bound of one packed flight is the
// iteration budget the Engine passes plan_iteration, and the segment bound comes from the
// Session's prefill_pack_limit(); what remains here is the pack-capacity contract itself.
#include "models/qwen3_5/prefill_chunk.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

// Pack capacity is >= one window, 128-aligned, and never beyond the token-axis capacity. The
// pack budget is decoupled from the lane count: the segment bound comes from
// kPrefillPackSegmentLimit alone, and the token bound makes one merged flight of the resident
// chunk reach 8192-16384 tokens.
void test_pack_capacity_bounds() {
    namespace q = ninfer::models::qwen3_5;
    require(q::kPrefillPackSegmentLimit >= 2, "segment budget fits a mixed flight");
    const auto flight = q::kResidentPrefillChunk * q::kPrefillPackChunks;
    require(flight >= 8192 && flight <= 16384, "one flight carries 8192-16384 resident tokens");
    for (const std::uint32_t chunk : {128u, 512u, 2048u, 4096u}) {
        for (const std::uint32_t capacity : {chunk, 262144u}) {
            const auto p = q::resolve_prefill_pack_tokens(chunk, capacity);
            require(p % 128 == 0, "pack tokens is a multiple of 128");
            require(p >= chunk, "pack tokens holds at least one chunk");
            if (capacity >= chunk)
                require(p <= capacity / 128 * 128,
                        "pack tokens bounded by the token-axis capacity");
        }
    }
    require(q::resolve_prefill_pack_tokens(128, 128) == 128,
            "pack == chunk when capacity == chunk");
    require(q::resolve_prefill_pack_tokens(2048, 262144) ==
                q::kPrefillPackChunks * 2048,
            "an unbounded capacity takes the full chunk budget");
}

} // namespace

int main() {
    try {
        test_pack_capacity_bounds();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
