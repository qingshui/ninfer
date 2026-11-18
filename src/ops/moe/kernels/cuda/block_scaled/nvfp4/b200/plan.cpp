#include "ops/moe/kernels/cuda/block_scaled/nvfp4/schedule.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::moe_cuda::block_scaled {
namespace {

// Tokens of one expert per tile. A 122B call routes tokens * 8 assignments over 256 experts, so an
// expert sees about tokens / 32 of them. Narrow tiles waste no weight bandwidth (the weights
// stream once per tile row), wide tiles amortize the per-tile epilogue and re-read the weights
// fewer times. Gate/up stops at 64: its two accumulator parts leave no tensor memory for a second
// accumulator buffer at 128, and the exposed epilogue costs more than the weights it saves.
int default_gate_up_tile_tokens(int tokens) {
    if (tokens <= 768) return 32;
    return 64;
}

int default_down_tile_tokens(int tokens) {
    if (tokens <= 256) return 16;
    if (tokens <= 768) return 32;
    if (tokens <= 1536) return 64;
    return 128;
}

void check_tile(int tile_tokens, const char* projection) {
    if (tile_tokens != 16 && tile_tokens != 32 && tile_tokens != 64 && tile_tokens != 128)
        throw std::invalid_argument(std::string("nvfp4 moe tcgen05: ") + projection +
                                    " tile_tokens must be 16, 32, 64 or 128, got " +
                                    std::to_string(tile_tokens));
}

} // namespace

Nvfp4Sm100Tuning resolve_nvfp4_sm100_tuning(const Nvfp4Sm100Tuning& requested, int tokens) {
    Nvfp4Sm100Tuning tuning = requested;
    if (tuning.gate_up_tile_tokens == 0)
        tuning.gate_up_tile_tokens = default_gate_up_tile_tokens(tokens);
    if (tuning.down_tile_tokens == 0) tuning.down_tile_tokens = default_down_tile_tokens(tokens);
    check_tile(tuning.gate_up_tile_tokens, "gate/up");
    check_tile(tuning.down_tile_tokens, "down");
    if (tuning.cluster != 1)
        throw std::invalid_argument("nvfp4 moe tcgen05: cluster must be 1, got " +
                                    std::to_string(tuning.cluster));
    if (tuning.ctas_per_sm != 1)
        throw std::invalid_argument("nvfp4 moe tcgen05: ctas_per_sm must be 1, got " +
                                    std::to_string(tuning.ctas_per_sm));
    return tuning;
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
