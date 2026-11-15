#include "ops/moe/kernels/cuda/block_scaled/nvfp4/schedule.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::moe_cuda::block_scaled {
namespace {

// Tokens of one expert per tile. A 122B call routes tokens * 8 assignments over 256 experts, so an
// expert sees about tokens / 32 of them; the tile is the smallest MMA N that keeps one tile per
// typical expert. Narrow tiles waste no weight bandwidth (the weights stream once per tile row),
// wide tiles amortize the per-tile epilogue.
int default_tile_tokens(int tokens) {
    if (tokens <= 256) return 16;
    if (tokens <= 1024) return 32;
    if (tokens <= 2048) return 64;
    return 128;
}

} // namespace

Nvfp4Sm100Tuning resolve_nvfp4_sm100_tuning(const Nvfp4Sm100Tuning& requested, int tokens) {
    Nvfp4Sm100Tuning tuning = requested;
    if (tuning.tile_tokens == 0) tuning.tile_tokens = default_tile_tokens(tokens);
    if (tuning.tile_tokens != 16 && tuning.tile_tokens != 32 && tuning.tile_tokens != 64 &&
        tuning.tile_tokens != 128)
        throw std::invalid_argument("nvfp4 moe tcgen05: tile_tokens must be 16, 32, 64 or 128, got " +
                                    std::to_string(tuning.tile_tokens));
    if (tuning.cluster != 1)
        throw std::invalid_argument("nvfp4 moe tcgen05: cluster must be 1, got " +
                                    std::to_string(tuning.cluster));
    if (tuning.ctas_per_sm != 1)
        throw std::invalid_argument("nvfp4 moe tcgen05: ctas_per_sm must be 1, got " +
                                    std::to_string(tuning.ctas_per_sm));
    return tuning;
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
