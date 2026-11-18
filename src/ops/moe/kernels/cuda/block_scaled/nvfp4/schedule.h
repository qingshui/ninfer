#pragma once

#include <optional>

namespace ninfer::ops::detail::moe_cuda::block_scaled {

// G16 scale rows are padded to eight bytes for aligned K128 loads.
#if defined(__CUDACC__)
__host__ __device__
#endif
    inline constexpr int
    nvfp4_activation_scale_stride(int features) {
    return ((features + 127) / 128) * 8;
}

enum class Nvfp4Pipeline { K128, K256Deep, K512, K256, K256SingleStage };

// Low bits bypass L1 for weight/input codes; high bits request L2 eviction
// priorities for streaming weights and reused activations. Scales stay cached.
enum class Nvfp4CachePolicy {
    Cached               = 0,
    BypassWeightL1       = 1,
    BypassInputL1        = 2,
    BypassL1             = 3,
    StreamingWeights     = 5,
    StreamingAll         = 7,
    RetainInputs         = 13,
    RetainInputsBypassL1 = 14,
};

// B200 (sm_100a) tcgen05 tile choice for the grouped NVFP4 GEMMs. A tile width is the MMA's N: the
// tokens of one expert that share a tile (16, 32, 64 or 128; 0 lets the b200 plan choose from the
// token count). The gate/up and down projections have separate widths because their accumulators
// differ: gate/up keeps two parts per tile and runs out of tensor memory for a second accumulator
// buffer (the epilogue overlapping the next tile's MMAs) above width 64. `cluster` is the CTA-group
// width and `ctas_per_sm` the persistent grid multiplier; the grouped kernels run one independent
// CTA per SM, so both must be 1.
struct Nvfp4Sm100Tuning {
    int gate_up_tile_tokens = 0;
    int down_tile_tokens    = 0;
    int cluster             = 1;
    int ctas_per_sm         = 1;
};

struct Nvfp4Tuning {
    Nvfp4Pipeline pipeline = Nvfp4Pipeline::K128;
    Nvfp4CachePolicy cache = Nvfp4CachePolicy::Cached;
    int down_row_window    = 0;
    Nvfp4Sm100Tuning sm100;
};

// Defined by the b200 target only (nvfp4/b200/plan.cpp). Returns `requested` with every unset
// field filled from the token count, and rejects a tile or grouping the grouped tcgen05 kernels
// do not provide.
Nvfp4Sm100Tuning resolve_nvfp4_sm100_tuning(const Nvfp4Sm100Tuning& requested, int tokens);

struct Nvfp4LoadSchedule {
    int gate_columns = 0, down_columns = 0;
    int max_load = 0x7fffffff;
    Nvfp4Tuning tuning;
    int queued_ctas_per_sm = 6;
    std::optional<Nvfp4Tuning> down_tuning;
};

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
