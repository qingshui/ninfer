#pragma once

// Persistent static tile scheduler for the SM100 GEMM mainloop.
//
// Tiles are numbered weight-tile major with the token tile varying fastest, so the CTAs that run
// together share one weight tile (read once from memory, then from L2) and walk the activation
// tiles. CTA i of a G-CTA grid owns tiles i, i + G, i + 2G, ... Producer, MMA and epilogue roles
// each walk the same sequence, which is what keeps their barrier phases in step.

#include <cstdint>

namespace ninfer::ops::common::sm100 {

struct TileCoord {
    int m; // weight-row tile
    int n; // token tile
};

struct StaticTileScheduler {
    int tiles_m;
    int tiles_n;

    __host__ __device__ constexpr int total() const { return tiles_m * tiles_n; }

    __device__ __forceinline__ int first() const { return static_cast<int>(blockIdx.x); }
    __device__ __forceinline__ int stride() const { return static_cast<int>(gridDim.x); }

    __host__ __device__ constexpr TileCoord coord(int tile) const {
        const int m = tile / tiles_n;
        return {m, tile - m * tiles_n};
    }
};

// CTAs to launch: one per SM, never more than there are tiles.
inline int persistent_grid(int tiles, int sm_count, int ctas_per_sm = 1) {
    const int resident = sm_count * ctas_per_sm;
    return tiles < resident ? tiles : resident;
}

} // namespace ninfer::ops::common::sm100
