#pragma once

// Persistent static tile scheduler for the SM100 GEMM mainloop.
//
// Tiles are numbered weight-tile major with the token tile varying fastest, so the CTAs that run
// together share one weight tile (read once from memory, then from L2) and walk the activation
// tiles. A launch may cut its last `split_tiles` tiles along K: each of them runs as `k_splits`
// units over consecutive K ranges, and split unit u accumulates FP32 partial block u for a later
// reduction. The other tiles each run as one unit over the whole K loop. Units are numbered tile
// by tile, whole tiles first, and CTA i of a G-CTA grid owns units i, i + G, i + 2G, ... Producer,
// MMA and epilogue roles each walk the same sequence, which is what keeps their barrier phases in
// step.

#include <cstdint>

namespace ninfer::ops::common::sm100 {

struct TileCoord {
    int m; // weight-row tile
    int n; // token tile
};

struct WorkUnit {
    TileCoord tile;
    int k_begin; // K iterations [k_begin, k_end)
    int k_end;
    int partial; // partial block of a split unit; -1 for a whole tile
};

struct StaticTileScheduler {
    int tiles_m;
    int tiles_n;
    int k_iters;
    int split_tiles = 0;
    int k_splits    = 1;

    __host__ __device__ constexpr int total() const { return tiles_m * tiles_n; }
    __host__ __device__ constexpr int whole_tiles() const { return total() - split_tiles; }
    __host__ __device__ constexpr int units() const {
        return whole_tiles() + split_tiles * k_splits;
    }

    __device__ __forceinline__ int first() const { return static_cast<int>(blockIdx.x); }
    __device__ __forceinline__ int stride() const { return static_cast<int>(gridDim.x); }

    __host__ __device__ constexpr TileCoord coord(int tile) const {
        const int m = tile / tiles_n;
        return {m, tile - m * tiles_n};
    }

    __host__ __device__ constexpr WorkUnit unit(int index) const {
        if (index < whole_tiles()) return {coord(index), 0, k_iters, -1};
        const int split = index - whole_tiles();
        const int part  = split % k_splits;
        return {coord(whole_tiles() + split / k_splits), part * k_iters / k_splits,
                (part + 1) * k_iters / k_splits, split};
    }
};

// CTAs to launch: one per SM, never more than there are tiles.
inline int persistent_grid(int tiles, int sm_count, int ctas_per_sm = 1) {
    const int resident = sm_count * ctas_per_sm;
    return tiles < resident ? tiles : resident;
}

} // namespace ninfer::ops::common::sm100
