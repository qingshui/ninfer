#pragma once

// Warp-specialized persistent tcgen05 GEMM mainloop (swap-AB form).
//
//   D[row, token] = sum_k W[row, k] * X[token, k]
//
// The weight is the MMA's M operand (A, 128 rows per tile) and the activation is its N operand
// (B, `TileN` tokens per tile); both are K-major and arrive through 128B-swizzled TMA tiles.
// Accumulators live in tensor memory, D[lane = weight row, column = token].
//
// One CTA per SM runs six roles (8 warps):
//   warp 0    TMA producer     fills the shared-memory stage ring
//   warp 1    MMA issuer       copies block scales shared -> tensor memory, issues tcgen05.mma
//   warp 2    TMEM allocator   allocates before and frees after the roles run
//   warps 4-7 epilogue         drain the accumulator (one 32-lane TMEM quadrant per warp)
// Stages are guarded by full/empty mbarriers; accumulator buffers by acc_full/acc_empty, so the
// epilogue of tile t overlaps the mainloop of tile t+1 when two accumulator buffers fit.
//
// The epilogue is a type supplied by the caller: it owns the numeric transform and the output
// layout. The mainloop only tells it where the finished tile is.

#include "core/pdl.cuh"
#include "ops/common/cuda/sm100/tcgen05.cuh"
#include "ops/common/cuda/sm100/tile_scheduler.cuh"
#include "ops/common/cuda/sm100/tma.cuh"
#include "ops/common/cuda/sm100/umma_descriptor.cuh"

#include <cuda.h>

#include <cstdint>

namespace ninfer::ops::common::sm100 {

enum class GemmKind { Nvfp4, Bf16, Fp8 };

__host__ __device__ constexpr int pow2_at_least(int value, int minimum) {
    int r = minimum;
    while (r < value) r <<= 1;
    return r;
}

// Compile-time geometry of one GEMM instance.
//   Kind    operand format
//   TileN   tokens per tile (the MMA's N)
//   Parts   weight tiles multiplied against the same activation tile (2 for gate/up SwiGLU)
//   EpilogueWarps  warps that drain the accumulator: four, one per TMEM lane quadrant, or eight,
//                  two per quadrant that each drain half of the tile's columns
template <GemmKind Kind, int TileN, int Parts = 1, int EpilogueWarps = 4>
struct GemmConfig {
    static_assert(TileN == 16 || TileN == 32 || TileN == 64 || TileN == 128 || TileN == 256);
    static_assert(Parts == 1 || Parts == 2);
    static_assert(EpilogueWarps == 4 || EpilogueWarps == 8);

    static constexpr GemmKind kKind    = Kind;
    static constexpr bool kScaled      = Kind == GemmKind::Nvfp4;
    static constexpr int kTileM        = 128;
    static constexpr int kTileN        = TileN;
    static constexpr int kParts        = Parts;
    static constexpr int kEpilogueWarps = EpilogueWarps;
    static constexpr int kThreads      = 32 * (4 + EpilogueWarps);

    // One stage is a 128-byte-wide K slab of every operand: 256 E2M1 values, 128 E4M3 values or
    // 64 BF16 values.
    static constexpr int kSlabBytes    = 128;
    static constexpr int kMmaPerStage  = 4; // 32 bytes of K per instruction
    static constexpr int kMmaKBytes    = 32;
    // TMA box inner extent in elements of the operand's tensor map, and the K step per stage.
    static constexpr int kBoxInner     = Kind == GemmKind::Bf16 ? 64 : 128;

    // Block scales: one 512-byte atom per 128 rows/tokens per 64 K values. A stage holds four.
    static constexpr int kScaleAtomBytes = 512;
    static constexpr int kScaleStageBytes = 4 * kScaleAtomBytes;
    static constexpr int kTokenAtoms   = (TileN + 127) / 128;

    static constexpr int kABytes       = kTileM * kSlabBytes;
    static constexpr int kBBytes       = TileN * kSlabBytes;
    static constexpr int kSfaBytes     = kScaled ? kScaleStageBytes : 0;
    static constexpr int kSfbBytes     = kScaled ? kScaleStageBytes * kTokenAtoms : 0;
    static constexpr int kStageBytes   = Parts * (kABytes + kSfaBytes) + kBBytes + kSfbBytes;
    static_assert(kStageBytes % 1024 == 0);

    static constexpr int kMaxSharedBytes = 227 * 1024;
    static constexpr int kScratchBytes   = kEpilogueWarps * 2048;
    static constexpr int kBarrierBytes   = 512;
    static constexpr int kAlignSlack     = 1024;
    static constexpr int kStagesFit =
        (kMaxSharedBytes - kAlignSlack - kScratchBytes - kBarrierBytes) / kStageBytes;
    static constexpr int kStages = kStagesFit > 8 ? 8 : kStagesFit;
    static_assert(kStages >= 2, "tile does not fit shared memory");

    // Tensor memory: accumulator buffers, then scale-factor copies (double buffered over k-iters,
    // four MMA slots each; one 4-column group per 128 rows of A parts and per token atom).
    static constexpr int kAccCols      = Parts * TileN;
    static constexpr int kSfColsPerMma = kScaled ? 4 * (Parts + kTokenAtoms) : 0;
    static constexpr int kSfColsPerBuf = kMmaPerStage * kSfColsPerMma;
    static constexpr int kSfBufs       = 2;
    static constexpr int kSfCols       = kScaled ? kSfBufs * kSfColsPerBuf : 0;
    static constexpr int kAccStages    = 2 * kAccCols + kSfCols <= kTmemColumns ? 2 : 1;
    static constexpr int kTmemNeeded   = kAccStages * kAccCols + kSfCols;
    static_assert(kTmemNeeded <= kTmemColumns, "tile does not fit tensor memory");
    static constexpr int kTmemAlloc    = pow2_at_least(kTmemNeeded, 32);

    static constexpr int kSharedBytes =
        kStages * kStageBytes + kScratchBytes + kBarrierBytes + kAlignSlack;

    __host__ __device__ static constexpr std::uint32_t idesc() {
        if constexpr (kScaled)
            return make_umma_idesc_mxf4nvf4(kTileM, TileN);
        else if constexpr (Kind == GemmKind::Fp8)
            return make_umma_idesc_f8f6f4(kTileM, TileN, UmmaF8F6F4Format::E4M3,
                                          UmmaF8F6F4Format::E4M3);
        else
            return make_umma_idesc_f16(kTileM, TileN, UmmaF16Format::BF16, UmmaF16Format::BF16);
    }
};

// Runtime arguments shared by every instance.
struct GemmArgs {
    int rows;                 // weight rows
    int tokens;               // activation rows
    int k_iters;              // K / (elements per stage)
    int tiles_m;              // weight tiles (per part)
    int tiles_n;              // token tiles
    int part_row_offset;      // row distance between a tile's parts (SwiGLU: rows / 2)
    int scale_k64;            // K / 64: block-scale atoms per row tile
    const std::uint8_t* a_scales; // weight scales, block_scale_k16_m128x4 layout
    const std::uint8_t* b_scales; // activation scales, K16M128x4 layout, padded to 128 tokens
    // Split K (see StaticTileScheduler): the last `split_tiles` tiles run as `k_splits` K ranges
    // whose FP32 accumulators fill `partials`, one [TileN tokens][128 rows] block per unit.
    int split_tiles = 0;
    int k_splits    = 1;
    float* partials = nullptr;
};

// What an epilogue warp is told about the tile it is draining.
struct EpilogueTile {
    std::uint32_t tmem;   // this warp's lane quadrant, accumulator part 0, column 0
    int part_stride;      // columns between parts
    int m_tile;
    int n_tile;
    int warp_q;           // TMEM quadrant of this warp = row group of 32 inside the 128-row tile
    int lane;
    int valid_tokens;     // tokens of this tile that exist (<= TileN)
    unsigned char* scratch; // 2 KiB private to this warp
};

template <class Cfg>
struct GemmSharedLayout {
    unsigned char* base;

    __device__ unsigned char* stage(int s) const { return base + s * Cfg::kStageBytes; }
    __device__ unsigned char* a(int s, int part) const {
        return stage(s) + part * Cfg::kABytes;
    }
    __device__ unsigned char* b(int s) const { return stage(s) + Cfg::kParts * Cfg::kABytes; }
    __device__ unsigned char* sfa(int s, int part) const {
        return b(s) + Cfg::kBBytes + part * Cfg::kSfaBytes;
    }
    __device__ unsigned char* sfb(int s) const {
        return b(s) + Cfg::kBBytes + Cfg::kParts * Cfg::kSfaBytes;
    }
    __device__ unsigned char* scratch(int warp_q) const {
        return base + Cfg::kStages * Cfg::kStageBytes + warp_q * 2048;
    }
    __device__ std::uint64_t* barriers() const {
        return reinterpret_cast<std::uint64_t*>(base + Cfg::kStages * Cfg::kStageBytes +
                                                Cfg::kScratchBytes);
    }
};

// A split unit's FP32 accumulators into its partial block, element [token][row] of the tile.
template <class Cfg>
__device__ __forceinline__ void drain_partial(const EpilogueTile& t, float* block) {
    constexpr int C = Cfg::kTileN >= 32 ? 32 : Cfg::kTileN;
    float* rows     = block + t.warp_q * 32 + t.lane;
    for (int c0 = 0; c0 < t.valid_tokens; c0 += C) {
        std::uint32_t r[C];
        tmem_load_32x32b<C>(t.tmem + c0, r);
        tmem_load_wait();
#pragma unroll
        for (int i = 0; i < C; ++i) {
            if (c0 + i < t.valid_tokens)
                rows[static_cast<std::int64_t>(c0 + i) * Cfg::kTileM] = __uint_as_float(r[i]);
        }
    }
}

// A SplitK instance runs the units of a split launch (see StaticTileScheduler). Any other instance
// runs every tile over the whole K loop and compiles without the split bookkeeping.
template <class Cfg, class Epilogue, bool SplitK = false>
__global__ __launch_bounds__(Cfg::kThreads, 1) void sm100_gemm_kernel(
    const __grid_constant__ CUtensorMap a_map, const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ GemmArgs args, const __grid_constant__ Epilogue epilogue) {
    extern __shared__ __align__(16) unsigned char shared_raw[];
    const unsigned raw_address = smem_addr(shared_raw);
    GemmSharedLayout<Cfg> shared{shared_raw + (((raw_address + 1023u) & ~1023u) - raw_address)};

    // Barriers: full[S] empty[S] acc_full[A] acc_empty[A], then the TMEM base slot.
    std::uint64_t* full      = shared.barriers();
    std::uint64_t* empty     = full + Cfg::kStages;
    std::uint64_t* acc_full  = empty + Cfg::kStages;
    std::uint64_t* acc_empty = acc_full + Cfg::kAccStages;
    auto* tmem_slot          = reinterpret_cast<std::uint32_t*>(acc_empty + Cfg::kAccStages);

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    if (warp == 0 && elect_one()) {
        tma_prefetch_descriptor(&a_map);
        tma_prefetch_descriptor(&b_map);
        for (int s = 0; s < Cfg::kStages; ++s) {
            mbarrier_init(&full[s], 1);
            mbarrier_init(&empty[s], 1);
        }
        for (int a = 0; a < Cfg::kAccStages; ++a) {
            mbarrier_init(&acc_full[a], 1);
            mbarrier_init(&acc_empty[a], Cfg::kEpilogueWarps);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 2) {
        tmem_alloc<1>(tmem_slot, Cfg::kTmemAlloc);
        tmem_relinquish_alloc_permit<1>();
    }
    __syncthreads();
    fence_after_thread_sync();
    // The reduction kernel of a split launch may take SM slots now; it waits for this grid.
    if constexpr (SplitK) pdl::trigger_dependents();
    const std::uint32_t tmem_base = *tmem_slot;
    const StaticTileScheduler scheduler{args.tiles_m, args.tiles_n, args.k_iters, args.split_tiles,
                                        args.k_splits};
    // Unit `index` and the end of its K range. The whole-K instance takes the bound from
    // args.k_iters rather than from the unit, which keeps its code that of a GEMM without split K.
    const int units    = SplitK ? scheduler.units() : scheduler.total();
    const auto unit_at = [&](int index) {
        return SplitK ? scheduler.unit(index)
                      : WorkUnit{scheduler.coord(index), 0, args.k_iters, -1};
    };
    const auto k_end = [&](const WorkUnit& unit) { return SplitK ? unit.k_end : args.k_iters; };
    constexpr std::uint32_t kSfBase = Cfg::kAccStages * Cfg::kAccCols;

    if (warp == 0) {
        // ---- TMA producer ----
        if (elect_one()) {
            const std::uint64_t hint = l2_retained_policy();
            int stage                = 0;
            std::uint32_t phase      = 0;
            for (int index = scheduler.first(); index < units; index += scheduler.stride()) {
                const WorkUnit unit = unit_at(index);
                const TileCoord c   = unit.tile;
                for (int kb = unit.k_begin; kb < k_end(unit); ++kb) {
                    mbarrier_wait(&empty[stage], phase ^ 1u);
                    mbarrier_arrive_expect_tx(&full[stage], Cfg::kStageBytes);
#pragma unroll
                    for (int p = 0; p < Cfg::kParts; ++p)
                        tma_load_tile_2d(shared.a(stage, p), &a_map, kb * Cfg::kBoxInner,
                                         c.m * Cfg::kTileM + p * args.part_row_offset,
                                         &full[stage], hint);
                    tma_load_tile_2d(shared.b(stage), &b_map, kb * Cfg::kBoxInner,
                                     c.n * Cfg::kTileN, &full[stage], hint);
                    if constexpr (Cfg::kScaled) {
#pragma unroll
                        for (int p = 0; p < Cfg::kParts; ++p) {
                            const std::int64_t atom =
                                static_cast<std::int64_t>(
                                    (c.m * Cfg::kTileM + p * args.part_row_offset) / 128) *
                                    args.scale_k64 +
                                kb * 4;
                            bulk_load(shared.sfa(stage, p), args.a_scales + atom * 512,
                                      Cfg::kScaleStageBytes, &full[stage]);
                        }
#pragma unroll
                        for (int t = 0; t < Cfg::kTokenAtoms; ++t) {
                            const std::int64_t atom =
                                static_cast<std::int64_t>(c.n * (Cfg::kTileN / 128 > 0
                                                                     ? Cfg::kTileN / 128
                                                                     : 1) +
                                                          t) *
                                    args.scale_k64 +
                                kb * 4;
                            bulk_load(shared.sfb(stage) + t * Cfg::kScaleStageBytes,
                                      args.b_scales + atom * 512, Cfg::kScaleStageBytes,
                                      &full[stage]);
                        }
                    }
                    if (++stage == Cfg::kStages) {
                        stage = 0;
                        phase ^= 1u;
                    }
                }
            }
        }
    } else if (warp == 1) {
        // ---- MMA issuer ----
        if (elect_one()) {
            constexpr std::uint32_t idesc = Cfg::idesc();
            int stage                     = 0;
            std::uint32_t phase           = 0;
            int acc                       = 0;
            std::uint32_t acc_phase       = 0;
            std::uint32_t k_count         = 0;
            for (int index = scheduler.first(); index < units; index += scheduler.stride()) {
                const WorkUnit unit = unit_at(index);
                mbarrier_wait(&acc_empty[acc], acc_phase ^ 1u);
                fence_after_thread_sync();
                for (int kb = unit.k_begin; kb < k_end(unit); ++kb) {
                    mbarrier_wait(&full[stage], phase);
                    fence_after_thread_sync();
                    const std::uint32_t sf_buf =
                        tmem_base + kSfBase + (k_count & 1u) * Cfg::kSfColsPerBuf;
                    if constexpr (Cfg::kScaled) {
#pragma unroll
                        for (int j = 0; j < Cfg::kMmaPerStage; ++j) {
                            const std::uint32_t sf_j = sf_buf + j * Cfg::kSfColsPerMma;
#pragma unroll
                            for (int p = 0; p < Cfg::kParts; ++p)
                                tmem_copy_32x128b_warpx4<1>(
                                    sf_j + p * 4,
                                    make_umma_scale_desc(smem_addr(shared.sfa(stage, p) +
                                                                   j * Cfg::kScaleAtomBytes)));
#pragma unroll
                            for (int t = 0; t < Cfg::kTokenAtoms; ++t)
                                tmem_copy_32x128b_warpx4<1>(
                                    sf_j + (Cfg::kParts + t) * 4,
                                    make_umma_scale_desc(smem_addr(
                                        shared.sfb(stage) + t * Cfg::kScaleStageBytes +
                                        j * Cfg::kScaleAtomBytes)));
                        }
                    }
                    const std::uint64_t b_desc0 =
                        make_umma_k_major_desc(smem_addr(shared.b(stage)), UmmaSwizzle::B128);
#pragma unroll
                    for (int j = 0; j < Cfg::kMmaPerStage; ++j) {
#pragma unroll
                        for (int p = 0; p < Cfg::kParts; ++p) {
                            const std::uint64_t a_desc = umma_desc_advance(
                                make_umma_k_major_desc(smem_addr(shared.a(stage, p)),
                                                       UmmaSwizzle::B128),
                                j * Cfg::kMmaKBytes);
                            const std::uint64_t b_desc =
                                umma_desc_advance(b_desc0, j * Cfg::kMmaKBytes);
                            const std::uint32_t d =
                                tmem_base + acc * Cfg::kAccCols + p * Cfg::kTileN;
                            const bool accumulate = ((kb - unit.k_begin) | j) != 0;
                            if constexpr (Cfg::kScaled) {
                                const std::uint32_t sf_j = sf_buf + j * Cfg::kSfColsPerMma;
                                umma_mxf4nvf4_ss<1>(d, a_desc, b_desc, idesc, sf_j + p * 4,
                                                    sf_j + Cfg::kParts * 4, accumulate);
                            } else if constexpr (Cfg::kKind == GemmKind::Fp8) {
                                umma_f8f6f4_ss<1>(d, a_desc, b_desc, idesc, accumulate);
                            } else {
                                umma_f16_ss<1>(d, a_desc, b_desc, idesc, accumulate);
                            }
                        }
                    }
                    umma_commit<1>(&empty[stage]);
                    ++k_count;
                    if (++stage == Cfg::kStages) {
                        stage = 0;
                        phase ^= 1u;
                    }
                }
                umma_commit<1>(&acc_full[acc]);
                if (++acc == Cfg::kAccStages) {
                    acc = 0;
                    acc_phase ^= 1u;
                }
            }
        }
    } else if (warp >= 4) {
        // ---- Epilogue warps ----
        const int q               = warp - 4;
        int acc                   = 0;
        std::uint32_t acc_phase   = 0;
        for (int index = scheduler.first(); index < units; index += scheduler.stride()) {
            const WorkUnit unit = unit_at(index);
            const TileCoord c   = unit.tile;
            mbarrier_wait(&acc_full[acc], acc_phase);
            fence_after_thread_sync();
            const int remaining = args.tokens - c.n * Cfg::kTileN;
            EpilogueTile t{tmem_address(tmem_base, q * 32, acc * Cfg::kAccCols),
                           Cfg::kTileN,
                           c.m,
                           c.n,
                           q,
                           lane,
                           remaining < Cfg::kTileN ? remaining : Cfg::kTileN,
                           shared.scratch(q)};
            if (!SplitK || unit.partial < 0)
                epilogue.template drain<Cfg>(t);
            else
                drain_partial<Cfg>(t, args.partials + static_cast<std::int64_t>(unit.partial) *
                                                          Cfg::kTileM * Cfg::kTileN);
            fence_before_thread_sync();
            __syncwarp();
            if (lane == 0) mbarrier_arrive(&acc_empty[acc]);
            if (++acc == Cfg::kAccStages) {
                acc = 0;
                acc_phase ^= 1u;
            }
        }
    }

    __syncthreads();
    if (warp == 2) {
        fence_after_thread_sync();
        tmem_dealloc<1>(tmem_base, Cfg::kTmemAlloc);
    }
}

} // namespace ninfer::ops::common::sm100
