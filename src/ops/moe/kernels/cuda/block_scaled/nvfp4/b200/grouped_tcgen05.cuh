#pragma once

// Grouped tcgen05 GEMMs for the Qwen3.5-122B NVFP4 routed experts on B200 (sm_100a).
//
// Tokens arrive grouped by expert (Groups: offsets/assignments). One persistent CTA per SM walks
// the (expert, weight row tile, token tile) space and runs the swap-AB GEMM
//
//     D[weight row, token] = sum_k W_e[row, k] * X[token, k]
//
// with the expert's weight rows as the MMA's M operand and up to TileN tokens as its N operand.
// The weight operands are consumed exactly as the artifact stores them: NVFP4 codes through a 3D
// TMA map over the expert-major bank and block_scale_k16_m128x4 scale atoms through bulk copies.
//
// Two kernels share the mainloop:
//   gate/up  B = quantized hidden states in packed-slot order (the launcher gathers them by
//            token id first). Two weight tiles
//            (gate rows, up rows) accumulate side by side; the epilogue computes
//            silu(gate) * up in FP32 and re-quantizes it to NVFP4 (G16, calibrated divisor) into
//            the packed-assignment-ordered product buffer.
//   down     B = that product buffer, contiguous per expert. The epilogue scales the FP32
//            accumulator and either atomically adds coefficient * value into the per-token sum or
//            stores it per assignment for the ordered reduction.
//
// Block scales of the activation operand are written in smem by a dedicated warp from the
// slot-major scale rows, in the 128-token K16M128x4 atom order tcgen05.cp expects.
//
// Roles (8 or 12 warps): 0 TMA producer, 1 MMA issuer, 2 TMEM allocator, 3 activation-scale
// loader, then 4 or 8 epilogue warps. A warp reaches the 32 TMEM lanes of quadrant (warp % 4); with
// eight epilogue warps the two warps of a quadrant drain the first and second half of the tile's
// token columns. Stages are guarded by full/empty mbarriers and
// accumulator buffers by acc_full/acc_empty, as in the dense mainloop.
//
// Each epilogue warp drains with one thread per accumulator lane, and its global loads queue
// behind the reductions it already issued. An epilogue therefore splits what it needs to know
// about a tile into `fetch` (independent loads) and `settle` (the loads and math that depend on
// them); both run for tile n + 1 while tile n drains, so `drain` finds them in registers.

#include "core/device.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/sm100/gemm_mainloop.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/nvfp4.cuh"

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4 {

namespace sm = ::ninfer::ops::common::sm100;


// Everything the mainloop needs besides the tensor maps. Pointers address the Groups arrays and
// the weight bank; none of it is repacked.
struct GroupedArgs {
    const int* offsets;            // [experts + 1] packed assignment offsets
    const int* assignments;        // [A] packed slot -> assignment id (token * top_k + k)
    const std::uint8_t* a_scales;  // weight scale plane of expert 0, block_scale_k16_m128x4
    std::uint64_t a_scale_stride;  // bytes between experts
    const std::uint8_t* b_scales;  // activation scales, token-major rows
    int b_scale_row_bytes;         // bytes per activation scale row (K / 16)
    int experts, top_k;
    int tiles_m;                   // weight row tiles per part
    int k_iters;                   // K / 256
    int scale_k64;                 // K / 64
    int part_row_offset;           // rows between the parts of one tile (gate -> up)
};

// What an epilogue warp is told about the tile it drains.
struct GroupTile {
    std::uint32_t tmem; // this warp's lane quadrant, accumulator part 0, column 0
    int part_stride;    // TMEM columns between parts
    int expert;
    int m;              // weight row tile
    int start;          // packed slot of the tile's first token
    int valid;          // tokens of the tile that exist (<= TileN)
    int warp_q;         // quadrant = 32-row group inside the 128-row tile
    int lane;
    unsigned char* scratch; // 2 KiB private to this warp
    int col_begin;      // first accumulator column this warp drains
    int col_end;        // one past the last
};

struct TileInfo {
    int expert, m, start, valid;
};

// Tiles are numbered expert-major, then weight row tile, then token tile (the token tile varies
// fastest so CTAs working on one weight tile share it through L2). Every role advances through the
// same increasing tile sequence, so the expert of its previous tile bounds the search for the next.
struct TileCursor {
    int expert = 0;
};

// The expert offsets and the first tile of every expert are staged in shared memory. A role finds
// a tile's expert by a binary search over the tile starts: walking the experts one by one costs a
// dependent shared load per expert, and with few routed tokens most of the 256 experts are empty,
// so that walk delayed every role by microseconds.
inline constexpr int kMaxExperts = 256;

template <int TileN>
__device__ __forceinline__ bool next_tile(TileCursor& cursor, int tile, const GroupedArgs& args,
                                          const int* offsets, const int* tile_start,
                                          TileInfo& info) {
    if (tile >= tile_start[args.experts]) return false;
    // tile_start[lo] <= tile < tile_start[hi]; expert `lo` ends up owning the tile.
    int lo = cursor.expert, hi = args.experts;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (tile_start[mid] <= tile) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    cursor.expert    = lo;
    const int begin  = offsets[lo];
    const int end    = offsets[lo + 1];
    const int ntiles = (end - begin + TileN - 1) / TileN;
    const int u      = tile - tile_start[lo];
    const int m      = u / ntiles;
    const int start  = begin + (u - m * ntiles) * TileN;
    info             = {lo, m, start, min(TileN, end - start)};
    return true;
}

// ---------------------------------------------------------------------------------------------
// TMA form the dense mainloop does not need: 3D expert-bank tiles.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ void tma_load_tile_3d(void* destination, const CUtensorMap* map, int c0,
                                                 int c1, int c2, std::uint64_t* barrier,
                                                 std::uint64_t cache_hint) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 ".L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
                 :
                 : "r"(common::smem_addr(destination)), "l"(map),
                   "r"(common::smem_addr(barrier)), "r"(c0), "r"(c1), "r"(c2), "l"(cache_hint)
                 : "memory");
}

// Adds four consecutive floats (16-byte aligned) in one L2 reduction.
__device__ __forceinline__ void red_add_v4(float* address, float x, float y, float z, float w) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(address), "f"(x), "f"(y),
                 "f"(z), "f"(w));
}

// ---------------------------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------------------------

template <class Cfg, class Epilogue>
__global__ __launch_bounds__(Cfg::kThreads, 1) void grouped_gemm_kernel(
    const __grid_constant__ CUtensorMap a_map, const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ GroupedArgs args, const __grid_constant__ Epilogue epilogue) {
    static_assert(Cfg::kScaled && Cfg::kTokenAtoms == 1);
    constexpr int TileN = Cfg::kTileN;
    // The activation scale stage is written by warp 3, not by TMA.
    constexpr std::uint32_t kTmaBytes = Cfg::kStageBytes - Cfg::kSfbBytes;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    const unsigned raw_address = common::smem_addr(shared_raw);
    sm::GemmSharedLayout<Cfg> shared{shared_raw + (((raw_address + 1023u) & ~1023u) - raw_address)};

    // Barriers: full[S] empty[S] acc_full[A] acc_empty[A], then the TMEM base slot.
    std::uint64_t* full      = shared.barriers();
    std::uint64_t* empty     = full + Cfg::kStages;
    std::uint64_t* acc_full  = empty + Cfg::kStages;
    std::uint64_t* acc_empty = acc_full + Cfg::kAccStages;
    auto* tmem_slot          = reinterpret_cast<std::uint32_t*>(acc_empty + Cfg::kAccStages);

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    __shared__ int offsets[kMaxExperts + 1];
    for (int i = static_cast<int>(threadIdx.x); i <= args.experts; i += Cfg::kThreads) {
        offsets[i] = args.offsets[i];
    }
    // One expert per thread: its tile count, scanned within the warp. The warp totals complete
    // the exclusive prefix after the barrier below.
    static_assert(Cfg::kThreads >= kMaxExperts);
    __shared__ int tile_start[kMaxExperts + 1];
    __shared__ int warp_tiles[Cfg::kThreads / 32];
    const int expert = static_cast<int>(threadIdx.x);
    int tiles_before = 0;
    {
        const int tiles = expert < args.experts ? (args.offsets[expert + 1] - args.offsets[expert] +
                                                   TileN - 1) / TileN * args.tiles_m
                                                : 0;
        int inclusive = tiles;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int value = __shfl_up_sync(0xffffffffu, inclusive, d);
            if (lane >= d) inclusive += value;
        }
        if (lane == 31) warp_tiles[warp] = inclusive;
        tiles_before = inclusive - tiles;
    }

    if (warp == 0 && lane == 0) {
        sm::tma_prefetch_descriptor(&a_map);
        sm::tma_prefetch_descriptor(&b_map);
        for (int s = 0; s < Cfg::kStages; ++s) {
            sm::mbarrier_init(&full[s], 2); // TMA producer (with tx bytes) + scale loader
            sm::mbarrier_init(&empty[s], 1);
        }
        for (int a = 0; a < Cfg::kAccStages; ++a) {
            sm::mbarrier_init(&acc_full[a], 1);
            sm::mbarrier_init(&acc_empty[a], Cfg::kEpilogueWarps);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 2) {
        sm::tmem_alloc<1>(tmem_slot, Cfg::kTmemAlloc);
        sm::tmem_relinquish_alloc_permit<1>();
    }
    __syncthreads();
    {
        int total = 0;
#pragma unroll
        for (int w = 0; w < Cfg::kThreads / 32; ++w) {
            const int tiles = warp_tiles[w];
            tiles_before += w < warp ? tiles : 0;
            total += tiles;
        }
        if (expert < args.experts) tile_start[expert] = tiles_before;
        if (expert == 0) tile_start[args.experts] = total;
    }
    __syncthreads();
    sm::fence_after_thread_sync();
    const std::uint32_t tmem_base = *tmem_slot;
    constexpr std::uint32_t kSfBase = Cfg::kAccStages * Cfg::kAccCols;

    if (warp == 0) {
        // ---- TMA producer ----
        const std::uint64_t weight_hint = common::l2_streaming_policy();
        const std::uint64_t input_hint  = common::l2_retained_policy();
        TileCursor cursor;
        TileInfo t;
        int stage           = 0;
        std::uint32_t phase = 0;
        for (int tile = static_cast<int>(blockIdx.x); next_tile<TileN>(cursor, tile, args, offsets, tile_start, t);
             tile += static_cast<int>(gridDim.x)) {
            for (int kb = 0; kb < args.k_iters; ++kb) {
                sm::mbarrier_wait(&empty[stage], phase ^ 1u);
                if (lane == 0) sm::mbarrier_arrive_expect_tx(&full[stage], kTmaBytes);
                __syncwarp();
                if (lane == 0) {
#pragma unroll
                    for (int p = 0; p < Cfg::kParts; ++p) {
                        const int row_tile = t.m + p * args.part_row_offset / 128;
                        tma_load_tile_3d(shared.a(stage, p), &a_map, kb * Cfg::kBoxInner,
                                         row_tile * 128, t.expert, &full[stage], weight_hint);
                        const std::int64_t atom =
                            static_cast<std::int64_t>(row_tile) * args.scale_k64 + kb * 4;
                        sm::bulk_load(shared.sfa(stage, p),
                                      args.a_scales + t.expert * args.a_scale_stride + atom * 512,
                                      Cfg::kScaleStageBytes, &full[stage]);
                    }
                    sm::tma_load_tile_2d(shared.b(stage), &b_map, kb * Cfg::kBoxInner, t.start,
                                         &full[stage], input_hint);
                }
                __syncwarp();
                if (++stage == Cfg::kStages) {
                    stage = 0;
                    phase ^= 1u;
                }
            }
        }
    } else if (warp == 3) {
        // ---- Activation scale loader ----
        // Rewrites the token-major scale rows (16 bytes = one 256-wide K stage per token) into
        // the K16M128x4 atoms: atom j holds groups 4j..4j+3, and token i sits at row i % 32,
        // word i / 32. The rows of the next kAhead stages are loaded before the slot they fill is
        // free, so a stage waits for its slot, not for a global load.
        constexpr int R      = (TileN + 31) / 32;
        constexpr int kAhead = 4;
        TileCursor cursor;
        TileInfo t;
        int stage           = 0;
        std::uint32_t phase = 0;
        for (int tile = static_cast<int>(blockIdx.x); next_tile<TileN>(cursor, tile, args, offsets, tile_start, t);
             tile += static_cast<int>(gridDim.x)) {
            std::int64_t row[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int i = lane + 32 * r;
                row[r]      = t.start + (i < t.valid ? i : 0);
            }
            const auto load_scales = [&](int kb, uint4 (&value)[R]) {
#pragma unroll
                for (int r = 0; r < R; ++r)
                    if (lane + 32 * r < TileN)
                        value[r] = *reinterpret_cast<const uint4*>(
                            args.b_scales + row[r] * args.b_scale_row_bytes + kb * 16);
            };
            uint4 pending[kAhead][R];
#pragma unroll
            for (int d = 0; d < kAhead; ++d)
                if (d < args.k_iters) load_scales(d, pending[d]);
            for (int kb = 0; kb < args.k_iters; kb += kAhead) {
#pragma unroll
                for (int d = 0; d < kAhead; ++d) {
                    if (kb + d < args.k_iters) {
                        sm::mbarrier_wait(&empty[stage], phase ^ 1u);
                        unsigned char* atoms = shared.sfb(stage);
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            const int i = lane + 32 * r;
                            if (i < TileN) {
                                unsigned char* word = atoms + (i & 31) * 16 + (i >> 5) * 4;
                                *reinterpret_cast<unsigned*>(word)        = pending[d][r].x;
                                *reinterpret_cast<unsigned*>(word + 512)  = pending[d][r].y;
                                *reinterpret_cast<unsigned*>(word + 1024) = pending[d][r].z;
                                *reinterpret_cast<unsigned*>(word + 1536) = pending[d][r].w;
                            }
                        }
                        sm::fence_proxy_async_shared();
                        __syncwarp();
                        if (lane == 0) sm::mbarrier_arrive(&full[stage]);
                        if (++stage == Cfg::kStages) {
                            stage = 0;
                            phase ^= 1u;
                        }
                        if (kb + d + kAhead < args.k_iters) load_scales(kb + d + kAhead, pending[d]);
                    }
                }
            }
        }
    } else if (warp == 1) {
        // ---- MMA issuer ----
        if (sm::elect_one()) {
            constexpr std::uint32_t idesc = Cfg::idesc();
            TileCursor cursor;
            TileInfo t;
            int stage               = 0;
            std::uint32_t phase     = 0;
            int acc                 = 0;
            std::uint32_t acc_phase = 0;
            std::uint32_t k_count   = 0;
            for (int tile = static_cast<int>(blockIdx.x); next_tile<TileN>(cursor, tile, args, offsets, tile_start, t);
                 tile += static_cast<int>(gridDim.x)) {
                sm::mbarrier_wait(&acc_empty[acc], acc_phase ^ 1u);
                sm::fence_after_thread_sync();
                for (int kb = 0; kb < args.k_iters; ++kb) {
                    sm::mbarrier_wait(&full[stage], phase);
                    sm::fence_after_thread_sync();
                    const std::uint32_t sf_buf =
                        tmem_base + kSfBase + (k_count & 1u) * Cfg::kSfColsPerBuf;
#pragma unroll
                    for (int j = 0; j < Cfg::kMmaPerStage; ++j) {
                        const std::uint32_t sf_j = sf_buf + j * Cfg::kSfColsPerMma;
#pragma unroll
                        for (int p = 0; p < Cfg::kParts; ++p)
                            sm::tmem_copy_32x128b_warpx4<1>(
                                sf_j + p * 4,
                                sm::make_umma_scale_desc(common::smem_addr(
                                    shared.sfa(stage, p) + j * Cfg::kScaleAtomBytes)));
                        sm::tmem_copy_32x128b_warpx4<1>(
                            sf_j + Cfg::kParts * 4,
                            sm::make_umma_scale_desc(
                                common::smem_addr(shared.sfb(stage) + j * Cfg::kScaleAtomBytes)));
                    }
                    const std::uint64_t b_desc0 = sm::make_umma_k_major_desc(
                        common::smem_addr(shared.b(stage)), sm::UmmaSwizzle::B128);
#pragma unroll
                    for (int j = 0; j < Cfg::kMmaPerStage; ++j) {
#pragma unroll
                        for (int p = 0; p < Cfg::kParts; ++p) {
                            const std::uint64_t a_desc = sm::umma_desc_advance(
                                sm::make_umma_k_major_desc(
                                    common::smem_addr(shared.a(stage, p)), sm::UmmaSwizzle::B128),
                                j * Cfg::kMmaKBytes);
                            const std::uint64_t b_desc =
                                sm::umma_desc_advance(b_desc0, j * Cfg::kMmaKBytes);
                            const std::uint32_t d =
                                tmem_base + acc * Cfg::kAccCols + p * Cfg::kTileN;
                            const std::uint32_t sf_j = sf_buf + j * Cfg::kSfColsPerMma;
                            sm::umma_mxf4nvf4_ss<1>(d, a_desc, b_desc, idesc, sf_j + p * 4,
                                                    sf_j + Cfg::kParts * 4, (kb | j) != 0);
                        }
                    }
                    sm::umma_commit<1>(&empty[stage]);
                    ++k_count;
                    if (++stage == Cfg::kStages) {
                        stage = 0;
                        phase ^= 1u;
                    }
                }
                sm::umma_commit<1>(&acc_full[acc]);
                if (++acc == Cfg::kAccStages) {
                    acc = 0;
                    acc_phase ^= 1u;
                }
            }
        }
    } else if (warp >= 4) {
        // ---- Epilogue warps ----
        using Ahead             = typename Epilogue::template Ahead<Cfg>;
        constexpr int kSlices   = Cfg::kEpilogueWarps / 4;
        constexpr int kSliceCols = TileN / kSlices;
        static_assert(kSliceCols % 16 == 0);
        const int ew            = warp - 4;
        const int q             = ew & 3;
        const int col_begin     = (ew >> 2) * kSliceCols;
        int acc                 = 0;
        std::uint32_t acc_phase = 0;
        TileCursor cursor;
        TileInfo t, upcoming;
        Ahead ahead{};
        bool more = next_tile<TileN>(cursor, static_cast<int>(blockIdx.x), args, offsets, tile_start, upcoming);
        if (more) {
            ahead = epilogue.template fetch<Cfg>(upcoming, lane);
            epilogue.template settle<Cfg>(ahead, upcoming, lane);
        }
        for (int tile = static_cast<int>(blockIdx.x); more; tile += static_cast<int>(gridDim.x)) {
            t    = upcoming;
            more = next_tile<TileN>(cursor, tile + static_cast<int>(gridDim.x), args, offsets, tile_start, upcoming);
            sm::mbarrier_wait(&acc_full[acc], acc_phase);
            sm::fence_after_thread_sync();
            Ahead following{};
            if (more) following = epilogue.template fetch<Cfg>(upcoming, lane);
            const GroupTile g{sm::tmem_address(tmem_base, q * 32, acc * Cfg::kAccCols),
                              Cfg::kTileN,
                              t.expert,
                              t.m,
                              t.start,
                              t.valid,
                              q,
                              lane,
                              shared.scratch(ew),
                              col_begin,
                              col_begin + kSliceCols};
            epilogue.template drain<Cfg>(g, ahead);
            if (more) {
                epilogue.template settle<Cfg>(following, upcoming, lane);
                ahead = following;
            }
            sm::fence_before_thread_sync();
            __syncwarp();
            if (lane == 0) sm::mbarrier_arrive(&acc_empty[acc]);
            if (++acc == Cfg::kAccStages) {
                acc = 0;
                acc_phase ^= 1u;
            }
        }
    }

    __syncthreads();
    if (warp == 2) {
        sm::fence_after_thread_sync();
        sm::tmem_dealloc<1>(tmem_base, Cfg::kTmemAlloc);
    }
}

// ---------------------------------------------------------------------------------------------
// Epilogues
// ---------------------------------------------------------------------------------------------

// Gate/up: accumulator part 0 holds gate rows, part 1 the matching up rows, so one thread owns
// both operands of an element. h = silu(gate * factor) * (up * factor) stays FP32; it is quantized
// to NVFP4 per 16 consecutive features and token with the calibrated down divisor, and written
// to the packed-slot row of the product buffer (codes [slot][I / 2], scales [slot][I / 16]).
struct GateUpEpilogue {
    std::uint8_t* codes;
    std::uint8_t* scales;
    const float* matrix_scales;
    std::uint64_t matrix_scale_stride;
    float activation_divisor; // divisor of the hidden-state quantization (gate/up input)
    float product_divisor;    // divisor of the product quantization (down input)
    int intermediate;

    template <class Cfg>
    struct Ahead {
        float factor; // fetch: the expert's matrix scalar; settle: reciprocal output scale
    };

    template <class Cfg>
    __device__ __forceinline__ Ahead<Cfg> fetch(const TileInfo& t, int /*lane*/) const {
        return {matrix_scales[t.expert * matrix_scale_stride]};
    }

    template <class Cfg>
    __device__ __forceinline__ void settle(Ahead<Cfg>& a, const TileInfo& /*t*/,
                                           int /*lane*/) const {
        a.factor = 1.0F / (activation_divisor * a.factor);
    }

    // The TMEM loads of the next 16-token chunk are in flight while this one is computed.
    template <class Cfg>
    __device__ __forceinline__ void drain(const GroupTile& t, const Ahead<Cfg>& a) const {
        static_assert(Cfg::kParts == 2);
        const int end = min(t.valid, t.col_end);
        if (t.col_begin >= end) return;
        std::uint32_t gate0[16], up0[16], gate1[16], up1[16];
        const auto load = [&](int c0, std::uint32_t (&gate)[16], std::uint32_t (&up)[16]) {
            sm::tmem_load_32x32b<16>(t.tmem + c0, gate);
            sm::tmem_load_32x32b<16>(t.tmem + t.part_stride + c0, up);
        };
        load(t.col_begin, gate0, up0);
        for (int c0 = t.col_begin; c0 < end; c0 += 32) {
            sm::tmem_load_wait();
            if (c0 + 16 < end) load(c0 + 16, gate1, up1);
            quantize_chunk<Cfg>(t, a.factor, gate0, up0, c0);
            if (c0 + 16 >= end) break;
            sm::tmem_load_wait();
            if (c0 + 32 < end) load(c0 + 32, gate0, up0);
            quantize_chunk<Cfg>(t, a.factor, gate1, up1, c0 + 16);
        }
    }

private:
    template <class Cfg>
    __device__ __forceinline__ void quantize_chunk(const GroupTile& t, float factor,
                                                   const std::uint32_t (&gate)[16],
                                                   const std::uint32_t (&up)[16], int c0) const {
        // 16 tokens x 32 features of FP32, bank-conflict free through an xor on the feature.
        auto* stage         = reinterpret_cast<float*>(t.scratch);
        const int feature0  = t.m * Cfg::kTileM + t.warp_q * 32;
        const int reader    = t.lane & 15; // token column this lane quantizes
        const int group     = t.lane >> 4; // which 16 features of the warp's 32
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float g = __uint_as_float(gate[i]) * factor;
            const float u = __uint_as_float(up[i]) * factor;
            stage[i * 32 + (t.lane ^ i)] = common::silu_approx(g) * u;
        }
        __syncwarp();
        const int column = c0 + reader;
        if (column < t.valid) {
            float2 h[8];
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                h[p].x = stage[reader * 32 + ((group * 16 + 2 * p) ^ reader)];
                h[p].y = stage[reader * 32 + ((group * 16 + 2 * p + 1) ^ reader)];
            }
            const auto q            = common::quantize_nvfp4_k16(h, product_divisor);
            const std::int64_t slot = t.start + column;
            const int feature       = feature0 + group * 16;
            *reinterpret_cast<uint2*>(codes + slot * (intermediate / 2) + feature / 2) =
                make_uint2(q.codes_lo, q.codes_hi);
            scales[slot * (intermediate / 16) + feature / 16] = q.scale;
        }
        __syncwarp();
    }
};

// Down: out[token, row] += coefficient * (acc * factor) (Atomic, into the per-token FP32 sum), or
// contributions[slot, row] = acc * factor (ordered reduction by the caller). Lane = output row.
// The ordered form writes 32 consecutive floats per column. The atomic form transposes 16
// columns through shared memory so that each lane reduces four consecutive rows of one token
// with a single vector reduction: the L2 atomic unit takes the same number of instructions
// whatever their width, and the scalar form was bound by them.
template <bool Atomic>
struct DownEpilogue {
    float* sum;           // [tokens][hidden] (Atomic)
    float* contributions; // [A][hidden]      (!Atomic)
    const int* assignments;
    const float* coefficients;
    const float* matrix_scales;
    std::uint64_t matrix_scale_stride;
    float activation_divisor; // divisor of the product quantization (down input)
    int hidden, top_k;

    // Lane j holds the route of column j + 32 r of the tile.
    template <class Cfg>
    struct Ahead {
        static constexpr int kRoutes = (Cfg::kTileN + 31) / 32;
        int assignment[kRoutes]; // set by fetch
        int token[kRoutes];      // set by settle
        float coefficient[kRoutes];
        float factor; // fetch: the expert's matrix scalar; settle: reciprocal output scale
    };

    template <class Cfg>
    __device__ __forceinline__ Ahead<Cfg> fetch(const TileInfo& t, int lane) const {
        Ahead<Cfg> a{};
        if constexpr (Atomic) {
#pragma unroll
            for (int r = 0; r < Ahead<Cfg>::kRoutes; ++r) {
                const int column = lane + 32 * r;
                a.assignment[r]  = column < t.valid ? assignments[t.start + column] : 0;
            }
        }
        a.factor = matrix_scales[t.expert * matrix_scale_stride];
        return a;
    }

    template <class Cfg>
    __device__ __forceinline__ void settle(Ahead<Cfg>& a, const TileInfo& t, int lane) const {
        if constexpr (Atomic) {
#pragma unroll
            for (int r = 0; r < Ahead<Cfg>::kRoutes; ++r) {
                const int assignment = a.assignment[r];
                a.coefficient[r]     = lane + 32 * r < t.valid ? coefficients[assignment] : 0.0F;
                a.token[r]           = assignment / top_k;
            }
        }
        a.factor = 1.0F / (activation_divisor * a.factor);
    }

    // The TMEM load of the next 16-column chunk is in flight while this one is reduced.
    template <class Cfg>
    __device__ __forceinline__ void drain(const GroupTile& t, const Ahead<Cfg>& a) const {
        static_assert(Cfg::kParts == 1 && Cfg::kTileN % 16 == 0);
        constexpr int kChunks = Cfg::kTileN / 16 / (Cfg::kEpilogueWarps / 4);
        const int row0        = t.m * Cfg::kTileM + t.warp_q * 32;
        auto* stage           = reinterpret_cast<float*>(t.scratch);
        if (t.col_begin >= t.valid) return;
        std::uint32_t r[2][16];
        sm::tmem_load_32x32b<16>(t.tmem + t.col_begin, r[0]);
#pragma unroll
        for (int chunk = 0; chunk < kChunks; ++chunk) {
            const int c0 = t.col_begin + chunk * 16;
            if (c0 >= t.valid) break;
            sm::tmem_load_wait();
            if (chunk + 1 < kChunks && c0 + 16 < t.valid)
                sm::tmem_load_32x32b<16>(t.tmem + c0 + 16, r[(chunk + 1) & 1]);
            if constexpr (Atomic) {
                // stage[column][row], rows in groups of four xor-swizzled by the column so both the
                // lane-per-row writes and the four-rows-per-lane reads are bank-conflict free.
                const int group = t.lane >> 2; // four consecutive rows
                const int quad  = t.lane & 3;  // which of the four columns of a round
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    stage[i * 32 + (((group ^ ((i & 3) << 1)) << 2) | (t.lane & 3))] =
                        __uint_as_float(r[chunk & 1][i]);
                __syncwarp();
#pragma unroll
                for (int round = 0; round < 4; ++round) {
                    const int column = c0 + 4 * round + quad;
                    const float4 v   = *reinterpret_cast<const float4*>(
                        stage + (4 * round + quad) * 32 + ((group ^ (quad << 1)) << 2));
                    const int route    = (c0 + 4 * round) >> 5;
                    const int source   = column & 31;
                    const int token    = __shfl_sync(0xffffffffu, a.token[route], source);
                    const float weight = __shfl_sync(0xffffffffu, a.coefficient[route], source);
                    if (column < t.valid) {
                        red_add_v4(sum + static_cast<std::int64_t>(token) * hidden + row0 +
                                       4 * group,
                                   v.x * a.factor * weight, v.y * a.factor * weight,
                                   v.z * a.factor * weight, v.w * a.factor * weight);
                    }
                }
                __syncwarp();
            } else {
                const int row = row0 + t.lane;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if (c0 + i < t.valid) {
                        contributions[static_cast<std::int64_t>(t.start + c0 + i) * hidden + row] =
                            __uint_as_float(r[chunk & 1][i]) * a.factor;
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------------

// Expert-major NVFP4 code plane as a 3D byte tensor [expert][row][k / 2]; the box is one 128-byte
// swizzle atom row wide and `box_rows` rows tall.
inline CUtensorMap make_expert_bank_map(const void* codes, std::uint64_t row_bytes,
                                        std::uint64_t rows, std::uint64_t experts,
                                        std::uint64_t expert_stride, std::uint32_t box_rows,
                                        const char* operation) {
    CUtensorMap map{};
    const std::uint64_t dims[]     = {row_bytes, rows, experts};
    const std::uint64_t strides[]  = {row_bytes, expert_stride};
    const std::uint32_t box[]      = {128, box_rows, 1};
    const std::uint32_t elements[] = {1, 1, 1};
    sm::check_driver(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
                                            const_cast<void*>(codes), dims, strides, box,
                                            elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                            CU_TENSOR_MAP_SWIZZLE_128B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                     operation);
    return map;
}

inline int device_sm_count() {
    int device = 0, count = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return count;
}

// Launches one persistent CTA per SM (never more than there can be tiles).
template <class Cfg, class Epilogue>
void launch_grouped(const CUtensorMap& a_map, const CUtensorMap& b_map, const GroupedArgs& args,
                    const Epilogue& epilogue, int assignments, cudaStream_t stream) {
    if (args.experts > kMaxExperts) throw std::runtime_error("moe tcgen05: too many experts");
    static bool configured[64] = {};
    int device                 = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= 64) throw std::runtime_error("moe tcgen05: device ordinal");
    if (!configured[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(grouped_gemm_kernel<Cfg, Epilogue>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Cfg::kSharedBytes));
        configured[device] = true;
    }
    // Sum of ceil(count_e / TileN) over experts is at most A / TileN + experts.
    const int max_tiles = args.tiles_m * (assignments / Cfg::kTileN + args.experts);
    const int grid      = sm::persistent_grid(max_tiles, device_sm_count());
    grouped_gemm_kernel<Cfg, Epilogue>
        <<<grid, Cfg::kThreads, Cfg::kSharedBytes, stream>>>(a_map, b_map, args, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4
