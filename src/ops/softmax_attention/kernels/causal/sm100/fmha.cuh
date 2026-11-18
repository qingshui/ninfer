#pragma once

// SM100 causal FMHA mainloop. See launch.h for the contract.
//
// Roles of one 320-thread CTA (one CTA per SM, ~210 KiB of shared memory):
//   warps 0-3   softmax / epilogue. Thread t owns packed query row t: the 128-lane TMEM accumulator
//               maps one row to one lane, so row maxima and sums never cross threads.
//   warp 4      MMA issue (one elected thread): S = Q.K^T into S[b], O += P.V
//   warp 5      TMEM allocation
//   warps 6-9   Q preparation, then the K/V tile producer (copy or dequantize into the swizzled
//               shared-memory layouts the UMMA descriptors read)
//
// Tensor memory: S double buffer (64 columns per buffer; two 64-column groups for native FP8), then
// the 256-column O accumulator. Shared memory: [K0 K1 V0 V1 P] ring, Q, per-row/key scales, barriers.
//
// Operand layouts (all 128B-swizzled, 16-byte slot s of row r stored at slot s ^ (r & 7)):
//   Q  (A, K-major)    chunk c = 64 bf16 (128 fp8) features, 128 rows x 128 B per chunk
//   K  (B, K-major)    chunk c likewise, 64 key rows per chunk
//   P  (A, K-major)    128 rows x 64 keys fp16, one chunk
//   V  (B, MN-major)   chunk c = 64 features, 64 key rows x 128 B per chunk

#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/sm100/tcgen05.cuh"
#include "ops/common/cuda/sm100/tma.cuh"
#include "ops/common/cuda/sm100/umma_descriptor.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/sm100/kv_policy.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"

#include <math_constants.h>

namespace ninfer::ops::detail::sm100_fmha {
namespace sm = ::ninfer::ops::common::sm100;

using Geometry = CausalD256H32Kv2;

inline constexpr int kGroup = Geometry::GroupSize; // query heads per KV head

template <KvDType Kv>
struct Cfg {
    using Policy                    = KvPolicy<Kv>;
    static constexpr bool kNative   = Policy::kNativeFp8;
    static constexpr int kRows      = kSm100TileQ;
    static constexpr int kKeys      = kSm100TileKv;
    static constexpr int kD         = 256;
    static constexpr int kThreads   = 320;
    static constexpr int kLoaderWarp0 = 6;

    static constexpr int kQBytes = kNative ? kRows * kD : kRows * kD * 2;
    static constexpr int kKBytes = kNative ? kKeys * kD : kKeys * kD * 2; // per stage
    static constexpr int kVBytes = kKeys * kD * 2;                        // per stage
    static constexpr int kPBytes = kRows * kKeys * 2;
    // Epilogue transposition scratch: 4 warps x 32 rows x (256 + 1) fp32.
    static constexpr int kScratchStride = 257;
    static constexpr int kScratchBytes  = 4 * 32 * kScratchStride * 4;
    static constexpr int kRingUsed      = 2 * kKBytes + 2 * kVBytes + kPBytes;
    static constexpr int kRingBytes =
        ((kRingUsed > kScratchBytes ? kRingUsed : kScratchBytes) + 1023) / 1024 * 1024;

    static constexpr int kKOff      = 0;
    static constexpr int kVOff      = 2 * kKBytes;
    static constexpr int kPOff      = kVOff + 2 * kVBytes;
    static constexpr int kQOff      = kRingBytes;
    static constexpr int kQScaleOff = kQOff + kQBytes;
    static constexpr int kKScaleOff = kQScaleOff + kRows * 4;
    static constexpr int kBarOff    = kKScaleOff + 2 * kKeys * 2 * 4;
    static constexpr int kSharedBytes = kBarOff + 256 + 1024;

    static constexpr int kSCols     = kNative ? 128 : 64;
    static constexpr int kOCol      = 2 * kSCols;
    static constexpr int kTmemAlloc = 512;
    static_assert(kOCol + 256 <= kTmemAlloc);
    static_assert(kSharedBytes <= 227 * 1024);
};

struct FmhaParams {
    const __nv_bfloat16* q;
    const std::int32_t* positions;
    __nv_bfloat16* out;
    CausalPartialView partial;
    Sm100KvCache cache;
    float scale;
    int width;
    int logical_capacity;
    int splits;
    int row_tiles;
    int key_shift;
};

// Entries of different widths on one token axis in a single launch (one split, BF16 output). Group
// g is `batch` consecutive entries of `width` columns starting at column token_base; entry b of the
// group uses cache metadata row `row + b`. The group owns the CTAs
// [cta_base, cta_base + batch * KVHeads * row_tiles(width)), and cta_base ascends.
inline constexpr int kMaxGroups = 72;

struct Group {
    int token_base, width, row, batch, cta_base;
};

struct GroupParams {
    FmhaParams base;
    int groups;
    Group group[kMaxGroups];
};

__host__ __device__ constexpr int row_tiles_of(int width) {
    return (width * kGroup + kSm100TileQ - 1) / kSm100TileQ;
}

// What one CTA owns: query rows [row_begin, row_begin + 128) of one entry, keys
// [split_start, end). The entry's columns are [token_base, token_base + width) of the token axis and
// its valid length and table row are entry `row` of the cache metadata. `batch` addresses the split
// partials, which only a launch with one width uses.
struct Work {
    int kv_head, split, batch, row, token_base, width, row_begin, live, split_start, end, n_tiles;
};

__device__ __forceinline__ Work make_work(const FmhaParams& p, int kv_head, int tile, int split,
                                          int batch, int row, int token_base, int width) {
    Work w{};
    w.kv_head    = kv_head;
    w.split      = split;
    w.batch      = batch;
    w.row        = row;
    w.token_base = token_base;
    w.width      = width;
    w.row_begin  = tile * kSm100TileQ;
    int live     = width;
    if (p.cache.valid_columns != nullptr) {
        live = p.cache.valid_columns[row];
        live = live < 0 ? 0 : (live < width ? live : width);
    }
    w.live = live;
    const std::int32_t* positions = p.positions + token_base;
    const int first_pos = positions[0];
    const int last_pos  = positions[width - 1];
    w.split_start = w.end = w.n_tiles = 0;
    const int tile_tokens_end =
        min(live, (w.row_begin + kSm100TileQ + kGroup - 1) / kGroup); // exclusive
    const int tile_tokens_begin = w.row_begin / kGroup;
    if (first_pos < 0 || last_pos < 0 || last_pos >= p.logical_capacity ||
        tile_tokens_end <= tile_tokens_begin)
        return w;
    const CausalKvPartition partition{p.splits, p.splits, p.key_shift};
    const int window = last_pos + 1;
    const int active = partition.active(window);
    if (w.split >= active) {
        w.n_tiles = -1; // The merge never reads this split.
        return w;
    }
    const int tiles       = (window + kSm100TileKv - 1) / kSm100TileKv;
    const int first_owned = w.split * tiles / active;
    const int end_owned   = (w.split + 1) * tiles / active;
    w.split_start         = first_owned * kSm100TileKv;
    const int split_end   = min(end_owned * kSm100TileKv, window);
    w.end                 = min(split_end, positions[tile_tokens_end - 1] + 1);
    w.n_tiles = w.end > w.split_start ? (w.end - w.split_start + kSm100TileKv - 1) / kSm100TileKv : 0;
    return w;
}

__device__ __forceinline__ Work make_work(const FmhaParams& p) {
    // Later query tiles see longer histories; launch them first.
    const int tile  = p.row_tiles - 1 - static_cast<int>(blockIdx.x) / Geometry::KVHeads;
    const int batch = static_cast<int>(blockIdx.z);
    return make_work(p, static_cast<int>(blockIdx.x) % Geometry::KVHeads, tile,
                     static_cast<int>(blockIdx.y), batch, batch, batch * p.width, p.width);
}

__device__ __forceinline__ Work make_work(const GroupParams& r) {
    const int cta = static_cast<int>(blockIdx.x);
    int low = 0, high = r.groups - 1; // the last group that starts at or before this CTA
    while (low < high) {
        const int mid = (low + high + 1) >> 1;
        if (r.group[mid].cta_base <= cta)
            low = mid;
        else
            high = mid - 1;
    }
    const Group& g       = r.group[low];
    const int row_tiles  = row_tiles_of(g.width);
    const int per_entry  = Geometry::KVHeads * row_tiles;
    const int local      = cta - g.cta_base;
    const int entry      = local / per_entry;
    const int within     = local - entry * per_entry;
    const int tile       = row_tiles - 1 - within / Geometry::KVHeads;
    return make_work(r.base, within % Geometry::KVHeads, tile, 0, 0, g.row + entry,
                     g.token_base + entry * g.width, g.width);
}

// Byte offset of 16-byte slot `slot` of row `row` inside a 128B-swizzled chunk of 128-byte rows.
__device__ __forceinline__ unsigned sw128(int row, int slot) {
    return static_cast<unsigned>(row * 128 + ((slot ^ (row & 7)) << 4));
}

__device__ __forceinline__ void tmem_store_32x32b_x32(std::uint32_t taddr,
                                                      const std::uint32_t (&r)[32]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
                 "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
                 "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
                 "%32};"
                 :
                 : "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]),
                   "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]),
                   "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
                   "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]),
                   "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]),
                   "r"(r[31])
                 : "memory");
}

template <class C>
struct Shared {
    unsigned char* k;
    unsigned char* v;
    unsigned char* p;
    unsigned char* q;
    unsigned char* scratch;
    float* q_scale;
    float* k_scale;
    std::uint64_t* q_full;
    std::uint64_t* kv_full;  // [2]
    std::uint64_t* kv_empty; // [2]
    std::uint64_t* s_full;   // [2]
    std::uint64_t* s_empty;  // [2]
    std::uint64_t* p_full;
    std::uint64_t* pv_done;
    std::uint64_t* o_full;
    std::uint32_t* tmem_slot;

    __device__ __forceinline__ unsigned char* k_stage(int s) const { return k + s * C::kKBytes; }
    __device__ __forceinline__ unsigned char* v_stage(int s) const { return v + s * C::kVBytes; }
};

template <class C>
__device__ __forceinline__ Shared<C> make_shared(unsigned char* base) {
    Shared<C> s{};
    s.k         = base + C::kKOff;
    s.v         = base + C::kVOff;
    s.p         = base + C::kPOff;
    s.q         = base + C::kQOff;
    s.scratch   = base;
    s.q_scale   = reinterpret_cast<float*>(base + C::kQScaleOff);
    s.k_scale   = reinterpret_cast<float*>(base + C::kKScaleOff);
    auto* bar   = reinterpret_cast<std::uint64_t*>(base + C::kBarOff);
    s.q_full    = bar;
    s.kv_full   = bar + 1;
    s.kv_empty  = bar + 3;
    s.s_full    = bar + 5;
    s.s_empty   = bar + 7;
    s.p_full    = bar + 9;
    s.pv_done   = bar + 10;
    s.o_full    = bar + 11;
    s.tmem_slot = reinterpret_cast<std::uint32_t*>(bar + 12);
    return s;
}

// Loader warps: the 128 query rows of this CTA into the Q operand tile. Rotated chains transform Q
// with the same normalized H256 the cache writer applied to K; the native FP8 chain additionally
// quantizes each row to E4M3 with a per-row scale (absmax / 448) kept for the score epilogue.
template <KvDType Kv>
__device__ __forceinline__ void fill_q(const FmhaParams& p, const Work& w,
                                       const Shared<Cfg<Kv>>& sh, int ltid) {
    using C        = Cfg<Kv>;
    const int warp = ltid >> 5, lane = ltid & 31;
    // `valid` is uniform over the warp. A row past the live tokens stays zero and skips the transform.
    const auto finish = [&](int row, float (&v)[8], bool valid) {
        if constexpr (C::Policy::kRotate) {
            if (valid) kv::normalized_hadamard_d256_inplace(v, lane);
        }
        if constexpr (!C::kNative) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane + 32 * r;
                *reinterpret_cast<__nv_bfloat16*>(sh.q + (d >> 6) * 16384 +
                                                  sw128(row, (d & 63) >> 3) + (d & 7) * 2) =
                    __float2bfloat16(v[r]);
            }
        } else {
            float amax = 0.0f;
            if (valid) {
#pragma unroll
                for (int r = 0; r < 8; ++r) amax = fmaxf(amax, fabsf(v[r]));
                amax = common::warp_max(amax, 0xffffffffu);
            }
            const float qs  = amax > 0.0f ? amax / common::kFp8E4M3MaxFinite : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane + 32 * r;
                sh.q[(d >> 7) * 16384 + sw128(row, (d & 127) >> 4) + (d & 15)] =
                    common::fp8_quant_code(v[r], inv);
            }
            if (lane == 0) sh.q_scale[row] = qs;
        }
    };
    // The loads of kBatch rows are in flight together: one memory round trip serves them all.
    constexpr int kBatch = 4;
    for (int row0 = warp; row0 < C::kRows; row0 += 4 * kBatch) {
        float v[kBatch][8];
        bool valid[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const int global_row = w.row_begin + row0 + 4 * b;
            const int token      = global_row / kGroup;
            const int head       = w.kv_head * kGroup + global_row % kGroup;
            valid[b]             = token < w.live;
            const __nv_bfloat16* src =
                p.q + causal_q_index<Geometry>(head, 0, valid[b] ? token + w.token_base : 0);
#pragma unroll
            for (int r = 0; r < 8; ++r)
                v[b][r] = valid[b] ? __bfloat162float(src[lane + 32 * r]) : 0.0f;
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b) finish(row0 + 4 * b, v[b], valid[b]);
    }
}

// One loader thread's share of a K/V tile, held in registers between its global loads and its
// shared-memory stores. Issuing every load of a tile before the first store makes a tile cost one
// memory round trip, and lets the next tile's loads run while the MMA and softmax work on this one.
template <KvDType Kv>
struct TileRegs {
    using C                      = Cfg<Kv>;
    using VRaw                   = typename C::Policy::VRaw;
    static constexpr int kVecK   = C::kNative ? 16 : 32;
    static constexpr int kVecV   = 32;
    static constexpr int kKIters = C::kKeys * kVecK / 128;
    static constexpr int kVIters = C::kKeys * kVecV / 128;
    static_assert(kKIters * 128 == C::kKeys * kVecK && kVIters * 128 == C::kKeys * kVecV);
    uint4 k[kKIters];
    VRaw v[kVIters];
    unsigned k_pair; // encoded K scale pair of key row ltid (native FP8)
    bool k_valid;
};

// Loader warps: global loads of keys [k0, k0 + 64) of this split. Rows at or beyond `end` are zero
// so padded cache pages never feed non-finite values into P.V. Nothing is decoded here: a decode
// that consumes a load would stall the issue of the loads behind it.
template <KvDType Kv>
__device__ __forceinline__ void load_kv(const FmhaParams& p, const Work& w, int tile,
                                        std::int64_t record, int ltid, TileRegs<Kv>& regs) {
    using C      = Cfg<Kv>;
    using Policy = typename C::Policy;
    using Regs   = TileRegs<Kv>;
    const int k0 = w.split_start + tile * C::kKeys;
#pragma unroll
    for (int it = 0; it < Regs::kKIters; ++it) {
        const int i     = ltid + it * 128;
        const int key_l = i / Regs::kVecK, vi = i % Regs::kVecK;
        regs.k[it]      = make_uint4(0, 0, 0, 0);
        if (k0 + key_l < w.end) {
            if constexpr (C::kNative)
                regs.k[it] = Policy::load_k16(p.cache, record, w.kv_head, key_l, vi * 16);
            else
                regs.k[it] = Policy::load_k8(p.cache, record, w.kv_head, key_l, vi * 8);
        }
    }
#pragma unroll
    for (int it = 0; it < Regs::kVIters; ++it) {
        const int i     = ltid + it * 128;
        const int key_l = i / Regs::kVecV, vi = i % Regs::kVecV;
        regs.v[it]      = typename Regs::VRaw{};
        if (k0 + key_l < w.end)
            regs.v[it] = Policy::load_v8_raw(p.cache, record, w.kv_head, key_l, vi * 8);
    }
    if constexpr (C::kNative) {
        regs.k_valid = ltid < C::kKeys && k0 + ltid < w.end;
        regs.k_pair  = regs.k_valid ? Policy::load_k_scale_pair(p.cache, record, w.kv_head, ltid) : 0u;
    }
}

// Loader warps: the staged tile into one K/V stage.
template <KvDType Kv>
__device__ __forceinline__ void store_kv(const TileRegs<Kv>& regs, int stage,
                                         const Shared<Cfg<Kv>>& sh, int ltid) {
    using C              = Cfg<Kv>;
    using Policy         = typename C::Policy;
    using Regs           = TileRegs<Kv>;
    unsigned char* k_dst = sh.k_stage(stage);
    unsigned char* v_dst = sh.v_stage(stage);
#pragma unroll
    for (int it = 0; it < Regs::kKIters; ++it) {
        const int i     = ltid + it * 128;
        const int key_l = i / Regs::kVecK, vi = i % Regs::kVecK;
        *reinterpret_cast<uint4*>(k_dst + (vi >> 3) * 8192 + sw128(key_l, vi & 7)) = regs.k[it];
    }
#pragma unroll
    for (int it = 0; it < Regs::kVIters; ++it) {
        const int i     = ltid + it * 128;
        const int key_l = i / Regs::kVecV, vi = i % Regs::kVecV;
        *reinterpret_cast<uint4*>(v_dst + (vi >> 3) * 8192 + sw128(key_l, vi & 7)) =
            Policy::decode_v8(regs.v[it]);
    }
    if constexpr (C::kNative) {
        if (ltid < C::kKeys) {
            float s0 = 0.0f, s1 = 0.0f;
            if (regs.k_valid) Policy::decode_k_scales(regs.k_pair, s0, s1);
            sh.k_scale[stage * 2 * C::kKeys + 2 * ltid]     = s0;
            sh.k_scale[stage * 2 * C::kKeys + 2 * ltid + 1] = s1;
        }
    }
}

// Warp 4: one elected thread issues every tcgen05.mma of the CTA in order
//   QK(j) -> S[j & 1];  PV(j - 1) -> O
// so the softmax of tile j overlaps the loads of tile j + 1 and the QK of tile j + 1.
template <KvDType Kv>
__device__ __forceinline__ void mma_role(const Shared<Cfg<Kv>>& sh, std::uint32_t tmem,
                                         int n_tiles) {
    using C = Cfg<Kv>;
    if (!sm::elect_one()) return;
    constexpr std::uint32_t idesc_qk =
        C::kNative ? sm::make_umma_idesc_f8f6f4(C::kRows, C::kKeys, sm::UmmaF8F6F4Format::E4M3,
                                                sm::UmmaF8F6F4Format::E4M3)
                   : sm::make_umma_idesc_f16(C::kRows, C::kKeys, sm::UmmaF16Format::BF16,
                                             sm::UmmaF16Format::BF16);
    constexpr std::uint32_t idesc_pv =
        sm::make_umma_idesc_f16(C::kRows, C::kD, sm::UmmaF16Format::F16, sm::UmmaF16Format::F16,
                                sm::UmmaMajor::K, sm::UmmaMajor::MN);
    const std::uint32_t q_addr = common::smem_addr(sh.q);
    const std::uint32_t p_addr = common::smem_addr(sh.p);
    const std::uint32_t o_addr = tmem + C::kOCol;

    sm::mbarrier_wait(sh.q_full, 0);
    sm::fence_after_thread_sync();

    const auto issue_pv = [&](int tile) {
        const int stage = tile & 1;
        sm::mbarrier_wait(sh.p_full, tile & 1);
        sm::fence_after_thread_sync();
        const std::uint32_t v_addr = common::smem_addr(sh.v_stage(stage));
#pragma unroll
        for (int t = 0; t < C::kKeys / 16; ++t) {
            const std::uint64_t a = sm::umma_desc_advance(
                sm::make_umma_k_major_desc(p_addr, sm::UmmaSwizzle::B128), t * 32);
            // MN-major V: 64-feature chunks are 8192 B apart (LBO), 8-key groups 1024 B (SBO).
            const std::uint64_t b = sm::make_umma_smem_desc(v_addr + t * 16 * 128, 8192, 1024,
                                                            sm::UmmaSwizzle::B128);
            sm::umma_f16_ss<1>(o_addr, a, b, idesc_pv, (tile | t) != 0);
        }
        sm::umma_commit<1>(sh.pv_done);
        sm::umma_commit<1>(&sh.kv_empty[stage]);
    };

    for (int j = 0; j < n_tiles; ++j) {
        const int stage = j & 1;
        const std::uint32_t parity = (j >> 1) & 1;
        sm::mbarrier_wait(&sh.kv_full[stage], parity);
        sm::mbarrier_wait(&sh.s_empty[stage], parity ^ 1u);
        sm::fence_after_thread_sync();
        const std::uint32_t k_addr = common::smem_addr(sh.k_stage(stage));
        const std::uint32_t s_addr = tmem + stage * C::kSCols;
        if constexpr (!C::kNative) {
#pragma unroll
            for (int c = 0; c < C::kD / 64; ++c)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const std::uint64_t a = sm::umma_desc_advance(
                        sm::make_umma_k_major_desc(q_addr + c * 16384, sm::UmmaSwizzle::B128),
                        t * 32);
                    const std::uint64_t b = sm::umma_desc_advance(
                        sm::make_umma_k_major_desc(k_addr + c * 8192, sm::UmmaSwizzle::B128),
                        t * 32);
                    sm::umma_f16_ss<1>(s_addr, a, b, idesc_qk, (c | t) != 0);
                }
        } else {
#pragma unroll
            for (int g = 0; g < 2; ++g)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const std::uint64_t a = sm::umma_desc_advance(
                        sm::make_umma_k_major_desc(q_addr + g * 16384, sm::UmmaSwizzle::B128),
                        t * 32);
                    const std::uint64_t b = sm::umma_desc_advance(
                        sm::make_umma_k_major_desc(k_addr + g * 8192, sm::UmmaSwizzle::B128),
                        t * 32);
                    sm::umma_f8f6f4_ss<1>(s_addr + g * C::kKeys, a, b, idesc_qk, t != 0);
                }
        }
        sm::umma_commit<1>(&sh.s_full[stage]);
        if (j > 0) issue_pv(j - 1);
    }
    issue_pv(n_tiles - 1);
    sm::umma_commit<1>(sh.o_full);
}

// Warps 0-3: online softmax, thread t = packed row t. Scores are kept in the log2 domain
// (x = s * scale * log2e); the running maximum m, running sum l and the O accumulator rescaling
// follow the usual FlashAttention recurrence. P is rounded to fp16 for the P.V MMA while l sums the
// FP32 probabilities, as in the mma.sync kernels.
template <KvDType Kv>
__device__ __forceinline__ void softmax_role(const FmhaParams& p, const Work& w,
                                             const Shared<Cfg<Kv>>& sh, std::uint32_t tmem, int tid,
                                             float& m_out, float& l_out) {
    using C            = Cfg<Kv>;
    const int warp     = tid >> 5;
    const int row_g    = w.row_begin + tid;
    const int token    = row_g / kGroup;
    const bool row_ok  = token < w.live;
    const std::int32_t* positions = p.positions + w.token_base;
    const int qabs     = row_ok ? positions[token] : -1;
    const int limit    = min(qabs + 1, w.end);
    const std::uint32_t t_lane = tmem + (static_cast<std::uint32_t>(warp * 32) << 16);

    float scale_log2 = p.scale * common::kLog2E;
    if constexpr (C::kNative) {
        sm::mbarrier_wait(sh.q_full, 0);
        scale_log2 *= sh.q_scale[tid];
    }
    const float kNegInf = -CUDART_INF_F;
    float m = kNegInf, l = 0.0f;

    for (int j = 0; j < w.n_tiles; ++j) {
        const int stage            = j & 1;
        const std::uint32_t parity = (j >> 1) & 1;
        sm::mbarrier_wait(&sh.s_full[stage], parity);
        if constexpr (C::kNative) sm::mbarrier_wait(&sh.kv_full[stage], parity);
        sm::fence_after_thread_sync();

        float x[C::kKeys];
        {
            std::uint32_t r0[32], r1[32];
            const std::uint32_t s_addr = t_lane + stage * C::kSCols;
            if constexpr (!C::kNative) {
                sm::tmem_load_32x32b<32>(s_addr, r0);
                sm::tmem_load_32x32b<32>(s_addr + 32, r1);
                sm::tmem_load_wait();
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    x[i]      = __uint_as_float(r0[i]);
                    x[i + 32] = __uint_as_float(r1[i]);
                }
            } else {
                // Two 128-feature groups: s = S0 * ks0 + S1 * ks1 per key.
                const float* ks = sh.k_scale + stage * 2 * C::kKeys;
                std::uint32_t g0[32], g1[32];
                sm::tmem_load_32x32b<32>(s_addr, r0);
                sm::tmem_load_32x32b<32>(s_addr + 32, r1);
                sm::tmem_load_32x32b<32>(s_addr + 64, g0);
                sm::tmem_load_32x32b<32>(s_addr + 96, g1);
                sm::tmem_load_wait();
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    x[i]      = __uint_as_float(r0[i]) * ks[2 * i] +
                                __uint_as_float(g0[i]) * ks[2 * i + 1];
                    x[i + 32] = __uint_as_float(r1[i]) * ks[2 * (i + 32)] +
                                __uint_as_float(g1[i]) * ks[2 * (i + 32) + 1];
                }
            }
        }
        sm::fence_before_thread_sync();
        sm::mbarrier_arrive(&sh.s_empty[stage]);

        const int k0 = w.split_start + j * C::kKeys;
        float tile_max = kNegInf;
#pragma unroll
        for (int i = 0; i < C::kKeys; ++i) {
            x[i]     = k0 + i < limit ? x[i] * scale_log2 : kNegInf;
            tile_max = fmaxf(tile_max, x[i]);
        }
        const float m_new = fmaxf(m, tile_max);
        const float alpha = m == kNegInf ? 0.0f : common::exp2_approx(m - m_new);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < C::kKeys; ++i) {
            x[i] = x[i] == kNegInf ? 0.0f : common::exp2_approx(x[i] - m_new);
            sum += x[i];
        }
        l = l * alpha + sum;
        m = m_new;

        if (j > 0) {
            sm::mbarrier_wait(sh.pv_done, (j - 1) & 1);
            sm::fence_after_thread_sync();
        }
#pragma unroll
        for (int v = 0; v < C::kKeys / 8; ++v) {
            const __half2 h0 = __floats2half2_rn(x[8 * v], x[8 * v + 1]);
            const __half2 h1 = __floats2half2_rn(x[8 * v + 2], x[8 * v + 3]);
            const __half2 h2 = __floats2half2_rn(x[8 * v + 4], x[8 * v + 5]);
            const __half2 h3 = __floats2half2_rn(x[8 * v + 6], x[8 * v + 7]);
            *reinterpret_cast<uint4*>(sh.p + sw128(tid, v)) =
                make_uint4(*reinterpret_cast<const unsigned*>(&h0),
                           *reinterpret_cast<const unsigned*>(&h1),
                           *reinterpret_cast<const unsigned*>(&h2),
                           *reinterpret_cast<const unsigned*>(&h3));
        }
        if (j > 0 && __any_sync(0xffffffffu, alpha != 1.0f)) {
            const std::uint32_t o_lane = t_lane + C::kOCol;
#pragma unroll 2
            for (int cc = 0; cc < C::kD / 32; ++cc) {
                std::uint32_t r[32];
                sm::tmem_load_32x32b<32>(o_lane + cc * 32, r);
                sm::tmem_load_wait();
#pragma unroll
                for (int i = 0; i < 32; ++i) r[i] = __float_as_uint(__uint_as_float(r[i]) * alpha);
                tmem_store_32x32b_x32(o_lane + cc * 32, r);
            }
            sm::tmem_store_wait();
        }
        sm::fence_proxy_async_shared();
        sm::fence_before_thread_sync();
        sm::mbarrier_arrive(sh.p_full);
    }
    m_out = m;
    l_out = l;
}

// Warps 0-3 when the CTA has no keys: a neutral partial (one split) or a zero output row.
__device__ __forceinline__ void write_empty(const FmhaParams& p, const Work& w, int tid) {
    const int row_g = w.row_begin + tid;
    const int token = row_g / kGroup;
    if (token >= w.width) return;
    const int head = w.kv_head * kGroup + row_g % kGroup;
    if (p.splits == 1) {
        auto* out = reinterpret_cast<uint4*>(
            p.out + causal_q_index<Geometry>(head, 0, token + w.token_base));
#pragma unroll
        for (int i = 0; i < 32; ++i) out[i] = make_uint4(0, 0, 0, 0);
    } else if (token < w.live) {
        const std::int64_t batch_acc =
            static_cast<std::int64_t>(w.batch) * 256 * Geometry::QHeads * p.width * p.splits;
        const std::int64_t batch_stat =
            static_cast<std::int64_t>(w.batch) * Geometry::QHeads * p.width * p.splits;
        auto* acc = reinterpret_cast<float4*>(
            p.partial.acc + batch_acc + causal_partial_index<Geometry>(head, 0, token, w.split, p.width));
#pragma unroll 8
        for (int i = 0; i < 64; ++i) acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const auto stat = batch_stat + causal_stat_index<Geometry>(head, token, w.split, p.width);
        p.partial.maximum[stat] = -CUDART_INF_F;
        p.partial.sum[stat]     = 0.0f;
    }
}

// Warps 0-3 after the last P.V: publish O.
//   splits == 1  normalized BF16 rows (rotated chains transform back through shared memory so a
//                warp owns complete D256 rows for the Hadamard), zero rows for masked columns
//   splits  > 1  FP32 numerator plus (natural-unit maximum, sum) for the shared merge
template <KvDType Kv>
__device__ __forceinline__ void epilogue(const FmhaParams& p, const Work& w,
                                         const Shared<Cfg<Kv>>& sh, std::uint32_t tmem, int tid,
                                         float m, float l) {
    using C        = Cfg<Kv>;
    const int warp = tid >> 5, lane = tid & 31;
    sm::mbarrier_wait(sh.o_full, 0);
    sm::fence_after_thread_sync();
    const std::uint32_t o_lane = tmem + (static_cast<std::uint32_t>(warp * 32) << 16) + C::kOCol;

    const int row_g  = w.row_begin + tid;
    const int token  = row_g / kGroup;
    const int head   = w.kv_head * kGroup + row_g % kGroup;
    const bool in_range = token < w.width;
    const bool row_ok   = token < w.live;

    if (p.splits > 1) {
        const std::int64_t batch_acc =
            static_cast<std::int64_t>(w.batch) * 256 * Geometry::QHeads * p.width * p.splits;
        const std::int64_t batch_stat =
            static_cast<std::int64_t>(w.batch) * Geometry::QHeads * p.width * p.splits;
        float* acc = p.partial.acc + batch_acc +
                     causal_partial_index<Geometry>(head, 0, token, w.split, p.width);
#pragma unroll 2
        for (int cc = 0; cc < C::kD / 32; ++cc) {
            std::uint32_t r[32];
            sm::tmem_load_32x32b<32>(o_lane + cc * 32, r);
            sm::tmem_load_wait();
            if (row_ok) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    *reinterpret_cast<float4*>(acc + cc * 32 + 4 * i) =
                        make_float4(__uint_as_float(r[4 * i]), __uint_as_float(r[4 * i + 1]),
                                    __uint_as_float(r[4 * i + 2]), __uint_as_float(r[4 * i + 3]));
            }
        }
        if (row_ok) {
            const auto stat = batch_stat + causal_stat_index<Geometry>(head, token, w.split, p.width);
            p.partial.maximum[stat] = m == -CUDART_INF_F ? m : m * 0.69314718055994531f;
            p.partial.sum[stat]     = l;
        }
        return;
    }

    const float inv = row_ok && l > 0.0f ? __frcp_rn(l) : 0.0f;
    if constexpr (!C::Policy::kInverseRotate) {
        __nv_bfloat16* out =
            p.out + causal_q_index<Geometry>(head, 0, in_range ? token + w.token_base : 0);
#pragma unroll 2
        for (int cc = 0; cc < C::kD / 32; ++cc) {
            std::uint32_t r[32];
            sm::tmem_load_32x32b<32>(o_lane + cc * 32, r);
            sm::tmem_load_wait();
            if (in_range) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint4 packed;
                    packed.x = common::pack_bf16x2(__uint_as_float(r[8 * i]) * inv,
                                                   __uint_as_float(r[8 * i + 1]) * inv);
                    packed.y = common::pack_bf16x2(__uint_as_float(r[8 * i + 2]) * inv,
                                                   __uint_as_float(r[8 * i + 3]) * inv);
                    packed.z = common::pack_bf16x2(__uint_as_float(r[8 * i + 4]) * inv,
                                                   __uint_as_float(r[8 * i + 5]) * inv);
                    packed.w = common::pack_bf16x2(__uint_as_float(r[8 * i + 6]) * inv,
                                                   __uint_as_float(r[8 * i + 7]) * inv);
                    *reinterpret_cast<uint4*>(out + cc * 32 + 8 * i) = packed;
                }
            }
        }
    } else {
        // The accumulator holds V-rotated coordinates. Transpose through shared memory (the K/V
        // ring is idle once O is final) so each warp reads whole rows, then rotate back.
        float* scratch = reinterpret_cast<float*>(sh.scratch) + warp * 32 * C::kScratchStride;
#pragma unroll 2
        for (int cc = 0; cc < C::kD / 32; ++cc) {
            std::uint32_t r[32];
            sm::tmem_load_32x32b<32>(o_lane + cc * 32, r);
            sm::tmem_load_wait();
#pragma unroll
            for (int i = 0; i < 32; ++i)
                scratch[lane * C::kScratchStride + cc * 32 + i] = __uint_as_float(r[i]);
        }
        __syncwarp();
        for (int r = 0; r < 32; ++r) {
            const int row2   = w.row_begin + warp * 32 + r;
            const int token2 = row2 / kGroup;
            const int head2  = w.kv_head * kGroup + row2 % kGroup;
            const float inv2 = __shfl_sync(0xffffffffu, inv, r);
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                v[i] = scratch[r * C::kScratchStride + lane + 32 * i] * inv2;
            kv::normalized_hadamard_d256_inplace(v, lane);
            if (token2 < w.width) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    causal_store_output(p.out + causal_q_index<Geometry>(
                                                    head2, lane + 32 * i, token2 + w.token_base),
                                        v[i]);
            }
        }
    }
}

template <KvDType Kv>
__device__ __forceinline__ void run_cta(const FmhaParams& p, const Work& w) {
    using C = Cfg<Kv>;
    extern __shared__ __align__(16) unsigned char raw[];
    const unsigned raw_address = common::smem_addr(raw);
    unsigned char* base        = raw + (((raw_address + 1023u) & ~1023u) - raw_address);
    const Shared<C> sh         = make_shared<C>(base);

    if (w.n_tiles < 0) return;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    if (w.n_tiles == 0) {
        if (tid < 128) write_empty(p, w, tid);
        return;
    }

    if (tid == 0) {
        sm::mbarrier_init(sh.q_full, 128);
        for (int s = 0; s < 2; ++s) {
            sm::mbarrier_init(&sh.kv_full[s], 128);
            sm::mbarrier_init(&sh.kv_empty[s], 1);
            sm::mbarrier_init(&sh.s_full[s], 1);
            sm::mbarrier_init(&sh.s_empty[s], 128);
        }
        sm::mbarrier_init(sh.p_full, 128);
        sm::mbarrier_init(sh.pv_done, 1);
        sm::mbarrier_init(sh.o_full, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 5) {
        sm::tmem_alloc<1>(sh.tmem_slot, C::kTmemAlloc);
        sm::tmem_relinquish_alloc_permit<1>();
    }
    __syncthreads();
    sm::fence_after_thread_sync();
    const std::uint32_t tmem = *sh.tmem_slot;

    if (warp < 4) {
        float m = 0.0f, l = 0.0f;
        softmax_role<Kv>(p, w, sh, tmem, tid, m, l);
        epilogue<Kv>(p, w, sh, tmem, tid, m, l);
    } else if (warp == 4) {
        mma_role<Kv>(sh, tmem, w.n_tiles);
    } else if (warp >= C::kLoaderWarp0) {
        const int ltid = tid - C::kLoaderWarp0 * 32;
        static_assert(C::kKeys == 64, "a cache record holds 64 keys");
        const std::int64_t* table =
            p.cache.tables +
            static_cast<std::int64_t>(p.cache.table_rows ? p.cache.table_rows[w.row] : 0) *
                p.cache.table_stride;
        const int page0 = w.split_start >> 6;
        // The first tile's loads overlap the Q preparation. A tile's page record is read two tiles
        // before the tile's loads need it.
        TileRegs<Kv> regs;
        std::int64_t record_next = w.n_tiles > 1 ? table[page0 + 1] : 0;
        load_kv<Kv>(p, w, 0, table[page0], ltid, regs);
        fill_q<Kv>(p, w, sh, ltid);
        sm::fence_proxy_async_shared();
        sm::mbarrier_arrive(sh.q_full);
        for (int j = 0; j < w.n_tiles; ++j) {
            const int stage = j & 1;
            const std::int64_t record_after = j + 2 < w.n_tiles ? table[page0 + j + 2] : 0;
            sm::mbarrier_wait(&sh.kv_empty[stage], ((j >> 1) & 1) ^ 1u);
            store_kv<Kv>(regs, stage, sh, ltid);
            sm::fence_proxy_async_shared();
            sm::mbarrier_arrive(&sh.kv_full[stage]);
            if (j + 1 < w.n_tiles) load_kv<Kv>(p, w, j + 1, record_next, ltid, regs);
            record_next = record_after;
        }
    }
    __syncthreads();
    if (warp == 5) {
        sm::fence_after_thread_sync();
        sm::tmem_dealloc<1>(tmem, C::kTmemAlloc);
    }
}

template <KvDType Kv>
__global__ __launch_bounds__(Cfg<Kv>::kThreads, 1) void fmha_kernel(const __grid_constant__ FmhaParams p) {
    run_cta<Kv>(p, make_work(p));
}

template <KvDType Kv>
__global__ __launch_bounds__(Cfg<Kv>::kThreads, 1) void fmha_group_kernel(const __grid_constant__ GroupParams r) {
    run_cta<Kv>(r.base, make_work(r));
}

} // namespace ninfer::ops::detail::sm100_fmha
