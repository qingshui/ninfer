#pragma once

// Epilogues and host-side launch plumbing shared by the NVFP4, BF16 and FP8 SM100 GEMMs.

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/cuda/sm100/gemm_mainloop.cuh"
#include "ops/linear/kernels/sm100/gemm.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::sm100_gemm {

namespace sm = ::ninfer::ops::common::sm100;

inline constexpr int kMaxSegments = 4;

// Identity, Add and Split share one epilogue: the accumulator of weight row r, token t is
// scaled, optionally added to the residual and written to the segment that owns row r. FP8
// operands carry their scales outside the MMA: the accumulator is multiplied by the weight row
// scale and the activation token scale before the residual is added.
struct PlainEpilogue {
    float alpha;
    bool add;
    const __nv_bfloat16* residual;
    __nv_bfloat16* outputs[kMaxSegments];
    int segment_begin[kMaxSegments + 1];
    int segment_count;
    int rows;
    int tokens;
    const float* row_scales;   // [rows], FP8 only
    const float* token_scales; // [tokens], FP8 only

    // The stored form of one accumulator: alpha, the FP8 row scale and token scale `token_scale()`,
    // the residual element `carry()` when adding, then the single BF16 rounding. The token scale
    // and the residual are read only where they apply.
    template <bool Fp8, class TokenScale, class Carry>
    __device__ __forceinline__ __nv_bfloat16 transform(float acc, float row_scale,
                                                       TokenScale token_scale, Carry carry) const {
        float v = acc * alpha;
        if constexpr (Fp8) v = v * row_scale * token_scale();
        if (add) v += carry();
        return __float2bfloat16_rn(v);
    }

    // What store_rows applies to the accumulators of weight rows [row, row + 4) of one token: the
    // FP8 scales and the residual. The GEMM writes none of them for the rows and tokens of a cut
    // tile, since it stores only whole tiles and a residual that aliases the output does so element
    // for element, so a reduction can load them before the partials are ready.
    struct RowOperands {
        float row_scale[4];
        float token_scale;
        uint2 carried; // four BF16 residual elements
    };

    template <class Cfg>
    __device__ __forceinline__ RowOperands row_operands(int row, int token) const {
        RowOperands o{{1.0f, 1.0f, 1.0f, 1.0f}, 1.0f, make_uint2(0, 0)};
        if constexpr (Cfg::kKind == sm::GemmKind::Fp8) {
            o.token_scale = token_scales[token];
#pragma unroll
            for (int i = 0; i < 4; ++i) o.row_scale[i] = row_scales[row + i];
        }
        if (add)
            o.carried = *reinterpret_cast<const uint2*>(
                residual + static_cast<std::int64_t>(token) * rows + row);
        return o;
    }

    // Stores the accumulators of weight rows [row, row + 4) for one token. A segment is a whole
    // number of 32-row groups, so the four rows share one.
    template <class Cfg>
    __device__ __forceinline__ void store_rows(const float (&acc)[4], const RowOperands& o, int row,
                                               int token) const {
        constexpr bool Fp8 = Cfg::kKind == sm::GemmKind::Fp8;
        int segment        = 0;
        while (segment + 1 < segment_count && row >= segment_begin[segment + 1]) ++segment;
        const int width   = segment_begin[segment + 1] - segment_begin[segment];
        const auto* carry = reinterpret_cast<const __nv_bfloat16*>(&o.carried);
        __align__(8) __nv_bfloat16 value[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            value[i] = transform<Fp8>(acc[i], o.row_scale[i], [&] { return o.token_scale; },
                                      [&] { return __bfloat162float(carry[i]); });
        *reinterpret_cast<uint2*>(outputs[segment] + static_cast<std::int64_t>(token) * width +
                                  row - segment_begin[segment]) =
            *reinterpret_cast<const uint2*>(value);
    }

    // The residual chunk is loaded in the store's lane mapping and staged in shared memory.
    template <class Cfg>
    __device__ __forceinline__ void drain(const sm::EpilogueTile& t) const {
        constexpr int C    = Cfg::kTileN >= 32 ? 32 : Cfg::kTileN;
        constexpr bool Fp8 = Cfg::kKind == sm::GemmKind::Fp8;
        const int row0     = t.m_tile * Cfg::kTileM + t.warp_q * 32;
        int segment        = 0;
        while (segment + 1 < segment_count && row0 >= segment_begin[segment + 1]) ++segment;
        __nv_bfloat16* out = outputs[segment];
        const int width    = segment_begin[segment + 1] - segment_begin[segment];
        const int column0  = row0 - segment_begin[segment];
        const int token0   = t.n_tile * Cfg::kTileN;
        const float row_scale = Fp8 ? row_scales[row0 + t.lane] : 1.0f;
        auto* stage        = reinterpret_cast<__nv_bfloat16*>(t.scratch);
        for (int c0 = 0; c0 < t.valid_tokens; c0 += C) {
            float token_scale = 0.0f;
            if constexpr (Fp8) {
                const int token = token0 + c0 + t.lane;
                if (token < tokens) token_scale = token_scales[token];
            }
            uint4 carried[C / 8];
            if (add) {
#pragma unroll
                for (int it = 0; it < C / 8; ++it) {
                    const int token = token0 + c0 + it * 8 + (t.lane >> 2);
                    carried[it]     = token < tokens
                                          ? *reinterpret_cast<const uint4*>(
                                                residual + static_cast<std::int64_t>(token) * rows +
                                                row0 + (t.lane & 3) * 8)
                                          : make_uint4(0, 0, 0, 0);
                }
            }
            std::uint32_t r[C];
            sm::tmem_load_32x32b<C>(t.tmem + c0, r);
            if (add) {
#pragma unroll
                for (int it = 0; it < C / 8; ++it)
                    *reinterpret_cast<uint4*>(stage + (it * 8 + (t.lane >> 2)) * 32 +
                                              (t.lane & 3) * 8) = carried[it];
                __syncwarp();
            }
            sm::tmem_load_wait();
#pragma unroll
            for (int i = 0; i < C; ++i) {
                stage[i * 32 + t.lane] = transform<Fp8>(
                    __uint_as_float(r[i]), row_scale,
                    [&] { return __shfl_sync(0xffffffffU, token_scale, i); },
                    [&] { return __bfloat162float(stage[i * 32 + t.lane]); });
            }
            __syncwarp();
#pragma unroll
            for (int it = 0; it < C / 8; ++it) {
                const int local = it * 8 + (t.lane >> 2);
                const int token = token0 + c0 + local;
                if (token < tokens) {
                    const uint4 value =
                        *reinterpret_cast<const uint4*>(stage + local * 32 + (t.lane & 3) * 8);
                    *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(token) * width +
                                              column0 + (t.lane & 3) * 8) = value;
                }
            }
            __syncwarp();
        }
    }
};

// Split K: the FP32 partial blocks of the cut tiles, reduced by split_finalize_kernel.
struct SplitFinalize {
    const float* partials;
    int tiles_n;
    int first_tile; // first cut tile of the launch
    int k_splits;   // at most kSm100MaxKSplits
};

// Sums the K partials of every cut tile in range order and stores the sum the way PlainEpilogue
// stores a whole-K accumulator. A warp owns the tile's 128 rows of one token, four per lane, and
// every lane issues all of its partial loads before the first addition; a block covers 8 tokens.
// It runs as a programmatic dependent of the GEMM that wrote the partials: the epilogue operands,
// which that GEMM does not write, are loaded before waiting for it.
template <class Cfg>
__global__ __launch_bounds__(256) void split_finalize_kernel(
    const __grid_constant__ PlainEpilogue epilogue, const __grid_constant__ SplitFinalize split) {
    const int local   = static_cast<int>(blockIdx.y) * 8 + static_cast<int>(threadIdx.x >> 5);
    const int group   = static_cast<int>(threadIdx.x & 31);
    const int tile    = split.first_tile + static_cast<int>(blockIdx.x);
    const int m       = tile / split.tiles_n;
    const int token   = (tile - m * split.tiles_n) * Cfg::kTileN + local;
    const int row     = m * Cfg::kTileM + group * 4;
    const bool active = local < Cfg::kTileN && token < epilogue.tokens;
    PlainEpilogue::RowOperands operands{};
    if (active) operands = epilogue.row_operands<Cfg>(row, token);
    pdl::wait_for_dependencies();
    if (!active) return;
    constexpr std::int64_t kBlock = static_cast<std::int64_t>(Cfg::kTileM) * Cfg::kTileN;
    const float* part = split.partials +
                        static_cast<std::int64_t>(blockIdx.x) * split.k_splits * kBlock +
                        local * Cfg::kTileM + group * 4;
    float4 parts[kSm100MaxKSplits];
#pragma unroll
    for (int s = 0; s < kSm100MaxKSplits; ++s)
        if (s < split.k_splits) parts[s] = *reinterpret_cast<const float4*>(part + s * kBlock);
    float acc[4] = {parts[0].x, parts[0].y, parts[0].z, parts[0].w};
#pragma unroll
    for (int s = 1; s < kSm100MaxKSplits; ++s) {
        if (s < split.k_splits) {
            acc[0] += parts[s].x;
            acc[1] += parts[s].y;
            acc[2] += parts[s].z;
            acc[3] += parts[s].w;
        }
    }
    epilogue.store_rows<Cfg>(acc, operands, row, token);
}

// Gate and up weight tiles accumulate side by side (accumulator part 0 / part 1), so one thread
// owns both operands of an output element: y = silu(gate * alpha) * (up * alpha).
struct SwiGluEpilogue {
    float alpha;
    __nv_bfloat16* out;
    int out_rows; // rows / 2
    int tokens;

    template <class Cfg>
    __device__ __forceinline__ void drain(const sm::EpilogueTile& t) const {
        static_assert(Cfg::kParts == 2);
        constexpr int C = Cfg::kTileN >= 32 ? 32 : Cfg::kTileN;
        const int row0  = t.m_tile * Cfg::kTileM + t.warp_q * 32;
        const int token0 = t.n_tile * Cfg::kTileN;
        auto* stage     = reinterpret_cast<__nv_bfloat16*>(t.scratch);
        for (int c0 = 0; c0 < t.valid_tokens; c0 += C) {
            std::uint32_t gate[C];
            std::uint32_t up[C];
            sm::tmem_load_32x32b<C>(t.tmem + c0, gate);
            sm::tmem_load_32x32b<C>(t.tmem + t.part_stride + c0, up);
            sm::tmem_load_wait();
#pragma unroll
            for (int i = 0; i < C; ++i) {
                const float g = __uint_as_float(gate[i]) * alpha;
                const float u = __uint_as_float(up[i]) * alpha;
                stage[i * 32 + t.lane] = __float2bfloat16_rn(g / (1.0f + __expf(-g)) * u);
            }
            __syncwarp();
#pragma unroll
            for (int it = 0; it < C / 8; ++it) {
                const int local = it * 8 + (t.lane >> 2);
                const int token = token0 + c0 + local;
                if (token < tokens) {
                    const uint4 value =
                        *reinterpret_cast<const uint4*>(stage + local * 32 + (t.lane & 3) * 8);
                    *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(token) * out_rows +
                                              row0 + (t.lane & 3) * 8) = value;
                }
            }
            __syncwarp();
        }
    }
};

// Router scores: FP32 accumulators stored as scores[token * stride + row] for the weight rows that
// exist; the lanes of a warp own consecutive rows, so each token's store is one contiguous run.
struct ScoresEpilogue {
    float* scores;
    int stride;
    int rows;

    template <class Cfg>
    __device__ __forceinline__ void drain(const sm::EpilogueTile& t) const {
        constexpr int C = Cfg::kTileN >= 32 ? 32 : Cfg::kTileN;
        const int row   = t.m_tile * Cfg::kTileM + t.warp_q * 32 + t.lane;
        const int token0 = t.n_tile * Cfg::kTileN;
        for (int c0 = 0; c0 < t.valid_tokens; c0 += C) {
            std::uint32_t r[C];
            sm::tmem_load_32x32b<C>(t.tmem + c0, r);
            sm::tmem_load_wait();
            if (row < rows) {
#pragma unroll
                for (int i = 0; i < C; ++i) {
                    if (c0 + i < t.valid_tokens) {
                        scores[static_cast<std::int64_t>(token0 + c0 + i) * stride + row] =
                            __uint_as_float(r[i]);
                    }
                }
            }
        }
    }
};

inline int device_sm_count() {
    static std::array<int, 64> counts{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= static_cast<int>(counts.size()))
        throw std::runtime_error("sm100 gemm: device ordinal out of range");
    if (counts[device] == 0)
        CUDA_CHECK(cudaDeviceGetAttribute(&counts[device], cudaDevAttrMultiProcessorCount, device));
    return counts[device];
}

// Split costs, calibrated on the B200 decode projections that pass partials: the reduction kernel
// that follows a split GEMM in K iterations of a 16-token tile, and the share of a round a cut
// last round still takes after full rounds.
inline constexpr double kSplitReduceCost = 4.0;
inline constexpr double kSplitTailFloor  = 0.4;

// The plan that finishes the call soonest, in whole-K rounds of a 16-token tile. The persistent
// grid runs ceil(tiles / SMs) rounds; a tile's K iteration takes about the same for 16- and
// 32-token tiles and longer for wider ones (measured relative costs below), so a narrow tile that
// keeps every weight tile in a single round beats a wide one that needs several. A two-part
// (SwiGLU) tile holds a gate and an up weight tile: it fits at most 128 tokens in tensor memory,
// and there its 128-token tile measures 2.0 rather than 2.2. With partials, a call of at most
// kSm100MaxSplitTokens tokens may run each tile of its last round as k_splits K ranges on the SMs
// that round leaves idle; such a call streams each weight tile about once, and the same
// calibration measures its one-part 64-token tile at 1.2. `only_tile` fixes the token tile. Ties
// keep the narrower tile and the fewer K ranges.
inline Sm100GemmPlan plan_gemm(int weight_tiles, int tokens, int k_iters, int weight_parts,
                               std::size_t partial_bytes, int only_tile = 0) {
    struct Candidate {
        int tile;
        double cost;
    };
    constexpr Candidate kCandidates[] = {{16, 1.0}, {32, 1.0}, {64, 1.4}, {128, 2.2}, {256, 3.8}};
    const int max_tile    = weight_parts == 2 ? 128 : 256;
    const long sms        = device_sm_count();
    const bool splittable = partial_bytes != 0 && tokens <= kSm100MaxSplitTokens;
    Sm100GemmPlan best{16};
    double lowest       = 0.0;
    const auto consider = [&](const Sm100GemmPlan& plan, double time) {
        if (lowest == 0.0 || time < lowest) {
            lowest = time;
            best   = plan;
        }
    };
    for (const auto& c : kCandidates) {
        if (c.tile > max_tile) break;
        if (only_tile != 0 && c.tile != only_tile) continue;
        double cost = c.cost;
        if (weight_parts == 2 && c.tile == 128) cost = 2.0;
        if (splittable && weight_parts == 1 && c.tile == 64) cost = 1.2;
        const long tiles  = static_cast<long>(weight_tiles) * ((tokens + c.tile - 1) / c.tile);
        const long rounds = (tiles + sms - 1) / sms;
        consider({c.tile}, static_cast<double>(rounds) * cost);
        const long rest = tiles - (rounds - 1) * sms; // tiles of the last round
        if (!splittable || c.tile > kSm100MaxSplitTokens || rest == sms) continue;
        const std::size_t unit_bytes = static_cast<std::size_t>(c.tile) * 128 * sizeof(float);
        const long units =
            std::min<long>(kSm100MaxSplitUnits, static_cast<long>(partial_bytes / unit_bytes));
        const long max_splits = std::min({sms / rest, static_cast<long>(k_iters), units / rest,
                                          static_cast<long>(kSm100MaxKSplits)});
        for (long splits = 2; splits <= max_splits; ++splits) {
            double tail = static_cast<double>((k_iters + splits - 1) / splits) / k_iters;
            if (rounds > 1) tail = std::max(tail, kSplitTailFloor);
            consider({c.tile, static_cast<int>(rest), static_cast<int>(splits)},
                     (static_cast<double>(rounds - 1) + tail) * cost + kSplitReduceCost / k_iters);
        }
    }
    return best;
}

template <class Cfg, class Epilogue, bool SplitK = false>
void launch_kernel(const CUtensorMap& a_map, const CUtensorMap& b_map, const sm::GemmArgs& args,
                   const Epilogue& epilogue, cudaStream_t stream) {
    static std::array<bool, 64> configured{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (!configured[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(sm::sm100_gemm_kernel<Cfg, Epilogue, SplitK>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Cfg::kSharedBytes));
        configured[device] = true;
    }
    const sm::StaticTileScheduler scheduler{args.tiles_m, args.tiles_n, args.k_iters,
                                            args.split_tiles, args.k_splits};
    const int grid =
        sm::persistent_grid(SplitK ? scheduler.units() : scheduler.total(), device_sm_count());
    sm::sm100_gemm_kernel<Cfg, Epilogue, SplitK>
        <<<grid, Cfg::kThreads, Cfg::kSharedBytes, stream>>>(a_map, b_map, args, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

[[noreturn]] inline void reject(const char* what) {
    throw std::invalid_argument(std::string("sm100 gemm: ") + what);
}

inline bool aligned(const void* p, std::uintptr_t a) {
    return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % a == 0;
}

// Runs a plain-epilogue GEMM with the split of `plan`. A split plan runs the split instance of the
// GEMM, which leaves the cut tiles' partials in `partials`, and split_finalize_kernel stores them;
// any other plan runs the whole-K instance.
template <class Cfg>
void launch_plain(const CUtensorMap& a_map, const CUtensorMap& b_map, sm::GemmArgs args,
                  const PlainEpilogue& epilogue, const Sm100GemmPlan& plan, float* partials,
                  cudaStream_t stream) {
    if (plan.split_tiles == 0) return launch_kernel<Cfg>(a_map, b_map, args, epilogue, stream);
    if constexpr (Cfg::kTileN > kSm100MaxSplitTokens) {
        reject("split K needs a token tile of at most kSm100MaxSplitTokens");
    } else {
        args.split_tiles = plan.split_tiles;
        args.k_splits    = plan.k_splits;
        args.partials    = partials;
        launch_kernel<Cfg, PlainEpilogue, true>(a_map, b_map, args, epilogue, stream);
        const SplitFinalize split{partials, args.tiles_n,
                                  args.tiles_m * args.tiles_n - plan.split_tiles, plan.k_splits};
        const pdl::LaunchConfig launch{dim3(static_cast<unsigned>(plan.split_tiles),
                                            (Cfg::kTileN + 7) / 8),
                                       dim3(256), 0, stream};
        CUDA_CHECK(pdl::launch_dependent(launch, split_finalize_kernel<Cfg>, epilogue, split));
    }
}

// Checks the epilogue against the problem and flattens it into kernel parameters.
inline PlainEpilogue make_plain_epilogue(const Sm100Epilogue& e, __nv_bfloat16* y, float alpha,
                                         int rows, int tokens) {
    if (e.kind == Sm100Epilogue::Scores) reject("the scores epilogue belongs to the BF16 GEMM");
    PlainEpilogue p{};
    p.alpha   = alpha;
    p.add     = e.kind == Sm100Epilogue::Add;
    p.residual = e.residual;
    p.rows    = rows;
    p.tokens  = tokens;
    if (p.add && !aligned(e.residual, 16))
        reject("Add epilogue requires a 16-byte aligned residual");
    if (e.kind == Sm100Epilogue::Split) {
        const int n = static_cast<int>(e.segments.size());
        if (n < 1 || n > kMaxSegments || e.outputs.size() != e.segments.size())
            reject("Split needs 1..4 segments with one output each");
        int begin = 0;
        for (int i = 0; i < n; ++i) {
            if (e.segments[i] <= 0 || e.segments[i] % 32 != 0)
                reject("segment widths must be positive multiples of 32");
            if (!aligned(e.outputs[i], 16)) reject("segment outputs must be 16-byte aligned");
            p.outputs[i]       = e.outputs[i];
            p.segment_begin[i] = begin;
            begin += e.segments[i];
        }
        if (begin != rows) reject("segments must sum to the weight rows");
        p.segment_begin[n] = begin;
        p.segment_count    = n;
    } else {
        if (!aligned(y, 16)) reject("output must be 16-byte aligned");
        p.outputs[0]       = y;
        p.segment_begin[0] = 0;
        p.segment_begin[1] = rows;
        p.segment_count    = 1;
    }
    return p;
}

} // namespace ninfer::ops::detail::sm100_gemm
