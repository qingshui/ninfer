#pragma once

#include "ops/sampling/kernels/random.cuh"
#include "ops/common/numeric/math.h"
#include "ops/sampling/kernels/workspace.h"
#include "ops/common/cuda/reduce.cuh"
#include "ninfer/ops/sampling.h"

#include <cub/block/block_merge_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops::sampling_detail {
using namespace common;

struct TokenRow {
    const std::int32_t* data = nullptr;
    std::int64_t stride      = 1;

    __host__ __device__ TokenRow() = default;

    __host__ __device__ TokenRow(const std::int32_t* data, std::int64_t stride = 1)
        : data(data), stride(stride) {}

    __device__ std::int32_t operator[](int column) const { return data[column * stride]; }

    __device__ explicit operator bool() const { return data != nullptr; }
};

struct TokenRows {
    const std::int32_t* data   = nullptr;
    std::int64_t column_stride = 1, row_stride = 0;

    __device__ TokenRow operator[](int row) const {
        return {data ? data + row * row_stride : nullptr, column_stride};
    }
};

inline TokenRows token_rows(const Tensor& tensor) {
    return {static_cast<const std::int32_t*>(tensor.data), tensor.nb[0] / 4, tensor.nb[1] / 4};
}

template <class S>
using SamplingPartialSort =
    cub::BlockMergeSort<unsigned long long, S::kSamplerBlock, S::kSamplerItemsPerThread>;
template <class S>
using SamplingGroupSort =
    cub::BlockMergeSort<unsigned long long, S::kSamplerGroupBlock, S::kSamplerGroupItemsPerThread>;

template <class S>
using SamplingTileWarpSort = cub::WarpMergeSort<unsigned long long, S::kSamplerItemsPerThread>;
template <class S>
using SamplingBf16TileWarpSort = cub::WarpMergeSort<unsigned int, S::kSamplerItemsPerThread>;

template <class S>
struct SamplingTileTopKStorage {
    typename SamplingTileWarpSort<S>::TempStorage warp[S::kSamplingTileWarps];
    unsigned long long candidates[S::kSamplingTileWarps * kSamplerCandidateCap];
};

template <class S>
struct SamplingBf16TileTopKStorage {
    typename SamplingBf16TileWarpSort<S>::TempStorage warp[S::kSamplingTileWarps];
    unsigned int candidates[S::kSamplingTileWarps * kSamplerCandidateCap];
};

template <class S>
union SamplingPartialTopKStorage {
    SamplingTileTopKStorage<S> fp32;
    SamplingBf16TileTopKStorage<S> bf16;
};

template <class S>
__device__ __forceinline__ unsigned int sampling_bf16_tile_sort_key(__nv_bfloat16 v,
                                                                    int tile_index) {
    unsigned int bits = __bfloat16_as_ushort(v);
    if ((bits & 0x7fffu) == 0u) { bits = 0u; }
    const unsigned int ordered = (bits & 0x8000u) ? ((~bits) & 0xffffu) : (bits ^ 0x8000u);
    return (ordered << S::kSamplingTileIndexBits) |
           (S::kSamplingTileIndexMask - static_cast<unsigned int>(tile_index));
}

template <class S>
__device__ __forceinline__ unsigned long long sampling_expand_bf16_tile_sort_key(unsigned int key,
                                                                                 int tile_start) {
    if (key == 0u) { return 0ull; }
    const unsigned int ordered = key >> S::kSamplingTileIndexBits;
    const unsigned int bits    = (ordered & 0x8000u) ? (ordered ^ 0x8000u) : ((~ordered) & 0xffffu);
    const int token            = tile_start + static_cast<int>(S::kSamplingTileIndexMask) -
                      static_cast<int>(key & S::kSamplingTileIndexMask);
    return score_id_order_key(__uint_as_float(bits << 16), token);
}

__device__ __forceinline__ int sampling_candidate_cap(const SamplingConfig& cfg,
                                                      std::int32_t vocab) {
    // top_k is clamped to the pipeline cap: top_k <= 0 (no explicit limit) or a
    // top_k larger than the cap both select the full kSamplerCandidateCap set.
    int cap = kSamplerCandidateCap;
    if (cfg.top_k > 0 && cfg.top_k < cap) { cap = cfg.top_k; }
    if (vocab < cap) { cap = vocab; }
    return cap;
}

__device__ __forceinline__ int sampling_partial_offset(const Workspace& workspace, int col,
                                                       int partial, int j) {
    return ((col * workspace.partial_stride + partial) * kSamplerCandidateCap) + j;
}

template <class S>
__device__ inline void
sampling_store_tile_topk(unsigned long long (&keys)[S::kSamplerItemsPerThread], int cap,
                         Workspace workspace, int col, int partial,
                         SamplingTileTopKStorage<S>& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SamplingTileWarpSort<S>(storage.warp[warp]).Sort(keys, ScoreIdOrderGreater{});
#pragma unroll
    for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
        const int rank = lane * S::kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    if (warp != 0) { return; }
    if (lane < S::kSamplingTileWarps) {
        constexpr unsigned int kMergeMask = (1u << S::kSamplingTileWarps) - 1u;
        int position                      = 0;
        for (int rank = 0; rank < cap; ++rank) {
            const unsigned long long key =
                storage.candidates[lane * kSamplerCandidateCap + position];
            const unsigned int high     = static_cast<unsigned int>(key >> 32);
            const unsigned int max_high = __reduce_max_sync(kMergeMask, high);
            const unsigned int low      = high == max_high ? static_cast<unsigned int>(key) : 0u;
            const unsigned int max_low  = __reduce_max_sync(kMergeMask, low);
            const unsigned int winners =
                __ballot_sync(kMergeMask, high == max_high && low == max_low);
            const int source = __ffs(static_cast<int>(winners)) - 1;
            if (lane == 0) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] =
                    (static_cast<unsigned long long>(max_high) << 32) | max_low;
            }
            if (lane == source) { ++position; }
        }
    }
}

template <class S>
__device__ inline void
sampling_store_bf16_tile_topk(unsigned int (&keys)[S::kSamplerItemsPerThread], int cap,
                              int tile_start, Workspace workspace, int col, int partial,
                              SamplingBf16TileTopKStorage<S>& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SamplingBf16TileWarpSort<S>(storage.warp[warp]).Sort(keys, ScoreIdOrderGreater{});
#pragma unroll
    for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
        const int rank = lane * S::kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    if (warp != 0) { return; }
    if (lane < S::kSamplingTileWarps) {
        constexpr unsigned int kMergeMask = (1u << S::kSamplingTileWarps) - 1u;
        int position                      = 0;
        for (int rank = 0; rank < cap; ++rank) {
            const unsigned int key     = storage.candidates[lane * kSamplerCandidateCap + position];
            const unsigned int best    = __reduce_max_sync(kMergeMask, key);
            const unsigned int winners = __ballot_sync(kMergeMask, key == best);
            const int source           = __ffs(static_cast<int>(winners)) - 1;
            if (lane == 0) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] =
                    sampling_expand_bf16_tile_sort_key<S>(best, tile_start);
            }
            if (lane == source) { ++position; }
        }
    }
}

__device__ __forceinline__ int sampling_dist_offset(int col, int j) {
    return col * kSamplerCandidateCap + j;
}

// Applies presence/frequency penalties to a raw logit. `overlay`/`overlay_len`
// carry a round-local count overlay: tokens already committed earlier in the
// current speculative round but not yet flushed to the global `token_counts`. For speculative
// verify column `col` the overlay is exactly drafts[0..col-1] (statically known,
// since column `col` is only consumed when every earlier draft was accepted), so
// the penalty at each column sees the same prefix a per-token sampler would.
// Non-speculative callers pass no overlay. The scan is bounded by k and
// only runs when penalties are active, so it is free on the no-penalty path.
__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v, const SamplingConfig& c,
                                                         TokenRow overlay = {},
                                                         int overlay_len  = 0) {
    if (c.mask.words && !(c.mask.words[overlay_len * c.mask.stride + v / 32] & (1u << (v % 32)))) {
        return -CUDART_INF_F;
    }
    float x = raw;
    if (c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f) { return x; }
    int cnt = c.token_counts != nullptr ? c.token_counts[v] : 0;
    for (int j = 0; j < overlay_len; ++j) {
        if (overlay[j] == v) { ++cnt; }
    }
    if (cnt > 0) { x -= c.presence_penalty; }
    if (c.frequency_penalty != 0.0f) { x -= c.frequency_penalty * static_cast<float>(cnt); }
    return x;
}

__device__ __forceinline__ void sampling_insert_candidate(float* vals, int* idxs, int cap, float v,
                                                          int idx) {
    if (cap <= 0 || !score_id_better(v, idx, vals[cap - 1], idxs[cap - 1])) { return; }
    int pos = cap - 1;
    while (pos > 0 && score_id_better(v, idx, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = v;
    idxs[pos] = idx;
}

template <class S>
__device__ inline void sampling_sort_tile_desc(float* vals, int* idxs) {
    const int tid = threadIdx.x;
    for (int size = 2; size <= S::kSamplerTileItems; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            __syncthreads();
            if (other > tid) {
                const float ov        = vals[other];
                const int oi          = idxs[other];
                const bool descending = ((tid & size) == 0);
                const bool swap       = descending ? score_id_better(ov, oi, vals[tid], idxs[tid])
                                                   : score_id_better(vals[tid], idxs[tid], ov, oi);
                if (swap) {
                    const float tv = vals[tid];
                    const int ti   = idxs[tid];
                    vals[tid]      = ov;
                    idxs[tid]      = oi;
                    vals[other]    = tv;
                    idxs[other]    = ti;
                }
            }
            __syncthreads();
        }
    }
}

__device__ inline void sampling_normalize_support(const SamplingConfig& cfg, float* cand_val,
                                                  int* cand_idx, float* prob, int* n_support,
                                                  int n) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        while (n > 0 && !isfinite(cand_val[n - 1])) { --n; }
        if (n == 0 || !isfinite(cand_val[0])) {
            // The caller provides nonempty masks. Missing finite support is a model execution
            // failure, not a distribution from which a placeholder token can be published.
            asm volatile("trap;");
        } else {
            const float inv_temp = 1.0f / cfg.temperature;
            const float m        = cand_val[0] * inv_temp;
            float sum            = 0.0f;
            for (int j = 0; j < n; ++j) {
                const float e = __expf(cand_val[j] * inv_temp - m);
                prob[j]       = e;
                sum += e;
            }
            const float e0           = prob[0];
            const float min_p_thresh = (cfg.min_p > 0.0f) ? cfg.min_p * e0 : -1.0f;
            const bool top_p_active  = (cfg.top_p < 1.0f);
            const float top_p_target = cfg.top_p * sum;
            float cum                = 0.0f;
            int support              = 0;
            for (int j = 0; j < n; ++j) {
                if (min_p_thresh >= 0.0f && prob[j] < min_p_thresh) { break; }
                cum += prob[j];
                support = j + 1;
                if (top_p_active && cum >= top_p_target) { break; }
            }
            if (support < 1) { support = 1; }
            float ssum = 0.0f;
            for (int j = 0; j < support; ++j) { ssum += prob[j]; }
            const float inv = 1.0f / ssum;
            for (int j = 0; j < support; ++j) { prob[j] *= inv; }
            *n_support = support;
        }
    }
    __syncthreads();
}

// All threads of the block must call. Small-column exact truncated support
// builder used by unit tests and any <=256-token column. It sorts one shared
// tile and then applies the same top-p/min-p renormalization as the large path.
template <class S>
__device__ inline void
sampling_build_truncated_small(const __nv_bfloat16* logits, std::int64_t base, std::int32_t vocab,
                               const SamplingConfig& cfg, float* tile_val, int* tile_idx,
                               float* cand_val, int* cand_idx, float* prob, int* n_support,
                               TokenRow overlay = {}, int overlay_len = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab);
    if (tid < S::kSamplerTileItems) {
        if (tid < vocab) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[base + tid]), tid, cfg,
                                                    overlay, overlay_len);
            tile_val[tid] = x;
            tile_idx[tid] = tid;
        } else {
            tile_val[tid] = -CUDART_INF_F;
            tile_idx[tid] = INT_MAX;
        }
    }
    __syncthreads();
    sampling_sort_tile_desc<S>(tile_val, tile_idx);
    if (tid < cap) {
        cand_val[tid] = tile_val[tid];
        cand_idx[tid] = tile_idx[tid];
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
}

// A warp's best `cap` candidates, best first: lane i holds entry i, so a candidate is inserted
// with shuffles only when it beats the cap-th entry and nothing lives in local memory.
struct SamplingWarpList {
    static constexpr unsigned kFull = 0xffffffffu;
    float entry_value;
    int entry_index;
    float worst_value;
    int worst_index;
    int cap;
    int lane;

    __device__ SamplingWarpList(int cap, int lane) : cap(cap), lane(lane) {
        entry_value = worst_value = -CUDART_INF_F;
        entry_index = worst_index = INT_MAX;
    }

    // Every lane of the warp calls together, each with its own candidate.
    __device__ __forceinline__ void offer(float value, int index, bool live) {
        unsigned pending =
            __ballot_sync(kFull, live && score_id_better(value, index, worst_value, worst_index));
        while (pending) {
            const int source = __ffs(pending) - 1;
            pending &= pending - 1;
            const float new_value = __shfl_sync(kFull, value, source);
            const int new_index   = __shfl_sync(kFull, index, source);
            // An earlier insertion of this group may have raised the cap-th entry.
            if (!score_id_better(new_value, new_index, worst_value, worst_index)) { continue; }
            const unsigned ahead = __ballot_sync(
                kFull, lane < cap && score_id_better(entry_value, entry_index, new_value,
                                                     new_index));
            const int rank          = __popc(ahead);
            const float above_value = __shfl_up_sync(kFull, entry_value, 1);
            const int above_index   = __shfl_up_sync(kFull, entry_index, 1);
            if (lane == rank) {
                entry_value = new_value;
                entry_index = new_index;
            } else if (lane > rank && lane < cap) {
                entry_value = above_value;
                entry_index = above_index;
            }
            worst_value = __shfl_sync(kFull, entry_value, cap - 1);
            worst_index = __shfl_sync(kFull, entry_index, cap - 1);
        }
    }
};

// Elements one warp reads per vector load: eight BF16 per lane.
inline constexpr int kSamplerChunk = 256;

// A lane's eight values of a row chunk, the elements first .. first + 7 for `first` a multiple of
// eight: masked, and also penalized when `penalize`. The eight elements share one mask word.
__device__ __forceinline__ void sampling_lane_values(const uint4& packed, int first,
                                                     const SamplingConfig& cfg, bool penalize,
                                                     float (&x)[8], TokenRow overlay = {},
                                                     int overlay_len = 0) {
    const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        x[2 * w]     = __uint_as_float(words[w] << 16);
        x[2 * w + 1] = __uint_as_float(words[w] & 0xffff0000u);
    }
    if (penalize) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = sampling_adjusted_logit(x[i], first + i, cfg, overlay, overlay_len);
        }
    } else if (cfg.mask.words) {
        const unsigned allowed =
            cfg.mask.words[overlay_len * cfg.mask.stride + first / 32] >> (first % 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (!((allowed >> i) & 1u)) { x[i] = -CUDART_INF_F; }
        }
    }
}

// A lower bound on the cap-th largest of the block's values: the largest value that at least `cap`
// of them reach, searched over the order keys that end in 16 zero bits, or -inf when fewer than
// `cap` of them exceed -inf. Every thread of the block calls with one value, which must not be
// NaN; the search is one block-wide count per remaining key bit.
__device__ inline float sampling_block_floor(float value, int cap) {
    const unsigned int key = ordered_score_bits(value);
    unsigned int floor_key = 0;
    for (int bit = 31; bit >= 16; --bit) {
        const unsigned int trial = floor_key | (1u << bit);
        if (__syncthreads_count(key >= trial) >= cap) { floor_key = trial; }
    }
    // The keys below that of -inf decode to NaN.
    const float floor = score_from_order_key(static_cast<std::uint64_t>(floor_key) << 32);
    return floor >= -CUDART_INF_F ? floor : -CUDART_INF_F;
}

// Single-block route for large columns when the finite multi-block workspace route cannot
// represent the launch. A 16-byte aligned row is read twice in 256-element chunks (eight BF16 per
// lane, the next group of chunks already in flight). The first pass keeps every lane's best
// element and every chunk's largest value. The lanes' best elements are distinct, so no element
// below the largest value that `cap` of them reach is a candidate, and the second pass revisits
// only the chunks that reach this floor: every warp offers their elements to a SamplingWarpList
// and the block then ranks the warps' lists against each other, so there are no thread-local
// arrays and no serial merge. A warp records the chunks of its first 32 groups and revisits every
// later chunk. Both passes apply the mask. Nonnegative penalties only lower values, so they are
// applied, with their token counts, to the lanes' best elements and to the revisited lanes whose
// unpenalized values reach both the floor and their warp list's cap-th value. The chunk tail and
// rows that are not 16-byte aligned take the scalar path.
__device__ inline void
sampling_build_truncated_block_fast(const __nv_bfloat16* logits, std::int64_t base,
                                    std::int32_t vocab, const SamplingConfig& cfg, float* merge_val,
                                    int* merge_idx, float* cand_val, int* cand_idx, float* prob,
                                    int* n_support, TokenRow overlay = {}, int overlay_len = 0) {
    static_assert(kSamplerFastCandidates <= 32, "a warp holds the candidate list, one per lane");
    constexpr unsigned kFull = 0xffffffffu;
    constexpr int kChunk     = kSamplerChunk;
    constexpr int kUnroll    = 4; // chunks in flight per warp
    const int tid            = threadIdx.x;
    const int lane           = tid & 31;
    const int warp           = tid >> 5;
    const int warps          = blockDim.x >> 5;
    // Always <= kSamplerFastCandidates.
    const int cap = sampling_candidate_cap(cfg, vocab);

    if (tid < cap) {
        cand_val[tid] = -CUDART_INF_F;
        cand_idx[tid] = INT_MAX;
    }

    SamplingWarpList list(cap, lane);

    const __nv_bfloat16* row = logits + base;
    const bool vector_row    = (reinterpret_cast<std::uintptr_t>(row) & 15) == 0;
    const int chunks         = vector_row ? vocab / kChunk : 0;
    const bool penalties     = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    // Nonnegative penalties only lower values.
    const bool bounded = penalties && cfg.presence_penalty >= 0.0f && cfg.frequency_penalty >= 0.0f;
    float floor        = -CUDART_INF_F;

    const auto load = [&](int chunk) {
        return __ldg(reinterpret_cast<const uint4*>(row + chunk * kChunk) + lane);
    };
    // A lane's eight values of a chunk, penalized unless bounded.
    const auto values = [&](const uint4& packed, int first, float (&x)[8]) {
        sampling_lane_values(packed, first, cfg, penalties && !bounded, x, overlay, overlay_len);
    };
    const auto offer_chunk = [&](const uint4& packed, int chunk) {
        const int first = chunk * kChunk + lane * 8;
        const float bar = fmaxf(floor, list.worst_value);
        float x[8];
        values(packed, first, x);
        float top = -CUDART_INF_F; // NaN values are never candidates
#pragma unroll
        for (int i = 0; i < 8; ++i) { top = fmaxf(top, x[i]); }
        const bool reaches = top >= bar;
        if (!__any_sync(kFull, reaches)) { return; }
        // A lane below the bar offers unpenalized values, which stay below it.
        if (bounded && reaches) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                x[i] = sampling_adjusted_logit(x[i], first + i, cfg, overlay, overlay_len);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) { list.offer(x[i], first + i, x[i] >= floor); }
    };
    if (chunks > 0) {
        // In its group g a warp reads the chunks warp + g * step + u * warps, u < kUnroll; lane g
        // records the order key of the largest value of each.
        constexpr int kRecorded = 32;
        const int step          = warps * kUnroll;
        uint4 current[kUnroll], upcoming[kUnroll];
        unsigned int peak[kUnroll];
        const auto fetch = [&](uint4* into, int group) {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                const int chunk = group + u * warps;
                if (chunk < chunks) { into[u] = load(chunk); }
            }
        };
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) { peak[u] = 0; } // below every value
        float best  = -CUDART_INF_F;
        int best_id = INT_MAX;
        fetch(current, warp);
        for (int group = warp, g = 0; group < chunks; group += step, ++g) {
            fetch(upcoming, group + step);
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                const int chunk = group + u * warps;
                if (chunk >= chunks) { continue; }
                const int first = chunk * kChunk + lane * 8;
                float x[8];
                values(current[u], first, x);
                float top = -CUDART_INF_F;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    top = fmaxf(top, x[i]);
                    if (x[i] > best) {
                        best    = x[i];
                        best_id = first + i;
                    }
                }
                const unsigned int largest = __reduce_max_sync(kFull, ordered_score_bits(top));
                if (lane == g) { peak[u] = largest; }
            }
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) { current[u] = upcoming[u]; }
        }
        // A bounded lane's best element contributes its penalized value.
        if (bounded && best_id != INT_MAX) {
            best = sampling_adjusted_logit(best, best_id, cfg, overlay, overlay_len);
        }
        floor = sampling_block_floor(best, cap);

        const unsigned int floor_key = ordered_score_bits(floor);
        unsigned int revisit[kUnroll], groups = 0;
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            revisit[u] = __ballot_sync(kFull, peak[u] >= floor_key);
            groups |= revisit[u];
        }
        const auto fetch_revisit = [&](uint4* into, int g) {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                if ((revisit[u] >> g) & 1u) { into[u] = load(warp + g * step + u * warps); }
            }
        };
        if (groups) {
            int g = __ffs(groups) - 1;
            groups &= groups - 1;
            fetch_revisit(current, g);
            for (;;) {
                const int next = groups ? __ffs(groups) - 1 : -1;
                if (next >= 0) {
                    groups &= groups - 1;
                    fetch_revisit(upcoming, next);
                }
#pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    if ((revisit[u] >> g) & 1u) {
                        offer_chunk(current[u], warp + g * step + u * warps);
                    }
                }
                if (next < 0) { break; }
                g = next;
#pragma unroll
                for (int u = 0; u < kUnroll; ++u) { current[u] = upcoming[u]; }
            }
        }
        for (int group = warp + kRecorded * step; group < chunks; group += step) {
            for (int u = 0; u < kUnroll; ++u) {
                const int chunk = group + u * warps;
                if (chunk < chunks) { offer_chunk(load(chunk), chunk); }
            }
        }
    }
    for (int start = chunks * kChunk + warp * 32; start < vocab; start += warps * 32) {
        const int id    = start + lane;
        const bool live = id < vocab;
        const float x   = live ? sampling_adjusted_logit(__bfloat162float(row[id]), id, cfg,
                                                         overlay, overlay_len)
                               : -CUDART_INF_F;
        list.offer(x, id, live && x >= floor);
    }

    if (lane < kSamplerFastCandidates) {
        merge_val[warp * kSamplerFastCandidates + lane] = list.entry_value;
        merge_idx[warp * kSamplerFastCandidates + lane] = list.entry_index;
    }
    __syncthreads();

    // The best `cap` of the warps' lists, ranked in parallel: a candidate's rank is the number of
    // candidates strictly ahead of it, and ranks below `cap` are the output slots. Each list is
    // ordered, so its candidates ahead of another list's candidate are a prefix, found by
    // bisection.
    for (int c = tid; c < warps * kSamplerFastCandidates; c += blockDim.x) {
        const int member = c % kSamplerFastCandidates;
        if (member >= cap) { continue; }
        const float value = merge_val[c];
        const int index   = merge_idx[c];
        if (index == INT_MAX) { continue; }
        const int owner = c / kSamplerFastCandidates;
        int rank        = member;
        for (int w = 0; w < warps && rank < cap; ++w) {
            if (w == owner) { continue; }
            const float* values_w = merge_val + w * kSamplerFastCandidates;
            const int* indices_w  = merge_idx + w * kSamplerFastCandidates;
            int low = 0, high = cap;
            while (low < high) {
                const int middle = (low + high) >> 1;
                if (score_id_better(values_w[middle], indices_w[middle], value, index)) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            rank += low;
        }
        if (rank < cap) {
            cand_val[rank] = value;
            cand_idx[rank] = index;
        }
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
}

// thread-0 helper: inverse-CDF pick over a normalized `prob[0..n-1]` support,
// optionally excluding `exclude` (a rejected draft token) and renormalizing over
// the remainder. `u` is a uniform in [0,1). Returns the chosen vocab id.
__device__ __forceinline__ int sampling_pick_from_support(const int* cand_idx, const float* prob,
                                                          int n, int exclude, float u) {
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        mass += prob[j];
    }
    if (mass <= 0.0f) {
        // Degenerate: support is only the excluded token. Return it (caller only
        // reaches this when accept probability was ~1, i.e. it will not happen).
        return cand_idx[0];
    }
    const float goal = u * mass;
    float acc        = 0.0f;
    int picked       = -1;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        acc += prob[j];
        picked = cand_idx[j];
        if (goal < acc) { return picked; }
    }
    return picked;
}

} // namespace ninfer::ops::sampling_detail
