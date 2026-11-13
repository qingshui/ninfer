#pragma once

#include "ops/softmax_attention/kernels/causal_sliding_window/constants.h"
#include "ops/softmax_attention/kernels/causal/bf16/epilogue.cuh"
#include "ops/softmax_attention/kernels/causal/bf16/schedule.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {

// Key positions p < L (the first live position of the row) come from the read-only ring at slot
// p mod 512. Positions p >= L are the row's live columns, p-L. A live V row crosses the same
// FP16 boundary as a stored ring row.
template <class G, class S>
__device__ __forceinline__ void
causal_swa_load_tile(__nv_bfloat16* key_tile, __half* value_tile, const __nv_bfloat16* ring_k,
                     const __half* ring_v, const __nv_bfloat16* new_k, const __nv_bfloat16* new_v,
                     int padded_capacity, int head, int begin, int end, int first_live, int tid) {
    constexpr int D = G::kHeadDim;
#pragma unroll 1
    for (int i = tid; i < S::kKeyRows * (D / 8); i += S::kThreads) {
        const int row = i / (D / 8), d = i % (D / 8) * 8, key = begin + row;
        const auto offset = row * D + common::swizzle_xor_64(row, d);
        auto* kd          = key_tile + offset;
        auto* vd          = value_tile + offset;
        if (key >= end) {
            common::store_vec(kd, make_int4(0, 0, 0, 0));
            common::store_vec(vd, make_int4(0, 0, 0, 0));
        } else if (key >= first_live) {
            const auto index = causal_new_index<G>(head, d, key - first_live);
            common::cp_async<16>(kd, new_k + index);
            common::store_vec(
                vd, common::bf16x8_bits_to_f16x8_bits(common::load_vec<int4>(new_v + index)));
        } else {
            const auto index =
                static_cast<std::int64_t>(D) * ((key & (kCausalSlidingWindow - 1)) +
                                                static_cast<std::int64_t>(padded_capacity) * head) +
                d;
            common::cp_async<16>(kd, ring_k + index);
            common::cp_async<16>(vd, ring_v + index);
        }
    }
}

// Packed query rows (token, group member) form one MMA tile per kv head. A CTA owns one row tile
// and one slice of the tile's key interval. Direct stores the BF16 result; otherwise the slice
// publishes FP32 partial statistics for the merge kernel.
template <class G, class S, bool Direct>
__launch_bounds__(S::kLaunchBoundThreads, S::kMinBlocks) __global__
    void causal_swa_kernel(const __nv_bfloat16* q, const __nv_bfloat16* new_k,
                           const __nv_bfloat16* new_v, const int* positions, const int* validity,
                           const std::int64_t* key_offsets, const std::int64_t* value_offsets,
                           const __nv_bfloat16* ring_k, const __half* ring_v, int padded_capacity,
                           int width, float scale, int splits, __nv_bfloat16* out,
                           CausalPartialView partial) {
    static_assert(!S::kFixedWidth);
    constexpr int D = G::kHeadDim, M = S::kQueryRows, N = S::kKeyRows;
    constexpr int NK = N / S::kWarpsKV, QKNt = NK / 8, QKKs = D / 16;
    constexpr int PVNt = D / 8, PVKs = NK / 16;
    using Storage = Bf16KvGroupedStorage<G, S>;
    static_assert(sizeof(Storage) <= 99 * 1024);
    Storage* storage;
    if constexpr (sizeof(Storage) <= 48 * 1024) {
        __shared__ Storage fixed;
        storage = &fixed;
    } else {
        extern __shared__ __align__(16) unsigned char dynamic[];
        storage = reinterpret_cast<Storage*>(dynamic);
    }
    auto* k_s     = reinterpret_cast<__nv_bfloat16*>(storage->qkv);
    auto* v_s     = reinterpret_cast<__half*>(storage->qkv + N * D);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_q = warp % S::kWarpsQ, warp_kv = warp / S::kWarpsQ;
    const int head = blockIdx.x % G::KVHeads, tile = blockIdx.x / G::KVHeads;
    const int split = blockIdx.y, batch = blockIdx.z;
    const int row_begin = tile * M, packed_rows = width * G::GroupSize;
    const int live = min(max(validity[batch], 0), width);
    q += static_cast<std::int64_t>(batch) * width * D * G::QHeads;
    positions += batch * width;
    new_k += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
    new_v += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
    if constexpr (Direct) {
        out += static_cast<std::int64_t>(batch) * width * D * G::QHeads;
    } else {
        partial.acc += static_cast<std::int64_t>(batch) * width * G::QHeads * D * splits;
        partial.maximum += static_cast<std::int64_t>(batch) * width * G::QHeads * splits;
        partial.sum += static_cast<std::int64_t>(batch) * width * G::QHeads * splits;
    }
    if (row_begin >= live * G::GroupSize) {
        // A tile with no live column: the merge zero-fills split outputs itself.
        if constexpr (Direct) {
            for (int i = tid; i < M * (D / 2); i += S::kThreads) {
                const int row = row_begin + i / (D / 2), d = i % (D / 2) * 2;
                if (row < packed_rows) {
                    bf16_kv_store_pair<G, false>(
                        partial, out, head * G::GroupSize + row % G::GroupSize, row / G::GroupSize,
                        d, width, 0, 0.0f, 0.0f, -CUDART_INF_F, 0.0f);
                }
            }
        }
        return;
    }
    const int first_live  = positions[0];
    const int first_token = row_begin / G::GroupSize;
    const int last_token  = min(live, common::div_up(row_begin + M, G::GroupSize)) - 1;
    // Union of the visible intervals of the tile's tokens.
    const int interval_begin = max(0, positions[first_token] - (kCausalSlidingWindow - 1));
    const int interval_end   = positions[last_token] + 1;
    const int slice = common::div_up(common::div_up(interval_end - interval_begin, splits), N) * N;
    const int start = interval_begin + split * slice;
    const int end   = min(interval_end, start + slice);
    if (start >= end) {
        if constexpr (!Direct) {
            for (int i = tid; i < M * (D / 2); i += S::kThreads) {
                const int row = row_begin + i / (D / 2), d = i % (D / 2) * 2;
                if (row < packed_rows) {
                    bf16_kv_store_pair<G, true>(
                        partial, nullptr, head * G::GroupSize + row % G::GroupSize,
                        row / G::GroupSize, d, width, split, 0.0f, 0.0f, -CUDART_INF_F, 0.0f);
                }
            }
        }
        return;
    }
    ring_k += key_offsets[batch] / sizeof(__nv_bfloat16);
    ring_v += value_offsets[batch] / sizeof(__half);
    for (int i = tid; i < M * (D / 8); i += S::kThreads) {
        const int row = i / (D / 8), d = i % (D / 8) * 8, packed = row_begin + row;
        const int token = packed / G::GroupSize, h = head * G::GroupSize + packed % G::GroupSize;
        auto* dst = k_s + row * D + common::swizzle_xor_64(row, d);
        if (token < live)
            common::cp_async<16>(dst, q + causal_q_index<G>(h, d, token));
        else
            common::store_vec(dst, make_int4(0, 0, 0, 0));
    }
    common::cp_commit();
    common::cp_wait<0>();
    __syncthreads();
    unsigned q_frag[QKKs][4];
    const int a_row = warp_q * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int d = k * 16 + (lane >> 4) * 8;
        common::ldmatrix_x4(q_frag[k][0], q_frag[k][1], q_frag[k][2], q_frag[k][3],
                            common::smem_addr(k_s + a_row * D + common::swizzle_xor_64(a_row, d)));
    }
    __syncthreads();
    float acc[PVNt][4] = {};
    OnlineSoftmaxRow state[2];
    const float scale_log2 = scale * common::kLog2E;
    const int gid = lane >> 2, lid = lane & 3;
    const int rows[2]   = {row_begin + warp_q * 16 + gid, row_begin + warp_q * 16 + gid + 8};
    const int tokens[2] = {rows[0] / G::GroupSize, rows[1] / G::GroupSize};
    const int qabs[2]   = {tokens[0] < live ? positions[tokens[0]] : -1,
                         tokens[1] < live ? positions[tokens[1]] : -1};
    for (int k0 = start; k0 < end; k0 += N) {
        causal_swa_load_tile<G, S>(k_s, v_s, ring_k, ring_v, new_k, new_v, padded_capacity, head,
                                   k0, end, first_live, tid);
        common::cp_commit();
        common::cp_wait<0>();
        __syncthreads();
        float score[QKNt][4] = {};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int row = warp_kv * NK + n * 8 + (lane & 7);
                const int d   = k * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x2(
                    bf[0], bf[1],
                    common::smem_addr(k_s + row * D + common::swizzle_xor_64(row, d)));
                common::mma_bf16(score[n][0], score[n][1], score[n][2], score[n][3], q_frag[k][0],
                                 q_frag[k][1], q_frag[k][2], q_frag[k][3], bf[0], bf[1]);
            }
        }
        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = k0 + warp_kv * NK + n * 8 + 2 * lid + (j & 1);
                // Visible: 0 <= qabs - key <= window - 1.
                if (key >= end || key > qabs[j / 2] || qabs[j / 2] - key >= kCausalSlidingWindow)
                    score[n][j] = -CUDART_INF_F;
                maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
            }
        }
        const float alpha[2] = {
            state[0].update(common::warp_max<4>(maximum[0], 0xffffffffu), scale_log2),
            state[1].update(common::warp_max<4>(maximum[1], 0xffffffffu), scale_log2)};
        unsigned pf[PVKs][4];
        float tile_sum[2] = {};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
            const float p0 = state[0].probability(score[n][0], scale_log2);
            const float p1 = state[0].probability(score[n][1], scale_log2);
            const float p2 = state[1].probability(score[n][2], scale_log2);
            const float p3 = state[1].probability(score[n][3], scale_log2);
            tile_sum[0] += p0 + p1;
            tile_sum[1] += p2 + p3;
            pf[n / 2][(n % 2) * 2]     = common::pack_f16x2(p0, p1);
            pf[n / 2][(n % 2) * 2 + 1] = common::pack_f16x2(p2, p3);
        }
        state[0].accumulate(alpha[0], tile_sum[0]);
        state[1].accumulate(alpha[1], tile_sum[1]);
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha[0];
            acc[n][1] *= alpha[0];
            acc[n][2] *= alpha[1];
            acc[n][3] *= alpha[1];
        }
        if constexpr (S::kWarpsQ >= 4) {
            unsigned vf[2][4];
            const auto load_v = [&](int i, int slot) {
                const int k = i / (PVNt / 2), n = i % (PVNt / 2) * 2;
                const int row = warp_kv * NK + k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int d   = n * 8 + (lane >> 4) * 8;
                common::ldmatrix_x4_t(
                    vf[slot][0], vf[slot][1], vf[slot][2], vf[slot][3],
                    common::smem_addr(v_s + row * D + common::swizzle_xor_64(row, d)));
            };
            load_v(0, 0);
#pragma unroll
            for (int i = 0; i < PVKs * (PVNt / 2); ++i) {
                const int k = i / (PVNt / 2), n = i % (PVNt / 2) * 2, slot = i & 1;
                if (i + 1 < PVKs * (PVNt / 2)) load_v(i + 1, slot ^ 1);
                common::mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1],
                                pf[k][2], pf[k][3], vf[slot][0], vf[slot][1]);
                common::mma_f16(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3],
                                pf[k][0], pf[k][1], pf[k][2], pf[k][3], vf[slot][2], vf[slot][3]);
            }
        } else {
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
#pragma unroll
                for (int k = 0; k < PVKs; ++k) {
                    unsigned vf[2];
                    const int row = warp_kv * NK + k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                    const int d   = n * 8;
                    common::ldmatrix_x2_t(
                        vf[0], vf[1],
                        common::smem_addr(v_s + row * D + common::swizzle_xor_64(row, d)));
                    common::mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1],
                                    pf[k][2], pf[k][3], vf[0], vf[1]);
                }
            }
        }
        __syncthreads();
    }
    state[0].finish();
    state[1].finish();
    float m[2]    = {state[0].maximum * scale_log2, state[1].maximum * scale_log2};
    float sums[2] = {state[0].sum, state[1].sum};
    if constexpr (S::kWarpsKV > 1) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = warp_kv * M + warp_q * 16 + gid + j * 8;
            if (lid == 0) {
                storage->maximum[row] = m[j];
                storage->sum[row]     = sums[j];
            }
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                const int d = n * 8 + 2 * lid;
                *reinterpret_cast<float2*>(storage->reduction + row * D + d) =
                    make_float2(acc[n][j * 2], acc[n][j * 2 + 1]);
            }
        }
        __syncthreads();
        if (warp_kv != 0) return;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = warp_q * 16 + gid + j * 8;
            m[j]          = -CUDART_INF_F;
            sums[j]       = 0;
#pragma unroll
            for (int w = 0; w < S::kWarpsKV; ++w) m[j] = fmaxf(m[j], storage->maximum[w * M + row]);
            float weights[S::kWarpsKV];
#pragma unroll
            for (int w = 0; w < S::kWarpsKV; ++w) {
                weights[w] = softmax_log2_state_weight(storage->maximum[w * M + row],
                                                       storage->sum[w * M + row], m[j]);
                sums[j] += weights[w] * storage->sum[w * M + row];
            }
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                float a = 0, b = 0;
#pragma unroll
                for (int w = 0; w < S::kWarpsKV; ++w) {
                    const auto x = *reinterpret_cast<const float2*>(
                        storage->reduction + (w * M + row) * D + n * 8 + 2 * lid);
                    a += weights[w] * x.x;
                    b += weights[w] * x.y;
                }
                acc[n][2 * j]     = a;
                acc[n][2 * j + 1] = b;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        if (tokens[j] < width) {
            const int h = head * G::GroupSize + rows[j] % G::GroupSize;
#pragma unroll
            for (int n = 0; n < PVNt; ++n)
                bf16_kv_store_pair<G, !Direct>(partial, out, h, tokens[j], n * 8 + 2 * lid, width,
                                               split, acc[n][2 * j], acc[n][2 * j + 1], m[j],
                                               sums[j]);
        }
    }
}

// Combines the slices of one (head, token). Tail columns are exact BF16 zero.
template <class G>
__launch_bounds__(G::kHeadDim) __global__
    void causal_swa_merge_kernel(const float* partial_acc, const float* partial_m,
                                 const float* partial_l, const int* validity, int width, int splits,
                                 __nv_bfloat16* out) {
    const int flat = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int batch = flat / width, token = flat - batch * width;
    if (token >= validity[batch]) {
        out[causal_q_index<G>(head, d, flat)] = __float2bfloat16(0.0f);
        return;
    }
    partial_acc += static_cast<std::int64_t>(batch) * G::kHeadDim * G::QHeads * width * splits;
    partial_m += static_cast<std::int64_t>(batch) * G::QHeads * width * splits;
    partial_l += static_cast<std::int64_t>(batch) * G::QHeads * width * splits;
    float maximum = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s)
        maximum = fmaxf(maximum, partial_m[causal_stat_index<G>(head, token, s, width)]);
    float denominator = 0.0f, numerator = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const auto stat = causal_stat_index<G>(head, token, s, width);
        const float l   = partial_l[stat];
        if (l > 0.0f) {
            const float weight = common::exp2_approx(partial_m[stat] - maximum);
            denominator += l * weight;
            numerator += weight * partial_acc[causal_partial_index<G>(head, d, token, s, width)];
        }
    }
    out[causal_q_index<G>(head, d, flat)] =
        __float2bfloat16(denominator > 0.0f ? numerator / denominator : 0.0f);
}

} // namespace ninfer::ops::detail
