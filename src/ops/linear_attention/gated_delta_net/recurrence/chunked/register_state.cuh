#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/workspace.h"

// Chunk recurrence for a 64-value stripe whose state never leaves the FP32 MMA accumulators.
// A warp owns sixteen value rows of S^T[value, key] and forms every product with the state or
// Delta as the A operand, so accumulator fragments feed the next MMA directly and only the
// K/Q/V fragments are read from shared memory. Operand precisions: TF32 for K.S and the two
// triangular products, BF16-rounded state for Q.S, BF16 Delta for the state update, FP32 state.
namespace ninfer::ops::detail::gated_delta_net::chunked::registers {

inline constexpr int kWarps       = 4;
inline constexpr int kThreads     = kWarps * 32;
inline constexpr int kValues      = kWarps * 16; // value columns of one CTA stripe
inline constexpr int kKeyTiles    = kStateDim / 8;
inline constexpr int kStages      = 3;
inline constexpr int kOutputPitch = 24; // BF16 elements per staged output row

using State = float[kKeyTiles][4];

// Chunk values are [token][value] with 16-byte groups XOR-permuted by token, so the
// transposing ldmatrix of eight tokens touches eight distinct bank groups.
__host__ __device__ constexpr int value_index(int row, int col) {
    return row * kValues + (((col >> 3) ^ (row & 7)) << 3) + (col & 7);
}

struct alignas(256) Stage {
    QkChunk qk;
    ControlChunk control;
    __nv_bfloat16 value[kChunkSize * kValues];
};

struct Shared {
    Stage stage[kStages];
    __nv_bfloat16 output[kWarps][kChunkSize * kOutputPitch];
};

static_assert(sizeof(Stage) == 12544);
static_assert(sizeof(Shared) == 40704);

__device__ __forceinline__ float2 load_pair(const float* source) {
    return common::load_vec<float2>(source);
}

__device__ __forceinline__ float2 load_pair(const __nv_bfloat16* source) {
    return common::bf16x2_to_float2(common::load_vec<__nv_bfloat162>(source));
}

__device__ __forceinline__ void store_pair(float* destination, float x, float y) {
    common::store_vec(destination, make_float2(x, y));
}

__device__ __forceinline__ void store_pair(__nv_bfloat16* destination, float x, float y) {
    common::store_vec(destination, common::pack_bf16x2(x, y));
}

// Fragment (g,t) holds keys 8*tile+2t, 2t+1 of value rows g and g+8. Word is the FP32 or BF16
// State storage; a BF16 store rounds to nearest even.
template <class Word>
__device__ __forceinline__ void load_state(State& state, const Word* head_state, int value_row,
                                           int lane) {
    const int g = lane >> 2, t = lane & 3;
    const Word* upper = head_state + static_cast<std::int64_t>(value_row + g) * kStateDim + 2 * t;
    const Word* lower = upper + 8 * kStateDim;
#pragma unroll
    for (int tile = 0; tile < kKeyTiles; ++tile) {
        const float2 a = load_pair(upper + tile * 8);
        const float2 b = load_pair(lower + tile * 8);
        state[tile][0] = a.x;
        state[tile][1] = a.y;
        state[tile][2] = b.x;
        state[tile][3] = b.y;
    }
}

template <class Word>
__device__ __forceinline__ void store_state(const State& state, Word* head_state, int value_row,
                                            int lane) {
    const int g = lane >> 2, t = lane & 3;
    Word* upper = head_state + static_cast<std::int64_t>(value_row + g) * kStateDim + 2 * t;
    Word* lower = upper + 8 * kStateDim;
#pragma unroll
    for (int tile = 0; tile < kKeyTiles; ++tile) {
        store_pair(upper + tile * 8, state[tile][0], state[tile][1]);
        store_pair(lower + tile * 8, state[tile][2], state[tile][3]);
    }
}

// Address of the 8x8 group of a Q/K chunk with tokens row..row+7 at key tile `tile`.
__device__ __forceinline__ unsigned vector_address(const __nv_bfloat16* base, int row, int tile) {
    return common::smem_addr(base + vector_index(row, tile * 8));
}

__device__ __forceinline__ float bits_to_float(unsigned bits) { return __uint_as_float(bits); }

// One 16-token chunk for the sixteen value rows of `warp`. `tile` is the warp's staging area for
// the output rows, `out` the first value of this warp in the chunk's first output token.
__device__ __forceinline__ void step(State& state, const Stage& stage, __nv_bfloat16* tile,
                                     __nv_bfloat16* out, std::int64_t out_pitch, int valid,
                                     float scale, int warp, int lane) {
    const int g = lane >> 2, t = lane & 3, r = lane & 7, m = lane >> 3;
    const QkChunk& qk       = stage.qk;
    const ControlChunk& ctl = stage.control;

    // P^T = S^T K^T through TF32 and H^T = S^T Q^T through the BF16-rounded state; token
    // columns are the two n8 tiles. The TF32 k order pairs keys (2t, 2t+1) as (t, t+4) in the
    // A fragment and in the B words alike.
    float p[2][2][4]{}, h[2][4]{};
#pragma unroll
    for (int j = 0; j < kKeyTiles; j += 2) {
        unsigned kb[4], qb[4];
        common::ldmatrix_x4(kb[0], kb[1], kb[2], kb[3],
                            vector_address(qk.k, 8 * (m & 1) + r, j + (m >> 1)));
        common::ldmatrix_x4(qb[0], qb[1], qb[2], qb[3],
                            vector_address(qk.q, 8 * (m >> 1) + r, j + (m & 1)));
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                unsigned b0, b1;
                common::unpack_bf16x2_to_fp32_bits(kb[2 * u + ni], b0, b1);
                common::mma_tf32_bits(p[u][ni][0], p[u][ni][1], p[u][ni][2], p[u][ni][3],
                                      __float_as_uint(state[j + u][0]),
                                      __float_as_uint(state[j + u][2]),
                                      __float_as_uint(state[j + u][1]),
                                      __float_as_uint(state[j + u][3]), b0, b1);
            }
        }
        const unsigned a0 = common::pack_bf16x2(state[j][0], state[j][1]);
        const unsigned a1 = common::pack_bf16x2(state[j][2], state[j][3]);
        const unsigned a2 = common::pack_bf16x2(state[j + 1][0], state[j + 1][1]);
        const unsigned a3 = common::pack_bf16x2(state[j + 1][2], state[j + 1][3]);
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            common::mma_bf16(h[ni][0], h[ni][1], h[ni][2], h[ni][3], a0, a1, a2, a3, qb[2 * ni],
                             qb[2 * ni + 1]);
        }
    }

    // R = V - prefix * P in the C layout (value g/g+8, token 8*ni+2t/2t+1); V comes transposed
    // straight from the [token][value] stage.
    const float2 prefix[2] = {common::load_vec<float2>(ctl.prefix + 2 * t),
                              common::load_vec<float2>(ctl.prefix + 8 + 2 * t)};
    unsigned vb[4];
    common::ldmatrix_x4_t(
        vb[0], vb[1], vb[2], vb[3],
        common::smem_addr(stage.value + value_index(r + 8 * (m >> 1), 16 * warp + 8 * (m & 1))));
    float res[2][4];
#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
        unsigned v00, v01, v10, v11;
        common::unpack_bf16x2_to_fp32_bits(vb[2 * ni], v00, v01);
        common::unpack_bf16x2_to_fp32_bits(vb[2 * ni + 1], v10, v11);
        res[ni][0] = bits_to_float(v00) - prefix[ni].x * (p[0][ni][0] + p[1][ni][0]);
        res[ni][1] = bits_to_float(v01) - prefix[ni].y * (p[0][ni][1] + p[1][ni][1]);
        res[ni][2] = bits_to_float(v10) - prefix[ni].x * (p[0][ni][2] + p[1][ni][2]);
        res[ni][3] = bits_to_float(v11) - prefix[ni].y * (p[0][ni][3] + p[1][ni][3]);
    }

    // Delta^T = R^T solve^T: R^T fragments are the A operands, solve rows the B words.
    float delta[2][4]{};
#pragma unroll
    for (int jp = 0; jp < 2; ++jp) {
#pragma unroll
        for (int no = 0; no < 2; ++no) {
            const float2 s = common::load_vec<float2>(ctl.solve + square_index(8 * no + g, 8 * jp + 2 * t));
            common::mma_tf32_bits(delta[no][0], delta[no][1], delta[no][2], delta[no][3],
                                  __float_as_uint(res[jp][0]), __float_as_uint(res[jp][2]),
                                  __float_as_uint(res[jp][1]), __float_as_uint(res[jp][3]),
                                  __float_as_uint(s.x), __float_as_uint(s.y));
        }
    }

    // O^T = prefix * H^T + Delta^T mqk^T.
    float o[2][4];
#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
        o[ni][0] = h[ni][0] * prefix[ni].x;
        o[ni][1] = h[ni][1] * prefix[ni].y;
        o[ni][2] = h[ni][2] * prefix[ni].x;
        o[ni][3] = h[ni][3] * prefix[ni].y;
    }
#pragma unroll
    for (int jp = 0; jp < 2; ++jp) {
#pragma unroll
        for (int no = 0; no < 2; ++no) {
            const float2 s = common::load_vec<float2>(ctl.mqk + square_index(8 * no + g, 8 * jp + 2 * t));
            common::mma_tf32_bits(o[no][0], o[no][1], o[no][2], o[no][3],
                                  __float_as_uint(delta[jp][0]), __float_as_uint(delta[jp][2]),
                                  __float_as_uint(delta[jp][1]), __float_as_uint(delta[jp][3]),
                                  __float_as_uint(s.x), __float_as_uint(s.y));
        }
    }
#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            tile[(8 * ni + 2 * t + (e & 1)) * kOutputPitch + g + 8 * (e >> 1)] =
                __float2bfloat16_rn(o[ni][e] * scale);
        }
    }
    __syncwarp();
    const int token = lane >> 1, half = lane & 1;
    const uint4 packed = common::load_vec<uint4>(tile + token * kOutputPitch + 8 * half);
    if (token < valid) common::store_vec(out + token * out_pitch + 8 * half, packed);
    __syncwarp();

    // S^T = decay S^T + Delta_s^T K, with Delta_s = bf16(suffix * Delta).
    const float2 suffix[2] = {common::load_vec<float2>(ctl.suffix + 2 * t),
                              common::load_vec<float2>(ctl.suffix + 8 + 2 * t)};
    const unsigned d0 = common::pack_bf16x2(suffix[0].x * delta[0][0], suffix[0].y * delta[0][1]);
    const unsigned d1 = common::pack_bf16x2(suffix[0].x * delta[0][2], suffix[0].y * delta[0][3]);
    const unsigned d2 = common::pack_bf16x2(suffix[1].x * delta[1][0], suffix[1].y * delta[1][1]);
    const unsigned d3 = common::pack_bf16x2(suffix[1].x * delta[1][2], suffix[1].y * delta[1][3]);
    const float decay = ctl.prefix[kChunkSize - 1];
#pragma unroll
    for (int nt = 0; nt < kKeyTiles; nt += 2) {
        unsigned kt[4];
        common::ldmatrix_x4_t(kt[0], kt[1], kt[2], kt[3],
                              vector_address(qk.k, 8 * (m & 1) + r, nt + (m >> 1)));
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
            for (int e = 0; e < 4; ++e) state[nt + u][e] *= decay;
            common::mma_bf16(state[nt + u][0], state[nt + u][1], state[nt + u][2],
                             state[nt + u][3], d0, d1, d2, d3, kt[2 * u], kt[2 * u + 1]);
        }
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked::registers
