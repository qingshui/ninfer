#pragma once

#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear_attention/gated_delta_net/operands.h"
#include "ops/common/numeric/math.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail::gated_delta_net {
struct HeadMap {
    int H_qk, H_v;
    common::FastDivisorU31 group;

    static HeadMap of(int qk, int value) {
        return {qk, value, common::FastDivisorU31::make(value / qk)};
    }

    __host__ __device__ int group_size() const { return H_v / H_qk; }

    __host__ __device__ int qk_head(int value_head) const {
        return static_cast<int>(group.divide(static_cast<std::uint32_t>(value_head)));
    }
};

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / common::kWarpSize;

static_assert(kStateDim % common::kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

// Recurrent state words are FP32 or BF16. Arithmetic is FP32 in both cases: a BF16 word widens
// exactly and a published state is rounded to nearest even.
template <class State>
inline constexpr bool kBf16State = std::is_same_v<State, __nv_bfloat16>;

template <class State>
__device__ __forceinline__ float represent(float value) {
    if constexpr (kBf16State<State>) return __bfloat162float(__float2bfloat16_rn(value));
    return value;
}

__device__ __forceinline__ float state_value(float word) { return word; }
__device__ __forceinline__ float state_value(__nv_bfloat16 word) { return __bfloat162float(word); }

template <class State>
__device__ __forceinline__ State state_word(float value) {
    if constexpr (kBf16State<State>) return __float2bfloat16_rn(value);
    else return value;
}

// Four consecutive state words; the address is aligned to four words.
__device__ __forceinline__ void load_state_quad(float* reg, const float* source) {
    common::store_vec(reg, common::load_vec<float4>(source));
}

__device__ __forceinline__ void load_state_quad(float* reg, const __nv_bfloat16* source) {
    const uint2 packed = common::load_vec<uint2>(source);
    const float2 low   = common::bf16x2_bits_to_float2(packed.x);
    const float2 high  = common::bf16x2_bits_to_float2(packed.y);
    common::store_vec(reg, make_float4(low.x, low.y, high.x, high.y));
}

__device__ __forceinline__ void store_state_quad(const float* reg, float* destination) {
    common::store_vec(destination, common::load_vec<float4>(reg));
}

__device__ __forceinline__ void store_state_quad(const float* reg, __nv_bfloat16* destination) {
    common::store_vec(destination, make_uint2(common::pack_bf16x2(reg[0], reg[1]),
                                              common::pack_bf16x2(reg[2], reg[3])));
}

template <class State>
__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const State* base,
                                             std::uint32_t dqk_base) {
    load_state_quad(reg, base + dqk_base);
}

template <class State>
__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], State* base,
                                              std::uint32_t dqk_base) {
    store_state_quad(reg, base + dqk_base);
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

struct RawQkLane {
    common::Bf16x4Pack bits;
    float value[kQkPerLane];
};

struct RawValueLane {
    __nv_bfloat16 bits;
    float value;
};

struct RawGatePair {
    uint2 bits;
    float g;
    float beta;
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = common::load_vec<common::Bf16x4Pack>(base + dqk_base);
    const float2 lo = common::bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = common::bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

template <bool Normalize>
__device__ __forceinline__ void normalize_qk_lane(float (&value)[kQkPerLane], int lane) {
    if constexpr (Normalize) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { sum += value[i] * value[i]; }
        sum       = common::warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(common::kFullWarpMask, inv, 0);
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { value[i] *= inv; }
    }
}

__device__ __forceinline__ RawValueLane load_value_lane(const __nv_bfloat16* base, int lane,
                                                        std::uint32_t dv_base) {
    RawValueLane out{__float2bfloat16(0.0f), 0.0f};
    if (lane < kDvPerWarp) {
        out.bits  = base[dv_base + lane];
        out.value = __bfloat162float(out.bits);
    }
    return out;
}

__device__ __forceinline__ RawGatePair load_source_gate(const float* g, const float* beta,
                                                        std::int64_t offset) {
    const float g_value    = g[offset];
    const float beta_value = beta[offset];
    return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value, beta_value};
}

__device__ __forceinline__ RawGatePair load_record_gate(const uint2* gate, std::int64_t offset) {
    const uint2 bits = common::load_vec<uint2>(gate + offset);
    return {bits, __uint_as_float(bits.x), __uint_as_float(bits.y)};
}

__device__ __forceinline__ void apply_gdn_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const float (&key)[kQkPerLane], float v_local,
                                                     float g, float beta) {
    const float alpha = expf(g);

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * key[c]; }
        partial = common::warp_sum<common::kWarpSize>(partial);

        const float v_r   = __shfl_sync(0xffffffff, v_local, r, common::kWarpSize);
        const float delta = beta * (v_r - alpha * partial);

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + delta * key[c]; }
    }
}

// Stores scale * S * q for the warp's value rows from the query lane in its consumed form.
__device__ __forceinline__ void readout_rows(const float (&state)[kDvPerWarp][kQkPerLane],
                                             const float (&query)[kQkPerLane],
                                             __nv_bfloat16* output, std::uint32_t dv_base,
                                             int lane, float scale) {
    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * query[c]; }
        partial = common::warp_sum<common::kWarpSize>(partial);
        if (lane == r) { attn_val = partial; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * scale); }
}

template <bool Normalize>
__device__ __forceinline__ void readout_and_store(float (&state)[kDvPerWarp][kQkPerLane],
                                                  const __nv_bfloat16* query, __nv_bfloat16* output,
                                                  std::uint32_t dqk_base, std::uint32_t dv_base,
                                                  int lane, float scale) {
    RawQkLane q = load_raw_qk_lane(query, dqk_base);
    normalize_qk_lane<Normalize>(q.value, lane);
    readout_rows(state, q.value, output, dv_base, lane, scale);
}

struct RecurrentCoordinates {
    int lane;
    int warp;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t state_tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    std::uint32_t dv_base;
    std::uint32_t dqk_base;
};

__device__ __forceinline__ RecurrentCoordinates make_coordinates(std::int32_t batch,
                                                                 std::int32_t layer,
                                                                 std::int32_t state_tile,
                                                                 HeadMap heads) {
    const int lane                 = threadIdx.x;
    const int warp                 = threadIdx.y;
    const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t qk_head =
        static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(value_head)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
    return {lane,    warp,       batch,
            layer,   state_tile, value_head,
            qk_head, dv_base,    static_cast<std::uint32_t>(lane * kQkPerLane)};
}

struct OutputEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access&, const RecurrentCoordinates&,
                                                       std::int32_t, const RawQkLane&) {}

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access&, const RecurrentCoordinates&, std::int32_t,
                       const RawValueLane&, const RawGatePair&) {}

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                   const RecurrentCoordinates& coord, std::int32_t token) {
        readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                           access.output_ptr(coord, token), coord.dqk_base,
                                           coord.dv_base, coord.lane, access.scale);
    }
};

template <class State>
__device__ __forceinline__ void load_state_tile(float (&state)[kDvPerWarp][kQkPerLane],
                                                const State* base,
                                                const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                     coord.dqk_base);
    }
}

template <class State>
__device__ __forceinline__ void store_state_tile(const float (&state)[kDvPerWarp][kQkPerLane],
                                                 State* base, const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                      coord.dqk_base);
    }
}

// Replaces the registers by the values a published State field would hold.
template <class State, int Rows, int Columns>
__device__ __forceinline__ void represent_state(float (&state)[Rows][Columns]) {
    if constexpr (kBf16State<State>) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
            for (int c = 0; c < Columns; ++c) state[r][c] = represent<State>(state[r][c]);
        }
    }
}

// TokenState is the representation every token's state takes before the next token: float keeps
// the FP32 registers of one call, a BF16 State repeats the one-token publication.
template <bool NormalizeInputs, class Effects, class TokenState = float, class Access>
__device__ __forceinline__ void
run_recurrent_sequence(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                       const RecurrentCoordinates& coord, std::int32_t valid) {
    RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    Effects::observe_key(access, coord, 0, key);
    normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);

    for (std::int32_t token = 0; token < valid; ++token) {
        const RawGatePair gate = access.load_gate(coord, token);
        const RawValueLane value =
            load_value_lane(access.value_ptr(coord, token), coord.lane, coord.dv_base);
        Effects::observe_value_gate(access, coord, token, value, gate);

        apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);

        if (token + 1 < valid) {
            key = load_raw_qk_lane(access.key_ptr(coord, token + 1), coord.dqk_base);
            Effects::observe_key(access, coord, token + 1, key);
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
        }

        Effects::template publish_output<NormalizeInputs>(state, access, coord, token);
        represent_state<TokenState>(state);
    }
}

template <class Access>
__device__ __forceinline__ void zero_output_suffix(const Access& access,
                                                   const RecurrentCoordinates& coord,
                                                   std::int32_t valid, std::int32_t width) {
    if (coord.lane < kDvPerWarp) {
        for (std::int32_t token = valid; token < width; ++token) {
            access.output_ptr(coord, token)[coord.dv_base + coord.lane] = __float2bfloat16(0.0f);
        }
    }
}

} // namespace ninfer::ops::detail::gated_delta_net
