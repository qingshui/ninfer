#pragma once

#include "ops/common/cuda/reduce.cuh"
#include "ops/linear_attention/gated_delta_net/kernels/transition.cuh"

namespace ninfer::ops::detail::gated_delta_net {
template <class StateWord>
struct DirectAccess {
    using State = StateWord;
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const State* state_read;
    State* state_write;
    __nv_bfloat16* out;
    HeadMap heads;
    std::int32_t width;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(0, 0, static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)coord;
        return token;
    }

    __device__ __forceinline__ const State*
    state_read_base(const RecurrentCoordinates& coord) const {
        return state_read + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ State* state_write_base(const RecurrentCoordinates& coord) const {
        return state_write + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

template <class StateWord>
struct BatchUpdateAccess {
    using State = StateWord;
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const State* states_read;
    State* states_write;
    const std::int64_t* source_offsets;
    const std::int64_t* destination_offsets;
    __nv_bfloat16* out;
    HeadMap heads;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates(std::int32_t state_tile) const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0, state_tile, heads);
    }

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return coordinates(static_cast<std::int32_t>(blockIdx.z));
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)token;
        return coord.batch;
    }

    __device__ __forceinline__ const State*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states_read + source_offsets[coord.batch] / sizeof(State) +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ State* state_write_base(const RecurrentCoordinates& coord) const {
        return states_write + destination_offsets[coord.batch] / sizeof(State) +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

template <bool NormalizeInputs, class State>
__global__ void __launch_bounds__(common::kWarpSize* kNumWarps, 2)
    recurrent_bf16_direct_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta,
                                 const State* __restrict__ state_read,
                                 State* __restrict__ state_write, __nv_bfloat16* __restrict__ out,
                                 std::int32_t width, HeadMap heads, float scale) {
    const DirectAccess<State> access{q,   k,     v,     g,     beta, state_read,
                                     state_write, out, heads, width, scale};
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<NormalizeInputs, OutputEffects>(state, access, coord, width);
    store_state_tile(state, access.state_write_base(coord), coord);
}

// A CTA updates Passes consecutive state tiles of one (value head, row), one tile per pass of its
// warps, and loads the next tile's state rows while the current pass computes. The passes share the
// key, gate and query, and each performs the arithmetic of run_recurrent_sequence, so the result is
// bit-identical to one tile per CTA. A pipelined CTA holds two tiles in registers and runs at eight
// CTAs per SM, which keeps enough state loads in flight for a BF16 state tile, whose own traffic is
// too small to hide the latency of its input loads.
template <bool NormalizeInputs, class State, int Passes>
__global__ void __launch_bounds__(common::kWarpSize* kNumWarps, Passes == 1 ? 2 : 8)
    recurrent_batch_update_kernel(BatchUpdateAccess<State> access) {
    constexpr int Buffers     = Passes == 1 ? 1 : 2;
    const std::int32_t tile0  = static_cast<std::int32_t>(blockIdx.z) * Passes;
    RecurrentCoordinates coord = access.coordinates(tile0);
    const State* source        = access.state_read_base(coord);
    State* destination         = access.state_write_base(coord);
    const __nv_bfloat16* values = access.value_ptr(coord, 0);
    __align__(16) float state[Buffers][kDvPerWarp][kQkPerLane];
    RawValueLane value[Buffers];
    load_state_tile(state[0], source, coord);
    value[0]      = load_value_lane(values, coord.lane, coord.dv_base);
    RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
    const RawGatePair gate = access.load_gate(coord, 0);
    RawQkLane query        = load_raw_qk_lane(access.query_ptr(coord, 0), coord.dqk_base);
    normalize_qk_lane<NormalizeInputs>(query.value, coord.lane);
    __nv_bfloat16* output = access.output_ptr(coord, 0);
#pragma unroll
    for (int pass = 0; pass < Passes; ++pass) {
        const int current = pass % Buffers;
        if (pass + 1 < Passes) {
            const RecurrentCoordinates next = access.coordinates(tile0 + pass + 1);
            load_state_tile(state[(pass + 1) % Buffers], source, next);
            value[(pass + 1) % Buffers] = load_value_lane(values, coord.lane, next.dv_base);
        }
        apply_gdn_transition(state[current], key.value, value[current].value, gate.g, gate.beta);
        readout_rows(state[current], query.value, output, coord.dv_base, coord.lane, access.scale);
        store_state_tile(state[current], destination, coord);
        if (pass + 1 < Passes) coord = access.coordinates(tile0 + pass + 1);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net
