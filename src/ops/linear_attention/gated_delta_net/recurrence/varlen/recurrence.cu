#include "ops/linear_attention/gated_delta_net/recurrence/varlen/launch.h"
#include "ops/common/host/launch.h"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear_attention/gated_delta_net/kernels/transition.cuh"

namespace ninfer::ops::detail::gated_delta_net::varlen {
namespace {

using chunked::ControlChunk;
using chunked::kChunkSize;
using chunked::QkChunk;
using chunked::square_index;
using chunked::vector_index;

template <int DV>
inline constexpr int kThreads = DV >= 32 ? 256 : 128;

constexpr int kPlanThreads = 1024;
static_assert(kPlanThreads >= kMaxRows);

__host__ __device__ inline int chunks_of(int length, int threshold) {
    return length >= threshold ? (length + kChunkSize - 1) / kChunkSize : 0;
}

// First index i in [0,count) with values[i] >= key.
__device__ __forceinline__ int lower_bound_i32(const std::int32_t* values, int count, int key) {
    int low = 0, high = count;
    while (low < high) {
        const int mid = (low + high) >> 1;
        if (values[mid] < key)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

// Plans the whole table in one block: one thread per row cuts the row at its boundaries,
// the block scans per-row chunk counts into chunk ids, and each row publishes its sub-segments.
// Sub-segment ids are row + boundary index, so row r owns ids [r+lb(r), r+lb(r+1)].
__global__ void __launch_bounds__(kPlanThreads)
    plan_kernel(Operands ops, PlanHeader* header, SubSegment* sub_segments,
                std::int32_t* row_first) {
    __shared__ int warp_totals[kPlanThreads / 32];
    const int row = threadIdx.x, lane = row & 31, warp = row >> 5;
    const bool live = row < ops.rows;
    int first = 0, last = 0, begin = 0, length = 0, own_chunks = 0;
    if (live) {
        first  = lower_bound_i32(ops.boundary_rows, ops.boundaries, row);
        last   = lower_bound_i32(ops.boundary_rows, ops.boundaries, row + 1);
        begin  = ops.token_offsets[row];
        length = ops.token_offsets[row + 1] - begin;
        int previous = 0;
        for (int i = first; i < last; ++i) {
            const int column = ops.boundary_columns[i];
            own_chunks += chunks_of(column - previous, ops.chunked_threshold);
            previous = column;
        }
        own_chunks += chunks_of(length - previous, ops.chunked_threshold);
    }
    int inclusive = own_chunks;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int up = __shfl_up_sync(0xffffffffU, inclusive, offset);
        if (lane >= offset) inclusive += up;
    }
    if (lane == 31) warp_totals[warp] = inclusive;
    __syncthreads();
    int warp_base = 0, total = 0;
    for (int w = 0; w < kPlanThreads / 32; ++w) {
        if (w < warp) warp_base += warp_totals[w];
        total += warp_totals[w];
    }
    int chunk_base = warp_base + inclusive - own_chunks;
    if (live) {
        row_first[row] = row + first;
        int previous   = 0;
        for (int i = first; i <= last; ++i) {
            const bool closes = i < last;
            const int column  = closes ? ops.boundary_columns[i] : length;
            const int count   = column - previous;
            sub_segments[row + i] = {begin + previous, count, chunk_base, row,
                                     closes ? ops.boundary_offsets[i] : std::int64_t{-1}};
            chunk_base += chunks_of(count, ops.chunked_threshold);
            previous = column;
        }
        if (row == ops.rows - 1) row_first[ops.rows] = ops.rows + ops.boundaries;
    }
    if (row == 0) header->total_chunks = total;
}

// The sub-segment owning chunk id `chunk`: the last entry with chunk_base <= chunk. Entries of the
// recurrent route share their successor's base, so the last such entry always owns chunks.
__device__ __forceinline__ int find_sub_segment(const SubSegment* sub_segments, int count,
                                                int chunk) {
    int low = 0, high = count;
    while (high - low > 1) {
        const int mid = (low + high) >> 1;
        if (sub_segments[mid].chunk_base <= chunk)
            low = mid;
        else
            high = mid;
    }
    return low;
}

struct PrepareShared {
    QkChunk qk;
    ControlChunk control;
    float prefix[kChunkSize];
    float beta[kChunkSize];
    float lower[kChunkSize * kChunkSize];
};

// Same arithmetic as chunked::prepare_kernel; only the chunk to token mapping differs.
template <bool Normalize>
__global__ __launch_bounds__(256, 2) void prepare_kernel(Operands ops, const PlanHeader* header,
                                                         const SubSegment* sub_segments,
                                                         int sub_segment_count, QkChunk* qk_ws,
                                                         ControlChunk* control_ws,
                                                         int chunk_capacity) {
    __shared__ PrepareShared sm;
    const int chunk_id = blockIdx.x / ops.value_heads;
    if (chunk_id >= header->total_chunks) return;
    const int tid = threadIdx.x, lane = tid & 31;
    const int head  = blockIdx.x % ops.value_heads;
    const auto& sub = sub_segments[find_sub_segment(sub_segments, sub_segment_count, chunk_id)];
    const int local = (chunk_id - sub.chunk_base) * kChunkSize;
    const int start = sub.start + local, length = min(kChunkSize, sub.length - local);
    const int group = ops.value_heads / ops.qk_heads, qh = head / group;
    const int row = tid / 16, col = (tid & 15) * 8;
    const auto qo = (static_cast<std::int64_t>(start + row) * ops.qk_heads + qh) * kStateDim;

    float q[8]{}, k[8]{};
    if (row < length) {
        const uint4 qp = common::load_vec<uint4>(ops.q + qo + col);
        const uint4 kp = common::load_vec<uint4>(ops.k + qo + col);
        const auto* qb = reinterpret_cast<const __nv_bfloat16*>(&qp);
        const auto* kb = reinterpret_cast<const __nv_bfloat16*>(&kp);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] = __bfloat162float(qb[i]);
            k[i] = __bfloat162float(kb[i]);
        }
    }
    if constexpr (Normalize) {
        float qs = 0, ks = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qs = fmaf(q[i], q[i], qs);
            ks = fmaf(k[i], k[i], ks);
        }
        qs             = common::warp_sum<16>(qs);
        ks             = common::warp_sum<16>(ks);
        const float qi = rsqrtf(qs + chunked::kNormEps), ki = rsqrtf(ks + chunked::kNormEps);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] *= qi;
            k[i] *= ki;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm.qk.q[vector_index(row, col + i)] = __float2bfloat16_rn(q[i]);
        sm.qk.k[vector_index(row, col + i)] = __float2bfloat16_rn(k[i]);
    }
    if (tid < 32) sm.control.padding[tid] = 0;
    if (tid < kChunkSize) {
        const auto off = static_cast<std::int64_t>(start + tid) * ops.value_heads + head;
        float g        = tid < length ? ops.g[off] : 0.0F;
        sm.beta[tid]   = tid < length ? ops.beta[off] : 0.0F;
#pragma unroll
        for (int offset = 1; offset < kChunkSize; offset <<= 1) {
            const float previous = __shfl_up_sync(0xffffU, g, offset, kChunkSize);
            if (tid >= offset) g += previous;
        }
        sm.prefix[tid]         = g;
        sm.control.prefix[tid] = common::exp2_approx(g * common::kLog2E);
        const float last       = __shfl_sync(0xffffU, g, kChunkSize - 1, kChunkSize);
        sm.control.suffix[tid] = common::exp2_approx((last - g) * common::kLog2E);
    }
    __syncthreads();

    // Both products consume the same represented Q/K that recurrence will load.
    if (tid < 64) {
        const auto* a = tid < 32 ? sm.qk.k : sm.qk.q;
        float acc[2][4]{};
#pragma unroll
        for (int offset = 0; offset < kStateDim; offset += 16) {
            unsigned af[4];
            common::ldmatrix_x4(
                af[0], af[1], af[2], af[3],
                common::smem_addr(a + vector_index((lane & 7) + ((lane >> 3) & 1) * 8,
                                                   offset + (lane >> 4) * 8)));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                unsigned bf[2];
                common::ldmatrix_x2(
                    bf[0], bf[1],
                    common::smem_addr(sm.qk.k + vector_index(n * 8 + (lane & 7),
                                                             offset + ((lane >> 3) & 1) * 8)));
                common::mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], af[0], af[1], af[2],
                                 af[3], bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = lane / 4 + (i / 2) * 8;
                const int c = n * 8 + (lane & 3) * 2 + (i & 1);
                // Do not evaluate upper-triangle exponentials: g can be strongly negative.
                const float value =
                    r >= c ? acc[n][i] *
                                 common::exp2_approx((sm.prefix[r] - sm.prefix[c]) * common::kLog2E)
                           : 0.0F;
                if (tid < 32)
                    sm.lower[r * kChunkSize + c] = r > c ? sm.beta[r] * value : 0.0F;
                else
                    sm.control.mqk[square_index(r, c)] = value;
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        const int c = lane & 15;
#pragma unroll
        for (int r = 0; r < kChunkSize; ++r) {
            float value = r == c ? 1.0F : 0.0F;
#pragma unroll
            for (int j = 0; j < r; ++j)
                value = fmaf(-sm.lower[r * kChunkSize + j], sm.control.solve[square_index(j, c)],
                             value);
            if (lane < 16) sm.control.solve[square_index(r, c)] = value;
            __syncwarp();
        }
    }
    __syncthreads();
    sm.control.solve[square_index(row, tid & 15)] *= sm.beta[tid & 15];
    __syncthreads();

    auto* ctrl = reinterpret_cast<uint4*>(control_ws +
                                          static_cast<std::int64_t>(head) * chunk_capacity +
                                          chunk_id);
    const auto* src = reinterpret_cast<const uint4*>(&sm.control);
    if (tid < sizeof(ControlChunk) / 16) ctrl[tid] = src[tid];
    // Only the first value head in a group publishes Q/K. Other producers never read this
    // workspace; the following kernel supplies the only producer/consumer synchronization.
    if (head % group == 0) {
        auto* dst = reinterpret_cast<uint4*>(
            qk_ws + static_cast<std::int64_t>(qh) * chunk_capacity + chunk_id);
        const auto* data = reinterpret_cast<const uint4*>(&sm.qk);
#pragma unroll
        for (int i = tid; i < sizeof(QkChunk) / 16; i += 256) dst[i] = data[i];
    }
}

template <int Width>
__device__ __forceinline__ int value_index(int row, int col) {
    if constexpr (Width == 8) return row * Width + col;
    if constexpr (Width == 16) return row * Width + (col ^ (((row >> 1) & 1) * 8));
    return row * Width + (col ^ ((row & 3) * 8));
}

__device__ __forceinline__ void load_vector_tf32_a(const __nv_bfloat16* p, int k, int lane,
                                                   unsigned (&a)[4]) {
    unsigned low, high;
    common::ldmatrix_x2(low, high,
                        common::smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k)));
    common::unpack_bf16x2_to_fp32_bits(low, a[0], a[2]);
    common::unpack_bf16x2_to_fp32_bits(high, a[1], a[3]);
}

__device__ __forceinline__ void load_vector_bf16_a(const __nv_bfloat16* p, int k, int lane,
                                                   unsigned (&a)[4]) {
    common::ldmatrix_x4(a[0], a[1], a[2], a[3],
                        common::smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8,
                                                           k + (lane >> 4) * 8)));
}

__device__ __forceinline__ void load_transposed_bf16_a(const __nv_bfloat16* p, int key, int lane,
                                                       unsigned (&a)[4]) {
    common::ldmatrix_x4_t(a[0], a[1], a[2], a[3],
                          common::smem_addr(p + vector_index((lane & 7) + (lane >> 4) * 8,
                                                             key + ((lane >> 3) & 1) * 8)));
}

template <int DV>
__device__ __forceinline__ void load_bf16_b(const float* p, int k, int n, int lane, unsigned& b0,
                                            unsigned& b1) {
    const int t = lane & 3, col = n * 8 + lane / 4;
    b0 = common::pack_bf16x2(p[value_index<DV>(k + 2 * t, col)],
                             p[value_index<DV>(k + 2 * t + 1, col)]);
    b1 = common::pack_bf16x2(p[value_index<DV>(k + 2 * t + 8, col)],
                             p[value_index<DV>(k + 2 * t + 9, col)]);
}

template <int DV>
struct RecurrenceShared {
    struct alignas(256) Stage {
        QkChunk qk;
        ControlChunk control;
        __nv_bfloat16 value[kChunkSize * DV];
    } stage[2];

    float snapshot[kStateDim * DV];
    float delta[kChunkSize * DV];
};

// Stages one chunk: `first` is the global token of its first row, `valid` its row count.
template <int DV>
__device__ __forceinline__ void
load_stage(typename RecurrenceShared<DV>::Stage& stage, const Operands& ops, const Workspace& ws,
           int head, int value_start, int chunk_id, int first, int valid, int tid) {
    const int qh       = head / (ops.value_heads / ops.qk_heads);
    const auto* qk_src = reinterpret_cast<const uint4*>(
        ws.qk + static_cast<std::int64_t>(qh) * ws.chunk_capacity + chunk_id);
    auto* qk_dst = reinterpret_cast<uint4*>(&stage.qk);
#pragma unroll
    for (int i = tid; i < sizeof(QkChunk) / 16; i += kThreads<DV>)
        common::cp_async<16, Cache::cg>(qk_dst + i, qk_src + i);
    const auto* ctrl_src = reinterpret_cast<const uint4*>(
        ws.control + static_cast<std::int64_t>(head) * ws.chunk_capacity + chunk_id);
    auto* ctrl_dst = reinterpret_cast<uint4*>(&stage.control);
#pragma unroll
    for (int i = tid; i < sizeof(ControlChunk) / 16; i += kThreads<DV>)
        common::cp_async<16, Cache::cg>(ctrl_dst + i, ctrl_src + i);
    constexpr int vectors = kChunkSize * DV / 8;
#pragma unroll
    for (int i = tid; i < vectors; i += kThreads<DV>) {
        const int row = i / (DV / 8), col = i % (DV / 8) * 8;
        const int valid_token = row < valid ? first + row : first;
        const auto off =
            (static_cast<std::int64_t>(valid_token) * ops.value_heads + head) * kStateDim +
            value_start + col;
        common::cp_async_zfill<16, Cache::cg>(stage.value + row * DV + col, ops.v + off,
                                              row < valid ? 16 : 0);
    }
    common::cp_commit();
}

template <int DV>
__device__ __forceinline__ void store_residual(float* delta, const __nv_bfloat16* value,
                                               const float (&prediction)[4], const float* prefix,
                                               int n, int lane) {
    const int row = lane / 4, col = n * 8 + (lane & 3) * 2;
    const float2 v0 = __bfloat1622float2(common::load_vec<__nv_bfloat162>(value + row * DV + col));
    const float2 v1 =
        __bfloat1622float2(common::load_vec<__nv_bfloat162>(value + (row + 8) * DV + col));
    common::store_vec(
        delta + value_index<DV>(row, col),
        make_float2(v0.x - prefix[row] * prediction[0], v0.y - prefix[row] * prediction[1]));
    common::store_vec(delta + value_index<DV>(row + 8, col),
                      make_float2(v1.x - prefix[row + 8] * prediction[2],
                                  v1.y - prefix[row + 8] * prediction[3]));
}

template <int DV>
__device__ __forceinline__ void matrix_delta(float (&acc)[4], const float* matrix,
                                             const float* delta, int n, int lane) {
    const int g = lane / 4, t = lane & 3;
#pragma unroll
    for (int k = 0; k < kChunkSize; k += 8) {
        unsigned af[4];
        common::ldmatrix_x4(
            af[0], af[1], af[2], af[3],
            common::smem_addr(
                matrix + square_index((lane & 7) + ((lane >> 3) & 1) * 8, k + (lane >> 4) * 4)));
        common::mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3],
                              __float_as_uint(delta[value_index<DV>(k + t, n * 8 + g)]),
                              __float_as_uint(delta[value_index<DV>(k + t + 4, n * 8 + g)]));
    }
}

// `first` is the global token of the chunk's first row; rows at or beyond `valid` are padding.
__device__ __forceinline__ void store_output(const Operands& ops, float (&acc)[4], int first,
                                             int valid, int head, int value_start, int n,
                                             int lane) {
    const int row = lane / 4, col = value_start + n * 8 + (lane & 3) * 2;
    const auto o0 =
        (static_cast<std::int64_t>(first + row) * ops.value_heads + head) * kStateDim + col;
    if (row < valid) {
        common::store_vec(ops.out + o0,
                          __floats2bfloat162_rn(acc[0] * ops.scale, acc[1] * ops.scale));
    }
    if (row + 8 < valid) {
        common::store_vec(ops.out + o0 +
                              static_cast<std::int64_t>(8) * ops.value_heads * kStateDim,
                          __floats2bfloat162_rn(acc[2] * ops.scale, acc[3] * ops.scale));
    }
}

template <int DV>
struct Tile {
    static constexpr int kWarps  = kThreads<DV> / 32;
    static constexpr int NT      = DV / 8;
    static constexpr int MW      = NT < kWarps ? kWarps / NT : 1;
    static constexpr int MPW     = (kStateDim / 16) / MW;
    static constexpr int NPW     = NT > kWarps ? NT / kWarps : 1;
    static constexpr int kGroups = DV / kDvPerWarp;
};

// MMA accumulators describe S^T[key,value]; external state remains [value,key].
template <int DV>
__device__ __forceinline__ void load_fragments(float (&state)[Tile<DV>::NPW][Tile<DV>::MPW][4],
                                               const float* base, int value_start, int warp,
                                               int lane) {
    using T = Tile<DV>;
    const int g = lane / 4, t = lane & 3;
    const int n_start = warp % T::NT, m_start = T::NT < T::kWarps ? warp / T::NT : 0;
#pragma unroll
    for (int ni = 0; ni < T::NPW; ++ni) {
        const int n = n_start + ni * T::kWarps;
#pragma unroll
        for (int mi = 0; mi < T::MPW; ++mi) {
            const int key   = (m_start + mi * T::MW) * 16 + g;
            const int value = value_start + n * 8 + t * 2;
            state[ni][mi][0] = base[value * kStateDim + key];
            state[ni][mi][1] = base[(value + 1) * kStateDim + key];
            state[ni][mi][2] = base[value * kStateDim + key + 8];
            state[ni][mi][3] = base[(value + 1) * kStateDim + key + 8];
        }
    }
}

template <int DV>
__device__ __forceinline__ void
store_fragments(const float (&state)[Tile<DV>::NPW][Tile<DV>::MPW][4], float* base,
                int value_start, int warp, int lane) {
    using T = Tile<DV>;
    const int g = lane / 4, t = lane & 3;
    const int n_start = warp % T::NT, m_start = T::NT < T::kWarps ? warp / T::NT : 0;
#pragma unroll
    for (int ni = 0; ni < T::NPW; ++ni) {
        const int n = n_start + ni * T::kWarps;
#pragma unroll
        for (int mi = 0; mi < T::MPW; ++mi) {
            const int key   = (m_start + mi * T::MW) * 16 + g;
            const int value = value_start + n * 8 + t * 2;
            base[value * kStateDim + key]           = state[ni][mi][0];
            base[(value + 1) * kStateDim + key]     = state[ni][mi][1];
            base[value * kStateDim + key + 8]       = state[ni][mi][2];
            base[(value + 1) * kStateDim + key + 8] = state[ni][mi][3];
        }
    }
}

// Recurrent-route access for one sub-segment; `token_base` is its first global token.
struct SubSegmentAccess {
    const __nv_bfloat16 *q, *k, *v;
    const float *g, *beta;
    __nv_bfloat16* out;
    int qk_heads, value_heads, token_base;
    float scale;

    __device__ __forceinline__ std::int64_t column(std::int32_t token) const {
        return static_cast<std::int64_t>(token_base) + token;
    }
    __device__ __forceinline__ const __nv_bfloat16*
    key_ptr(const RecurrentCoordinates& coord, std::int32_t token) const {
        return k + (column(token) * qk_heads + coord.qk_head) * kStateDim;
    }
    __device__ __forceinline__ const __nv_bfloat16*
    value_ptr(const RecurrentCoordinates& coord, std::int32_t token) const {
        return v + (column(token) * value_heads + coord.value_head) * kStateDim;
    }
    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(token) * value_heads + coord.value_head);
    }
    __device__ __forceinline__ const __nv_bfloat16*
    query_ptr(const RecurrentCoordinates& coord, std::int32_t token) const {
        return q + (column(token) * qk_heads + coord.qk_head) * kStateDim;
    }
    __device__ __forceinline__ __nv_bfloat16*
    output_ptr(const RecurrentCoordinates& coord, std::int32_t token) const {
        return out + (column(token) * value_heads + coord.value_head) * kStateDim;
    }
};

// Same per-row arithmetic as recurrent_bf16_direct_kernel. A warp owns four value rows at a
// time and rows are independent, so any assignment of row groups to warps is bit-identical.
template <int DV>
__device__ __forceinline__ void run_recurrent_sub_segment(const Operands& ops,
                                                          const SubSegment& sub, int head,
                                                          int value_start, const float* read,
                                                          float* destination, float* checkpoint,
                                                          bool normalize) {
    using T = Tile<DV>;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const SubSegmentAccess access{ops.q,         ops.k,           ops.v,
                                  ops.g,         ops.beta,        ops.out,
                                  ops.qk_heads,  ops.value_heads, sub.start,
                                  ops.scale};
    for (int group = warp; group < T::kGroups; group += T::kWarps) {
        RecurrentCoordinates coord{};
        coord.lane       = lane;
        coord.warp       = warp;
        coord.value_head = static_cast<std::uint32_t>(head);
        coord.qk_head    = static_cast<std::uint32_t>(head / (ops.value_heads / ops.qk_heads));
        coord.dv_base    = static_cast<std::uint32_t>(value_start + group * kDvPerWarp);
        coord.dqk_base   = static_cast<std::uint32_t>(lane * kQkPerLane);
        __align__(16) float state[kDvPerWarp][kQkPerLane];
        load_state_tile(state, read, coord);
        if (normalize)
            run_recurrent_sequence<true, OutputEffects>(state, access, coord, sub.length);
        else
            run_recurrent_sequence<false, OutputEffects>(state, access, coord, sub.length);
        store_state_tile(state, destination, coord);
        if (checkpoint) store_state_tile(state, checkpoint, coord);
    }
}

// One CTA owns one (row, value head, value tile) and walks the row's sub-segments in order. The
// state stays in registers between consecutive chunked sub-segments; FP32 registers and FP32
// memory round-trip exactly, so this is bit-identical to publishing and reloading it. A
// switch between the chunked and recurrent routes (different thread mappings) hands the state
// over through the row destination record, which also stages every intermediate state.
template <int DV>
__global__ __launch_bounds__(kThreads<DV>, (DV >= 32 ? 2 : 4)) void recurrence_kernel(
    Operands ops, Workspace ws, bool normalize) {
    using T = Tile<DV>;
    extern __shared__ __align__(256) unsigned char backing[];
    auto& sm      = *reinterpret_cast<RecurrenceShared<DV>*>(backing);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int g = lane / 4, t = lane & 3;
    const int head        = blockIdx.x / (kStateDim / DV);
    const int value_start = blockIdx.x % (kStateDim / DV) * DV;
    const int row         = blockIdx.y;
    const int n_start     = warp % T::NT;
    const int m_start     = T::NT < T::kWarps ? warp / T::NT : 0;

    const auto head_offset = static_cast<std::int64_t>(head) * kStateDim * kStateDim;
    const auto* source = reinterpret_cast<const float*>(ops.source_base + ops.source_offsets[row]) +
                         head_offset;
    auto* destination =
        reinterpret_cast<float*>(ops.destination_base + ops.destination_offsets[row]) + head_offset;
    const int first_sub = ws.row_first[row], end_sub = ws.row_first[row + 1];

    float state[T::NPW][T::MPW][4];
    bool resident = false;
    for (int s = first_sub; s < end_sub; ++s) {
        const SubSegment sub = ws.sub_segments[s];
        const bool last      = s + 1 == end_sub;
        const float* read    = s == first_sub ? source : destination;
        float* checkpoint =
            sub.checkpoint >= 0
                ? reinterpret_cast<float*>(ops.checkpoint_base + sub.checkpoint) + head_offset
                : nullptr;
        if (sub.length < ops.chunked_threshold) {
            __syncthreads(); // the state may have been staged by threads of the other mapping
            run_recurrent_sub_segment<DV>(ops, sub, head, value_start, read, destination,
                                          checkpoint, normalize);
            __syncthreads();
            continue;
        }
        if (!resident) {
            load_fragments<DV>(state, read, value_start, warp, lane);
            resident = true;
        }
        const int chunks = chunks_of(sub.length, ops.chunked_threshold);
        load_stage<DV>(sm.stage[0], ops, ws, head, value_start, sub.chunk_base, sub.start,
                       min(kChunkSize, sub.length), tid);
        if (chunks > 1)
            load_stage<DV>(sm.stage[1], ops, ws, head, value_start, sub.chunk_base + 1,
                           sub.start + kChunkSize, min(kChunkSize, sub.length - kChunkSize), tid);

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int first = sub.start + chunk * kChunkSize;
            const int valid = min(kChunkSize, sub.length - chunk * kChunkSize);
            if (chunk + 1 < chunks)
                common::cp_wait<1>();
            else
                common::cp_wait<0>();
#pragma unroll
            for (int ni = 0; ni < T::NPW; ++ni) {
                const int n = n_start + ni * T::kWarps;
#pragma unroll
                for (int mi = 0; mi < T::MPW; ++mi) {
                    const int key = (m_start + mi * T::MW) * 16 + g;
                    const int col = n * 8 + t * 2;
                    common::store_vec(sm.snapshot + value_index<DV>(key, col),
                                      make_float2(state[ni][mi][0], state[ni][mi][1]));
                    common::store_vec(sm.snapshot + value_index<DV>(key + 8, col),
                                      make_float2(state[ni][mi][2], state[ni][mi][3]));
                }
            }
            __syncthreads();
            const auto& stage     = sm.stage[chunk & 1];
            const ControlChunk& p = stage.control;
            const QkChunk& qk     = stage.qk;
            float history[T::NPW][4]{};

            if constexpr (T::NT >= T::kWarps) {
                float prediction[T::NPW][4]{};
#pragma unroll
                for (int k = 0; k < kStateDim; k += 16) {
                    unsigned ka0[4], ka1[4], qa[4];
                    load_vector_tf32_a(qk.k, k, lane, ka0);
                    load_vector_tf32_a(qk.k, k + 8, lane, ka1);
                    load_vector_bf16_a(qk.q, k, lane, qa);
#pragma unroll
                    for (int ni = 0; ni < T::NPW; ++ni) {
                        const int n    = warp + ni * T::kWarps;
                        const float b0 = sm.snapshot[value_index<DV>(k + 2 * t, n * 8 + g)];
                        const float b1 = sm.snapshot[value_index<DV>(k + 2 * t + 1, n * 8 + g)];
                        const float b2 = sm.snapshot[value_index<DV>(k + 2 * t + 8, n * 8 + g)];
                        const float b3 = sm.snapshot[value_index<DV>(k + 2 * t + 9, n * 8 + g)];
                        common::mma_tf32_bits(prediction[ni][0], prediction[ni][1],
                                              prediction[ni][2], prediction[ni][3], ka0[0], ka0[1],
                                              ka0[2], ka0[3], __float_as_uint(b0),
                                              __float_as_uint(b1));
                        common::mma_tf32_bits(prediction[ni][0], prediction[ni][1],
                                              prediction[ni][2], prediction[ni][3], ka1[0], ka1[1],
                                              ka1[2], ka1[3], __float_as_uint(b2),
                                              __float_as_uint(b3));
                        common::mma_bf16(history[ni][0], history[ni][1], history[ni][2],
                                         history[ni][3], qa[0], qa[1], qa[2], qa[3],
                                         common::pack_bf16x2(b0, b1), common::pack_bf16x2(b2, b3));
                    }
                }
#pragma unroll
                for (int ni = 0; ni < T::NPW; ++ni) {
                    store_residual<DV>(sm.delta, stage.value, prediction[ni], p.prefix,
                                       warp + ni * T::kWarps, lane);
                }
            } else {
                if (warp < 2 * T::NT) {
                    const bool query = warp >= T::NT;
                    const int n      = warp % T::NT;
                    if (query) {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 16) {
                            unsigned af[4], b0, b1;
                            load_vector_bf16_a(qk.q, k, lane, af);
                            load_bf16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                            common::mma_bf16(history[0][0], history[0][1], history[0][2],
                                             history[0][3], af[0], af[1], af[2], af[3], b0, b1);
                        }
                    } else {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 8) {
                            unsigned af[4];
                            load_vector_tf32_a(qk.k, k, lane, af);
                            common::mma_tf32_bits(
                                history[0][0], history[0][1], history[0][2], history[0][3], af[0],
                                af[1], af[2], af[3],
                                __float_as_uint(sm.snapshot[value_index<DV>(k + 2 * t, n * 8 + g)]),
                                __float_as_uint(
                                    sm.snapshot[value_index<DV>(k + 2 * t + 1, n * 8 + g)]));
                        }
                        store_residual<DV>(sm.delta, stage.value, history[0], p.prefix, n, lane);
                    }
                }
            }
            __syncthreads();

            // Each value tile reads only its own R columns, so R can become Delta in place.
#pragma unroll
            for (int n = warp; n < T::NT; n += T::kWarps) {
                float delta[4]{};
                matrix_delta<DV>(delta, p.solve, sm.delta, n, lane);
                const int col = n * 8 + t * 2;
                __syncwarp();
                common::store_vec(sm.delta + value_index<DV>(g, col),
                                  make_float2(delta[0], delta[1]));
                common::store_vec(sm.delta + value_index<DV>(g + 8, col),
                                  make_float2(delta[2], delta[3]));
            }
            __syncthreads();

            if constexpr (T::NT >= T::kWarps) {
#pragma unroll
                for (int ni = 0; ni < T::NPW; ++ni) {
                    const int n = warp + ni * T::kWarps;
                    history[ni][0] *= p.prefix[g];
                    history[ni][1] *= p.prefix[g];
                    history[ni][2] *= p.prefix[g + 8];
                    history[ni][3] *= p.prefix[g + 8];
                    matrix_delta<DV>(history[ni], p.mqk, sm.delta, n, lane);
                    store_output(ops, history[ni], first, valid, head, value_start, n, lane);
                }
            } else if (warp >= T::NT && warp < 2 * T::NT) {
                const int n = warp - T::NT;
                history[0][0] *= p.prefix[g];
                history[0][1] *= p.prefix[g];
                history[0][2] *= p.prefix[g + 8];
                history[0][3] *= p.prefix[g + 8];
                matrix_delta<DV>(history[0], p.mqk, sm.delta, n, lane);
                store_output(ops, history[0], first, valid, head, value_start, n, lane);
            }

            // S^T = gamma S^T + K^T diag(suffix) Delta. The master state stays FP32.
            // Delta is converted to BF16 only for this MMA; the output branch above consumed the
            // original FP32 Delta through TF32 MMA.
            unsigned delta_b[T::NPW][2];
#pragma unroll
            for (int ni = 0; ni < T::NPW; ++ni) {
                const int col  = (n_start + ni * T::kWarps) * 8 + g;
                delta_b[ni][0] = common::pack_bf16x2(
                    p.suffix[2 * t] * sm.delta[value_index<DV>(2 * t, col)],
                    p.suffix[2 * t + 1] * sm.delta[value_index<DV>(2 * t + 1, col)]);
                delta_b[ni][1] = common::pack_bf16x2(
                    p.suffix[2 * t + 8] * sm.delta[value_index<DV>(2 * t + 8, col)],
                    p.suffix[2 * t + 9] * sm.delta[value_index<DV>(2 * t + 9, col)]);
            }
#pragma unroll
            for (int mi = 0; mi < T::MPW; ++mi) {
                const int key     = (m_start + mi * T::MW) * 16 + g;
                const float decay = p.prefix[kChunkSize - 1];
#pragma unroll
                for (int ni = 0; ni < T::NPW; ++ni) {
                    state[ni][mi][0] *= decay;
                    state[ni][mi][1] *= decay;
                    state[ni][mi][2] *= decay;
                    state[ni][mi][3] *= decay;
                }
                unsigned af[4];
                load_transposed_bf16_a(qk.k, key - g, lane, af);
#pragma unroll
                for (int ni = 0; ni < T::NPW; ++ni) {
                    common::mma_bf16(state[ni][mi][0], state[ni][mi][1], state[ni][mi][2],
                                     state[ni][mi][3], af[0], af[1], af[2], af[3], delta_b[ni][0],
                                     delta_b[ni][1]);
                }
            }
            __syncthreads();
            if (chunk + 2 < chunks) {
                load_stage<DV>(sm.stage[chunk & 1], ops, ws, head, value_start,
                               sub.chunk_base + chunk + 2, first + 2 * kChunkSize,
                               min(kChunkSize, sub.length - (chunk + 2) * kChunkSize), tid);
            }
        }

        // The state after this sub-segment: checkpoint it, publish it at the row end, or stage
        // it in the destination when the next sub-segment uses the recurrent mapping.
        if (checkpoint) store_fragments<DV>(state, checkpoint, value_start, warp, lane);
        if (last || ws.sub_segments[s + 1].length < ops.chunked_threshold) {
            store_fragments<DV>(state, destination, value_start, warp, lane);
            resident = false;
        }
    }
}

template <int DV>
void launch_tile(const Operands& ops, const Workspace& ws, bool normalize, cudaStream_t stream) {
    constexpr int shared_bytes = sizeof(RecurrenceShared<DV>);
    (void)common::prepare_shared<shared_bytes, recurrence_kernel<DV>, true>();
    const dim3 grid(static_cast<unsigned>(ops.value_heads * (kStateDim / DV)),
                    static_cast<unsigned>(ops.rows));
    recurrence_kernel<DV><<<grid, kThreads<DV>, shared_bytes, stream>>>(ops, ws, normalize);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch(const Operands& ops, const Workspace& ws, bool normalize_qk, int value_tile,
            cudaStream_t stream) {
    plan_kernel<<<1, kPlanThreads, 0, stream>>>(ops, ws.header, ws.sub_segments, ws.row_first);
    CUDA_CHECK(cudaGetLastError());
    if (ws.chunk_capacity > 0) {
        const unsigned blocks =
            static_cast<unsigned>(static_cast<std::int64_t>(ops.value_heads) * ws.chunk_capacity);
        const int count = ops.rows + ops.boundaries;
        if (normalize_qk)
            prepare_kernel<true><<<blocks, 256, 0, stream>>>(ops, ws.header, ws.sub_segments,
                                                             count, ws.qk, ws.control,
                                                             ws.chunk_capacity);
        else
            prepare_kernel<false><<<blocks, 256, 0, stream>>>(ops, ws.header, ws.sub_segments,
                                                              count, ws.qk, ws.control,
                                                              ws.chunk_capacity);
        CUDA_CHECK(cudaGetLastError());
    }
    switch (value_tile) {
    case 64:
        return launch_tile<64>(ops, ws, normalize_qk, stream);
    case 32:
        return launch_tile<32>(ops, ws, normalize_qk, stream);
    case 16:
        return launch_tile<16>(ops, ws, normalize_qk, stream);
    case 8:
        return launch_tile<8>(ops, ws, normalize_qk, stream);
    }
    throw std::logic_error("linear attention: invalid varlen recurrence value tile");
}

} // namespace ninfer::ops::detail::gated_delta_net::varlen
