#pragma once

#include "core/device.h"
#include "ninfer/ops/attention_geometry.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"


#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SlidingWindowAttentionExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

/**
 * Symmetric non-causal sliding-window grouped-query attention.
 *
 * The registered profiles are D=128, Hq=32, Hkv=8 (group 4), window/cyclic capacity 2048 or
 * 4096, scale=1/sqrt(128), T=1..16, and B=1..8. q/out are contiguous BF16 [128,32,T,B],
 * query_k/query_v are contiguous BF16 [128,8,T,B], positions is contiguous device I32 [T,B], and
 * Context K/V select dense [128,padded_capacity,8] fields with contiguous Device I64 [B]
 * byte offsets. Every selected field is 16-byte aligned. valid_columns is Device I32 [B].
 * Row b has V=valid_columns[b] live query columns and positions[i,b]=L+i for i<V.
 * Context K/V offsets select its cyclic-cache records; inactive rows need no valid record. Columns
 * i>=V are an inert physical tail and produce exact BF16 zero.
 *
 * The read-only cyclic BFloat16 context stores BF16 K and FP16 V for committed absolute positions
 * [max(0,L-window),L), with absolute position p stored at slot p mod window. Temporary query K/V is
 * a separate BF16 segment at [L,L+V). For live query position p_i, a populated context or
 * temporary key at p_j is visible
 * exactly when abs(p_j-p_i)<window. Thus distance window-1 is included and distance window is
 * excluded; every live query also sees every live temporary query row in its batch row. Query
 * head h reads KV head floor(h/4).
 *
 * Over the visible set J, score[j]=scale*dot(q,K[j]) and out=BF16(sum_j softmax(score)[j]*V[j]).
 * Inputs are interpreted in their represented dtype, including BF16 temporary V and FP16 cached V.
 * q/query_k/query_v/positions/valid_columns, addressed context fields, out, and live workspace
 * allocations are pairwise non-overlapping. The K/V offset tables may share the same Tensor and
 * overlap no writable storage. Inputs and context remain unchanged; out is the only observable
 * mutation and is completely overwritten.
 *
 * The caller guarantees envelope.min_context <= L <= envelope.max_context, sequential nonnegative
 * live positions, and that the selected record contains the declared committed interval. The
 * envelope is a host launch/workspace resource promise; it never changes the admitted key set or
 * numerical result.
 */
void sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& positions, const Tensor& valid_columns,
                              AttentionHeadGeometry geometry, std::uint32_t window, float scale,
                              const ConstCyclicKVCacheLayerView& context,
                              SlidingWindowAttentionExecutionEnvelope envelope,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

/**
 * Return transient capacity for every T in the inclusive optimized interval at the exact batch
 * size. Geometry, window, and execution envelope are fixed profile inputs; invalid inputs throw.
 */
[[nodiscard]] std::size_t sliding_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window,
    SlidingWindowAttentionExecutionEnvelope envelope, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t batch_size);

/**
 * Causal sliding-window grouped-query attention over a read-only cyclic ring plus the call's own
 * live columns (chunked prefill and decode/verification with the "attend, then append" order).
 *
 * The registered profile is D=256, Hq=8, Hkv=2 (group 4), window/cyclic capacity 512, and
 * scale=1/sqrt(256). q/out are contiguous BF16 [256,8,T,B]; k/v are contiguous BF16 [256,2,T,B];
 * positions is contiguous device I32 [T,B]; valid_columns is contiguous device I32 [B]. B=1
 * accepts every positive T. B=2..16 accepts T=1..16. All tensors that the kernels vectorize
 * (q, k, v, out, and every addressed ring field) are 16-byte aligned.
 *
 * Row b has V=valid_columns[b] live columns and positions[i,b]=L+i for i<V. Query column i attends
 * a key at absolute position pj exactly when 0 <= pi-pj <= 511, so distance 511 is visible and
 * distance 512 is not. Keys come from two sources: committed positions [max(0,L-512),L) read from
 * the ring at slot p mod 512 (BF16 K, FP16 V), and the row's live columns j<=i, whose K is taken
 * as BF16 and whose V crosses the same BF16->FP16 round-to-nearest boundary as a stored ring row.
 * A chunk longer than the window therefore never reads its own positions from the ring: ring slots
 * that alias the chunk still hold the older history. The Op does not append; the caller appends
 * with kv_cache_append_prefix after attending. Query head h reads KV head floor(h/4).
 *
 * Over the visible set J, score[j]=scale*dot(q,K[j]) and out=BF16(sum_j softmax(score)[j]*V[j]).
 * Columns i>=V are an inert physical tail and produce exact BF16 zero; a row with V=0 reads no
 * ring record. The ring fields select one record per row through device I64 [B] byte offsets
 * (K and V tables may be the same Tensor). Inputs and ring remain unchanged; out is completely
 * overwritten and is the only mutation. q/k/v/positions/valid_columns, the addressed ring
 * fields, out, and live workspace allocations are pairwise non-overlapping.
 *
 * The caller guarantees sequential nonnegative live positions and that each live row's selected
 * ring record holds the committed interval [max(0,L-512),L). The Op is replayable under CUDA
 * Graph for fixed tensor views, T, B, and SM count.
 */
void causal_sliding_window_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     AttentionHeadGeometry geometry, std::uint32_t window,
                                     float scale, const ConstCyclicKVCacheLayerView& context,
                                     MemoryArena& workspace, Tensor& out,
                                     CudaExecutionView execution);

/**
 * Return transient capacity for every T in [min_tokens,max_tokens] at the exact batch size and
 * physical SM count. Geometry and window are fixed profile inputs; invalid inputs throw. The
 * returned capacity may be zero.
 */
[[nodiscard]] std::size_t causal_sliding_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window, std::int32_t batch_size,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t multiprocessor_count);

} // namespace ninfer::ops
