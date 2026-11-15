#pragma once

// SM100 (B200, sm_100a) causal softmax attention over the paged KV cache.
//
// One CTA computes 128 packed query rows (token-major, GQA group inside a token) of one KV head
// for one KV split. Q.K^T and P.V run on tcgen05 with the accumulators in tensor memory and an
// online softmax between them. A call that needs a single split writes the final BF16 output;
// otherwise every split publishes an FP32 partial (numerator, natural-unit maximum, sum) that the
// shared natural merge combines.

#include "ops/softmax_attention/backend.h"
#include "ops/softmax_attention/kernels/causal/operands.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Sm100AttentionPlan {
    int q_heads;
    int kv_heads;
    int head_dim;
    int tile_q;  // packed query rows per CTA (the MMA's M)
    int tile_kv; // keys per pipeline step (the QK MMA's N)
    int splits;  // KV partitions per query tile; 1 selects the direct BF16 epilogue
};

inline constexpr int kSm100TileQ  = 128;
inline constexpr int kSm100TileKv = 64;

// Paged cache planes, type-erased: the KV dtype argument of the launch selects the element types.
struct Sm100KvCache {
    const void* keys;
    const void* values;
    const void* key_scales;   // null for BF16
    const void* value_scales; // null for BF16
    const std::int64_t* tables;
    const std::int32_t* valid_columns; // null when every column is live
    const std::int32_t* table_rows;    // null selects table row 0
    int table_stride;
    int kv_heads;
};

// CausalAttentionOperands plus the cache planes and the split workspace. `partial` is unused when
// the plan has one split.
struct Sm100CausalOperands : CausalAttentionOperands {
    Sm100KvCache cache;
    CausalPartialView partial;
};

// The cache must already contain every key the queries attend to (the append is a separate write).
void launch_sm100_causal_attention(const Sm100CausalOperands& operands, KvDType kv,
                                   const Sm100AttentionPlan& plan, cudaStream_t stream);

} // namespace ninfer::ops::detail
