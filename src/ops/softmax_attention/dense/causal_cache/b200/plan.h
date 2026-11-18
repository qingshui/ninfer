#pragma once

// Causal softmax attention on B200: the tcgen05 FMHA behind every KV dtype of the (32, 2) D256
// service geometry. Host side planning, workspace sizing and the append / cached entry points.

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/backend.h"
#include "ops/softmax_attention/kernels/causal/sm100/launch.h"

namespace ninfer::ops::detail {

// Query tiles are 128 packed rows (8 tokens x 16 heads). Calls whose query tiles already fill the
// SMs run one split and write the final output; the rest split the key range (at least two splits,
// at most one per two 64-key tiles of the envelope) and merge.
Sm100AttentionPlan select_sm100_attention_plan(int query_heads, int width, int batch,
                                               CausalAttentionExecutionEnvelope envelope,
                                               int multiprocessor_count);

std::size_t sm100_attention_workspace_bytes(int query_heads, int batch, int min_width,
                                            int max_width, CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count);

// Write the new K/V rows into the paged cache, then attend over the completed cache.
void sm100_kv_append_attention(KvDType kv, const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid_columns,
                               const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                               Tensor& out, CudaExecutionView execution);

// The groups' K/V rows are written group by group, then one launch attends for all of them. The
// tensors are those of causal_softmax_attention_groups and carry the whole token axis.
void sm100_kv_append_attention_groups(KvDType kv, const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& positions, const Tensor& valid_columns,
                                      const Tensor& table_rows, float scale,
                                      PagedKVBatchLayerView cache,
                                      std::span<const CausalAttentionGroup> groups, Tensor& out,
                                      CudaExecutionView execution);

void sm100_kv_cached_attention(KvDType kv, const Tensor& q, const Tensor& positions, float scale,
                               const ConstPagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                               Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops::detail
