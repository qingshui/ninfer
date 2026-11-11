#pragma once

#include "models/spark2_5/execution/parameters.h"
#include "models/spark2_5/state/kv_cache.h"
#include "models/spark2_5/state/window_state.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/softmax_attention.h"

#include <functional>
#include <variant>

namespace ninfer::models::spark2_5 {

struct TextExecutionView {
    const Parameters& parameters;
    MemoryArena& workspace;
    CudaExecutionView execution;
    const FullKVCache& cache;
    CyclicKVCacheView& window;
};

// One traversal of B rows of T columns each. All tensors are contiguous and on the execution
// device:
//   ids, positions     I32 [T,B]   token ids and absolute positions; live positions are sequential
//   valid_columns      I32 [B]     live columns of each row; the rest is an inert physical tail
//   kv_table_rows      I32 [B]     execution-table row of each row's full-layer KV history
//   window_offsets     I64 [B]     StateImage record byte offset of each row's sliding rings
// The caller guarantees that the rings and full-layer pages cover [0, positions + T).
struct TextInputs {
    Tensor ids, positions, valid_columns, kv_table_rows, window_offsets;
};

// What a layer's attention observed. Tensors are BF16 views into the traversal's workspace and
// stay valid only during on_attention. q and k are rotated.
//   q, out    [256,8,T,B]   out is the attention result before the gate
//   k, v      [256,2,T,B]   this call's own columns
//   context   Sliding: the ring holding the committed history [max(0,L-512),L) of each row,
//                      not yet advanced by this call's columns.
//             Full:    the paged history through this call's columns, already appended.
struct AttentionCall {
    Tensor q, k, v, positions;
    std::variant<ConstCyclicKVCacheLayerView, ConstPagedKVBatchLayerView> context;
    Tensor out;
};

// Called after the layer's attention output exists. The probe runs on the host thread between
// submissions; it synchronizes the stream itself before reading and must not mutate any tensor.
struct AttentionProbe {
    virtual ~AttentionProbe() = default;
    // `layer` is the index among all layers.
    virtual void on_attention(std::uint32_t layer, LayerKind kind, const AttentionCall& call) = 0;
};

// Sliding-window attention of one sliding layer (`sliding_layer` counts sliding layers only).
// Every query attends the keys at distance 0..511 from the ring's committed history and its own
// live columns, then the call's live columns are appended to the ring. q is [256,8,T,B]; k, v
// are [256,2,T,B]; out is [256,8,T,B].
void sliding_attention(const TextExecutionView&, std::uint32_t sliding_layer, const Tensor& q,
                       const Tensor& k, const Tensor& v, const TextInputs&, Tensor& out);

// As above. `after_attend` runs once `out` is submitted and before the append, with the ring that
// still holds the history before this call.
using AfterAttend = std::function<void(const ConstCyclicKVCacheLayerView& ring)>;
void sliding_attention(const TextExecutionView&, std::uint32_t sliding_layer, const Tensor& q,
                       const Tensor& k, const Tensor& v, const TextInputs&, Tensor& out,
                       const AfterAttend& after_attend);

// Embeds ids, runs every layer with a BF16 residual and writes the final-norm output into
// `hidden`, contiguous BF16 [hidden, T*B] (the column of row b, token t is t + T*b). Attention
// state is advanced for the live columns. `envelope` bounds the keys visible to full layers.
void forward_text(TextExecutionView, const TextInputs&, ops::CausalAttentionExecutionEnvelope,
                  Tensor& hidden, AttentionProbe* probe = nullptr);

// logits[:, c] = head * hidden[:, c]; hidden is BF16 [hidden, N] and logits BF16 [vocabulary, N].
void output_logits(const Parameters&, const Tensor& hidden, Tensor& logits, MemoryArena&,
                   CudaExecutionView);

} // namespace ninfer::models::spark2_5
