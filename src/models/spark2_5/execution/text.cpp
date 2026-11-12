#include "models/spark2_5/execution/text.h"

#include "models/spark2_5/execution/workspace.h"

#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_geglu.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/residual_add_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/sliding_window_attention.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::spark2_5 {
namespace {

constexpr ops::AttentionHeadGeometry kGeometry{kAttentionHeadDim, kQueryHeads, kKvHeads};

float attention_scale() {
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(kAttentionHeadDim)));
}

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Spark2_5 forward_text: " + message);
}

void require_input(const Tensor& tensor, DType dtype, std::int32_t extent, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != extent || tensor.ne[2] != 1 || tensor.ne[3] != 1 ||
        !tensor.is_contiguous())
        invalid(std::string(name) + " has the wrong type, extent or layout");
}

struct Extents {
    std::int32_t tokens, batch, columns;
};

Extents validate(const TextConfig& c, const TextInputs& in, const Tensor& hidden) {
    const Extents e{in.ids.ne[0], in.ids.ne[1], in.ids.ne[0] * in.ids.ne[1]};
    if (e.tokens <= 0 || e.batch <= 0 || in.ids.dtype != DType::I32 || in.ids.ne[2] != 1 ||
        in.ids.ne[3] != 1 || !in.ids.is_contiguous())
        invalid("ids must be contiguous I32 [T,B] with positive extents");
    if (in.positions.dtype != DType::I32 || in.positions.ne[0] != e.tokens ||
        in.positions.ne[1] != e.batch || in.positions.ne[2] != 1 || in.positions.ne[3] != 1 ||
        !in.positions.is_contiguous())
        invalid("positions must be contiguous I32 [T,B] like ids");
    require_input(in.valid_columns, DType::I32, e.batch, "valid_columns");
    require_input(in.kv_table_rows, DType::I32, e.batch, "kv_table_rows");
    require_input(in.window_offsets, DType::I64, e.batch, "window_offsets");
    if (hidden.dtype != DType::BF16 || !hidden.is_contiguous() ||
        hidden.numel() != std::int64_t(c.hidden) * e.columns)
        invalid("hidden must be contiguous BF16 [hidden, T*B]");
    return e;
}

void run_full_attention(const TextExecutionView& view, std::uint32_t full_layer, const Tensor& q,
                        const Tensor& k, const Tensor& v, const TextInputs& in,
                        ops::CausalAttentionExecutionEnvelope envelope, Tensor& out) {
    auto scratch = view.workspace.scope();
    ops::causal_softmax_attention(
        q, k, v, in.positions, in.valid_columns, in.kv_table_rows, kGeometry, attention_scale(),
        view.cache.batch_layer_view(full_layer), envelope, view.workspace, out, view.execution);
}

// Attention sub-block of one layer: writes the output projection into `roots.branch`.
void attention_block(const TextExecutionView& view, std::uint32_t layer, const TextInputs& in,
                     const Extents& e, ops::CausalAttentionExecutionEnvelope envelope,
                     workspace::Roots& roots, AttentionProbe* probe) {
    const auto& p      = view.parameters;
    const auto& c      = p.config;
    const auto& state  = c.layers[layer];
    const auto& weight = p.layers[layer].attention;
    auto& work         = view.workspace;
    const auto cuda    = view.execution;
    auto scope         = work.scope();
    auto b             = workspace::attention(work, c, e.columns);
    const auto d = kAttentionHeadDim, qh = kQueryHeads, kvh = kKvHeads;
    {
        auto scratch = work.scope();
        const std::array outputs{b.query, b.key, b.value};
        ops::linear_split(roots.normalized, weight.qkv, outputs, work, cuda);
    }
    {
        auto scratch = work.scope();
        ops::linear(roots.normalized, weight.gate.weight, b.gate, weight.gate.policy, work, cuda);
    }
    {
        // Only the leading rotary_dim channels rotate; the others pass through untouched.
        const auto& rope = c.rope(state.kind);
        auto query       = b.query.view({d, qh, e.columns});
        auto key         = b.key.view({d, kvh, e.columns});
        ops::rope(in.positions.view({e.columns}), dimension(rope.rotary_dim), rope.theta, query,
                  key, cuda);
    }
    const auto q = b.query.view({d, qh, e.tokens, e.batch});
    const auto k = b.key.view({d, kvh, e.tokens, e.batch});
    const auto v = b.value.view({d, kvh, e.tokens, e.batch});
    auto out     = b.output.view({d, qh, e.tokens, e.batch});
    if (state.kind == LayerKind::Sliding) {
        sliding_attention(
            view, state.state_index, q, k, v, in, out,
            [&](const ConstCyclicKVCacheLayerView& ring) {
                if (probe)
                    probe->on_attention(layer, state.kind, {q, k, v, in.positions, ring, out});
            });
    } else {
        run_full_attention(view, state.state_index, q, k, v, in, envelope, out);
        if (probe)
            probe->on_attention(
                layer, state.kind,
                {q, k, v, in.positions,
                 ConstPagedKVBatchLayerView(view.cache.batch_layer_view(state.state_index)), out});
    }
    // sigmoid(g) is one scalar per head and scales that head's whole output before out_proj.
    auto gated = b.output.view({dimension(c.query_width()), e.columns});
    ops::headwise_sigmoid_mul(b.gate, gated, cuda);
    {
        auto scratch = work.scope();
        ops::linear(gated, weight.output.weight, roots.branch, weight.output.policy, work, cuda);
    }
}

// MLP sub-block: writes the down projection into `roots.branch`.
void mlp_block(const TextExecutionView& view, std::uint32_t layer, const Extents& e,
               workspace::Roots& roots) {
    const auto& p   = view.parameters;
    const auto& mlp = p.layers[layer].mlp;
    auto& work      = view.workspace;
    auto scope      = work.scope();
    auto product    = workspace::product(work, p.config, e.columns);
    {
        auto scratch = work.scope();
        ops::linear_geglu(roots.normalized, mlp.gate_up.weight, product, mlp.gate_up.policy, work,
                          view.execution);
    }
    auto scratch = work.scope();
    ops::linear(product, mlp.down.weight, roots.branch, mlp.down.policy, work, view.execution);
}

// The last full-attention layer when only the last column of the traversal is consumed. Keys and
// values still cover every column, so the projection, rotary embedding and cache append run at
// full width; everything downstream of attention (query attend, gate, out_proj, MLP, final norm)
// runs on the last column alone. Single row (B == 1) and T > 1. Reads `roots.normalized` and
// `roots.residual` as the layer's input and writes the final-norm output of the last column.
void last_layer_last_column(const TextExecutionView& view, std::uint32_t layer, const TextInputs& in,
                            const Extents& e, ops::CausalAttentionExecutionEnvelope envelope,
                            workspace::Roots& roots, Tensor& hidden) {
    const auto& p      = view.parameters;
    const auto& c      = p.config;
    const auto& state  = c.layers[layer];
    const auto& weight = p.layers[layer].attention;
    auto& work         = view.workspace;
    const auto cuda    = view.execution;
    const auto d = kAttentionHeadDim, qh = kQueryHeads, kvh = kKvHeads;
    const auto tail = e.columns - 1;
    workspace::Roots last{roots.residual.slice(1, tail, 1), roots.normalized.slice(1, tail, 1),
                          roots.branch.slice(1, tail, 1)};
    {
        auto scope = work.scope();
        const auto query_width = dimension(c.query_width()), kv_width = dimension(c.kv_width());
        auto query = work.alloc(DType::BF16, {query_width, e.columns});
        auto key   = work.alloc(DType::BF16, {kv_width, e.columns});
        auto value = work.alloc(DType::BF16, {kv_width, e.columns});
        auto gate  = work.alloc(DType::BF16, {dimension(c.heads), 1});
        auto out   = work.alloc(DType::BF16, {query_width, 1});
        {
            auto scratch = work.scope();
            const std::array outputs{query, key, value};
            ops::linear_split(roots.normalized, weight.qkv, outputs, work, cuda);
        }
        {
            auto scratch = work.scope();
            ops::linear(last.normalized, weight.gate.weight, gate, weight.gate.policy, work, cuda);
        }
        {
            const auto& rope = c.rope(state.kind);
            auto rq          = query.view({d, qh, e.columns});
            auto rk          = key.view({d, kvh, e.columns});
            ops::rope(in.positions.view({e.columns}), dimension(rope.rotary_dim), rope.theta, rq,
                      rk, cuda);
        }
        const auto k = key.view({d, kvh, e.tokens, e.batch});
        const auto v = value.view({d, kvh, e.tokens, e.batch});
        ops::kv_cache_append_prefix(k, v, in.positions, in.valid_columns, in.kv_table_rows,
                                    {0, std::uint32_t(e.tokens)},
                                    view.cache.batch_layer_view(state.state_index), cuda);
        // Dense width-1 attend: the call re-appends the last K/V row it already finds in the
        // cache, then attends over positions [0, position of the last column].
        const auto q1 = query.view({d, qh, e.tokens, e.batch}).slice(2, tail, 1);
        const auto k1 = k.slice(2, tail, 1);
        const auto v1 = v.slice(2, tail, 1);
        auto o1       = out.view({d, qh, 1, 1});
        {
            auto scratch = work.scope();
            ops::causal_softmax_attention(
                q1, k1, v1, in.positions.slice(0, tail, 1), Tensor{}, in.kv_table_rows, kGeometry,
                attention_scale(), view.cache.batch_layer_view(state.state_index), envelope, work,
                o1, cuda);
        }
        ops::headwise_sigmoid_mul(gate, out, cuda);
        auto scratch = work.scope();
        ops::linear(out, weight.output.weight, last.branch, weight.output.policy, work, cuda);
    }
    ops::residual_add_rmsnorm(last.residual, last.branch, p.layers[layer].post_attention_norm,
                              c.norm_epsilon, last.residual, last.normalized, cuda);
    mlp_block(view, layer, Extents{1, 1, 1}, last);
    auto target = hidden.view({dimension(c.hidden), e.columns}).slice(1, tail, 1);
    ops::residual_add_rmsnorm(last.residual, last.branch, p.final_norm, c.norm_epsilon,
                              last.residual, target, cuda);
}

} // namespace

void sliding_attention(const TextExecutionView& view, std::uint32_t sliding_layer, const Tensor& q,
                       const Tensor& k, const Tensor& v, const TextInputs& in, Tensor& out) {
    sliding_attention(view, sliding_layer, q, k, v, in, out, {});
}

void sliding_attention(const TextExecutionView& view, std::uint32_t sliding_layer, const Tensor& q,
                       const Tensor& k, const Tensor& v, const TextInputs& in, Tensor& out,
                       const AfterAttend& after_attend) {
    const auto ring = view.window.layer_view(sliding_layer, in.window_offsets);
    {
        auto scratch = view.workspace.scope();
        // The Op's `window` is the number of positions a query sees, itself included.
        ops::causal_sliding_window_attention(q, k, v, in.positions, in.valid_columns, kGeometry,
                                             kWindowCapacity, attention_scale(), ring,
                                             view.workspace, out, view.execution);
    }
    if (after_attend) after_attend(ring);
    // Attend reads only history from before this call, so the append may overwrite old slots.
    ops::kv_cache_append_prefix(k, v, in.positions, in.valid_columns, {0, std::uint32_t(q.ne[2])},
                                ring, view.execution);
}

void forward_text(TextExecutionView view, const TextInputs& in,
                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                  AttentionProbe* probe, TextOutput output) {
    const auto& p   = view.parameters;
    const auto& c   = p.config;
    const auto e    = validate(c, in, hidden);
    const auto cuda = view.execution;
    auto frame      = view.workspace.scope();
    auto roots      = workspace::roots(view.workspace, c, e.columns);
    auto residual   = roots.residual;
    auto normalized = roots.normalized;

    ops::embedding(in.ids.view({e.columns}), p.token_embedding, residual, cuda);
    ops::rmsnorm(residual, p.layers.front().input_norm, c.norm_epsilon, false, normalized, cuda);
    const auto count = static_cast<std::uint32_t>(p.layers.size());
    // The narrow tail needs a dense single-row traversal, a full-attention last layer and no
    // observer; otherwise every column is written, which also satisfies LastColumn.
    const bool narrow = output == TextOutput::LastColumn && !probe && e.batch == 1 &&
                        e.tokens > 1 && c.layers[count - 1].kind == LayerKind::Full;
    const auto wide = narrow ? count - 1 : count;
    for (std::uint32_t i = 0; i < wide; ++i) {
        const auto& block = p.layers[i];
        attention_block(view, i, in, e, envelope, roots, probe);
        ops::residual_add_rmsnorm(residual, roots.branch, block.post_attention_norm, c.norm_epsilon,
                                  residual, normalized, cuda);
        mlp_block(view, i, e, roots);
        const bool last = i + 1 == count;
        auto target     = last ? hidden.view({dimension(c.hidden), e.columns}) : normalized;
        ops::residual_add_rmsnorm(residual, roots.branch,
                                  last ? p.final_norm : p.layers[i + 1].input_norm, c.norm_epsilon,
                                  residual, target, cuda);
    }
    if (narrow) last_layer_last_column(view, count - 1, in, e, envelope, roots, hidden);
}

void output_logits(const Parameters& p, const Tensor& hidden, Tensor& logits, MemoryArena& work,
                   CudaExecutionView cuda) {
    auto scope = work.scope();
    ops::linear(hidden, p.output_head.weight, logits, p.output_head.policy, work, cuda);
}

} // namespace ninfer::models::spark2_5
