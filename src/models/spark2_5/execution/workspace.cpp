#include "models/spark2_5/execution/workspace.h"

#include "models/spark2_5/state/window_state.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_geglu.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/sliding_window_attention.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::spark2_5 {
namespace {

constexpr ops::AttentionHeadGeometry kGeometry{kAttentionHeadDim, kQueryHeads, kKvHeads};

std::size_t linear(const LinearParameters& p, std::int32_t first, std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                                first, last);
}

} // namespace

std::size_t text_workspace_bytes(const Parameters& p, const TextWorkspaceShape& s) {
    if (s.batch <= 0 || s.min_tokens <= 0 || s.max_tokens < s.min_tokens)
        throw std::invalid_argument("Spark2_5 workspace: invalid batch or token interval");
    const auto& c    = p.config;
    const auto first = s.batch * s.min_tokens;
    const auto last  = s.batch * s.max_tokens;
    const auto rows  = p.qkv_rows();
    WorkspaceLayoutBuilder a;
    const auto scratch = [&](std::size_t bytes) {
        auto scope = a.scope();
        (void)a.alloc_bytes(bytes);
    };
    (void)workspace::roots(a, c, last);
    bool sliding = false, full = false;
    for (const auto& layer : c.layers) (layer.kind == LayerKind::Full ? full : sliding) = true;
    for (const auto& layer : p.layers) {
        auto scope = a.scope();
        {
            auto attention = a.scope();
            (void)workspace::attention(a, c, last);
            scratch(
                ops::linear_split_workspace_capacity_bytes(layer.attention.qkv, rows, first, last));
            scratch(linear(layer.attention.gate, first, last));
            if (sliding)
                scratch(ops::causal_sliding_window_attention_workspace_capacity_bytes(
                    kGeometry, kWindowCapacity, s.batch, s.min_tokens, s.max_tokens,
                    s.multiprocessor_count));
            if (full)
                scratch(ops::causal_softmax_attention_workspace_capacity_bytes(
                    kGeometry, s.kv_storage, s.envelope, s.batch, s.min_tokens, s.max_tokens,
                    s.multiprocessor_count));
            scratch(linear(layer.attention.output, first, last));
        }
        auto mlp = a.scope();
        (void)workspace::product(a, c, last);
        scratch(ops::linear_geglu_workspace_capacity_bytes(
            layer.mlp.gate_up.weight.qtype, layer.mlp.gate_up.weight.n, layer.mlp.gate_up.weight.k,
            layer.mlp.gate_up.policy, first, last));
        scratch(linear(layer.mlp.down, first, last));
    }
    return a.peak_bytes();
}

std::size_t output_workspace_bytes(const Parameters& p, std::int32_t min_columns,
                                   std::int32_t max_columns) {
    return linear(p.output_head, min_columns, max_columns);
}

} // namespace ninfer::models::spark2_5
