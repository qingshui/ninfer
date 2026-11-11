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

std::size_t linear(const ops::LinearProfile& p, std::int32_t first, std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last);
}

} // namespace

BlockProfile describe_block(const BlockParameters& b) {
    return {ops::projection_profile(b.attention.qkv),
            ops::linear_profile(b.attention.gate),
            ops::linear_profile(b.attention.output),
            ops::linear_profile(b.mlp.gate_up),
            ops::linear_profile(b.mlp.down)};
}

std::size_t text_workspace_bytes(const Parameters& p, const TextWorkspaceShape& s) {
    std::vector<BlockProfile> layers;
    layers.reserve(p.layers.size());
    for (const auto& layer : p.layers) layers.push_back(describe_block(layer));
    return text_workspace_bytes(p.config, layers, s);
}

std::size_t text_workspace_bytes(const TextConfig& c, std::span<const BlockProfile> layers,
                                 const TextWorkspaceShape& s) {
    if (s.batch <= 0 || s.min_tokens <= 0 || s.max_tokens < s.min_tokens)
        throw std::invalid_argument("Spark2_5 workspace: invalid batch or token interval");
    const auto first = s.batch * s.min_tokens;
    const auto last  = s.batch * s.max_tokens;
    const std::array<std::int32_t, 3> rows{dimension(c.query_width()), dimension(c.kv_width()),
                                           dimension(c.kv_width())};
    WorkspaceLayoutBuilder a;
    const auto scratch = [&](std::size_t bytes) {
        auto scope = a.scope();
        (void)a.alloc_bytes(bytes);
    };
    (void)workspace::roots(a, c, last);
    bool sliding = false, full = false;
    for (const auto& layer : c.layers) (layer.kind == LayerKind::Full ? full : sliding) = true;
    for (const auto& layer : layers) {
        auto scope = a.scope();
        {
            auto attention = a.scope();
            (void)workspace::attention(a, c, last);
            scratch(
                ops::linear_split_workspace_capacity_bytes(layer.qkv, rows, first, last));
            scratch(linear(layer.gate, first, last));
            if (sliding)
                scratch(ops::causal_sliding_window_attention_workspace_capacity_bytes(
                    kGeometry, kWindowCapacity, s.batch, s.min_tokens, s.max_tokens,
                    s.multiprocessor_count));
            if (full)
                scratch(ops::causal_softmax_attention_workspace_capacity_bytes(
                    kGeometry, s.kv_storage, s.envelope, s.batch, s.min_tokens, s.max_tokens,
                    s.multiprocessor_count));
            scratch(linear(layer.output, first, last));
        }
        auto mlp = a.scope();
        (void)workspace::product(a, c, last);
        scratch(ops::linear_geglu_workspace_capacity_bytes(layer.gate_up.qtype, layer.gate_up.n,
                                                           layer.gate_up.k, layer.gate_up.policy,
                                                           first, last));
        scratch(linear(layer.down, first, last));
    }
    return a.peak_bytes();
}

std::size_t output_workspace_bytes(const Parameters& p, std::int32_t min_columns,
                                   std::int32_t max_columns) {
    return output_workspace_bytes(ops::linear_profile(p.output_head), min_columns, max_columns);
}

std::size_t output_workspace_bytes(const ops::LinearProfile& head, std::int32_t min_columns,
                                   std::int32_t max_columns) {
    return linear(head, min_columns, max_columns);
}

} // namespace ninfer::models::spark2_5
