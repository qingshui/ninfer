#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "models/spark2_5/config.h"
#include "models/spark2_5/execution/parameters.h"

#include "ninfer/ops/softmax_attention.h"

#include <array>
#include <cstddef>
#include <span>

namespace ninfer::models::spark2_5 {

// Columns are the token axis of a call: W tokens for each of B rows. The same allocation
// sequence serves execution (MemoryArena) and sizing (WorkspaceLayoutBuilder).
namespace workspace {

struct Roots {
    Tensor residual, normalized, branch;
};

template <class Arena>
Roots roots(Arena& a, const TextConfig& c, std::int32_t columns) {
    const auto h = dimension(c.hidden);
    return {a.alloc(DType::BF16, {h, columns}), a.alloc(DType::BF16, {h, columns}),
            a.alloc(DType::BF16, {h, columns})};
}

struct AttentionBuffers {
    Tensor query, key, value, gate, output;
};

template <class Arena>
AttentionBuffers attention(Arena& a, const TextConfig& c, std::int32_t columns) {
    const auto q = dimension(c.query_width()), kv = dimension(c.kv_width());
    return {a.alloc(DType::BF16, {q, columns}), a.alloc(DType::BF16, {kv, columns}),
            a.alloc(DType::BF16, {kv, columns}),
            a.alloc(DType::BF16, {dimension(c.heads), columns}),
            a.alloc(DType::BF16, {q, columns})};
}

template <class Arena>
Tensor product(Arena& a, const TextConfig& c, std::int32_t columns) {
    return a.alloc(DType::BF16, {dimension(c.intermediate), columns});
}

} // namespace workspace

// Bounds of one forward_text call family: `batch` rows of every token count in the inclusive
// interval, attention envelope and KV format fixed.
struct TextWorkspaceShape {
    std::int32_t batch      = 1;
    std::int32_t min_tokens = 1;
    std::int32_t max_tokens = 1;
    ops::CausalAttentionExecutionEnvelope envelope;
    KvCacheStorage kv_storage         = KvCacheStorage::BFloat16;
    std::int32_t multiprocessor_count = 0;
};

// The projection facts workspace sizing needs from one block: format, shape and activation
// policy. They derive from the definition alone, so planning needs no materialized shard.
struct BlockProfile {
    ops::ProjectionProfile qkv;
    ops::LinearProfile gate, output, gate_up, down;
};

[[nodiscard]] BlockProfile describe_block(const BlockParameters&);

// Peak arena bytes forward_text needs for any call within the shape.
[[nodiscard]] std::size_t text_workspace_bytes(const Parameters&, const TextWorkspaceShape&);
[[nodiscard]] std::size_t text_workspace_bytes(const TextConfig&,
                                               std::span<const BlockProfile> layers,
                                               const TextWorkspaceShape&);

// Peak arena bytes output_logits needs for any column count in the inclusive interval.
[[nodiscard]] std::size_t output_workspace_bytes(const Parameters&, std::int32_t min_columns,
                                                 std::int32_t max_columns);
[[nodiscard]] std::size_t output_workspace_bytes(const ops::LinearProfile& head,
                                                 std::int32_t min_columns,
                                                 std::int32_t max_columns);

} // namespace ninfer::models::spark2_5
