#pragma once

#include "artifact/reader.h"
#include "artifact/schema.h"
#include "ninfer/types.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::models::spark2_5 {

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (!value || value > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
        throw std::overflow_error("Spark2_5 tensor extent is outside int32");
    return static_cast<std::int32_t>(value);
}

// Both kinds use the same projection shapes; they differ in the attention window and RoPE.
enum class LayerKind { Sliding, Full };

struct LayerConfig {
    LayerKind kind;
    // Position among the layers of the same kind, for per-kind attention state.
    std::uint32_t state_index;
};

struct RopeConfig {
    std::uint32_t rotary_dim = 0;
    float theta              = 0;
};

struct TextConfig {
    std::uint32_t hidden, heads, kv_heads, head_dim, intermediate, vocabulary, max_context;
    std::uint32_t sliding_window;
    float norm_epsilon;
    bool tie_word_embeddings;
    TokenId bos_token_id, eos_token_id, pad_token_id;
    // Kind of each layer exactly as listed by layer_types.
    std::vector<LayerConfig> layers;
    std::uint32_t full_layers = 0, sliding_layers = 0;
    // A kind absent from layer_types keeps the zero RopeConfig.
    RopeConfig sliding_rope, full_rope;

    [[nodiscard]] const RopeConfig& rope(LayerKind kind) const noexcept {
        return kind == LayerKind::Full ? full_rope : sliding_rope;
    }

    [[nodiscard]] std::uint64_t query_width() const { return std::uint64_t(heads) * head_dim; }

    [[nodiscard]] std::uint64_t kv_width() const { return std::uint64_t(kv_heads) * head_dim; }
};

// Validates the artifact's mathematical text config; errors name the offending field.
[[nodiscard]] TextConfig parse_text_config(const artifact::Json& config);
[[nodiscard]] TextConfig parse_text_config(const artifact::Reader&);

} // namespace ninfer::models::spark2_5
