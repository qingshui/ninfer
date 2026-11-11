#include "models/spark2_5/execution/parameters.h"

#include "models/spark2_5/state/window_state.h"

#include "ninfer/ops/weight_input.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::spark2_5 {
namespace {

// The registered Op geometry is the model's support range; anything else is rejected here, before
// any device work, with the field that differs.
void require_supported(const TextConfig& c) {
    const auto require = [](bool value, const char* what) {
        if (!value) throw std::invalid_argument(std::string("Spark2_5 execution: ") + what);
    };
    require(c.head_dim == std::uint32_t(kAttentionHeadDim) &&
                c.heads == std::uint32_t(kQueryHeads) && c.kv_heads == std::uint32_t(kKvHeads),
            "attention geometry must be head_dim 256 with 8 query and 2 KV heads");
    require(c.sliding_window == kWindowCapacity, "sliding_window must be 512");
}

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

} // namespace

Parameters::Parameters(const ModelShard& source)
    : model(source), config(source.definition().config()) {
    require_supported(config);
    const auto& w     = source.definition().weights().text;
    const auto linear = [&](WeightId id) {
        return with_context(source.weight(id).name,
                            [&] { return ops::prepare_linear_weight(source.input(id)); });
    };
    const auto tensor = [&](WeightId id) {
        return with_context(source.weight(id).name,
                            [&] { return parameter_tensor(source.weight(id)); });
    };
    token_embedding = native_weight(source.weight(w.token_embedding).view);
    output_head     = linear(w.output_head);
    final_norm      = tensor(w.final_norm);
    layers.reserve(w.layers.size());
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& l = w.layers[i];
        with_context("text/layers/" + std::to_string(i), [&] {
            const std::array qkv{source.input(l.attention.query), source.input(l.attention.key),
                                 source.input(l.attention.value)};
            const std::array gate_up{source.input(l.mlp.gate), source.input(l.mlp.up)};
            BlockParameters block;
            block.input_norm          = tensor(l.input_norm);
            block.post_attention_norm = tensor(l.post_attention_norm);
            block.attention = {ops::prepare_projection_weights(qkv), linear(l.attention.gate),
                               linear(l.attention.output)};
            block.mlp       = {ops::prepare_linear_weight(gate_up), linear(l.mlp.down)};
            layers.push_back(std::move(block));
            return 0;
        });
    }
}

std::array<std::int32_t, 3> Parameters::qkv_rows() const noexcept {
    return {dimension(config.query_width()), dimension(config.kv_width()),
            dimension(config.kv_width())};
}

} // namespace ninfer::models::spark2_5
