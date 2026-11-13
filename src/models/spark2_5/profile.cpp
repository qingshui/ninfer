#include "models/spark2_5/profile.h"

#include "ninfer/ops/weight_input.h"

#include <array>

namespace ninfer::models::spark2_5 {

std::shared_ptr<const ModelProfile>
describe_model(std::shared_ptr<const ModelDefinition> definition) {
    const auto& d    = *definition;
    const auto& text = d.resources().text;
    auto profile                 = std::make_shared<ModelProfile>();
    profile->vocabulary          = d.config().vocabulary;
    profile->public_tokens       = text.public_token_count;
    profile->control_token_limit = text.control_tokens ? std::uint32_t(text.control_tokens->size()) : 0;
    const auto& w                = d.weights().text;
    const std::array head{d.input(w.output_head)};
    profile->planning_head       = ops::describe_linear_weight(head);
    for (const auto& l : w.layers) {
        const std::array qkv{d.input(l.attention.query), d.input(l.attention.key),
                             d.input(l.attention.value)};
        const std::array gate_up{d.input(l.mlp.gate), d.input(l.mlp.up)};
        const auto single = [&](WeightId id) {
            const std::array one{d.input(id)};
            return ops::describe_linear_weight(one);
        };
        profile->planning_layers.push_back({ops::describe_projection_weights(qkv),
                                            single(l.attention.gate), single(l.attention.output),
                                            ops::describe_linear_weight(gate_up),
                                            single(l.mlp.down)});
    }
    profile->definition = std::move(definition);
    return profile;
}

} // namespace ninfer::models::spark2_5
