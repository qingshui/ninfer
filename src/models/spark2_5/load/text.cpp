#include "models/spark2_5/load.h"

#include <algorithm>
#include <string>

namespace ninfer::models::spark2_5::loading {
namespace {

AttentionWeights attention(Bindings& b, const TextConfig& c, const std::string& layer) {
    const auto p     = layer + "attention/";
    const auto input = std::vector<std::string>{layer + "mixer_input"};
    return {b.parameter(p + "query", {c.query_width(), c.hidden}, input),
            b.parameter(p + "key", {c.kv_width(), c.hidden}, input),
            b.parameter(p + "value", {c.kv_width(), c.hidden}, input),
            b.parameter(p + "gate", {c.heads, c.hidden}, input),
            b.parameter(p + "output", {c.hidden, c.query_width()}, {p + "gated_output"})};
}

MlpWeights mlp(Bindings& b, const TextConfig& c, const std::string& layer) {
    const auto p     = layer + "mlp/";
    const auto input = std::vector<std::string>{layer + "ffn_input"};
    return {b.parameter(p + "gate", {c.intermediate, c.hidden}, input),
            b.parameter(p + "up", {c.intermediate, c.hidden}, input),
            b.parameter(p + "down", {c.hidden, c.intermediate}, {p + "product"})};
}

} // namespace

void require_shared(const artifact::ParameterReference& parameter,
                    const artifact::ParameterReference& target) {
    const auto& parts  = parameter.binding.parts;
    const auto& stored = target.binding.parts;
    if (!std::equal(parts.begin(), parts.end(), stored.begin(), stored.end(),
                    [](const auto& a, const auto& b) {
                        return a.object == b.object && a.begin == b.begin && a.end == b.end;
                    }))
        throw artifact::ArtifactError(parameter.name + ": must share " + target.name +
                                      " weight storage");
}

TextWeights bind_text(Bindings& b, const TextConfig& c) {
    TextWeights out;
    // The embedding is a gather table; the Embedding Op owns the supported table formats.
    out.token_embedding = b.parameter("text/token_embedding", {c.vocabulary, c.hidden});
    out.output_head =
        b.parameter("text/output_head", {c.vocabulary, c.hidden}, {"text/final_hidden"});
    if (c.tie_word_embeddings)
        require_shared(b.at(out.output_head).reference, b.at(out.token_embedding).reference);
    out.final_norm = b.direct("text/final_norm", {c.hidden});
    out.layers.reserve(c.layers.size());
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
        const auto p = "text/layers/" + std::to_string(i) + "/";
        LayerWeights layer;
        layer.input_norm          = b.direct(p + "input_norm", {c.hidden});
        layer.attention           = attention(b, c, p);
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {c.hidden});
        layer.mlp                 = mlp(b, c, p);
        out.layers.push_back(layer);
    }
    return out;
}

} // namespace ninfer::models::spark2_5::loading
