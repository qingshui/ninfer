// Binds a real Spark2_5 artifact (BF16 or FP8) named by NINFER_SPARK2_5_ARTIFACT.
#include "models/spark2_5/definition.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace ninfer;
using namespace ninfer::models;
namespace spark = models::spark2_5;

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

bool same_storage(const artifact::ParameterReference& a, const artifact::ParameterReference& b) {
    const auto& x = a.binding.parts;
    const auto& y = b.binding.parts;
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i].object != y[i].object || x[i].begin != y[i].begin || x[i].end != y[i].end)
            return false;
    return true;
}

void require_use(const spark::ModelDefinition& d, WeightId id, const std::string& input) {
    const auto uses = d.uses(id);
    require(uses.size() == 1 && uses[0].input == input,
            d.parameter(id).name + ": Use differs from " + input);
}

void require_shape(const spark::ModelDefinition& d, WeightId id, artifact::Shape shape) {
    require(d.parameter(id).shape == shape, d.parameter(id).name + ": shape differs");
}

void require_adjacent(const spark::ModelDefinition& d, std::initializer_list<WeightId> ids) {
    const WeightParent* parent = nullptr;
    std::uint64_t end          = 0;
    for (const auto id : ids) {
        const auto region = contiguous_weight_region(d.weight_data()[id.index].view);
        require(!parent || (parent == region.parent && region.begin == end),
                d.parameter(id).name + ": merged projection has a gap or another parent");
        parent = region.parent;
        end    = region.end;
    }
}

void rejects_options(const artifact::Reader& reader) {
    for (const LoadOptions options :
         {LoadOptions{.vision = true}, LoadOptions{.speculative = SpeculativeBackend::Mtp}}) {
        bool rejected = false;
        try {
            (void)spark::read_definition(reader, options);
        } catch (const artifact::ArtifactError&) { rejected = true; }
        require(rejected, "unsupported component selection was accepted");
    }
}

void binds(const artifact::Reader& reader) {
    const auto definition = spark::read_definition(reader);
    const auto& c         = definition->config();
    const auto& w         = definition->weights().text;
    const auto& raw       = reader.directory().component("text").config;
    const auto& kinds     = raw.at("layer_types");
    require(c.layers.size() == kinds.size() && w.layers.size() == kinds.size(),
            "layer count differs from layer_types");
    std::uint32_t full = 0, sliding = 0;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        const bool is_full = kinds[i] == "full_attention";
        require(c.layers[i].kind == (is_full ? spark::LayerKind::Full : spark::LayerKind::Sliding),
                "layer " + std::to_string(i) + " kind differs from layer_types");
        require(c.layers[i].state_index == (is_full ? full : sliding),
                "layer " + std::to_string(i) + " state index differs");
        (is_full ? full : sliding)++;
    }
    require(full == c.full_layers && sliding == c.sliding_layers, "per-kind counts differ");

    const std::uint64_t h = c.hidden, q = c.query_width(), kv = c.kv_width(), ff = c.intermediate;
    require_shape(*definition, w.token_embedding, {c.vocabulary, h});
    require_shape(*definition, w.output_head, {c.vocabulary, h});
    require_shape(*definition, w.final_norm, {h});
    require_use(*definition, w.output_head, "text/final_hidden");
    if (c.tie_word_embeddings)
        require(same_storage(definition->parameter(w.output_head),
                             definition->parameter(w.token_embedding)),
                "tied output head does not share the embedding payload");
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto p      = "text/layers/" + std::to_string(i) + "/";
        const auto& layer = w.layers[i];
        const auto& a     = layer.attention;
        const auto& m     = layer.mlp;
        require_shape(*definition, layer.input_norm, {h});
        require_shape(*definition, layer.post_attention_norm, {h});
        require_shape(*definition, a.query, {q, h});
        require_shape(*definition, a.key, {kv, h});
        require_shape(*definition, a.value, {kv, h});
        require_shape(*definition, a.gate, {c.heads, h});
        require_shape(*definition, a.output, {h, q});
        require_shape(*definition, m.gate, {ff, h});
        require_shape(*definition, m.up, {ff, h});
        require_shape(*definition, m.down, {h, ff});
        for (const auto id : {a.query, a.key, a.value, a.gate})
            require_use(*definition, id, p + "mixer_input");
        require_use(*definition, a.output, p + "attention/gated_output");
        require_use(*definition, m.gate, p + "ffn_input");
        require_use(*definition, m.up, p + "ffn_input");
        require_use(*definition, m.down, p + "mlp/product");
        require_adjacent(*definition, {a.query, a.key, a.value});
        require_adjacent(*definition, {m.gate, m.up});
    }
}
} // namespace

int main() {
    try {
        const char* path = std::getenv("NINFER_SPARK2_5_ARTIFACT");
        if (!path || !*path) {
            std::cout << "SKIP NINFER_SPARK2_5_ARTIFACT is not set\n";
            return 77;
        }
        artifact::Reader reader(path);
        binds(reader);
        rejects_options(reader);
        std::cout << "PASS Spark2_5 definition binding\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
