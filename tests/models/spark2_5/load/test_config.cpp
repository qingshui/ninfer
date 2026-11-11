#include "models/registry.h"
#include "models/spark2_5/config.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace ninfer;
namespace spark = models::spark2_5;
using artifact::Json;

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

// Layer kinds are deliberately not periodic: only layer_types may decide them.
Json valid() {
    return Json{{"architectures", {"Spark2_5ForCausalLM"}},
                {"model_type", "spark2_5_text"},
                {"hidden_size", 2048},
                {"num_hidden_layers", 5},
                {"num_attention_heads", 8},
                {"num_key_value_heads", 2},
                {"head_dim", 256},
                {"intermediate_size", 6656},
                {"vocab_size", 131072},
                {"max_position_embeddings", 1048576},
                {"rms_norm_eps", 1e-6},
                {"hidden_act", "gelu"},
                {"attention_gate", "headwise_sigmoid"},
                {"sliding_window", 512},
                {"layer_types",
                 {"full_attention", "sliding_attention", "sliding_attention", "full_attention",
                  "full_attention"}},
                {"rope",
                 {{"full_attention", {{"rotary_dim", 64}, {"theta", 5000000.0}}},
                  {"sliding_attention", {{"rotary_dim", 256}, {"theta", 10000.0}}}}},
                {"tie_word_embeddings", true},
                {"bos_token_id", 1},
                {"eos_token_id", 2},
                {"pad_token_id", 0}};
}

void rejects(const std::string& label, const std::string& field,
             const std::function<void(Json&)>& mutate) {
    auto config = valid();
    mutate(config);
    try {
        (void)spark::parse_text_config(config);
    } catch (const std::invalid_argument& error) {
        require(std::string(error.what()).find(field) != std::string::npos,
                label + ": error does not name " + field + ": " + error.what());
        return;
    }
    throw std::runtime_error(label + ": accepted an invalid config");
}

void accepts_valid_config() {
    const auto c = spark::parse_text_config(valid());
    require(c.hidden == 2048 && c.heads == 8 && c.kv_heads == 2 && c.head_dim == 256 &&
                c.intermediate == 6656 && c.vocabulary == 131072 && c.sliding_window == 512 &&
                c.max_context == 1048576 && c.tie_word_embeddings && c.norm_epsilon == 1e-6F,
            "scalar fields differ");
    require(c.query_width() == 2048 && c.kv_width() == 512, "projection widths differ");
    require(c.bos_token_id == 1 && c.eos_token_id == 2 && c.pad_token_id == 0, "token ids differ");
    constexpr auto F                    = spark::LayerKind::Full;
    constexpr auto S                    = spark::LayerKind::Sliding;
    const spark::LayerKind kinds[]      = {F, S, S, F, F};
    const std::uint32_t state_indices[] = {0, 0, 1, 1, 2};
    require(c.layers.size() == 5 && c.full_layers == 3 && c.sliding_layers == 2,
            "layer counts differ");
    for (std::size_t i = 0; i < c.layers.size(); ++i)
        require(c.layers[i].kind == kinds[i] && c.layers[i].state_index == state_indices[i],
                "layer " + std::to_string(i) + " kind or state index differs");
    require(c.rope(F).rotary_dim == 64 && c.rope(F).theta == 5.0e6F &&
                c.rope(S).rotary_dim == 256 && c.rope(S).theta == 1.0e4F,
            "rope tables differ");

    // Fields outside the model's math (the FP8 checkpoint carries extras) do not change the result.
    auto extra                 = valid();
    extra["quantization_config"] = {{"quant_method", "fp8"}};
    extra["rope"]["rope_type"]   = "default";
    const auto e                 = spark::parse_text_config(extra);
    require(e.full_rope.theta == c.full_rope.theta && e.sliding_rope.rotary_dim == 256 &&
                e.layers.size() == c.layers.size() && e.sliding_window == c.sliding_window,
            "extra fields changed the parsed config");

    auto only_full                 = valid();
    only_full["num_hidden_layers"] = 1;
    only_full["layer_types"]       = {"full_attention"};
    only_full["rope"].erase("sliding_attention");
    const auto f = spark::parse_text_config(only_full);
    require(f.sliding_layers == 0 && f.rope(S).rotary_dim == 0, "absent kind keeps a rope");
}

void rejects_invalid_fields() {
    rejects("architecture", "architectures", [](Json& c) { c["architectures"] = {"Other"}; });
    rejects("model_type", "model_type", [](Json& c) { c["model_type"] = "spark2_5"; });
    rejects("activation", "hidden_act", [](Json& c) { c["hidden_act"] = "silu"; });
    rejects("gate", "attention_gate", [](Json& c) { c["attention_gate"] = "elementwise"; });
    rejects("missing head_dim", "head_dim", [](Json& c) { c.erase("head_dim"); });
    rejects("zero hidden", "hidden_size", [](Json& c) { c["hidden_size"] = 0; });
    rejects("fractional heads", "num_attention_heads",
            [](Json& c) { c["num_attention_heads"] = 8.5; });
    rejects("grouping", "num_attention_heads", [](Json& c) { c["num_key_value_heads"] = 3; });
    rejects("window", "sliding_window", [](Json& c) { c["sliding_window"] = -1; });
    rejects("one-token window", "sliding_window", [](Json& c) { c["sliding_window"] = 1; });
    rejects("narrow window", "sliding_window", [](Json& c) { c["sliding_window"] = 256; });
    rejects("wide window", "sliding_window", [](Json& c) { c["sliding_window"] = 1024; });
    rejects("epsilon", "rms_norm_eps", [](Json& c) { c["rms_norm_eps"] = 0.0; });
    rejects("tie", "tie_word_embeddings", [](Json& c) { c["tie_word_embeddings"] = "yes"; });
    rejects("eos", "eos_token_id", [](Json& c) { c["eos_token_id"] = 131072; });
    rejects("layer count", "layer_types", [](Json& c) { c["num_hidden_layers"] = 6; });
    rejects("layer kind", "layer_types[1]", [](Json& c) { c["layer_types"][1] = "linear"; });
    rejects("missing rope", "rope.sliding_attention",
            [](Json& c) { c["rope"].erase("sliding_attention"); });
    rejects("odd rotary", "rope.full_attention.rotary_dim",
            [](Json& c) { c["rope"]["full_attention"]["rotary_dim"] = 63; });
    rejects("wide rotary", "rope.sliding_attention.rotary_dim",
            [](Json& c) { c["rope"]["sliding_attention"]["rotary_dim"] = 512; });
    rejects("theta", "rope.full_attention.theta",
            [](Json& c) { c["rope"]["full_attention"]["theta"] = -1.0; });
    rejects("missing theta", "rope.sliding_attention.theta",
            [](Json& c) { c["rope"]["sliding_attention"].erase("theta"); });
}

void resolves_architecture() {
    using models::Architecture;
    require(models::resolve_architecture("Spark2_5ForCausalLM", "spark2_5_text") ==
                Architecture::Spark2_5,
            "Spark2_5 does not resolve");
    require(models::architecture_name(Architecture::Spark2_5) == "Spark2_5ForCausalLM",
            "Spark2_5 name differs");
    for (const auto model_type : {"spark2_5", "qwen3_5_text"}) {
        bool rejected = false;
        try {
            (void)models::resolve_architecture("Spark2_5ForCausalLM", model_type);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, std::string("accepted Spark2_5 with model_type ") + model_type);
    }
}
} // namespace

int main() {
    try {
        accepts_valid_config();
        rejects_invalid_fields();
        resolves_architecture();
        std::cout << "PASS Spark2_5 config and registry\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
