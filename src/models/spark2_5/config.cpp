#include "models/spark2_5/config.h"

#include "artifact/reader.h"
#include "models/spark2_5/state/window_state.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::spark2_5 {
namespace {
using artifact::Json;

[[noreturn]] void fail(const std::string& field, const std::string& message) {
    throw std::invalid_argument("Spark2_5 " + field + ": " + message);
}

const Json& field(const Json& object, const std::string& name, const std::string& label) {
    if (!object.is_object()) fail(label.empty() ? "config" : label, "expected an object");
    const auto found = object.find(name);
    if (found == object.end()) fail(label.empty() ? name : label + "." + name, "missing");
    return *found;
}

const Json& field(const Json& object, const char* name) { return field(object, name, {}); }

void expect_string(const Json& config, const char* name, const char* expected) {
    const auto& value = field(config, name);
    if (!value.is_string() || value.get<std::string>() != expected)
        fail(name, std::string("expected \"") + expected + "\"");
}

std::uint32_t integer(const Json& value, const std::string& name, bool positive) {
    const bool nonnegative =
        value.is_number_unsigned() || (value.is_number_integer() && value.get<std::int64_t>() >= 0);
    if (!nonnegative) fail(name, "expected a nonnegative integer");
    const auto n = value.get<std::uint64_t>();
    if (positive && !n) fail(name, "expected a positive integer");
    if (n > std::numeric_limits<std::int32_t>::max()) fail(name, "exceeds int32");
    return static_cast<std::uint32_t>(n);
}

std::uint32_t extent(const Json& config, const char* name) {
    return integer(field(config, name), name, true);
}

float positive_f32(const Json& value, const std::string& name) {
    if (!value.is_number()) fail(name, "expected a number");
    const auto result = static_cast<float>(value.get<double>());
    if (!std::isfinite(result) || !(result > 0)) fail(name, "expected a positive finite F32");
    return result;
}

TokenId token_id(const Json& config, const char* name, std::uint32_t vocabulary) {
    const auto id = integer(field(config, name), name, false);
    if (id >= vocabulary) fail(name, "outside the vocabulary");
    return static_cast<TokenId>(id);
}

RopeConfig rope_config(const Json& rope, const char* kind, std::uint32_t head_dim) {
    const std::string label = std::string("rope.") + kind;
    const auto& table       = field(rope, kind, "rope");
    RopeConfig out;
    out.rotary_dim = integer(field(table, "rotary_dim", label), label + ".rotary_dim", true);
    if (out.rotary_dim % 2 || out.rotary_dim > head_dim)
        fail(label + ".rotary_dim", "must be even and not exceed head_dim");
    out.theta = positive_f32(field(table, "theta", label), label + ".theta");
    return out;
}

} // namespace

TextConfig parse_text_config(const Json& c) {
    const auto& names = field(c, "architectures");
    if (!names.is_array() || names.size() != 1 || names[0] != "Spark2_5ForCausalLM")
        fail("architectures", "expected [\"Spark2_5ForCausalLM\"]");
    expect_string(c, "model_type", "spark2_5_text");
    expect_string(c, "hidden_act", "gelu");
    expect_string(c, "attention_gate", "headwise_sigmoid");

    TextConfig out{};
    out.hidden       = extent(c, "hidden_size");
    out.heads        = extent(c, "num_attention_heads");
    out.kv_heads     = extent(c, "num_key_value_heads");
    out.head_dim     = extent(c, "head_dim");
    out.intermediate = extent(c, "intermediate_size");
    out.vocabulary   = extent(c, "vocab_size");
    out.max_context  = extent(c, "max_position_embeddings");
    if (out.heads % out.kv_heads)
        fail("num_attention_heads", "must be divisible by num_key_value_heads");
    out.sliding_window = extent(c, "sliding_window");
    if (out.sliding_window <= 1) fail("sliding_window", "must exceed 1");
    // The window ring holds exactly the positions a sliding layer attends.
    if (out.sliding_window != kWindowCapacity)
        fail("sliding_window", "must be " + std::to_string(kWindowCapacity));
    out.norm_epsilon   = positive_f32(field(c, "rms_norm_eps"), "rms_norm_eps");
    const auto& tied   = field(c, "tie_word_embeddings");
    if (!tied.is_boolean()) fail("tie_word_embeddings", "expected a boolean");
    out.tie_word_embeddings = tied.get<bool>();
    out.bos_token_id        = token_id(c, "bos_token_id", out.vocabulary);
    out.eos_token_id        = token_id(c, "eos_token_id", out.vocabulary);
    out.pad_token_id        = token_id(c, "pad_token_id", out.vocabulary);

    const auto layers = extent(c, "num_hidden_layers");
    const auto& kinds = field(c, "layer_types");
    if (!kinds.is_array() || kinds.size() != layers)
        fail("layer_types", "must describe every layer");
    out.layers.reserve(layers);
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        if (kinds[i] == "full_attention")
            out.layers.push_back({LayerKind::Full, out.full_layers++});
        else if (kinds[i] == "sliding_attention")
            out.layers.push_back({LayerKind::Sliding, out.sliding_layers++});
        else
            fail("layer_types[" + std::to_string(i) + "]", "unknown layer type");
    }

    const auto& rope = field(c, "rope");
    if (!rope.is_object()) fail("rope", "expected an object");
    if (out.full_layers) out.full_rope = rope_config(rope, "full_attention", out.head_dim);
    if (out.sliding_layers) out.sliding_rope = rope_config(rope, "sliding_attention", out.head_dim);
    return out;
}

TextConfig parse_text_config(const artifact::Reader& reader) {
    return parse_text_config(reader.directory().component("text").config);
}

} // namespace ninfer::models::spark2_5
