#include "models/spark2_5/frontend/resources.h"

#include "artifact/binder.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ninfer::models::spark2_5 {

void parse_resources(FrontendResources& resources, std::uint32_t embedding_rows) {
    models::frontend::parse_text_resources(resources.text, embedding_rows);
    models::frontend::prepare_control_tokens(resources.text, "</think>", "</think>");
    const auto generation =
        artifact::parse_json(resources.text.generation_config_json, "generation_config.json");
    SamplingPreset sampling;
    try {
        sampling.temperature = generation.value("temperature", 1.0F);
        sampling.top_p       = generation.value("top_p", 0.95F);
        if (const auto top_k = generation.find("top_k"); top_k != generation.end()) {
            if (!top_k->is_number_integer() || *top_k < std::numeric_limits<std::int32_t>::min() ||
                *top_k > std::numeric_limits<std::int32_t>::max())
                throw artifact::ArtifactError("generation_config.json top_k must fit int32");
            // The checkpoint writes -1 for "no top-k"; the sampling contract spells that 0.
            sampling.top_k = std::max<std::int32_t>(top_k->get<std::int32_t>(), 0);
        }
        sampling.min_p            = generation.value("min_p", sampling.min_p);
        sampling.presence_penalty = generation.value("presence_penalty", sampling.presence_penalty);
        sampling.frequency_penalty =
            generation.value("frequency_penalty", sampling.frequency_penalty);
        if (!generation.value("do_sample", true)) sampling.temperature = 0.0F;
    } catch (const nlohmann::json::exception& error) {
        throw artifact::ArtifactError(std::string("invalid generation defaults: ") + error.what());
    }
    for (const float value : {sampling.temperature, sampling.top_p, sampling.min_p,
                              sampling.presence_penalty, sampling.frequency_penalty})
        if (!std::isfinite(value))
            throw artifact::ArtifactError("generation defaults must fit finite float values");
    resources.sampling = sampling;
}

FrontendResources bind_resources(artifact::Binder& binder, std::uint32_t embedding_rows) {
    const auto bytes = [&](std::string_view role) {
        const auto data = binder.resource_bytes(binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(data.data()), data.size());
    };
    FrontendResources resources;
    resources.text.tokenizer_json         = bytes("tokenizer.json");
    resources.text.tokenizer_config_json  = bytes("tokenizer_config.json");
    resources.text.chat_template_jinja    = bytes("chat_template.jinja");
    resources.text.generation_config_json = bytes("generation_config.json");
    parse_resources(resources, embedding_rows);
    return resources;
}

} // namespace ninfer::models::spark2_5
