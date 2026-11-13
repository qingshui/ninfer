#pragma once

#include "models/spark2_5/frontend/prepared_prompt.h"
#include "models/spark2_5/frontend/resources.h"
#include "ninfer/types.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace spark_fixture {
using namespace ninfer;
using Json      = nlohmann::ordered_json;
namespace spark = models::spark2_5;

inline const auto fixtures =
    std::filesystem::path(NINFER_SOURCE_DIR) / "tests/models/spark2_5/frontend/fixtures";

inline std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}

inline void require(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error(message);
}

template <class Action>
void rejects(Action action, std::string_view reason) {
    try {
        action();
    } catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()).find(reason) != std::string_view::npos,
                "unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection: " + std::string(reason));
}

template <class Action>
void fails(Action action, RequestErrorKind kind) {
    try {
        action();
    } catch (const RequestError& error) {
        require(error.kind() == kind, std::string("unexpected request error: ") + error.what());
        return;
    }
    throw std::runtime_error("expected request error");
}

inline ChatRole role(std::string_view name) {
    if (name == "system") return ChatRole::System;
    if (name == "developer") return ChatRole::Developer;
    if (name == "user") return ChatRole::User;
    if (name == "assistant") return ChatRole::Assistant;
    if (name == "tool") return ChatRole::Tool;
    throw std::invalid_argument("invalid fixture role");
}

inline ChatMessage message(ChatRole role, std::string content) {
    return {.role = role, .parts = {{.kind = MessagePartKind::Text, .text = std::move(content)}}};
}

// Engine input of a golden case: content is a string or text blocks.
inline PromptInput prompt(const Json& input) {
    PromptInput result;
    for (const auto& item : input.at("messages")) {
        ChatMessage m;
        m.role              = role(item.at("role").get<std::string>());
        const auto& content = item.at("content");
        if (content.is_string()) {
            m.parts.push_back({.kind = MessagePartKind::Text, .text = content.get<std::string>()});
        } else {
            for (const auto& part : content) {
                require(part.at("type") == "text", "invalid fixture part");
                m.parts.push_back(
                    {.kind = MessagePartKind::Text, .text = part.at("text").get<std::string>()});
            }
        }
        m.reasoning_content = item.value("reasoning_content", std::string{});
        m.tool_call_id      = item.value("tool_call_id", std::string{});
        for (const auto& call : item.value("tool_calls", Json::array()))
            m.tool_calls.push_back({.id   = call.at("id").get<std::string>(),
                                    .name = call.at("function").at("name").get<std::string>(),
                                    .arguments_json = call.at("function").at("arguments").dump()});
        result.messages.push_back(std::move(m));
    }
    const auto& opts = input.at("options");
    auto& options    = result.options;
    if (opts.value("continuation", false))
        options.continuation = PromptContinuationMode::ContinueFinalAssistant;
    if (opts.contains("enable_thinking"))
        options.enable_thinking = opts.at("enable_thinking").get<bool>();
    if (opts.contains("reasoning_effort")) {
        const auto name = opts.at("reasoning_effort").get<std::string>();
        for (auto effort : {ReasoningEffort::None, ReasoningEffort::Minimal, ReasoningEffort::Low,
                            ReasoningEffort::Medium, ReasoningEffort::High, ReasoningEffort::XHigh,
                            ReasoningEffort::Max})
            if (reasoning_effort_name(effort) == name) options.reasoning_effort = effort;
        require(options.reasoning_effort.has_value(), "invalid fixture effort");
    }
    for (const auto& tool : opts.value("tools", Json::array()))
        options.tool_jsons.push_back(tool.dump());
    return result;
}

inline Json tokenizer_config() {
    return {{"bos_token", "<｜start▁of▁sentence｜>"},
            {"eos_token", "<｜end▁of▁sentence｜>"},
            {"pad_token", "<｜▁pad▁｜>"}};
}

// Parsed resources over the small fixture tokenizer unless `root` names a checkpoint directory.
struct Resources {
    std::string tokenizer, config, generation, chat_template;
    spark::FrontendResources parsed;
};

inline std::unique_ptr<Resources> resources(const std::filesystem::path& tokenizer_path,
                                            const std::string& config = tokenizer_config().dump(),
                                            std::string generation    = {}) {
    auto result           = std::make_unique<Resources>();
    result->tokenizer     = read(tokenizer_path);
    result->config        = config;
    result->chat_template = read(fixtures / "chat_template.jinja");
    const auto json       = Json::parse(result->tokenizer);
    std::uint32_t rows    = 0;
    std::int64_t eos      = -1;
    for (const auto& item : json["model"]["vocab"].items())
        rows = std::max(rows, item.value().get<std::uint32_t>() + 1);
    for (const auto& token : json["added_tokens"]) {
        rows = std::max(rows, token["id"].get<std::uint32_t>() + 1);
        if (token["content"] == "<｜end▁of▁sentence｜>") eos = token["id"].get<std::int64_t>();
    }
    require(eos >= 0, "fixture tokenizer has no end of sentence");
    result->generation =
        generation.empty()
            ? Json{{"eos_token_id", eos}, {"temperature", 1.0}, {"top_p", 0.95}, {"top_k", -1}}
                  .dump()
            : std::move(generation);
    result->parsed.text = {.tokenizer_json         = result->tokenizer,
                           .tokenizer_config_json  = result->config,
                           .chat_template_jinja    = result->chat_template,
                           .generation_config_json = result->generation};
    spark::parse_resources(result->parsed, rows);
    return result;
}

inline spark::Frontend frontend(const Resources& resources, std::uint32_t max_context = 16384) {
    return spark::make_frontend(resources.parsed,
                                {.vision_enabled = false, .max_context = max_context});
}

inline Json tool_schema(Json properties, std::vector<std::string> required = {}) {
    return {{"type", "object"},
            {"properties", std::move(properties)},
            {"required", std::move(required)},
            {"additionalProperties", false}};
}

inline std::string declaration(const Json& schema, bool strict = false, std::string name = "call") {
    return Json{{"type", "function"},
                {"function", {{"name", name}, {"parameters", schema}, {"strict", strict}}}}
        .dump();
}

inline PromptInput input(std::vector<std::string> tools = {}) {
    PromptInput p;
    p.messages.push_back(message(ChatRole::User, "question"));
    p.options.tool_jsons = std::move(tools);
    return p;
}

struct Output {
    std::string reasoning, content;
    std::vector<GeneratedToolCall> calls;

    void append(const std::vector<OutputEvent>& events) {
        for (const auto& event : events) {
            if (const auto* text = std::get_if<OutputDelta>(&event))
                (text->channel == OutputChannel::Reasoning ? reasoning : content) += text->text;
            else if (const auto* call = std::get_if<ToolCallStart>(&event)) {
                require(call->index == calls.size(), "call event index is not append-only");
                calls.push_back({call->name, {}, ToolCallStatus::InProgress});
            } else if (const auto* args = std::get_if<ToolArgumentsDelta>(&event))
                calls.at(args->index).arguments_json += args->text;
            else if (const auto* end = std::get_if<ToolCallEnd>(&event))
                calls.at(end->index).status = end->status;
        }
    }
};

inline void commit(models::frontend::OutputSession& session, std::span<const TokenId> ids,
                   Output& result, std::uint32_t budget = 100000) {
    (void)session.preview_model(ids, budget, FinishReason::OutputLimit);
    result.append(session.commit_preview());
    if (auto error = session.output_error()) std::rethrow_exception(error->exception());
}
} // namespace spark_fixture
