#include "models/spark2_5/frontend/chat_template.h"

#include "models/common/frontend/media_cache.h"
#include "models/spark2_5/frontend/prompt_layout.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::spark2_5 {
namespace {
using Json = nlohmann::ordered_json;

struct Role {
    ChatRole role;
    std::string_view marker;
};

constexpr Role kRoles[] = {{ChatRole::System, kSystemRole},
                           {ChatRole::User, kUserRole},
                           {ChatRole::Assistant, kBotRole},
                           {ChatRole::Tool, kToolRole}};

// The template exposes two reasoning levels besides disabled thinking.
std::string_view supported_effort(std::string_view name) {
    if (name == "none" || name == "low" || name == "high") return name;
    throw std::invalid_argument("Spark2_5 reasoning_effort must be none, low or high");
}

Json template_parameters(const PromptOptions& options, const Json& tokens) {
    auto context = models::frontend::template_context(options.chat_template_kwargs_json, tokens);
    for (const char* key : {"enable_thinking", "reasoning_effort"})
        if (context.contains(key) && context[key].is_null()) context.erase(key);
    if (options.enable_thinking)
        models::frontend::merge_template_option(context, "enable_thinking",
                                                *options.enable_thinking);
    if (options.reasoning_effort) {
        const auto name = reasoning_effort_name(*options.reasoning_effort);
        if (name.empty()) throw std::invalid_argument("invalid reasoning effort");
        models::frontend::merge_template_option(context, "reasoning_effort", name);
    }
    if (context.contains("enable_thinking") && !context["enable_thinking"].is_boolean())
        throw std::invalid_argument("enable_thinking must be a boolean");
    if (options.preserve_thinking && !*options.preserve_thinking)
        throw std::invalid_argument(
            "Spark2_5 always keeps earlier reasoning; preserve_thinking=false is unsupported");
    if (options.add_vision_id) throw std::invalid_argument("Spark2_5 does not support vision");

    if (context.contains("reasoning_effort")) {
        const auto& effort = context["reasoning_effort"];
        if (!effort.is_string()) throw std::invalid_argument("reasoning_effort must be a string");
        const auto name     = supported_effort(effort.get_ref<const std::string&>());
        const bool thinking = name != "none";
        if (context.contains("enable_thinking") && context["enable_thinking"] != thinking)
            throw RequestError(RequestErrorKind::ConflictingPromptOptions,
                               "reasoning_effort conflicts with enable_thinking",
                               "reasoning_effort");
        context["enable_thinking"] = thinking;
        // The template has no "none" level; disabled thinking is expressed by enable_thinking.
        if (!thinking) context.erase("reasoning_effort");
    }
    if (!context.contains("enable_thinking") && options.defaults.enable_thinking)
        context["enable_thinking"] = *options.defaults.enable_thinking;
    context.update(tokens);
    return context;
}

std::string_view role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
    case ChatRole::Developer:
        return "system";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("invalid chat role");
}

struct TemplateInput {
    Json context;
    bool thinking = true;
    std::vector<text::TemplateInputRegion> regions;
    std::vector<MessageSources> messages;
    std::vector<std::uint32_t> tools;

    std::uint32_t tag(std::string pointer) {
        const auto id = static_cast<std::uint32_t>(regions.size() + 1);
        regions.push_back({std::move(pointer), id});
        return id;
    }

    void append_message(const ChatMessage& message) {
        if (message.role != ChatRole::Assistant &&
            (!message.reasoning_content.empty() || !message.tool_calls.empty()))
            throw std::invalid_argument("reasoning and tool calls require an assistant message");
        if (message.role != ChatRole::Tool &&
            (!message.tool_call_id.empty() || message.tool_result_is_error))
            throw std::invalid_argument("tool result fields require a tool message");

        const auto pointer = "/messages/" + std::to_string(messages.size());
        Json value{{"role", role_name(message.role)}};
        MessageSources sources;
        sources.parts.resize(message.parts.size());
        std::string content = message.tool_result_is_error ? "[tool_error]\n" : "";
        const auto id       = tag(pointer + "/content");
        for (std::size_t i = 0; i < message.parts.size(); ++i) {
            const auto& part = message.parts[i];
            if (part.kind == MessagePartKind::Media)
                throw std::invalid_argument("Spark2_5 does not support media input");
            if (part.kind != MessagePartKind::Text)
                throw std::invalid_argument("invalid message part kind");
            const auto begin = content.size();
            content += part.text;
            sources.parts[i] = TextSource{id, {begin, content.size()}};
        }
        sources.content  = TextSource{id, {0, content.size()}};
        value["content"] = std::move(content);
        if (message.role == ChatRole::Assistant) {
            // An explicit field keeps literal <think> text in the content from being read as
            // reasoning by templates that import it.
            value["reasoning_content"] = message.reasoning_content;
            sources.reasoning          = {tag(pointer + "/reasoning_content"),
                                          {0, message.reasoning_content.size()}};
        }
        if (!message.tool_calls.empty()) {
            value["tool_calls"] = Json::array();
            for (const auto& call : message.tool_calls) {
                auto arguments =
                    call.arguments_json.empty() ? Json::object() : Json::parse(call.arguments_json);
                if (!arguments.is_object())
                    throw std::invalid_argument("tool call arguments must be a JSON object");
                sources.tool_calls.push_back(
                    tag(pointer + "/tool_calls/" + std::to_string(value["tool_calls"].size())));
                value["tool_calls"].push_back(
                    {{"id", call.id},
                     {"type", "function"},
                     {"function", {{"name", call.name}, {"arguments", std::move(arguments)}}}});
            }
        }
        if (!message.tool_call_id.empty()) value["tool_call_id"] = message.tool_call_id;
        context["messages"].push_back(std::move(value));
        messages.push_back(std::move(sources));
    }
};

void truncate(text::TemplateOutput& output, std::size_t end) {
    output.text.resize(end);
    std::erase_if(output.regions, [&](const auto& region) { return region.begin > end; });
    for (auto& region : output.regions) region.end = std::min(region.end, end);
    std::erase_if(output.literal_spans, [&](auto span) { return span.begin >= end; });
    if (!output.literal_spans.empty())
        output.literal_spans.back().end = std::min(output.literal_spans.back().end, end);
    std::erase_if(output.boundary_mappings,
                  [&](const auto& mapping) { return mapping.offset > end; });
}

// The final assistant message was rendered whole; reopen it after its completed reasoning.
void finish_continuation(RenderedChat& result, const Block& block, std::string_view content) {
    auto& output         = result.output;
    const auto& rendered = output.text;
    const auto end       = rendered.size() - kSentenceEnd.size();
    if (!std::string_view(rendered).ends_with(kSentenceEnd) ||
        text::overlaps(output.literal_spans, end, rendered.size()))
        throw std::invalid_argument("Spark2_5 chat template must close the final assistant turn");
    std::size_t close = std::string::npos;
    if (find_control(output, kThinkOpen, block.content, block.content + kThinkOpen.size()) ==
        block.content)
        close = find_control(output, kThinkClose, block.content + kThinkOpen.size(), end);
    else if (find_control(output, kThinkClose, block.content, block.content + kThinkClose.size()) ==
             block.content)
        close = block.content;
    if (close == std::string::npos)
        throw RequestError(RequestErrorKind::UnsupportedPromptContinuation,
                           "Spark2_5 continuation requires completed reasoning");
    const auto content_begin = close + kThinkClose.size();
    if (end < content.size() || end - content.size() != content_begin ||
        std::string_view(rendered).substr(content_begin, content.size()) != content)
        throw RequestError(RequestErrorKind::UnsupportedPromptContinuation,
                           "Spark2_5 chat template must expose the final assistant text once");
    result.continuation_content = std::string(content);
    truncate(output, end);
}

RenderedChat render_input(const TemplateInput& input,
                          const models::frontend::ChatTemplate& compiled, bool continuation,
                          const PreparationControl& control) {
    std::vector<std::string> control_variables;
    for (const auto& item : compiled.special_tokens.items())
        control_variables.push_back(item.key());
    const text::TemplateRenderOptions execution{
        .checkpoint =
            [&] { models::frontend::check_preparation_control(control, "Spark2_5 chat template"); },
        .regions           = input.regions,
        .control_variables = control_variables};
    RenderedChat result{.output   = compiled.program.render(input.context, execution),
                        .messages = input.messages,
                        .tools    = input.tools};
    const auto blocks = scan_blocks(result.output);
    if (blocks.empty() || blocks.back().role != ChatRole::Assistant)
        throw std::invalid_argument("Spark2_5 chat template must end in an assistant message");
    const auto& last = blocks.back();
    if (continuation) {
        finish_continuation(
            result, last,
            input.context["messages"].back()["content"].get_ref<const std::string&>());
        return result;
    }
    // The generation prefix either opens reasoning or closes it at once.
    const auto prefix = input.thinking ? kThinkOpen : kThinkClose;
    if (find_control(result.output, prefix, last.content) != last.content ||
        last.content + prefix.size() != result.output.text.size())
        throw std::invalid_argument("Spark2_5 generation prefix must end with " +
                                    std::string(prefix));
    result.starts_in_reasoning = input.thinking;
    return result;
}
} // namespace

std::size_t find_control(const text::TemplateOutput& output, std::string_view marker,
                         std::size_t begin, std::size_t end) {
    end = std::min(end, output.text.size());
    for (auto pos = output.text.find(marker, begin); pos != std::string::npos && pos < end;
         pos      = output.text.find(marker, pos + 1))
        if (pos + marker.size() <= end &&
            !text::overlaps(output.literal_spans, pos, pos + marker.size()))
            return pos;
    return std::string::npos;
}

std::vector<Block> scan_blocks(const text::TemplateOutput& output) {
    std::vector<Block> blocks;
    for (auto pos = find_control(output, kSentenceStart); pos != std::string::npos;
         pos      = find_control(output, kSentenceStart, pos + kSentenceStart.size())) {
        const auto body = pos + kSentenceStart.size();
        for (const auto& [role, marker] : kRoles) {
            if (!std::string_view(output.text).substr(body).starts_with(marker) ||
                text::overlaps(output.literal_spans, body, body + marker.size()))
                continue;
            if (!blocks.empty()) blocks.back().bytes.end = pos;
            blocks.push_back({role, {pos, output.text.size()}, body + marker.size()});
            break;
        }
    }
    return blocks;
}

RenderedChat CompiledChatTemplate::render(std::span<const ChatMessage> messages,
                                          const PromptOptions& options,
                                          const PreparationControl& control) const {
    return render_impl(messages, options, {}, nullptr, control);
}

RenderedChat CompiledChatTemplate::render(std::span<const ChatMessage> messages,
                                          const PromptOptions& options,
                                          std::span<const PromptCacheMarker> markers,
                                          PromptBoundaries& boundaries,
                                          const PreparationControl& control) const {
    return render_impl(messages, options, markers, &boundaries, control);
}

RenderedChat CompiledChatTemplate::render_impl(std::span<const ChatMessage> messages,
                                               const PromptOptions& options,
                                               std::span<const PromptCacheMarker> markers,
                                               PromptBoundaries* boundaries,
                                               const PreparationControl& control) const {
    models::frontend::check_preparation_control(control, "Spark2_5 chat template");
    if (messages.empty()) throw std::invalid_argument("chat requires at least one message");
    bool continuation = false;
    switch (options.continuation) {
    case PromptContinuationMode::NewAssistantTurn:
        break;
    case PromptContinuationMode::ContinueFinalAssistant:
        continuation = true;
        if (messages.back().role != ChatRole::Assistant || !messages.back().tool_calls.empty() ||
            std::any_of(messages.back().parts.begin(), messages.back().parts.end(),
                        [](const auto& part) { return part.kind != MessagePartKind::Text; }))
            throw RequestError(
                RequestErrorKind::UnsupportedPromptContinuation,
                "Spark2_5 continuation requires a final text-only assistant message");
        break;
    default:
        throw std::invalid_argument("invalid prompt continuation mode");
    }
    TemplateInput input{.context = template_parameters(options, compiled_.special_tokens)};
    input.thinking                          = input.context.value("enable_thinking", true);
    input.context["messages"]               = Json::array();
    input.context["add_generation_prompt"]  = !continuation;
    input.context["continue_final_message"] = continuation;
    for (const auto& message : messages) input.append_message(message);
    input.context["tools"] = Json::array();
    for (const auto& source : options.tool_jsons) {
        auto tool = Json::parse(source);
        if (!tool.is_object())
            throw std::invalid_argument("tool declaration must be a JSON object");
        input.tools.push_back(input.tag("/tools/" + std::to_string(input.tools.size())));
        input.context["tools"].push_back(std::move(tool));
    }
    auto result = render_input(input, compiled_, continuation, control);
    if (boundaries) {
        PromptProbes probes;
        probes.content_prefix = [&](const PromptCacheMarker& marker) {
            auto probe       = input;
            const auto index = marker.after_message_count - 1;
            auto& source     = probe.messages[index];
            auto& value      = probe.context["messages"][index];
            const auto bytes = marker.after_message_part_count
                                   ? source.parts[marker.after_message_part_count - 1]->bytes.end
                                   : 0;
            value["content"].get_ref<std::string&>().resize(bytes);
            source.content->bytes.end = bytes;
            source.parts.resize(marker.after_message_part_count);
            value["reasoning_content"].get_ref<std::string&>().resize(marker.reasoning_bytes);
            source.reasoning.bytes.end = marker.reasoning_bytes;
            if (value.contains("tool_calls")) {
                auto& calls = value["tool_calls"];
                calls.erase(calls.begin() + marker.after_tool_call_count, calls.end());
            }
            source.tool_calls.resize(marker.after_tool_call_count);
            return render_input(probe, compiled_, continuation, control);
        };
        probes.next_turn = [&] {
            bool has_response = false;
            for (const auto& message : messages) {
                if (message.role == ChatRole::User)
                    has_response = false;
                else if (message.role == ChatRole::Assistant)
                    has_response = true;
            }
            auto probe = input;
            std::optional<RenderedChat> synthetic;
            if (!has_response) {
                probe.append_message(ChatMessage{.role              = ChatRole::Assistant,
                                                 .parts             = {{.text = "response"}},
                                                 .reasoning_content = "reasoning"});
                synthetic = render_input(probe, compiled_, false, control);
            }
            const auto& before = synthetic ? *synthetic : result;
            const auto end     = scan_blocks(before.output).back().bytes.begin;
            probe.append_message(ChatMessage{.role = ChatRole::User, .parts = {{.text = ""}}});
            auto after         = render_input(probe, compiled_, false, control);
            const auto retains = models::frontend::same_prefix(before.output, after.output, end);
            return TurnProbe{retains, std::move(after.output)};
        };
        *boundaries = resolve_prompt_boundaries(result, messages, continuation, markers, probes);
    }
    return result;
}

} // namespace ninfer::models::spark2_5
