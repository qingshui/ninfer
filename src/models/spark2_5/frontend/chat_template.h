#pragma once

#include "models/common/frontend/chat_template.h"
#include "ninfer/types.h"

#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::spark2_5 {

struct PromptBoundaries;

// Control text of the checkpoint's chat template. Occurrences inside message strings are
// literal input and never count as control text.
inline constexpr std::string_view kSentenceStart = "<｜start▁of▁sentence｜>";
inline constexpr std::string_view kSentenceEnd   = "<｜end▁of▁sentence｜>";
inline constexpr std::string_view kSystemRole    = "<|System|>";
inline constexpr std::string_view kUserRole      = "<|User|>";
inline constexpr std::string_view kBotRole       = "<|Bot|>";
inline constexpr std::string_view kToolRole      = "<|Tool|>";
inline constexpr std::string_view kThinkOpen     = "<think>";
inline constexpr std::string_view kThinkClose    = "</think>";
inline constexpr std::string_view kToolCallOpen  = "<tool_call>";
inline constexpr std::string_view kToolCallClose = "</tool_call>";
inline constexpr std::string_view kResponseOpen  = "<tool_response>";
inline constexpr std::string_view kResponseClose = "</tool_response>";
inline constexpr std::string_view kToolsOpen     = "<tools>";
inline constexpr std::string_view kToolsClose    = "</tools>";
static_assert(kSentenceStart.size() == 29 && kSentenceEnd.size() == 27);

// Byte coordinates refer to the tagged template input, before trimming or serialization.
struct TextSource {
    std::uint32_t tag = 0;
    text::ByteSpan bytes;
};

struct MessageSources {
    std::optional<TextSource> content;
    // Indexed by the original message's parts.
    std::vector<std::optional<TextSource>> parts;
    TextSource reasoning;
    std::vector<std::uint32_t> tool_calls;
};

// One role section: sentence start, role marker, body and (except the last) sentence end.
struct Block {
    ChatRole role = ChatRole::System;
    text::ByteSpan bytes;
    // First byte after the role marker.
    std::size_t content = 0;
};

[[nodiscard]] std::vector<Block> scan_blocks(const text::TemplateOutput& output);

// First template-owned occurrence of `marker` in [begin, end), or npos.
[[nodiscard]] std::size_t find_control(const text::TemplateOutput& output, std::string_view marker,
                                       std::size_t begin = 0, std::size_t end = std::string::npos);

struct RenderedChat {
    text::TemplateOutput output;
    std::vector<MessageSources> messages;
    std::vector<std::uint32_t> tools;
    bool starts_in_reasoning = false;
    std::string continuation_content;
};

class CompiledChatTemplate {
public:
    explicit CompiledChatTemplate(models::frontend::ChatTemplate compiled)
        : compiled_(std::move(compiled)) {}

    // New turns start after the generation prefix, in reasoning when thinking is enabled.
    // Continuation resumes the final assistant's text after its completed reasoning block.
    // All returned text, source mappings and literal spans are owned.
    [[nodiscard]] RenderedChat render(std::span<const ChatMessage> messages,
                                      const PromptOptions& options      = {},
                                      const PreparationControl& control = {}) const;

    [[nodiscard]] RenderedChat render(std::span<const ChatMessage> messages,
                                      const PromptOptions& options,
                                      std::span<const PromptCacheMarker> markers,
                                      PromptBoundaries& boundaries,
                                      const PreparationControl& control = {}) const;

private:
    RenderedChat render_impl(std::span<const ChatMessage> messages, const PromptOptions& options,
                             std::span<const PromptCacheMarker> markers,
                             PromptBoundaries* boundaries, const PreparationControl& control) const;
    models::frontend::ChatTemplate compiled_;
};

} // namespace ninfer::models::spark2_5
