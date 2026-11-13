#include "models/spark2_5/frontend/frontend.h"

#include "models/common/frontend/media_cache.h"
#include "models/spark2_5/definition.h"
#include "models/spark2_5/frontend/chat_template.h"
#include "models/spark2_5/frontend/prepared_prompt.h"
#include "models/spark2_5/frontend/prompt_layout.h"
#include "models/spark2_5/frontend/resources.h"
#include "runtime/contract/request.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace ninfer::models::spark2_5 {
namespace {
using Clock = std::chrono::steady_clock;

std::uint32_t checked_count(std::size_t count) {
    if (!count || count > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("prepared prompt requires a nonempty uint32 token count");
    return static_cast<std::uint32_t>(count);
}

void require_text_only(std::span<const ChatMessage> messages) {
    for (const auto& message : messages)
        for (const auto& part : message.parts)
            if (part.kind == MessagePartKind::Media)
                throw std::invalid_argument("Spark2_5 does not support media input");
}

void sort_unique(std::vector<std::uint32_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

void encode_prompt(PreparedPromptData& output, const text::Tokenizer& tokenizer,
                   const RenderedChat& chat, const PromptBoundaries& boundaries,
                   const ContextCacheHints& hints, std::uint32_t max_context) {
    std::vector<std::size_t> offsets = boundaries.semantic;
    offsets.insert(offsets.end(), boundaries.structural.begin(), boundaries.structural.end());
    offsets.insert(offsets.end(), boundaries.execution.begin(), boundaries.execution.end());
    for (const auto marker : boundaries.markers)
        if (marker) offsets.push_back(*marker);
    if (boundaries.rewrite) offsets.push_back(boundaries.rewrite->frontier);
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    const auto start                = Clock::now();
    auto encoded                    = tokenizer.encode_with_boundaries(chat.output.text, offsets,
                                                                       {.max_tokens = std::size_t(max_context) + 1},
                                                                       chat.output.literal_spans);
    output.prepare.tokenize_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (encoded.input_ids.size() > max_context)
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            runtime::context_length_error_message(encoded.input_ids.size(), max_context, true));
    (void)checked_count(encoded.input_ids.size());
    const auto frontier = [&](std::size_t offset) -> std::optional<std::uint32_t> {
        const auto i = std::lower_bound(offsets.begin(), offsets.end(), offset) - offsets.begin();
        const auto value = encoded.boundaries.at(i).exact_frontier;
        if (!value || !*value) return {};
        return static_cast<std::uint32_t>(*value);
    };
    const auto convert = [&](const auto& source) {
        std::vector<std::uint32_t> values;
        for (const auto offset : source)
            if (const auto value = frontier(offset)) values.push_back(*value);
        sort_unique(values);
        return values;
    };
    std::vector<std::optional<std::uint32_t>> markers;
    for (const auto marker : boundaries.markers)
        markers.push_back(marker ? frontier(*marker) : std::nullopt);
    output.context_cache = models::frontend::prepare_context_cache(
        hints, markers, convert(boundaries.semantic), convert(boundaries.structural));
    output.identity.execution_frontiers = convert(boundaries.execution);
    if (boundaries.rewrite) {
        if (const auto value = frontier(boundaries.rewrite->frontier))
            output.identity.rewrite_checkpoint = models::frontend::RewriteCheckpointSpec{
                .kind = boundaries.rewrite->kind, .frontier = *value, .recovery_frontier = *value};
    }
    output.token_ids = std::move(encoded.input_ids);
}
} // namespace

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, models::frontend::FrontendOptions options)
        : chat(models::frontend::load_chat_template(resources.text, options.chat_template_path)),
          tokenizer(resources.text.tokenizer), sampling(resources.sampling),
          output(resources.text, options.grammar_cache_bytes), max_context(options.max_context) {
        if (!tokenizer)
            throw std::invalid_argument("Spark2_5 frontend requires parsed text resources");
        if (!max_context) throw std::invalid_argument("frontend max_context must be nonzero");
        // Boundaries are reported in tokens, so every structural marker must be one token.
        for (const auto marker :
             {kSentenceStart, kSentenceEnd, kSystemRole, kUserRole, kBotRole, kToolRole, kThinkOpen,
              kThinkClose, kToolCallOpen, kToolCallClose, kResponseOpen, kResponseClose}) {
            const auto ids = tokenizer->encode(marker);
            if (ids.size() != 1 || tokenizer->decode(ids) != marker)
                throw std::invalid_argument(
                    "Spark2_5 template marker must be one tokenizer token: " + std::string(marker));
        }
    }

    CompiledChatTemplate chat;
    std::shared_ptr<const text::Tokenizer> tokenizer;
    SamplingPreset sampling;
    glm5next::OutputResources output;
    std::uint32_t max_context;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

PromptSummary PreparedPrompt::summary() const {
    const auto& data = PreparedPromptAccess::view(*this);
    return {.starts_in_reasoning = data.output.starts_in_reasoning,
            .prompt_tokens       = checked_count(data.token_ids.size()),
            .has_media           = false};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    return data_ ? data_->prepare.public_stats() : PromptPreparationStats{};
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (!prompt.data_) throw std::invalid_argument("prepared prompt is empty");
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (!prompt.data_) throw std::invalid_argument("prepared prompt is empty");
    auto data = std::move(prompt.data_);
    return std::move(*data);
}

Frontend make_frontend(const FrontendResources& resources,
                       models::frontend::FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, std::move(options)));
}

Frontend create_frontend(const ModelDefinition& definition, const EngineOptions& o) {
    return make_frontend(definition.resources(),
                         {.chat_template_path       = o.chat_template_path,
                          .grammar_cache_bytes      = o.grammar_cache_bytes,
                          .vision_enabled           = false,
                          .max_context              = o.max_context,
                          .media_cache_bytes        = o.media_cache_bytes,
                          .media_live_bytes         = o.media_live_bytes,
                          .media_preprocess_threads = o.media_preprocess_threads});
}

std::shared_ptr<const void> Frontend::identity() const noexcept { return impl_; }

ModelSamplingDefaults Frontend::sampling_defaults() const noexcept {
    return {.thinking = impl_->sampling, .non_thinking = impl_->sampling};
}

MediaCacheSummary Frontend::media_cache_summary() const { return {}; }

SamplingPreset Frontend::sampling_for(const PreparedPrompt& prompt) const {
    return PreparedPromptAccess::view(prompt).sampling;
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    const auto start = Clock::now();
    models::frontend::check_preparation_control(control);
    models::frontend::validate_context_cache_hints(input.context_cache, input.messages.size());
    require_text_only(input.messages);
    PromptBoundaries boundaries;
    auto rendered = impl_->chat.render(input.messages, input.options, input.context_cache.markers,
                                       boundaries, control);
    auto data     = std::make_unique<PreparedPromptData>();
    encode_prompt(*data, *impl_->tokenizer, rendered, boundaries, input.context_cache,
                  impl_->max_context);
    models::frontend::check_preparation_control(control, "Spark2_5 tokenization");
    data->output.starts_in_reasoning  = rendered.starts_in_reasoning;
    data->output.continuation_content = std::move(rendered.continuation_content);
    data->output.tools    = glm5next::build_tool_call_output_contract(input.options.tool_jsons);
    data->sampling        = impl_->sampling;
    data->prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(data));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    models::frontend::check_preparation_control(control);
    require_text_only(input.messages);
    const auto rendered = impl_->chat.render(input.messages, input.options, control);
    const auto encoded  = impl_->tokenizer->encode_with_boundaries(rendered.output.text, {}, {},
                                                                   rendered.output.literal_spans);
    models::frontend::check_preparation_control(control, "Spark2_5 tokenization");
    return checked_count(encoded.input_ids.size());
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> tokens,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    if (tokens.size() > impl_->max_context)
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            runtime::context_length_error_message(tokens.size(), impl_->max_context));
    (void)checked_count(tokens.size());
    for (const auto token : tokens)
        if (!impl_->tokenizer->is_valid_token(token))
            throw std::invalid_argument("prompt token is outside the checkpoint vocabulary");
    auto data                                = std::make_unique<PreparedPromptData>();
    data->token_ids                          = std::move(tokens);
    data->identity.reusable                  = allow_prefix_identity;
    data->context_cache.update_session_index = false;
    data->sampling                           = impl_->sampling;
    data->prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(data));
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    return impl_->tokenizer->encode(text);
}

models::frontend::OutputSession
Frontend::make_output_session(const PreparedPrompt& prompt, const StopPolicy& stop,
                              const OutputOptions& output, const ThinkingControlOptions& thinking,
                              const std::optional<OutputConstraint>& constraint,
                              const ToolChoice& choice) const {
    return glm5next::create_output_session(impl_->output, PreparedPromptAccess::view(prompt).output,
                                           stop, output, thinking, constraint, choice);
}

const StopPolicy& Frontend::default_stop_policy() const noexcept {
    return impl_->output.text.defaults;
}

} // namespace ninfer::models::spark2_5
