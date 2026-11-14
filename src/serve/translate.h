#pragma once

// Adapter between the wire-independent protocol request and the public engine API.

#include "ninfer/types.h"
#include "serve/request.h"
#include "product/config/options.h"

#include <functional>

namespace ninfer::serve {

// Media acquisition is a product-layer concern. Translation preserves part order
// and asks the caller to turn each wire source into owning bytes before the
// target frontend sees it.
using MediaAcquirer = std::function<ninfer::OwnedMedia(const ContentPart&)>;

struct PromptOptionInputs {
    std::optional<bool> enable_thinking;
    std::optional<ninfer::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
    PromptDefaults defaults;
};

PromptOptionInputs collect_prompt_options(const GenerationRequest& req,
                                          const product::ServeOptions& server);

ninfer::PromptInput to_prompt_input(const GenerationRequest& req,
                                    const PromptOptionInputs& semantics,
                                    const MediaAcquirer& acquire_media);

// Build public request options (output budget, thinking, stop policy, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// Explicit request fields override startup defaults, including temperature. Prefix-reuse
// participation is resolved by GenerationService and supplied explicitly because
// operational requests do not inherit the external-traffic policy. publish_continuation
// separately controls whether the request's final context is retained as a source later
// requests may adopt; a request may adopt a source without publishing its own result.
ninfer::RequestOptions to_request_options(const GenerationRequest& req,
                                          const product::ServeOptions& server,
                                          bool allow_prefix_reuse,
                                          bool publish_prefix_continuation = true);

} // namespace ninfer::serve
