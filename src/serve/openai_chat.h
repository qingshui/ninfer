#pragma once

// OpenAI Chat Completions wire adapter. Parsing produces one protocol envelope plus an executable
// GenerationRequest; response builders consume protocol-neutral GenerationOutcome values.

#include "serve/generation_service.h"
#include "serve/request.h"
#include "serve/request_json.h"
#include "serve/tool_stream_state.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

struct OpenAIChatRequest {
    std::string model;
    GenerationRequest generation;
    // OpenAI `n`: independently sampled sequences fan out from this one request.
    int sample_count           = 1;
    bool stream                = false;
    bool include_usage         = false;
    bool output_tokens_explicit = false;
    // llama.cpp-compatible response observations. Final timings remain unconditional;
    // timings_per_token controls only cumulative timing snapshots on streamed output chunks.
    bool timings_per_token = false;
    bool return_progress   = false;
};

OpenAIChatRequest parse_chat_completion_request(const RequestJson& body,
                                                const RequestLimits& limits);

struct OpenAIChatResponseIdentity {
    std::string id;
    std::string model;
    std::int64_t created = 0;
};

OpenAIChatResponseIdentity make_openai_chat_response_identity(std::string model);
// One choice per outcome, indexed by sample order. Usage reports the single shared prompt and
// the sum of every sample's completion tokens.
std::string make_chat_completion_response(const OpenAIChatResponseIdentity& identity,
                                          const std::vector<GenerationOutcome>& outcomes);
// Single-sample form; identical to the multi-sample form with exactly one outcome.
std::string make_chat_completion_response(const OpenAIChatResponseIdentity& identity,
                                          const GenerationOutcome& outcome);

// Multi-sample fan-out of one Chat Completions request. One prepared request per requested
// sample, each consumed exactly once. This owner keeps the prepared samples, the request that
// renders their shared prompt, and their transport flags alive across the chunked-content
// provider.
struct ChatGenerationStream {
    ChatGenerationStream(std::vector<PreparedRequest> requests, GenerationRequest generation);

    std::vector<PreparedRequest> prepared;
    GenerationRequest generation;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};
};

// Prepares sample `index` of a request that asked for `sample_count` samples. Every sample
// renders the same prompt; a sample is prepared only after the previous sample's generation
// completed, so it adopts the previous sample's published prompt source instead of the Engine
// prefilling the same prompt again. The first sample (and a single-sample request) publishes
// its final context as a continuation source; a later sample publishes unless it is the
// request's last, so a finished request leaves no retained source behind.
[[nodiscard]] PreparedRequest
prepare_chat_sample(const GenerationService& service, const GenerationRequest& generation,
                    int sample_count, int index, GenerationConsumerMode consumer_mode,
                    const ninfer::GenerationObservationOptions& observation,
                    std::function<bool()> is_cancelled);

// Runs the samples of one non-streaming request in sample order, preparing each sample only
// after the previous sample's generation completed so it adopts the previous sample's published
// prompt source, and returns one outcome per sample in sample order. `stream` carries the
// already prepared first sample and the request that renders the shared prompt.
[[nodiscard]] std::vector<GenerationOutcome>
run_sequential_samples(GenerationService& service, ChatGenerationStream& stream,
                       int sample_count, const ninfer::GenerationObservationOptions& observation,
                       const std::function<bool()>& is_cancelled);

class OpenAIChatStream {
public:
    OpenAIChatStream(OpenAIChatResponseIdentity identity, bool include_usage,
                     bool timings_per_token = false, bool return_progress = false,
                     int sample_index = 0);

    std::string start();
    void note_start(const ninfer::GenerationStart& start);
    std::string initial_prompt_progress();
    std::string prompt_progress(const ninfer::PromptProgress& progress);
    void note_timing(const ninfer::GenerationTimingObservation& timing);
    std::string reasoning_delta(const std::string& text);
    std::string content_delta(const std::string& text);
    std::vector<std::string> tool_start(const ninfer::ToolCallStart& event);
    std::vector<std::string> tool_arguments(const ninfer::ToolArgumentsDelta& event);
    std::vector<std::string> tool_end(const ninfer::ToolCallEnd& event);
    // Terminal events for one choice of a multi-sample request: buffered suffixes plus this
    // choice's finish chunk. Aggregate usage and the [DONE] sentinel are emitted separately by
    // make_chat_stream_sample_terminator.
    std::vector<std::string> finish_sample(const GenerationOutcome& outcome);
    // Complete terminal events for a single-sample request.
    std::vector<std::string> finish(const GenerationOutcome& outcome);

private:
    nlohmann::json live_timings_json() const;

    OpenAIChatResponseIdentity identity_;
    ToolStreamState tools_;
    std::string reasoning_;
    std::string content_;
    std::optional<ninfer::GenerationTimingObservation> live_timing_;
    std::uint32_t prompt_tokens_            = 0;
    std::uint32_t cached_tokens_            = 0;
    std::uint32_t last_progress_tokens_     = 0;
    std::uint64_t last_progress_elapsed_ns_ = 0;
    int sample_index_                       = 0;
    bool include_usage_                     = false;
    bool timings_per_token_                 = false;
    bool return_progress_                   = false;
    bool started_                           = false;
    bool admitted_                          = false;
    bool progress_started_                  = false;
    bool content_started_                   = false;
    bool finished_                          = false;
};

// Aggregate terminal events for a multi-sample streaming request: one usage chunk built from
// every sample (only when include_usage; prompt tokens are the single shared prompt, completion
// tokens are the sum) followed by the [DONE] sentinel. Timings and constraint state report the
// first sample's terminal accounting.
std::vector<std::string>
make_chat_stream_sample_terminator(const OpenAIChatResponseIdentity& identity,
                                   const std::vector<GenerationOutcome>& outcomes,
                                   bool include_usage);

} // namespace ninfer::serve
