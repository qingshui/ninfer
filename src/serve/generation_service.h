#pragma once

// Product-side adapter from one protocol-neutral generation request to the public Engine. Wire
// adapters normalize before this layer and render IDs, usage, and response events after it.

#include "ninfer/engine.h"
#include "product/config/options.h"
#include "serve/request.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct RequestLifetime;
struct RequestCapacity;

// Wall-clock boundaries of one HTTP generation request, in unix epoch milliseconds. The serve
// layer marks each boundary at most once (the first mark wins); a boundary that never occurs
// stays null. The branch threads of one sample group mark concurrently, so access is
// serialized. The Engine never participates in this timeline.
class RequestTimeline {
public:
    struct Snapshot {
        std::optional<std::uint64_t> admitted_at_ms, first_token_at_ms, completed_at_ms,
            http_first_flush_at_ms, http_done_at_ms;
    };

    void mark_admitted();
    void mark_first_token();
    void mark_completed();
    void mark_http_first_flush();
    void mark_http_done();
    [[nodiscard]] Snapshot take() const;

private:
    void mark(std::optional<std::uint64_t>& boundary);

    mutable std::mutex mutex_;
    std::optional<std::uint64_t> admitted_, first_token_, completed_, http_first_flush_,
        http_done_;
};

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    std::vector<ninfer::TokenId> generated_token_ids;
    std::vector<ninfer::GeneratedToolCall> tool_calls;
    ninfer::ToolCallParseDiagnostics tool_call_parse;
    int prompt_tokens                  = 0;
    int completion_tokens              = 0;
    int reasoning_tokens               = 0;
    ninfer::FinishReason finish_reason = ninfer::FinishReason::OutputLimit;
    std::optional<std::string> matched_stop_string;
    RequestStats stats;
    double service_prepare_seconds = 0.0;
    // Serve-owned measurement timeline shared by every branch of the request's sample group.
    std::shared_ptr<RequestTimeline> timeline;
};

// A transport may report an interrupted stream through a callback or is_cancelled().
// Generation still settles through Engine cancellation and returns its actual work statistics.
class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

struct StreamSink {
    std::function<void(const ninfer::GenerationStart& start)> on_start;
    std::function<void(const ninfer::PromptProgress& progress)> on_progress;
    std::function<void(const ninfer::GenerationTimingObservation& timing)> on_timing;
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<void(const ninfer::ToolCallStart&)> on_tool_start;
    std::function<void(const ninfer::ToolArgumentsDelta&)> on_tool_arguments;
    std::function<void(const ninfer::ToolCallEnd&)> on_tool_end;
    std::function<bool()> is_cancelled;
};

enum class GenerationConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract.
ApiError
request_error_to_api_error(const ninfer::RequestError& exception,
                           std::string_view constraint_param = "structured_outputs.grammar",
                           std::span<const std::string> tool_schema_params = {});

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    std::string constraint_param;
    std::vector<std::string> tool_schema_params;
    ninfer::GenerationHandle generation;
    ninfer::ResolvedSamplingParameters sampling;
    // Service input acquisition/bookkeeping; Engine timings own prompt and constraint preparation.
    double service_prepare_seconds = 0.0;
    double acquisition_seconds     = 0.0;
    PromptPreparationStats preparation;
    int prompt_tokens    = 0;
    bool enable_thinking = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ninfer::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::shared_ptr<RequestLifetime> lifetime;
    // Shared by every branch of one sample group; null for requests prepared without one.
    std::shared_ptr<RequestTimeline> timeline;

    // Copies the per-request facts shared by every branch of one sample group: the move-only
    // generation handle and its resolved sampling stay unset and are installed per branch.
    [[nodiscard]] PreparedRequest clone_branch_base() const;
};

class GenerationService {
public:
    explicit GenerationService(product::ServeOptions options,
                               StartupObserver startup_observer = {});

    [[nodiscard]] const product::ServeOptions& options() const noexcept { return options_; }

    // Engine owns the once-normalized startup configuration. Serving diagnostics must use this
    // value instead of reinterpreting optional defaults from product::ServeOptions.
    [[nodiscard]] const ninfer::EngineOptions& engine_options() const { return engine_->options(); }

    [[nodiscard]] const EngineDescription& description() const { return engine_->description(); }

    [[nodiscard]] const StartupReport& startup_report() const { return engine_->startup_report(); }

    [[nodiscard]] EngineSnapshot snapshot(SnapshotOptions options = {}) const {
        return engine_->snapshot(options);
    }

    [[nodiscard]] bool is_available() const { return engine_->is_available(); }

    [[nodiscard]] ninfer::ModelSamplingDefaults sampling_defaults() const {
        return engine_->sampling_defaults();
    }

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          GenerationConsumerMode consumer_mode,
                                          ninfer::GenerationObservationOptions observation = {},
                                          std::function<bool()> is_cancelled               = {},
                                          ContextCacheHints context_cache = {}) const;
    // Prepares the request's prompt once and submits `n` independently sampled branches as one
    // Engine sample group: the group occupies its lanes atomically, a forking Session prefills
    // the shared prefix once and forks one child sequence per branch, and branch i derives its
    // sampling seed from (the request seed, i). Returns one prepared request per branch, in
    // branch order; every branch shares the request's lifetime and preparation facts. `n` of
    // one prepares an ordinary single submission.
    [[nodiscard]] std::vector<PreparedRequest>
    prepare_samples(const GenerationRequest& req, int n, GenerationConsumerMode consumer_mode,
                    ninfer::GenerationObservationOptions observation = {},
                    std::function<bool()> is_cancelled               = {},
                    ContextCacheHints context_cache                  = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;

    // Establishes admission before an HTTP adapter commits its streaming response.
    void wait_for_admission(PreparedRequest& prepared,
                            std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    void warmup();

    void request_stop() noexcept { engine_->request_stop(); }

private:
    [[nodiscard]] std::function<bool()>
    preparation_cancellation(std::function<bool()> external) const;

    enum class CacheParticipation : std::uint8_t {
        Disabled,
        ReadWrite,
    };

    enum class DeadlinePolicy : std::uint8_t {
        ClientPendingTimeout,
        UnboundedStartup,
    };

    [[nodiscard]] std::vector<PreparedRequest>
    prepare_impl(const GenerationRequest& req, int sample_count,
                 GenerationConsumerMode consumer_mode,
                 ninfer::GenerationObservationOptions observation,
                 std::function<bool()> is_cancelled, ContextCacheHints context_cache,
                 CacheParticipation cache_participation, DeadlinePolicy deadline_policy,
                 bool publish_prefix_continuation) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_request_lifetime(DeadlinePolicy deadline_policy) const;

    product::ServeOptions options_;
    std::unique_ptr<ninfer::Engine> engine_;
    std::shared_ptr<RequestCapacity> request_capacity_;
};

} // namespace ninfer::serve
