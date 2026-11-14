#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

// One Chat Completions request fans out to one prepared request per requested sample. Every
// sample is prepared up front and consumed exactly once; all samples render the same prompt, so
// the Engine's prefix reuse and context cache share their prefill. This owner keeps the
// prepared samples and their transport flags alive across the chunked-content provider.
struct ChatGenerationStream {
    explicit ChatGenerationStream(std::vector<PreparedRequest> requests)
        : prepared(std::move(requests)) {}

    std::vector<PreparedRequest> prepared;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};
};

// Request-level accounting for one HTTP request: the first sample's content with every sample's
// completion tokens. Measurement logs stay one record per HTTP request.
GenerationOutcome aggregate_outcomes(const std::vector<GenerationOutcome>& outcomes) {
    if (outcomes.empty()) { throw std::logic_error("chat request has no sample outcomes"); }
    GenerationOutcome aggregate = outcomes.front();
    for (std::size_t index = 1; index < outcomes.size(); ++index) {
        aggregate.completion_tokens += outcomes[index].completion_tokens;
        aggregate.reasoning_tokens += outcomes[index].reasoning_tokens;
    }
    return aggregate;
}

// Runs every prepared sample of one request in parallel, one thread per sample, and returns one
// outcome per sample in sample order. A failing sample signals its siblings to stop; the
// first-originating failure is rethrown only after every sample settled, so cancellation errors
// raised by the stop signal never mask the failure that caused it.
std::vector<GenerationOutcome>
run_parallel_samples(GenerationService& service, std::vector<PreparedRequest>& prepared,
                     const std::function<bool()>& is_cancelled) {
    std::vector<GenerationOutcome> outcomes(prepared.size());
    std::vector<std::thread> workers;
    workers.reserve(prepared.size());
    std::atomic<bool> stop{false};
    std::atomic<bool> failure_recorded{false};
    std::exception_ptr failure;
    const auto run_sample = [&](std::size_t index) {
        try {
            outcomes[index] = service.run(
                prepared[index], nullptr,
                [&] {
                    return stop.load(std::memory_order_acquire) ||
                           (is_cancelled && is_cancelled());
                });
        } catch (...) {
            if (!failure_recorded.exchange(true, std::memory_order_acq_rel)) {
                failure = std::current_exception();
            }
            stop.store(true, std::memory_order_release);
        }
    };
    try {
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            workers.emplace_back(run_sample, index);
        }
    } catch (...) {
        stop.store(true, std::memory_order_release);
        for (std::thread& worker : workers) { worker.join(); }
        throw;
    }
    for (std::thread& worker : workers) { worker.join(); }
    if (failure) { std::rethrow_exception(failure); }
    return outcomes;
}

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.launch.generation.max_tokens;
        request                   = parse_chat_completion_request(parse_json_body(req), limits);
        validate_openai_model(request.model, public_model_id_);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    RequestLogMetadata metadata{.http_request_id        = res.get_header_value("x-request-id"),
                                .model                  = request.model,
                                .stream                 = request.stream,
                                .output_tokens_explicit = request.output_tokens_explicit};
    observations_.preparing(req_id, "openai_chat_completions", metadata);
    std::vector<PreparedRequest> prepared;
    ninfer::GenerationObservationOptions observation;
    try {
        observation = ninfer::GenerationObservationOptions{
            .constraint_timings = true,
            .live_timings       = request.stream && request.timings_per_token,
            .prompt_progress    = request.stream && request.return_progress,
            .scheduling         = scheduling_observer(req_id, metadata.http_request_id),
            .first_token        = first_token_observer(req_id),
        };
        // All samples of one request are prepared up front and run in parallel: every sample
        // renders the same prompt, so the Engine's prefix reuse and context cache share their
        // prefill instead of the request chaining one sample's published source into the next.
        prepared.reserve(static_cast<std::size_t>(request.sample_count));
        for (int index = 0; index < request.sample_count; ++index) {
            prepared.push_back(service_->prepare_sample(
                request.generation, request.sample_count, index,
                request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate,
                observation, [&req] { return client_disconnected(req); }));
        }
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    metadata.response_id                      = identity.id;
    auto lifecycle                            = begin_request(make_request_log_context(
        req_id, "openai_chat_completions", request.generation, metadata, prepared.front()));

    std::vector<GenerationOutcome> outcomes;
    try {
        if (request.stream) {
            for (PreparedRequest& sample : prepared) {
                service_->wait_for_admission(sample,
                                             [&req] { return client_disconnected(req); });
            }
        } else {
            outcomes = run_parallel_samples(*service_, prepared,
                                            [&req] { return client_disconnected(req); });
        }
    } catch (const ApiException& exception) {
        lifecycle->failure(make_generation_request_failure(exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }

    if (!request.stream) {
        lifecycle->done(aggregate_outcomes(outcomes));
        try {
            set_owned_json_content(res, make_chat_completion_response(identity, outcomes),
                                   prepared.front().lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
        }
        return;
    }

    try {
        const bool return_progress   = request.return_progress;
        const bool timings_per_token = request.timings_per_token;
        const bool include_usage     = request.include_usage;
        // Every sample is already prepared; the provider runs them in parallel and interleaves
        // their SSE chunks by choice index.
        auto stream = std::make_shared<ChatGenerationStream>(std::move(prepared));
        std::vector<std::shared_ptr<OpenAIChatStream>> encoders;
        encoders.reserve(stream->prepared.size());
        for (std::size_t index = 0; index < stream->prepared.size(); ++index) {
            encoders.push_back(std::make_shared<OpenAIChatStream>(
                identity, include_usage, timings_per_token, return_progress,
                static_cast<int>(index)));
        }

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, identity, stream, encoders, lifecycle, return_progress, timings_per_token,
             include_usage](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        render_and_write(transport, [&] { return sse_error_event(error); });
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, exception.what()));
                        return false;
                    }
                };
                try {
                    for (const std::shared_ptr<OpenAIChatStream>& encoder : encoders) {
                        render_and_write(transport, [&] { return encoder->start(); });
                    }
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                // One worker per sample consumes its prepared request; the shared transport
                // mutex keeps every SSE chunk atomic, so chunks of different choices interleave
                // by arrival while each choice keeps its own event order. A failing sample
                // signals its siblings to stop; the first-originating failure is rethrown after
                // every sample settled.
                const std::size_t sample_count = encoders.size();
                std::vector<GenerationOutcome> outcomes(sample_count);
                std::mutex write_mutex;
                std::atomic<bool> stop{false};
                std::atomic<bool> failure_recorded{false};
                std::exception_ptr failure;
                std::vector<std::thread> workers;
                workers.reserve(sample_count);
                const auto run_sample = [&](std::size_t index) {
                    try {
                        OpenAIChatStream& encoder = *encoders[index];
                        StreamSink output;
                        output.on_start = [&](const ninfer::GenerationStart& start) {
                            const std::lock_guard lock(write_mutex);
                            encoder.note_start(start);
                            if (return_progress) {
                                render_and_write(
                                    transport, [&] { return encoder.initial_prompt_progress(); });
                            }
                        };
                        if (return_progress) {
                            output.on_progress = [&](const ninfer::PromptProgress& progress) {
                                const std::lock_guard lock(write_mutex);
                                render_and_write(transport,
                                                 [&] { return encoder.prompt_progress(progress); });
                            };
                        }
                        if (timings_per_token) {
                            output.on_timing = [&](const ninfer::GenerationTimingObservation& timing) {
                                const std::lock_guard lock(write_mutex);
                                encoder.note_timing(timing);
                            };
                        }
                        output.on_content = [&](const std::string& text) {
                            const std::lock_guard lock(write_mutex);
                            render_and_write(transport,
                                             [&] { return encoder.content_delta(text); });
                        };
                        output.on_reasoning = [&](const std::string& text) {
                            const std::lock_guard lock(write_mutex);
                            render_and_write(transport,
                                             [&] { return encoder.reasoning_delta(text); });
                        };
                        output.on_tool_start = [&](const ninfer::ToolCallStart& event) {
                            const std::lock_guard lock(write_mutex);
                            render_and_write(transport, [&] { return encoder.tool_start(event); });
                        };
                        output.on_tool_arguments = [&](const ninfer::ToolArgumentsDelta& event) {
                            const std::lock_guard lock(write_mutex);
                            render_and_write(transport,
                                             [&] { return encoder.tool_arguments(event); });
                        };
                        output.on_tool_end = [&](const ninfer::ToolCallEnd& event) {
                            const std::lock_guard lock(write_mutex);
                            render_and_write(transport, [&] { return encoder.tool_end(event); });
                        };
                        output.is_cancelled = [&] {
                            const std::lock_guard lock(write_mutex);
                            return stop.load(std::memory_order_acquire) || transport.poll();
                        };
                        outcomes[index] = service_->run(stream->prepared[index], &output);
                    } catch (...) {
                        if (!failure_recorded.exchange(true, std::memory_order_acq_rel)) {
                            failure = std::current_exception();
                        }
                        stop.store(true, std::memory_order_release);
                    }
                };
                try {
                    try {
                        for (std::size_t index = 0; index < sample_count; ++index) {
                            workers.emplace_back(run_sample, index);
                        }
                    } catch (...) {
                        // Thread creation failed: stop and settle already-started samples before
                        // leaving, so no worker outlives this provider invocation.
                        stop.store(true, std::memory_order_release);
                        for (std::thread& worker : workers) { worker.join(); }
                        throw;
                    }
                    for (std::thread& worker : workers) { worker.join(); }
                    if (failure) { std::rethrow_exception(failure); }
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    lifecycle->failure(make_generation_request_failure(exception.error()));
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                bool cancelled = false;
                for (const GenerationOutcome& outcome : outcomes) {
                    cancelled |= outcome.finish_reason == ninfer::FinishReason::Cancelled;
                }
                lifecycle->done(aggregate_outcomes(outcomes));
                if (cancelled || stream->cancelled.load(std::memory_order_acquire)) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
                std::vector<std::string> terminal;
                try {
                    for (std::size_t index = 0; index < encoders.size(); ++index) {
                        const std::vector<std::string> events =
                            encoders[index]->finish_sample(outcomes[index]);
                        terminal.insert(terminal.end(), events.begin(), events.end());
                    }
                    const std::vector<std::string> tail =
                        make_chat_stream_sample_terminator(identity, outcomes, include_usage);
                    terminal.insert(terminal.end(), tail.begin(), tail.end());
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }
                try {
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace ninfer::serve
