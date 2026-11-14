#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

// One Chat Completions request fans out to one prepared request per requested sample. This owner
// keeps the prepared samples and their transport flags alive across the chunked-content provider.
// The first sample is prepared before the response starts; later samples are prepared while the
// response streams, each only after the previous sample's generation completed, so every sample
// adopts the previous sample's published prompt source instead of the Engine prefilling the same
// prompt again. `generation` keeps the request that renders that shared prompt.
struct ChatGenerationStream {
    ChatGenerationStream(std::vector<PreparedRequest> requests, GenerationRequest generation)
        : prepared(std::move(requests)), generation(std::move(generation)) {}

    std::vector<PreparedRequest> prepared;
    GenerationRequest generation;
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
        // The first sample is prepared up front; later samples of one request are prepared
        // only after the previous sample's generation completed, so each adopts the previous
        // sample's published prompt source instead of the Engine prefilling the same prompt
        // once per sample.
        prepared.push_back(service_->prepare_sample(
            request.generation, request.sample_count, 0,
            request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate,
            observation, [&req] { return client_disconnected(req); }));
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
            outcomes.reserve(static_cast<std::size_t>(request.sample_count));
            for (int index = 0; index < request.sample_count; ++index) {
                if (static_cast<int>(prepared.size()) <= index) {
                    // The previous sample completed; prepare this one now so it adopts the
                    // previous sample's published prompt source. The last sample publishes
                    // nothing: no later sample adopts its result, so the request leaves no
                    // retained source behind.
                    prepared.push_back(service_->prepare_sample(
                        request.generation, request.sample_count, index,
                        GenerationConsumerMode::Aggregate, observation,
                        [&req] { return client_disconnected(req); }, {},
                        index + 1 < request.sample_count));
                }
                outcomes.push_back(
                    service_->run(prepared.back(), nullptr,
                                  [&req] { return client_disconnected(req); }));
            }
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
        const bool return_progress    = request.return_progress;
        const bool timings_per_token  = request.timings_per_token;
        const bool include_usage      = request.include_usage;
        // The first sample is already prepared; the provider prepares each later sample
        // only after the previous sample's generation completed.
        const std::size_t sample_count = static_cast<std::size_t>(request.sample_count);
        auto stream = std::make_shared<ChatGenerationStream>(std::move(prepared),
                                                             std::move(request.generation));
        std::vector<std::shared_ptr<OpenAIChatStream>> encoders;
        encoders.reserve(sample_count);
        for (std::size_t index = 0; index < sample_count; ++index) {
            encoders.push_back(std::make_shared<OpenAIChatStream>(
                identity, include_usage, timings_per_token, return_progress,
                static_cast<int>(index)));
        }

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, identity, stream, encoders, lifecycle, return_progress, timings_per_token,
             include_usage, observation, sample_count](std::size_t, httplib::DataSink& sink) -> bool {
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

                std::vector<GenerationOutcome> outcomes;
                bool cancelled = false;
                try {
                    outcomes.reserve(encoders.size());
                    for (std::size_t index = 0; index < encoders.size(); ++index) {
                        if (stream->prepared.size() <= index) {
                            // The previous sample's generation completed; prepare this sample
                            // now so it adopts the previous sample's published prompt source
                            // instead of the Engine prefilling the same prompt again. The
                            // last sample publishes nothing: no later sample adopts its
                            // result, so the request leaves no retained source behind.
                            const auto sample_index = static_cast<int>(stream->prepared.size());
                            stream->prepared.push_back(service_->prepare_sample(
                                stream->generation, static_cast<int>(sample_count), sample_index,
                                GenerationConsumerMode::Streaming, observation,
                                [stream] { return stream->cancelled.load(std::memory_order_acquire); },
                                {}, sample_index < static_cast<int>(sample_count) - 1));
                        }
                        OpenAIChatStream& encoder = *encoders[index];
                        StreamSink output;
                        output.on_start = [&](const ninfer::GenerationStart& start) {
                            encoder.note_start(start);
                            if (return_progress) {
                                render_and_write(
                                    transport, [&] { return encoder.initial_prompt_progress(); });
                            }
                        };
                        if (return_progress) {
                            output.on_progress = [&](const ninfer::PromptProgress& progress) {
                                render_and_write(transport,
                                                 [&] { return encoder.prompt_progress(progress); });
                            };
                        }
                        if (timings_per_token) {
                            output.on_timing = [&](const ninfer::GenerationTimingObservation& timing) {
                                encoder.note_timing(timing);
                            };
                        }
                        output.on_content = [&](const std::string& text) {
                            render_and_write(transport,
                                             [&] { return encoder.content_delta(text); });
                        };
                        output.on_reasoning = [&](const std::string& text) {
                            render_and_write(transport,
                                             [&] { return encoder.reasoning_delta(text); });
                        };
                        output.on_tool_start = [&](const ninfer::ToolCallStart& event) {
                            render_and_write(transport, [&] { return encoder.tool_start(event); });
                        };
                        output.on_tool_arguments = [&](const ninfer::ToolArgumentsDelta& event) {
                            render_and_write(transport,
                                             [&] { return encoder.tool_arguments(event); });
                        };
                        output.on_tool_end = [&](const ninfer::ToolCallEnd& event) {
                            render_and_write(transport, [&] { return encoder.tool_end(event); });
                        };
                        output.is_cancelled = [&] { return transport.poll(); };

                        outcomes.push_back(service_->run(stream->prepared[index], &output));
                        cancelled |=
                            outcomes.back().finish_reason == ninfer::FinishReason::Cancelled;
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
