#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
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

ApiError internal_error(const std::exception& exception) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = exception.what();
    return error;
}

// One merged stream of one sample group. Branch workers render their own encoder's SSE events
// and publish them in arrival order; the transport thread drains the batches, so it stays the
// only writer to the client. A failed branch stops its siblings through `abort`; `abort` is the
// group's own signal and never marks the client gone, so the failure that stopped the group can
// still reach the client as an error event. `settled` reaches the branch count only after every
// branch outcome or failure lands.
struct SampleMerge {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::vector<std::string>> pending;
    std::vector<std::optional<GenerationOutcome>> outcomes;
    std::exception_ptr failure;
    std::size_t settled = 0;
    std::atomic<bool> abort{false};
};

constexpr auto kMergePollInterval = std::chrono::milliseconds(250);

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
    // One serve-owned timeline per HTTP request, marked at each boundary it reaches; the first
    // mark of a boundary wins and unreachable boundaries stay null.
    const auto timeline = std::make_shared<RequestTimeline>();
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
        if (observation.first_token) {
            observation.first_token = [timeline,
                                       inner = std::move(observation.first_token)](
                                          const ninfer::GenerationFirstTokenObservation& first) {
                timeline->mark_first_token();
                inner(first);
            };
        }
        const auto is_cancelled = [&req] { return client_disconnected(req); };
        const GenerationConsumerMode consumer_mode =
            request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate;
        // A single-sample request keeps the ordinary submission; an n>1 request submits one
        // Engine sample group so every branch executes in parallel behind a shared prompt.
        if (request.sample_count == 1) {
            prepared.push_back(
                service_->prepare(request.generation, consumer_mode, observation, is_cancelled));
        } else {
            prepared = service_->prepare_samples(request.generation, request.sample_count,
                                                 consumer_mode, observation, is_cancelled);
        }
        for (PreparedRequest& sample : prepared) { sample.timeline = timeline; }
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata,
            internal_error(exception)));
        write_openai_error(res, internal_error(exception));
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    metadata.response_id                      = identity.id;
    auto lifecycle                            = begin_request(make_request_log_context(
        req_id, "openai_chat_completions", request.generation, metadata, prepared.front()));

    // The sample group gains its lanes together; waiting on each branch in turn establishes
    // admission before the response commits. Each successful wait marks the shared timeline.
    std::vector<GenerationOutcome> outcomes;
    try {
        for (PreparedRequest& sample : prepared) {
            service_->wait_for_admission(sample, [&req] { return client_disconnected(req); });
        }
        if (!request.stream) {
            // The group already executes in parallel; each branch's result is consumed in turn.
            outcomes.reserve(prepared.size());
            for (PreparedRequest& sample : prepared) {
                outcomes.push_back(
                    service_->run(sample, nullptr, [&req] { return client_disconnected(req); }));
            }
        }
    } catch (const ApiException& exception) {
        lifecycle->failure(make_generation_request_failure(exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        write_openai_error(res, internal_error(exception));
        return;
    }

    if (!request.stream) {
        timeline->mark_completed();
        // The whole JSON body is the response's single flush; the render boundary is the last
        // serve-owned point before cpp-httplib writes the response.
        bool rendered = false;
        std::string render_failure;
        std::string body;
        try {
            body     = make_chat_completion_response(identity, outcomes);
            rendered = true;
        } catch (const std::exception& exception) {
            render_failure = exception.what();
        }
        if (rendered) {
            timeline->mark_http_first_flush();
            timeline->mark_http_done();
        }
        lifecycle->done(aggregate_outcomes(outcomes));
        if (!rendered) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, render_failure));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = render_failure;
            write_openai_error(res, error);
            return;
        }
        try {
            set_owned_json_content(res, std::move(body), prepared.front().lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            write_openai_error(res, internal_error(exception));
        }
        return;
    }

    try {
        const bool return_progress     = request.return_progress;
        const bool timings_per_token   = request.timings_per_token;
        const bool include_usage       = request.include_usage;
        const std::size_t sample_count = static_cast<std::size_t>(request.sample_count);
        auto stream = std::make_shared<ChatGenerationStream>(std::move(prepared));
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
            [this, identity, stream, encoders, lifecycle, timeline, return_progress,
             timings_per_token, include_usage,
             sample_count](std::size_t, httplib::DataSink& sink) -> bool {
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
                // One branch worker per sample: renders its own encoder's SSE events and
                // publishes them to the merge queue; the transport thread writes them to the
                // client in arrival order. A closed socket marks the shared cancelled flag; a
                // failed branch stops its siblings through the merge's own abort flag.
                auto merge = std::make_shared<SampleMerge>();
                merge->outcomes.resize(sample_count);
                const auto publish = [merge](std::vector<std::string> events) {
                    if (events.empty()) { return; }
                    std::lock_guard lock(merge->mutex);
                    merge->pending.push_back(std::move(events));
                    merge->cv.notify_one();
                };
                const auto fail_branch = [merge](std::exception_ptr error) {
                    std::lock_guard lock(merge->mutex);
                    if (!merge->failure) { merge->failure = std::move(error); }
                    ++merge->settled;
                    merge->abort.store(true, std::memory_order_release);
                    merge->cv.notify_one();
                };
                const auto run_branch = [this, stream, encoders, merge, publish, fail_branch,
                                         return_progress, timings_per_token](std::size_t index) {
                    const auto render = [](auto render_events) -> std::vector<std::string> {
                        try {
                            return render_events();
                        } catch (const ResponseRenderFailure&) {
                            throw;
                        } catch (const std::exception& exception) {
                            throw ResponseRenderFailure(exception.what());
                        }
                    };
                    try {
                        OpenAIChatStream& encoder = *encoders[index];
                        StreamSink output;
                        output.on_start = [&](const ninfer::GenerationStart& start) {
                            encoder.note_start(start);
                            if (return_progress) {
                                publish(render([&] {
                                    return std::vector<std::string>{
                                        encoder.initial_prompt_progress()};
                                }));
                            }
                        };
                        if (return_progress) {
                            output.on_progress = [&](const ninfer::PromptProgress& progress) {
                                publish(render([&] {
                                    return std::vector<std::string>{
                                        encoder.prompt_progress(progress)};
                                }));
                            };
                        }
                        if (timings_per_token) {
                            output.on_timing =
                                [&](const ninfer::GenerationTimingObservation& timing) {
                                    encoder.note_timing(timing);
                                };
                        }
                        output.on_content = [&](const std::string& text) {
                            publish(render([&] {
                                return std::vector<std::string>{encoder.content_delta(text)};
                            }));
                        };
                        output.on_reasoning = [&](const std::string& text) {
                            publish(render([&] {
                                return std::vector<std::string>{encoder.reasoning_delta(text)};
                            }));
                        };
                        output.on_tool_start = [&](const ninfer::ToolCallStart& event) {
                            publish(render([&] { return encoder.tool_start(event); }));
                        };
                        output.on_tool_arguments = [&](const ninfer::ToolArgumentsDelta& event) {
                            publish(render([&] { return encoder.tool_arguments(event); }));
                        };
                        output.on_tool_end = [&](const ninfer::ToolCallEnd& event) {
                            publish(render([&] { return encoder.tool_end(event); }));
                        };
                        output.is_cancelled = [&] {
                            return stream->cancelled.load(std::memory_order_acquire) ||
                                   merge->abort.load(std::memory_order_acquire);
                        };
                        GenerationOutcome outcome =
                            service_->run(stream->prepared[index], &output);
                        const std::vector<std::string> finish =
                            render([&] { return encoder.finish_sample(outcome); });
                        {
                            std::lock_guard lock(merge->mutex);
                            merge->outcomes[index].emplace(std::move(outcome));
                            if (!finish.empty()) {
                                merge->pending.push_back(std::move(finish));
                            }
                            ++merge->settled;
                            merge->cv.notify_one();
                        }
                    } catch (...) {
                        fail_branch(std::current_exception());
                    }
                };

                std::vector<GenerationOutcome> outcomes;
                try {
                    for (const std::shared_ptr<OpenAIChatStream>& encoder : encoders) {
                        render_and_write(transport, [&] { return encoder->start(); });
                    }
                    // The start chunks are the response's first HTTP flush.
                    timeline->mark_http_first_flush();
                    std::vector<std::thread> branches;
                    branches.reserve(sample_count);
                    for (std::size_t index = 0; index < sample_count; ++index) {
                        branches.emplace_back(run_branch, index);
                    }
                    const auto join_branches = [&branches] {
                        for (std::thread& branch : branches) {
                            if (branch.joinable()) { branch.join(); }
                        }
                    };
                    try {
                        for (;;) {
                            std::vector<std::vector<std::string>> batches;
                            std::exception_ptr failure;
                            bool settled_all = false;
                            {
                                std::unique_lock lock(merge->mutex);
                                merge->cv.wait_for(lock, kMergePollInterval, [&] {
                                    return !merge->pending.empty() ||
                                           merge->settled == sample_count;
                                });
                                while (!merge->pending.empty()) {
                                    batches.push_back(std::move(merge->pending.front()));
                                    merge->pending.pop_front();
                                }
                                failure     = merge->failure;
                                settled_all = merge->settled == sample_count;
                            }
                            for (const std::vector<std::string>& batch : batches) {
                                transport.write(batch);
                            }
                            if (settled_all) {
                                join_branches();
                                if (failure) { std::rethrow_exception(failure); }
                                break;
                            }
                            // Heartbeat the quiet stream and observe a closed socket while
                            // the branches generate. A failed probe marks the shared
                            // cancelled flag; the branches settle cancelled through their
                            // own wait loops.
                            if (transport.poll()) { continue; }
                        }
                    } catch (...) {
                        // Whatever leaves the merge loop (a failed branch, a closed socket)
                        // stops the remaining branches before it propagates; only a closed
                        // socket marks the client gone.
                        merge->abort.store(true, std::memory_order_release);
                        join_branches();
                        throw;
                    }
                    outcomes.reserve(sample_count);
                    for (std::size_t index = 0; index < sample_count; ++index) {
                        outcomes.push_back(std::move(*merge->outcomes[index]));
                    }
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(internal_error(exception));
                } catch (const ApiException& exception) {
                    lifecycle->failure(make_generation_request_failure(exception.error()));
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    return send_error(internal_error(exception));
                }

                timeline->mark_completed();
                // Aggregate request state (usage, [DONE]) rides one terminator chunk after
                // every branch's finish events, with the same semantics as a single sample.
                // Only a client that is gone omits it: the transport refuses the write then.
                std::vector<std::string> terminal;
                try {
                    terminal = make_chat_stream_sample_terminator(identity, outcomes,
                                                                  include_usage);
                } catch (const std::exception& exception) {
                    lifecycle->done(aggregate_outcomes(outcomes));
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(internal_error(exception));
                }
                try {
                    transport.write(terminal);
                } catch (const ClientDisconnected&) {
                    lifecycle->done(aggregate_outcomes(outcomes));
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
                timeline->mark_http_done();
                lifecycle->done(aggregate_outcomes(outcomes));
                sink.done();
                return true;
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
        write_openai_error(res, internal_error(exception));
    }
}

} // namespace ninfer::serve
