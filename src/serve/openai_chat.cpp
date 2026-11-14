#include "serve/openai_chat.h"

// Multi-sample fan-out of one OpenAI Chat Completions request: every requested sample is
// prepared up front when the request is admitted and consumed exactly once; all samples render
// the same prompt, so the Engine's prefix reuse and context cache share their prefill.

#include "serve/generation_service.h"

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::serve {

ChatGenerationStream::ChatGenerationStream(std::vector<PreparedRequest> requests)
    : prepared(std::move(requests)) {}

std::vector<PreparedRequest>
prepare_chat_samples(const GenerationService& service, const GenerationRequest& generation,
                     int sample_count, GenerationConsumerMode consumer_mode,
                     const ninfer::GenerationObservationOptions& observation,
                     std::function<bool()> is_cancelled) {
    std::vector<PreparedRequest> prepared;
    prepared.reserve(static_cast<std::size_t>(sample_count));
    for (int index = 0; index < sample_count; ++index) {
        prepared.push_back(service.prepare_sample(generation, sample_count, index, consumer_mode,
                                                  observation, is_cancelled));
    }
    return prepared;
}

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
        // Thread creation failed: stop and settle already-started samples before leaving, so no
        // worker outlives this call.
        stop.store(true, std::memory_order_release);
        for (std::thread& worker : workers) { worker.join(); }
        throw;
    }
    for (std::thread& worker : workers) { worker.join(); }
    if (failure) { std::rethrow_exception(failure); }
    return outcomes;
}

} // namespace ninfer::serve
