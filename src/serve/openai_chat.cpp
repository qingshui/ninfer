#include "serve/openai_chat.h"

// Multi-sample fan-out of one OpenAI Chat Completions request: one prepared request per
// requested sample, each consumed exactly once. Samples render the same prompt and run in
// sample order; each sample after the first is prepared only after the previous sample's
// generation completed, so it adopts the previous sample's published prompt source instead of
// the Engine prefilling the same prompt again.

#include "serve/generation_service.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::serve {

ChatGenerationStream::ChatGenerationStream(std::vector<PreparedRequest> requests,
                                           GenerationRequest generation)
    : prepared(std::move(requests)), generation(std::move(generation)) {}

PreparedRequest
prepare_chat_sample(const GenerationService& service, const GenerationRequest& generation,
                    int sample_count, int index, GenerationConsumerMode consumer_mode,
                    const ninfer::GenerationObservationOptions& observation,
                    std::function<bool()> is_cancelled) {
    if (sample_count < 1 || index < 0 || index >= sample_count) {
        throw std::logic_error("sample index outside the request's sample count");
    }
    // The first sample (and a single-sample request) publishes its final context as a
    // continuation source; a later sample publishes unless it is the request's last, so a
    // finished request leaves no retained source behind.
    return service.prepare_sample(generation, sample_count, index, consumer_mode, observation,
                                  std::move(is_cancelled), {},
                                  index == 0 || index + 1 < sample_count);
}

std::vector<GenerationOutcome>
run_sequential_samples(GenerationService& service, ChatGenerationStream& stream,
                       int sample_count,
                       const ninfer::GenerationObservationOptions& observation,
                       const std::function<bool()>& is_cancelled) {
    std::vector<GenerationOutcome> outcomes;
    outcomes.reserve(static_cast<std::size_t>(sample_count));
    for (int index = 0; index < sample_count; ++index) {
        if (static_cast<int>(stream.prepared.size()) <= index) {
            // The previous sample completed; prepare this one now so it adopts the previous
            // sample's published prompt source.
            stream.prepared.push_back(
                prepare_chat_sample(service, stream.generation, sample_count, index,
                                    GenerationConsumerMode::Aggregate, observation,
                                    is_cancelled));
        }
        outcomes.push_back(service.run(stream.prepared[static_cast<std::size_t>(index)], nullptr,
                                       is_cancelled));
    }
    return outcomes;
}

} // namespace ninfer::serve
