#include "serve/openai_chat.h"

// Multi-sample fan-out of one OpenAI Chat Completions request: one prepared request per
// requested sample, submitted to the Engine as one sample group so every branch executes in
// parallel. The request's prompt is prepared once; the group occupies its lanes atomically and
// the Engine derives each branch's sampling seed from (the request seed, branch index).

#include "serve/generation_service.h"

#include <utility>
#include <vector>

namespace ninfer::serve {

ChatGenerationStream::ChatGenerationStream(std::vector<PreparedRequest> requests)
    : prepared(std::move(requests)) {}

} // namespace ninfer::serve
