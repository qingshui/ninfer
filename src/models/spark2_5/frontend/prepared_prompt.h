#pragma once

#include "models/glm5next/frontend/output.h"
#include "models/glm5next/prefix_identity.h"
#include "models/spark2_5/frontend/frontend.h"

namespace ninfer::models::spark2_5 {

struct PromptIdentity {
    bool reusable = true;
    std::optional<models::frontend::RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> execution_frontiers;
};

struct PreparedPromptData : models::frontend::PromptData {
    PromptIdentity identity;
    // The tool-call framing is shared with GLM; reasoning markers are the same <think> pair.
    glm5next::PreparedOutput output;
    SamplingPreset sampling;

    [[nodiscard]] glm5next::InputPrefixView prefix_view() const noexcept {
        return {token_ids, {}, identity.execution_frontiers};
    }
};

class PreparedPromptAccess {
public:
    static const PreparedPromptData& view(const PreparedPrompt& prompt);
    static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::models::spark2_5
