#pragma once

#include "models/glm5next/prefix_identity.h"
#include "models/spark2_5/frontend/prepared_prompt.h"

namespace ninfer::models::spark2_5 {

// The ordered input of a sequence: prompt tokens followed by what it generated. The model has no
// media, so identity is tokens and the required execution frontiers. The shortlist digests are
// the shared prefix-digest scheme; exact token equality still decides reuse.
struct InputIdentity {
    std::vector<TokenId> tokens;
    std::vector<std::uint32_t> execution_frontiers;
    glm5next::PrefixDigests digests;
    std::uint32_t tag = 0;

    InputIdentity() = default;
    InputIdentity(const PreparedPromptData&, std::uint32_t tag);

    [[nodiscard]] glm5next::InputPrefixView view() const noexcept {
        return {tokens, {}, execution_frontiers};
    }

    void reserve(std::size_t count);
    void prepare_generated(std::size_t count, std::optional<std::uint32_t> split);
    void append_generated(std::span<const TokenId>, std::optional<std::uint32_t> split);
    void truncate(std::uint32_t frontier);
};

} // namespace ninfer::models::spark2_5
