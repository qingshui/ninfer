#include "models/spark2_5/state/identity.h"

#include <algorithm>

namespace ninfer::models::spark2_5 {

InputIdentity::InputIdentity(const PreparedPromptData& p, std::uint32_t domain)
    : tokens(p.token_ids), execution_frontiers(p.identity.execution_frontiers),
      digests(glm5next::InputPrefixView{p.token_ids, {}, p.identity.execution_frontiers}),
      tag(domain) {}

void InputIdentity::reserve(std::size_t n) {
    tokens.reserve(n);
    digests.reserve(n);
}

void InputIdentity::prepare_generated(std::size_t count, std::optional<std::uint32_t> split) {
    if (split && (!*split || *split > count))
        throw std::invalid_argument("Spark2_5 generated boundary exceeds input");
    reserve(tokens.size() + count);
    if (split) execution_frontiers.reserve(execution_frontiers.size() + 1);
}

void InputIdentity::append_generated(std::span<const TokenId> ids,
                                     std::optional<std::uint32_t> split) {
    prepare_generated(ids.size(), split);
    const auto begin = tokens.size();
    digests.append_generated(ids, split);
    tokens.insert(tokens.end(), ids.begin(), ids.end());
    if (split) execution_frontiers.push_back(begin + *split);
}

void InputIdentity::truncate(std::uint32_t frontier) {
    tokens.resize(frontier);
    digests.truncate(frontier);
    std::erase_if(execution_frontiers, [&](auto f) { return f > frontier; });
}

} // namespace ninfer::models::spark2_5
