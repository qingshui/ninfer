#pragma once

#include "models/spark2_5/execution/workspace.h"

#include <memory>
#include <vector>

namespace ninfer::models::spark2_5 {

// Facts the Program and Session derive from the definition alone, before any weight is placed.
// The prepared weights carry shapes and policies for workspace sizing; they own no device
// storage and borrow the definition, which the profile keeps alive.
struct ModelProfile {
    std::shared_ptr<const ModelDefinition> definition;
    std::uint32_t vocabulary          = 0;
    std::uint32_t public_tokens       = 0;
    std::uint32_t control_token_limit = 0;
    std::vector<BlockProfile> planning_layers;
    ops::LinearProfile planning_head;
};

[[nodiscard]] std::shared_ptr<const ModelProfile>
describe_model(std::shared_ptr<const ModelDefinition> definition);

// Parameters plus the profile; the Engine constructs it from the materialized shard.
class RankParameters : public Parameters {
public:
    RankParameters(const ModelShard& shard, std::shared_ptr<const ModelProfile> p)
        : Parameters(shard), profile(std::move(p)) {}

    const std::shared_ptr<const ModelProfile> profile;
};

} // namespace ninfer::models::spark2_5
