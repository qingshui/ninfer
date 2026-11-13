#pragma once

#include "models/common/frontend/resources.h"

namespace ninfer::artifact {
class Binder;
}

namespace ninfer::models::spark2_5 {

struct FrontendResources {
    models::frontend::TextResources text;
    SamplingPreset sampling;
};

// Raw views borrow Definition resource storage; parsed tokenizer and defaults own their values.
void parse_resources(FrontendResources& resources, std::uint32_t embedding_rows);
FrontendResources bind_resources(artifact::Binder& binder, std::uint32_t embedding_rows);

} // namespace ninfer::models::spark2_5
