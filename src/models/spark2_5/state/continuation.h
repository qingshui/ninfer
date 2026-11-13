#pragma once

#include "runtime/contract/placement.h"

#include <cstdint>

namespace ninfer::models::spark2_5 {

// What a committed prefix records beyond its StateImage and full-layer KV history. The only
// placement is the single rank, but the Context contract names one for every record.
struct ContinuationMetadata {
    std::uint32_t main_frontier = 0;
    runtime::PlacementId placement;
};

} // namespace ninfer::models::spark2_5
