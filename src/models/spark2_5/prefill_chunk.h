#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::models::spark2_5 {

// Prefill windows are multiples of this; the attention Ops tile the token axis by it.
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;
inline constexpr std::uint32_t kDefaultPrefillChunk   = 4096;

// An explicit EngineOptions::prefill_chunk wins over the model default. The result is clamped to
// max_context, so a window never exceeds what a request can hold.
[[nodiscard]] inline std::uint32_t resolve_prefill_chunk(const EngineOptions& options) {
    const auto base = options.prefill_chunk != 0 ? options.prefill_chunk : kDefaultPrefillChunk;
    if (base % kPrefillChunkAlignment != 0)
        throw std::invalid_argument("prefill_chunk must be a multiple of " +
                                    std::to_string(kPrefillChunkAlignment));
    return std::min(base, options.max_context);
}

} // namespace ninfer::models::spark2_5
