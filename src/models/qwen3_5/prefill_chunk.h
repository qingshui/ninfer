#pragma once

#include "models/common/startup.h"
#include "models/qwen3_5/planning.h"
#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5 {

// Default prefill chunks. This header is the only place that decides which chunk a placement runs
// with when the caller does not set one; the sequence planner accepts only a resolved value.
inline constexpr std::uint32_t kResidentPrefillChunk    = 2048;
inline constexpr std::uint32_t kHostExpertsPrefillChunk = 4096;

// One packed prefill execution carries at most this many chunks of tokens in total, summed over
// its sequence segments. With the resident chunk (2048) one flight carries up to 16384 tokens.
// The Session reports the resulting capacity through Session::prefill_pack_tokens(); the Engine
// derives its Throughput allowance from that and holds no copy of this value.
inline constexpr std::uint32_t kPrefillPackChunks = 8;

// Upper bound on the number of sequence segments one packed prefill execution carries. It is the
// only authority for that bound and is decoupled from the lane count: a flight merges whichever
// waiting segments fit the token budget, whether few lanes each carry a long window or many lanes
// each carry a short one. Masked draft backends and causal scoring execute one segment at a time
// and keep their single-segment constraint.
inline constexpr std::uint32_t kPrefillPackSegmentLimit = 64;

// Explicit EngineOptions::prefill_chunk wins over the placement default. The result is clamped
// to max_context and must be a multiple of kPrefillChunkAlignment; a max_context that is not
// itself aligned keeps the largest aligned chunk that fits (a chunk is batching granularity,
// never a context bound, so rounding down preserves every contract).
[[nodiscard]] inline std::uint32_t resolve_prefill_chunk(const EngineOptions& options,
                                                         WeightPlacement placement) {
    const auto base  = options.prefill_chunk != 0                  ? options.prefill_chunk
                       : placement == WeightPlacement::HostExperts ? kHostExpertsPrefillChunk
                                                                   : kResidentPrefillChunk;
    const auto room = options.max_context / kPrefillChunkAlignment * kPrefillChunkAlignment;
    const auto chunk = std::min(base, room);
    if (chunk == 0)
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of " +
                                    std::to_string(kPrefillChunkAlignment));
    return chunk;
}

// Upper bound of the summed segment length of one packed prefill execution. It is the only
// authority for that bound: planning sizes the token-axis workspace and the execution frame from
// it, and the Session admits segments against it. `token_capacity` is the token-axis capacity of
// the KV/GDN state pool, derived by the sequence planning from the pool backing; the result is a
// multiple of kPrefillChunkAlignment, at least `prefill_chunk` (so a single window always fits)
// and never above the aligned capacity.
[[nodiscard]] inline std::uint32_t resolve_prefill_pack_tokens(std::uint32_t prefill_chunk,
                                                               std::uint32_t token_capacity) {
    if (prefill_chunk == 0 || prefill_chunk % kPrefillChunkAlignment != 0)
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of " +
                                    std::to_string(kPrefillChunkAlignment));
    const std::uint64_t aligned_capacity =
        std::uint64_t{token_capacity} / kPrefillChunkAlignment * kPrefillChunkAlignment;
    const std::uint64_t packed = std::uint64_t{prefill_chunk} * kPrefillPackChunks;
    return std::max(prefill_chunk, static_cast<std::uint32_t>(std::min(packed, aligned_capacity)));
}

// Segment counts of the prebuilt prefill launch plans. A plan covers a flight of exactly S
// segments that each carry one full `prefill_chunk` window, so its token extent is S * chunk (the
// 2048/4096/8192/16384 widths of the resident chunk). S runs over the powers of two whose extent
// stays within `pack_tokens` and whose segments fit the `lanes` of the resolved concurrency.
// Planning sizes the plans' staging State rows from the largest S and the program builds one
// plan per entry; no other place derives this set.
[[nodiscard]] inline std::vector<std::uint32_t>
prefill_plan_segment_counts(std::uint32_t prefill_chunk, std::uint32_t pack_tokens,
                            std::uint32_t lanes) {
    std::vector<std::uint32_t> counts;
    if (prefill_chunk == 0) return counts;
    for (std::uint64_t segments = 1;
         segments <= lanes && segments * prefill_chunk <= pack_tokens; segments <<= 1)
        counts.push_back(static_cast<std::uint32_t>(segments));
    return counts;
}

// Upper bound of the token axis of one mixed execution: a packed prefill (at most `pack_tokens`)
// followed by the verify block of up to `mixed_decode_rows` decode rows, each `draft_window + 1`
// columns wide. Zero decode rows means the model does not mix and the result is zero. Planning
// sizes the mixed workspace and the execution frame from it, and the Program admits work against
// it; no other place derives this extent.
[[nodiscard]] inline std::uint32_t resolve_mixed_tokens(std::uint32_t pack_tokens,
                                                        std::uint32_t mixed_decode_rows,
                                                        std::uint32_t draft_window) {
    if (mixed_decode_rows == 0) return 0;
    if (pack_tokens == 0 || pack_tokens % kPrefillChunkAlignment != 0)
        throw std::invalid_argument("pack_tokens must be a nonzero multiple of " +
                                    std::to_string(kPrefillChunkAlignment));
    const std::uint64_t tokens =
        std::uint64_t{pack_tokens} + std::uint64_t{mixed_decode_rows} * (draft_window + 1ULL);
    if (tokens > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("mixed token axis exceeds the Tensor domain");
    return static_cast<std::uint32_t>(tokens);
}

} // namespace ninfer::models::qwen3_5
