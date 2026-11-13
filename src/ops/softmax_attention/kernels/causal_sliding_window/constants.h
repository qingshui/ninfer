#pragma once

namespace ninfer::ops::detail {

inline constexpr int kCausalSlidingWindow          = 512;
inline constexpr int kCausalSlidingWindowMaxSplits = 8;
inline constexpr int kCausalSlidingWindowKvHeads   = 2;
inline constexpr int kCausalSlidingWindowGroup     = 4;
inline constexpr int kCausalSlidingWindowMaxBatch  = 16;
// B > 1 serves speculative verification and decode: at most this many query columns.
inline constexpr int kCausalSlidingWindowMaxBatchedTokens = 16;

} // namespace ninfer::ops::detail
