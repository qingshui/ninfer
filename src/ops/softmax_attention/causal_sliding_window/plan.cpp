#include "ops/softmax_attention/causal_sliding_window/plan.h"

#include "ops/softmax_attention/kernels/causal_sliding_window/constants.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {

CausalSlidingWindowPlan select_causal_sliding_window_plan(std::int32_t tokens, std::int32_t batch,
                                                          std::int32_t multiprocessor_count) {
    if (tokens < 1 || batch < 1 || batch > kCausalSlidingWindowMaxBatch ||
        (batch > 1 && tokens > kCausalSlidingWindowMaxBatchedTokens) || multiprocessor_count < 1) {
        throw std::invalid_argument(
            "causal_sliding_window_attention plan: invalid B, T or SM count");
    }
    constexpr int kGroup           = kCausalSlidingWindowGroup;
    const int rows                 = tokens * kGroup;
    const auto instance            = rows <= 16   ? CausalSlidingWindowInstance::Rows16
                                     : rows <= 32 ? CausalSlidingWindowInstance::Rows32
                                                  : CausalSlidingWindowInstance::Rows64;
    const int query_rows           = instance == CausalSlidingWindowInstance::Rows16   ? 16
                                     : instance == CausalSlidingWindowInstance::Rows32 ? 32
                                                                                       : 64;
    const std::int64_t tiles       = (rows + query_rows - 1) / query_rows;
    const std::int64_t independent = tiles * kCausalSlidingWindowKvHeads * batch;
    // Two CTAs per SM fill the machine; the window bounds the useful slice count.
    const std::int64_t wanted = std::max<std::int64_t>(1, 2 * multiprocessor_count / independent);
    return {
        instance, tokens, batch, query_rows,
        static_cast<std::int32_t>(std::min<std::int64_t>(wanted, kCausalSlidingWindowMaxSplits))};
}

} // namespace ninfer::ops::detail
