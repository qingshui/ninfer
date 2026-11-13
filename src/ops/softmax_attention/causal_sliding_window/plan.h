#pragma once

#include "ninfer/ops/sliding_window_attention.h"

namespace ninfer::ops::detail {

// Query rows pack the four query heads of one kv head with each token.
enum class CausalSlidingWindowInstance { Rows16, Rows32, Rows64 };

struct CausalSlidingWindowPlan {
    CausalSlidingWindowInstance instance;
    std::int32_t tokens;
    std::int32_t batch;
    std::int32_t query_rows; // MMA rows per CTA
    std::int32_t splits;     // 1 stores BF16 directly; larger values publish partials
};

[[nodiscard]] CausalSlidingWindowPlan
select_causal_sliding_window_plan(std::int32_t tokens, std::int32_t batch,
                                  std::int32_t multiprocessor_count);

} // namespace ninfer::ops::detail
