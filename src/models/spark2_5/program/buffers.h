#pragma once

#include "core/layout.h"
#include "models/spark2_5/config.h"
#include "ninfer/types.h"

#include <optional>

namespace ninfer::models::spark2_5 {

// Rank-local persistent tensors of one traversal. Prefill uses the first `chunk` columns of the
// token axis; a decode round of B rows uses the first B. Everything between `ingress.offset` and
// the end of the last metadata tensor is mirrored in pinned memory and uploaded as one block.
struct BufferLayout {
    TensorRegion ids, positions;
    TensorRegion valid_columns, kv_rows, destination_offsets;
    TensorRegion sampling_positions, count_slots, count_columns;
    std::optional<TensorRegion> sampling;
    LayoutRegion ingress;
    TensorRegion hidden;
    std::optional<TensorRegion> logits, sampled_ids, token_counts, grammar_masks;
    std::uint32_t max_positions = 0;
};

[[nodiscard]] BufferLayout plan_buffers(LayoutBuilder&, const TextConfig&, const EngineOptions&,
                                        std::uint32_t chunk, std::uint32_t public_tokens);

} // namespace ninfer::models::spark2_5
