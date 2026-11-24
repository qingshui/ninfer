#pragma once

#include "core/dtype.h"
#include "core/layout.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/workspace.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Varlen recurrence: one launch sequence covers a whole row table. A row is cut into
// sub-segments at its traversal boundaries; every sub-segment is evaluated exactly as the
// distinct-state gated_delta_net would evaluate a call of that length (recurrent route below the
// chunked threshold, chunks aligned to the sub-segment start otherwise).
namespace ninfer::ops::detail::gated_delta_net::varlen {

inline constexpr int kMaxRows       = 1024;
inline constexpr int kMaxBoundaries = 1024;

// One sub-segment of the packed token axis, in (row, column) order.
struct SubSegment {
    std::int32_t start;      // global first token
    std::int32_t length;     // tokens, >= 1
    std::int32_t chunk_base; // first chunk id; zero chunks for the recurrent route
    std::int32_t row;
    std::int64_t checkpoint; // checkpoint record byte offset after this sub-segment, or -1
};
static_assert(sizeof(SubSegment) == 24);

struct PlanHeader {
    std::int32_t total_chunks;
    std::int32_t padding[3];
};
static_assert(sizeof(PlanHeader) == 16);

struct WorkspaceLayout {
    LayoutRegion plan;
    TensorRegion qk;
    TensorRegion control;
    std::size_t total_bytes;
    std::int32_t chunk_capacity; // chunk ids reserved; 0 when no chunk can exist
    std::int32_t sub_segments;   // R + C
};

// Smallest sub-segment length evaluated by the chunked route. Derived from the single-segment
// route selector so the two paths can not disagree.
int chunked_min_tokens(int qk_heads, int value_heads);

// Upper bound on chunk ids for any call with at most this many tokens/rows/boundaries.
WorkspaceLayout workspace_layout(int qk_heads, int value_heads, int rows, int tokens,
                                 int boundaries, int chunked_threshold);

struct Operands {
    const __nv_bfloat16 *q, *k, *v;
    const float *g, *beta;
    __nv_bfloat16* out;
    const std::int32_t* token_offsets;
    const std::byte* source_base;    // source storage + field offset
    const std::int64_t* source_offsets;
    std::byte* destination_base;     // destination storage + field offset
    const std::int64_t* destination_offsets;
    std::byte* checkpoint_base;      // destination storage + checkpoint field offset
    const std::int32_t* boundary_rows;
    const std::int32_t* boundary_columns;
    const std::int64_t* boundary_offsets;
    int qk_heads, value_heads;
    int tokens, rows, boundaries;
    int chunked_threshold;
    float scale;
    DType state_dtype; // FP32 or BF16 words of source, destination and checkpoint records
};

struct Workspace {
    PlanHeader* header;
    SubSegment* sub_segments;
    std::int32_t* row_first;
    chunked::QkChunk* qk;
    chunked::ControlChunk* control;
    int chunk_capacity;
};

void launch(const Operands& operands, const Workspace& workspace, bool normalize_qk,
            int value_tile, cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net::varlen
