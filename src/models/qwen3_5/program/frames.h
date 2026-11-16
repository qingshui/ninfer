#pragma once

#include "core/buffer.h"
#include "core/layout.h"
#include "core/memory.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::models::qwen3_5 {

struct FrameSpec {
    std::int32_t hidden         = 0;
    std::int32_t output_rows    = 0;
    std::int32_t public_tokens  = 0;
    std::uint32_t prefill_chunk = 0;
    // Summed token extent and segment count of one packed prefill execution: pack_tokens >=
    // prefill_chunk, pack_segments <= kPrefillPackSegmentLimit (a budget decoupled from
    // batch_capacity; the lanes a flight merges stay bounded by batch_capacity at admission).
    // Per-sequence consumers stay at one chunk.
    std::uint32_t pack_tokens   = 0;
    std::uint32_t pack_segments = 1;
    // A mixed execution appends the verify block of up to mixed_decode_rows decode rows to a
    // packed prefill on the same token axis; mixed_tokens (0 when the frame does not mix) is the
    // resolved extent of that axis, see resolve_mixed_tokens(). The decode rows are bounded by
    // the lane count and are independent of pack_segments. The ingress row metadata of such a
    // frame holds the decode rows and the prefill segments side by side, see
    // FrameSpec::ingress_rows().
    std::uint32_t mixed_decode_rows   = 0;
    std::uint32_t mixed_tokens        = 0;
    std::uint32_t batch_capacity      = 1;
    std::uint32_t draft_window        = 0;
    SpeculativeBackend backend        = SpeculativeBackend::None;
    bool causal_scoring               = false;
    bool vision                       = false;
    std::uint32_t control_token_limit = 0;

    // Compact rows of the ingress arrays: the lanes of a decode execution, and for a mixed
    // execution the decode rows plus the prefill segments.
    [[nodiscard]] std::uint32_t ingress_rows() const noexcept {
        return std::max(batch_capacity, pack_segments + mixed_decode_rows);
    }
};

// Byte offsets of the lane tables of one frame region, resolved for its concurrency. One
// layout serves the device region and the pinned host mirror of the same tables, so both bind
// every member at the same offset.
struct FrameIngressLayout {
    std::uint32_t lanes = 0;
    // Capacity of the varlen boundary tables (execution splits of one packed execution).
    std::uint32_t boundaries = 0;
    std::size_t anchors = 0, frontiers = 0, backend_frontiers = 0, extents = 0,
        valid_columns = 0, proposal_valid_columns = 0, budgets = 0, main_table_rows = 0,
        backend_table_rows = 0, state_source_offsets = 0, state_destination_offsets = 0,
        residency_slots = 0, rope_deltas = 0, sampling = 0, sampling_positions = 0,
        history_skip = 0, previous_rope = 0, decode_positions = 0, decode_rope = 0,
        step_positions = 0, step_valid = 0, step_rope = 0;
    // Segment table of a packed or mixed prefill: token_offsets of its rows (lanes + 1 entries)
    // and the traversal boundaries inside them, each with the state image that receives the
    // state after its column (-1: none).
    std::size_t segment_offsets = 0, boundary_rows = 0, boundary_columns = 0,
        boundary_destinations = 0;
    std::size_t bytes = 0;
};

struct FrameCommitLayout {
    std::uint32_t lanes = 0;
    std::size_t state_destination_offsets = 0, last_columns = 0, accepted_columns = 0,
        count_columns = 0, draft_append_starts = 0, draft_append_ends = 0;
    std::size_t bytes = 0;
};

struct FrameEgressLayout {
    std::uint32_t lanes = 0;
    std::size_t tokens = 0, counts = 0;
    std::size_t bytes = 0;
};

[[nodiscard]] FrameIngressLayout plan_frame_ingress(std::uint32_t lanes,
                                                    std::uint32_t boundaries);
[[nodiscard]] FrameCommitLayout plan_frame_commit(std::uint32_t lanes);
[[nodiscard]] FrameEgressLayout plan_frame_egress(std::uint32_t lanes);

// Arrays are compact-row indexed. Tensor views bind only the current exact-B prefix.
struct FrameIngress {
    std::span<TokenId> anchors;
    std::span<std::int32_t> frontiers, backend_frontiers, extents, valid_columns;
    std::span<std::int32_t> proposal_valid_columns, budgets, main_table_rows,
        backend_table_rows;
    std::span<std::int64_t> state_source_offsets, state_destination_offsets;
    std::span<std::int32_t> residency_slots, rope_deltas;
    std::span<ops::SamplingConfig> sampling;
    std::span<std::int32_t> sampling_positions, history_skip;
    std::span<std::int32_t> previous_rope, decode_positions, decode_rope;
    std::span<std::int32_t> step_positions, step_valid, step_rope;
    std::span<std::int32_t> segment_offsets, boundary_rows, boundary_columns;
    std::span<std::int64_t> boundary_destinations;

    [[nodiscard]] static FrameIngress bind(MemorySpan region,
                                           const FrameIngressLayout& layout) noexcept;
};

struct FrameCommit {
    std::span<std::int64_t> state_destination_offsets;
    std::span<std::int32_t> last_columns, accepted_columns, count_columns,
        draft_append_starts, draft_append_ends;

    [[nodiscard]] static FrameCommit bind(MemorySpan region,
                                          const FrameCommitLayout& layout) noexcept;
};

struct FrameEgress {
    std::span<TokenId> tokens;
    std::span<std::int32_t> counts;

    [[nodiscard]] static FrameEgress bind(MemorySpan region,
                                          const FrameEgressLayout& layout) noexcept;
};

struct MtpFrameLayout {
    TensorRegion base_conditioning;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion drafts;
    TensorRegion proposal_logits;
    TensorRegion history_conditioning;
    TensorRegion history_ids;
    TensorRegion history_positions, history_columns;
    TensorRegion history_rope_positions;
};

struct DFlashFrameLayout {
    TensorRegion proposal_ids;
    TensorRegion proposal_positions;
    TensorRegion append_positions;
    TensorRegion append_counts;
    TensorRegion drafts;
    std::optional<TensorRegion> candidate_ids;
    std::optional<TensorRegion> proposal_q;
};

struct FrameHostLayout {
    LayoutRegion ingress;
    LayoutRegion commit;
    LayoutRegion egress;
    TensorRegion input_ids;
    TensorRegion cache_positions;
    TensorRegion rope_positions;
    std::optional<TensorRegion> visual_indices;
    std::optional<TensorRegion> control_tokens;
    std::optional<TensorRegion> masks;
    std::optional<TensorRegion> drafts;
    std::optional<TensorRegion> score_logprobs;
    std::size_t bytes = 0;
};

struct FrameLayout {
    FrameSpec spec;
    // Byte layouts of the ingress/commit/egress lane tables, resolved for spec.batch_capacity.
    FrameIngressLayout ingress_arrays;
    FrameCommitLayout commit_arrays;
    FrameEgressLayout egress_arrays;
    // Token columns of the Main input/output planes; staged_capacity bounds one window's
    // per-sequence consumers (visual embeddings, MTP history).
    std::int32_t input_capacity  = 0;
    std::int32_t staged_capacity = 0;
    LayoutRegion ingress;
    LayoutRegion commit;
    LayoutRegion egress;
    TensorRegion input_ids;
    TensorRegion cache_positions;
    TensorRegion rope_positions;
    TensorRegion hidden;
    std::optional<TensorRegion> input_embeddings;
    std::optional<TensorRegion> visual_indices;
    std::optional<TensorRegion> logits;
    std::optional<TensorRegion> target_argmax;
    std::optional<TensorRegion> masks;
    std::optional<TensorRegion> score_targets;
    std::optional<TensorRegion> score_logprobs;
    std::optional<TensorRegion> control_ids;
    std::optional<MtpFrameLayout> mtp;
    std::optional<DFlashFrameLayout> dflash;
    FrameHostLayout host;
};

[[nodiscard]] FrameLayout plan_frames(LayoutBuilder& builder, const FrameSpec& spec);

struct MtpStepView {
    Tensor ids, positions, rope_positions, valid_columns;
};

struct MtpFrame {
    Tensor base_conditioning, ar_hidden, next_hidden, drafts, proposal_logits;
    Tensor history_conditioning, history_ids, history_positions, history_rope_positions,
        history_columns;
    Tensor step_positions, step_rope_positions, step_valid_columns;

    MtpFrame(MemorySpan backing, const MtpFrameLayout& layout);
    [[nodiscard]] MtpStepView step_view(std::int32_t step, std::int32_t batch) const;
    [[nodiscard]] Tensor proposals(std::int32_t batch) const;
};

struct DFlashFrame {
    Tensor proposal_ids, proposal_positions, append_positions, append_counts, drafts;
    Tensor candidate_ids, proposal_q;

    DFlashFrame(MemorySpan backing, const DFlashFrameLayout& layout);
};

struct FrameInputView {
    Tensor ids, cache_positions, rope_positions, hidden;
};

// One stable device frame and its pinned transfer storage. The execution ticket retains both
// until all device, transfer and control-side consumers finish.
class ExecutionFrame {
public:
    // `padding_state_offset` names the scratch State row (StateImageViews::scratch_row): every
    // unbound row of an ingress fill points its State destination there, so a captured decode
    // graph replaying its bucket writes a padding row's GDN State into the scratch row instead
    // of the zeroed offset zero, which names the first pooled record a live sequence may own.
    ExecutionFrame(MemorySpan backing, const FrameLayout& layout,
                   std::int64_t padding_state_offset);
    ExecutionFrame(const ExecutionFrame&)            = delete;
    ExecutionFrame& operator=(const ExecutionFrame&) = delete;

    // Packed prefill columns: segments are consecutive token ranges of one [sum, 1] plane, with
    // one valid length and one table row per segment in the ingress arrays.
    [[nodiscard]] FrameInputView input_view(std::int32_t width, std::int32_t batch) const;
    [[nodiscard]] FrameInputView host_input_view(std::int32_t width, std::int32_t batch) const;
    [[nodiscard]] FrameInputView decode_view(std::int32_t width, std::int32_t batch) const;
    [[nodiscard]] FrameInputView host_decode_view(std::int32_t width, std::int32_t batch) const;
    // Verify block of a mixed execution: columns [column, column + width * batch) of the shared
    // token axis. Ids and hidden are those columns; positions and RoPE are the dense decode
    // planes, exactly as decode_view binds them.
    [[nodiscard]] FrameInputView verify_view(std::int32_t column, std::int32_t width,
                                             std::int32_t batch) const;

    const FrameSpec spec;
    const std::int32_t input_capacity;
    const std::int32_t staged_capacity;
    const FrameIngressLayout ingress_arrays;
    const FrameCommitLayout commit_arrays;
    const FrameEgressLayout egress_arrays;
    // Scratch State row offset carried by every unbound ingress row; see the constructor.
    const std::int64_t padding_state_offset;
    PinnedHostBuffer host_storage;
    // Host mirrors over the pinned lane tables, one span per table; `host_ingress` and friends
    // point at them so device-side code shares one spelling with the tensor bindings.
    FrameIngress ingress_tables;
    FrameCommit commit_tables;
    FrameEgress egress_tables;
    FrameIngress* host_ingress = nullptr;
    FrameCommit* host_commit   = nullptr;
    FrameEgress* host_egress   = nullptr;
    MemorySpan host_ingress_region, host_commit_region, host_egress_region;
    Tensor host_input_ids, host_cache_positions, host_rope_positions;
    Tensor host_visual_indices;
    Tensor host_control_tokens;
    Tensor control_ids;
    Tensor host_masks, host_drafts, host_score_logprobs;
    Tensor host_step_positions, host_step_rope_positions, host_step_valid_columns;
    MemorySpan ingress, commit, egress;
    Tensor anchors, frontiers, backend_frontiers, extents, valid_columns;
    Tensor proposal_valid_columns, budgets, main_table_rows, backend_table_rows;
    Tensor state_source_offsets, state_destination_offsets, residency_slots, rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampling_positions, history_skip, previous_rope;
    Tensor decode_positions, decode_rope;
    Tensor segment_offsets, boundary_rows, boundary_columns, boundary_destinations;
    Tensor commit_destination_offsets, last_columns, count_columns, accepted_columns;
    Tensor draft_append_starts, draft_append_ends;
    Tensor output_tokens, output_counts;
    Tensor input_ids, cache_positions, rope_positions, hidden;
    Tensor input_embeddings;
    Tensor visual_indices;
    Tensor logits, target_argmax, masks, score_targets, score_logprobs;
    std::optional<MtpFrame> mtp;
    std::optional<DFlashFrame> dflash;

private:
    void validate_input_shape(std::int32_t width, std::int32_t batch) const;

public:
    // Byte offset of the egress counts table inside the egress region; a speculative readback
    // copies it plus one count per row.
    [[nodiscard]] std::size_t egress_counts_offset() const noexcept {
        return egress_arrays.counts;
    }
    // Zero every lane table of one host mirror. Replaces the trivially copyable structs'
    // value-initialization of the fixed-layout tables.
    void clear_host_ingress() noexcept;
    void clear_host_commit() noexcept;
    void clear_host_egress() noexcept;
};

} // namespace ninfer::models::qwen3_5
