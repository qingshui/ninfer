#include "models/qwen3_5/program/frames.h"

#include "models/load_options.h"
#include "models/qwen3_5/prefill_chunk.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::size_t kAlignment  = 256;
constexpr std::int32_t kScoreTile = 1024;

// Cursor over one frame region's lane tables: one contiguous table per member, back to back at
// the member's natural alignment, sized by the frame's resolved concurrency.
class FrameArraysCursor {
public:
    explicit FrameArraysCursor(std::uint32_t lanes) : lanes_(lanes) {
        if (lanes == 0) throw std::invalid_argument("Qwen3.5 frame concurrency must be positive");
    }

    template <class T>
    std::size_t table(std::size_t per_lane = 1) {
        const auto alignment = alignof(T);
        offset_              = (offset_ + alignment - 1) / alignment * alignment;
        const auto at        = offset_;
        offset_ += sizeof(T) * per_lane * lanes_;
        return at;
    }

    [[nodiscard]] std::size_t finish() const {
        constexpr auto alignment = alignof(std::max_align_t);
        return (offset_ + alignment - 1) / alignment * alignment;
    }

private:
    std::size_t offset_ = 0;
    std::size_t lanes_  = 0;
};

template <class T>
std::span<T> lane_span(MemorySpan region, std::size_t offset, std::uint32_t lanes) noexcept {
    return {static_cast<T*>(static_cast<void*>(static_cast<std::byte*>(region.data) + offset)),
            lanes};
}

TensorRegion add(LayoutBuilder& builder, DType dtype, std::initializer_list<std::int32_t> shape,
                 const char* label) {
    return builder.add_tensor(dtype, shape, kAlignment, label);
}

std::int32_t dimension(std::uint64_t value) {
    if (value == 0 || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("Qwen3.5 frame dimension is outside the Tensor domain");
    }
    return static_cast<std::int32_t>(value);
}

void validate(const FrameSpec& spec) {
    if (spec.hidden <= 0 || spec.output_rows <= 0 || spec.public_tokens <= 0 ||
        spec.public_tokens > spec.output_rows || spec.prefill_chunk == 0 ||
        spec.pack_tokens < spec.prefill_chunk || spec.pack_segments == 0 ||
        spec.batch_capacity == 0 || spec.batch_capacity > kMaximumConcurrency ||
        spec.pack_segments > kPrefillPackSegmentLimit) {
        throw std::invalid_argument("Qwen3.5 frame geometry is invalid");
    }
    if (spec.mixed_tokens !=
            resolve_mixed_tokens(spec.pack_tokens, spec.mixed_decode_rows, spec.draft_window) ||
        (spec.mixed_decode_rows != 0 &&
         (spec.mixed_decode_rows + 1 > spec.pack_segments || spec.causal_scoring ||
          is_masked_draft_backend(spec.backend)))) {
        throw std::invalid_argument("Qwen3.5 frame mixed geometry is invalid");
    }
    if ((spec.backend == SpeculativeBackend::None && spec.draft_window != 0) ||
        (spec.backend == SpeculativeBackend::Mtp &&
         (spec.draft_window == 0 || spec.draft_window > 5)) ||
        (is_masked_draft_backend(spec.backend) &&
         (spec.draft_window == 0 || spec.draft_window > 15))) {
        throw std::invalid_argument("Qwen3.5 frame draft extent does not match its backend");
    }
    if (spec.causal_scoring &&
        (spec.backend != SpeculativeBackend::None || spec.batch_capacity != 1)) {
        throw std::invalid_argument("Qwen3.5 scoring requires one non-speculative row");
    }
}

Tensor prefix(const Tensor& tensor, std::initializer_list<std::int32_t> shape) {
    return Tensor(MemorySpan{tensor.data, tensor.bytes(), tensor.device}, tensor.dtype, shape);
}

} // namespace

FrameIngressLayout plan_frame_ingress(std::uint32_t lanes) {
    FrameArraysCursor cursor(lanes);
    FrameIngressLayout out;
    out.lanes                     = lanes;
    out.anchors                   = cursor.table<TokenId>();
    out.frontiers                  = cursor.table<std::int32_t>();
    out.backend_frontiers          = cursor.table<std::int32_t>();
    out.extents                    = cursor.table<std::int32_t>();
    out.valid_columns              = cursor.table<std::int32_t>();
    out.proposal_valid_columns     = cursor.table<std::int32_t>();
    out.budgets                    = cursor.table<std::int32_t>();
    out.main_table_rows            = cursor.table<std::int32_t>();
    out.backend_table_rows         = cursor.table<std::int32_t>();
    out.state_source_offsets       = cursor.table<std::int64_t>();
    out.state_destination_offsets  = cursor.table<std::int64_t>();
    out.residency_slots            = cursor.table<std::int32_t>();
    out.rope_deltas                = cursor.table<std::int32_t>();
    out.sampling                   = cursor.table<ops::SamplingConfig>();
    out.sampling_positions         = cursor.table<std::int32_t>();
    out.history_skip               = cursor.table<std::int32_t>();
    out.previous_rope              = cursor.table<std::int32_t>(3);
    out.decode_positions           = cursor.table<std::int32_t>(16);
    out.decode_rope                = cursor.table<std::int32_t>(48);
    out.step_positions             = cursor.table<std::int32_t>(5);
    out.step_valid                 = cursor.table<std::int32_t>(5);
    out.step_rope                  = cursor.table<std::int32_t>(15);
    out.bytes                      = cursor.finish();
    return out;
}

FrameCommitLayout plan_frame_commit(std::uint32_t lanes) {
    FrameArraysCursor cursor(lanes);
    FrameCommitLayout out;
    out.lanes                    = lanes;
    out.state_destination_offsets = cursor.table<std::int64_t>();
    out.last_columns              = cursor.table<std::int32_t>();
    out.accepted_columns          = cursor.table<std::int32_t>();
    out.count_columns             = cursor.table<std::int32_t>();
    out.draft_append_starts       = cursor.table<std::int32_t>();
    out.draft_append_ends         = cursor.table<std::int32_t>();
    out.bytes                     = cursor.finish();
    return out;
}

FrameEgressLayout plan_frame_egress(std::uint32_t lanes) {
    FrameArraysCursor cursor(lanes);
    FrameEgressLayout out;
    out.lanes  = lanes;
    out.tokens = cursor.table<TokenId>(16);
    out.counts = cursor.table<std::int32_t>();
    out.bytes  = cursor.finish();
    return out;
}

FrameIngress FrameIngress::bind(MemorySpan region, const FrameIngressLayout& layout) noexcept {
    const auto lanes = layout.lanes;
    FrameIngress out;
    out.anchors                    = lane_span<TokenId>(region, layout.anchors, lanes);
    out.frontiers                  = lane_span<std::int32_t>(region, layout.frontiers, lanes);
    out.backend_frontiers          = lane_span<std::int32_t>(region, layout.backend_frontiers, lanes);
    out.extents                    = lane_span<std::int32_t>(region, layout.extents, lanes);
    out.valid_columns              = lane_span<std::int32_t>(region, layout.valid_columns, lanes);
    out.proposal_valid_columns     = lane_span<std::int32_t>(region, layout.proposal_valid_columns, lanes);
    out.budgets                    = lane_span<std::int32_t>(region, layout.budgets, lanes);
    out.main_table_rows            = lane_span<std::int32_t>(region, layout.main_table_rows, lanes);
    out.backend_table_rows         = lane_span<std::int32_t>(region, layout.backend_table_rows, lanes);
    out.state_source_offsets       = lane_span<std::int64_t>(region, layout.state_source_offsets, lanes);
    out.state_destination_offsets  = lane_span<std::int64_t>(region, layout.state_destination_offsets, lanes);
    out.residency_slots            = lane_span<std::int32_t>(region, layout.residency_slots, lanes);
    out.rope_deltas                = lane_span<std::int32_t>(region, layout.rope_deltas, lanes);
    out.sampling                   = lane_span<ops::SamplingConfig>(region, layout.sampling, lanes);
    out.sampling_positions         = lane_span<std::int32_t>(region, layout.sampling_positions, lanes);
    out.history_skip               = lane_span<std::int32_t>(region, layout.history_skip, lanes);
    out.previous_rope              = lane_span<std::int32_t>(region, layout.previous_rope, 3 * lanes);
    out.decode_positions           = lane_span<std::int32_t>(region, layout.decode_positions, 16 * lanes);
    out.decode_rope                = lane_span<std::int32_t>(region, layout.decode_rope, 48 * lanes);
    out.step_positions             = lane_span<std::int32_t>(region, layout.step_positions, 5 * lanes);
    out.step_valid                 = lane_span<std::int32_t>(region, layout.step_valid, 5 * lanes);
    out.step_rope                  = lane_span<std::int32_t>(region, layout.step_rope, 15 * lanes);
    return out;
}

FrameCommit FrameCommit::bind(MemorySpan region, const FrameCommitLayout& layout) noexcept {
    const auto lanes = layout.lanes;
    FrameCommit out;
    out.state_destination_offsets = lane_span<std::int64_t>(region, layout.state_destination_offsets, lanes);
    out.last_columns             = lane_span<std::int32_t>(region, layout.last_columns, lanes);
    out.accepted_columns         = lane_span<std::int32_t>(region, layout.accepted_columns, lanes);
    out.count_columns            = lane_span<std::int32_t>(region, layout.count_columns, lanes);
    out.draft_append_starts      = lane_span<std::int32_t>(region, layout.draft_append_starts, lanes);
    out.draft_append_ends        = lane_span<std::int32_t>(region, layout.draft_append_ends, lanes);
    return out;
}

FrameEgress FrameEgress::bind(MemorySpan region, const FrameEgressLayout& layout) noexcept {
    FrameEgress out;
    out.tokens = lane_span<TokenId>(region, layout.tokens, 16 * layout.lanes);
    out.counts = lane_span<std::int32_t>(region, layout.counts, layout.lanes);
    return out;
}

FrameLayout plan_frames(LayoutBuilder& builder, const FrameSpec& spec) {
    validate(spec);
    const auto batch = dimension(spec.batch_capacity);
    const auto width = dimension(std::uint64_t(spec.draft_window) + 1);
    const auto chunk = dimension(spec.prefill_chunk);
    FrameLayout out;
    const auto pack     = dimension(spec.pack_tokens);
    out.spec            = spec;
    out.staged_capacity = dimension(std::max<std::uint64_t>(chunk, std::uint64_t(width) * batch));
    out.input_capacity =
        std::max({out.staged_capacity, pack, static_cast<std::int32_t>(spec.mixed_tokens)});
    const auto inputs   = out.input_capacity;
    const auto staged   = out.staged_capacity;
    out.ingress_arrays  = plan_frame_ingress(spec.batch_capacity);
    out.commit_arrays   = plan_frame_commit(spec.batch_capacity);
    out.egress_arrays   = plan_frame_egress(spec.batch_capacity);
    out.ingress         = builder.add(out.ingress_arrays.bytes, kAlignment, "execution ingress");
    out.commit          = builder.add(out.commit_arrays.bytes, kAlignment, "execution commit");
    out.egress          = builder.add(out.egress_arrays.bytes, kAlignment, "execution egress");
    out.input_ids       = add(builder, DType::I32, {inputs}, "execution input ids");
    out.cache_positions = add(builder, DType::I32, {inputs}, "execution cache positions");
    out.rope_positions  = add(builder, DType::I32, {inputs, 3}, "execution rope positions");
    out.hidden = add(builder, DType::BF16, {spec.hidden, inputs}, "execution target hidden");
    if (spec.vision) {
        out.input_embeddings =
            add(builder, DType::BF16, {spec.hidden, staged}, "execution input embeddings");
        out.visual_indices = add(builder, DType::I32, {chunk}, "execution visual indices");
    }
    if (spec.causal_scoring) {
        out.score_targets  = add(builder, DType::I32, {kScoreTile}, "scoring target ids");
        out.score_logprobs = add(builder, DType::FP32, {kScoreTile}, "scoring log probabilities");
    } else {
        out.logits =
            add(builder, DType::BF16, {spec.output_rows, width, batch}, "execution target logits");
        out.target_argmax = add(builder, DType::I32, {width, batch}, "execution target argmax");
        out.masks         = add(builder, DType::I32,
                                {dimension((std::uint64_t(spec.public_tokens) + 31) / 32), width, batch},
                                "execution grammar masks");
    }
    if (spec.backend == SpeculativeBackend::Mtp) {
        auto& mtp         = out.mtp.emplace();
        const auto drafts = dimension(spec.draft_window);
        mtp.base_conditioning =
            add(builder, DType::BF16, {spec.hidden, batch}, "MTP base conditioning");
        mtp.ar_hidden =
            add(builder, DType::BF16, {spec.hidden, batch}, "MTP autoregressive hidden");
        mtp.next_hidden = add(builder, DType::BF16, {spec.hidden, batch}, "MTP next hidden");
        mtp.drafts      = add(builder, DType::I32, {batch, drafts}, "MTP current drafts");
        mtp.proposal_logits =
            add(builder, DType::BF16, {spec.output_rows, batch}, "MTP proposal logits");
        mtp.history_conditioning =
            add(builder, DType::BF16, {spec.hidden, staged}, "MTP history conditioning");
        mtp.history_ids       = add(builder, DType::I32, {staged}, "MTP history ids");
        mtp.history_positions = add(builder, DType::I32, {staged}, "MTP history positions");
        mtp.history_columns   = add(builder, DType::I32, {batch}, "MTP history lengths");
        mtp.history_rope_positions =
            add(builder, DType::I32, {staged, 3}, "MTP history rope positions");
    }
    if (is_masked_draft_backend(spec.backend)) {
        auto& draft        = out.dflash.emplace();
        draft.proposal_ids = add(builder, DType::I32, {width, batch}, "DFlash proposal ids");
        draft.proposal_positions =
            add(builder, DType::I32, {width, batch}, "DFlash proposal positions");
        draft.append_positions =
            add(builder, DType::I32, {width, batch}, "DFlash context append positions");
        draft.append_counts = add(builder, DType::I32, {batch}, "DFlash context append counts");
        draft.drafts        = add(builder, DType::I32, {width - 1, batch}, "DFlash current drafts");
        if (spec.backend == SpeculativeBackend::DFlash2) {
            draft.candidate_ids =
                add(builder, DType::I32, {16, width - 1, batch}, "DFlash2 proposal candidates");
            draft.proposal_q =
                add(builder, DType::FP32, {16, width - 1, batch}, "DFlash2 proposal probabilities");
        }
    }

    LayoutBuilder host;
    out.host.ingress         = host.add(out.ingress_arrays.bytes, kAlignment, "Host execution ingress");
    out.host.commit          = host.add(out.commit_arrays.bytes, kAlignment, "Host execution commit");
    out.host.egress          = host.add(out.egress_arrays.bytes, kAlignment, "Host execution egress");
    out.host.input_ids       = add(host, DType::I32, {inputs}, "Host input ids");
    out.host.cache_positions = add(host, DType::I32, {inputs}, "Host cache positions");
    out.host.rope_positions  = add(host, DType::I32, {inputs, 3}, "Host rope positions");
    if (spec.vision) {
        out.host.visual_indices = add(host, DType::I32, {chunk}, "Host visual indices");
    }
    if (!spec.causal_scoring) {
        const auto control      = dimension(std::max(spec.control_token_limit, 1U));
        out.host.control_tokens = add(host, DType::I32, {control, batch}, "Host control tokens");
        out.control_ids         = add(builder, DType::I32, {control}, "control token counts input");
    }
    if (out.masks) {
        out.host.masks =
            add(host, DType::I32, {out.masks->shape[0], width, batch}, "Host grammar masks");
    }
    if (spec.draft_window) {
        out.host.drafts = add(host, DType::I32, {width - 1, batch}, "Host draft handoff");
    }
    if (spec.causal_scoring) {
        out.host.score_logprobs =
            add(host, DType::FP32, {kScoreTile}, "Host score log probabilities");
    }
    out.host.bytes = host.finish(kAlignment, "Host execution frame");
    return out;
}

MtpFrame::MtpFrame(MemorySpan backing, const MtpFrameLayout& layout)
    : base_conditioning(layout.base_conditioning.bind(backing)),
      ar_hidden(layout.ar_hidden.bind(backing)), next_hidden(layout.next_hidden.bind(backing)),
      drafts(layout.drafts.bind(backing)), proposal_logits(layout.proposal_logits.bind(backing)),
      history_conditioning(layout.history_conditioning.bind(backing)),
      history_ids(layout.history_ids.bind(backing)),
      history_positions(layout.history_positions.bind(backing)),
      history_rope_positions(layout.history_rope_positions.bind(backing)),
      history_columns(layout.history_columns.bind(backing)) {}

Tensor MtpFrame::proposals(std::int32_t batch) const {
    return drafts.slice(0, 0, batch).permute({1, 0, 2, 3});
}

MtpStepView MtpFrame::step_view(std::int32_t step, std::int32_t batch) const {
    if (step < 0 || step >= drafts.ne[1] || batch <= 0 || batch > drafts.ne[0])
        throw std::invalid_argument("MTP step exceeds its frame capacity");
    const auto view = [&](const Tensor& value) {
        return value.slice(1, step, 1).slice(0, 0, batch).view({batch});
    };
    return {.ids       = view(drafts),
            .positions = view(step_positions),
            .rope_positions =
                step_rope_positions.slice(1, step, 1).slice(0, 0, 3 * batch).view({batch, 3}),
            .valid_columns = view(step_valid_columns)};
}

DFlashFrame::DFlashFrame(MemorySpan backing, const DFlashFrameLayout& layout)
    : proposal_ids(layout.proposal_ids.bind(backing)),
      proposal_positions(layout.proposal_positions.bind(backing)),
      append_positions(layout.append_positions.bind(backing)),
      append_counts(layout.append_counts.bind(backing)), drafts(layout.drafts.bind(backing)) {
    if (layout.candidate_ids) candidate_ids = layout.candidate_ids->bind(backing);
    if (layout.proposal_q) proposal_q = layout.proposal_q->bind(backing);
}

ExecutionFrame::ExecutionFrame(MemorySpan backing, const FrameLayout& layout)
    : spec(layout.spec), input_capacity(layout.input_capacity),
      staged_capacity(layout.staged_capacity), ingress_arrays(layout.ingress_arrays),
      commit_arrays(layout.commit_arrays), egress_arrays(layout.egress_arrays),
      host_storage(layout.host.bytes),
      ingress(layout.ingress.bind(backing)), commit(layout.commit.bind(backing)),
      egress(layout.egress.bind(backing)), input_ids(layout.input_ids.bind(backing)),
      cache_positions(layout.cache_positions.bind(backing)),
      rope_positions(layout.rope_positions.bind(backing)), hidden(layout.hidden.bind(backing)) {
    const auto batch = static_cast<std::int32_t>(spec.batch_capacity);
    const auto width = static_cast<std::int32_t>(spec.draft_window + 1);
    const auto in    = [&](std::size_t offset) {
        return Tensor(ingress.subspan(offset), DType::I32, {batch});
    };
    sampling_positions = in(ingress_arrays.sampling_positions);
    history_skip       = in(ingress_arrays.history_skip);
    previous_rope =
        Tensor(ingress.subspan(ingress_arrays.previous_rope), DType::I32, {batch, 3});
    decode_positions =
        Tensor(ingress.subspan(ingress_arrays.decode_positions), DType::I32, {16 * batch});
    decode_rope =
        Tensor(ingress.subspan(ingress_arrays.decode_rope), DType::I32, {16 * batch, 3});
    anchors                = in(ingress_arrays.anchors);
    frontiers              = in(ingress_arrays.frontiers);
    backend_frontiers      = in(ingress_arrays.backend_frontiers);
    extents                = in(ingress_arrays.extents);
    valid_columns          = in(ingress_arrays.valid_columns);
    proposal_valid_columns = in(ingress_arrays.proposal_valid_columns);
    budgets                = in(ingress_arrays.budgets);
    main_table_rows        = in(ingress_arrays.main_table_rows);
    backend_table_rows     = in(ingress_arrays.backend_table_rows);
    state_source_offsets =
        Tensor(ingress.subspan(ingress_arrays.state_source_offsets), DType::I64, {batch});
    state_destination_offsets = Tensor(
        ingress.subspan(ingress_arrays.state_destination_offsets), DType::I64, {batch});
    residency_slots = in(ingress_arrays.residency_slots);
    rope_deltas     = in(ingress_arrays.rope_deltas);
    sampling        = static_cast<const ops::SamplingConfig*>(
        ingress.subspan(ingress_arrays.sampling).data);
    const auto committed = [&](std::size_t offset) {
        return Tensor(commit.subspan(offset), DType::I32, {batch});
    };
    commit_destination_offsets = Tensor(
        commit.subspan(commit_arrays.state_destination_offsets), DType::I64, {batch});
    last_columns        = committed(commit_arrays.last_columns);
    accepted_columns    = committed(commit_arrays.accepted_columns);
    count_columns       = committed(commit_arrays.count_columns);
    draft_append_starts = committed(commit_arrays.draft_append_starts);
    draft_append_ends   = committed(commit_arrays.draft_append_ends);
    output_tokens =
        Tensor(egress.subspan(egress_arrays.tokens), DType::I32, {width, batch});
    output_counts = Tensor(egress.subspan(egress_arrays.counts), DType::I32, {batch});
    if (layout.logits) logits = layout.logits->bind(backing);
    if (layout.input_embeddings) input_embeddings = layout.input_embeddings->bind(backing);
    if (layout.visual_indices) visual_indices = layout.visual_indices->bind(backing);
    if (layout.target_argmax) target_argmax = layout.target_argmax->bind(backing);
    if (layout.masks) masks = layout.masks->bind(backing);
    if (layout.score_targets) score_targets = layout.score_targets->bind(backing);
    if (layout.score_logprobs) score_logprobs = layout.score_logprobs->bind(backing);
    if (layout.mtp) mtp.emplace(backing, *layout.mtp);
    if (layout.dflash) dflash.emplace(backing, *layout.dflash);

    const auto host      = host_storage.span();
    host_ingress_region  = layout.host.ingress.bind(host);
    host_commit_region   = layout.host.commit.bind(host);
    host_egress_region   = layout.host.egress.bind(host);
    ingress_tables       = FrameIngress::bind(host_ingress_region, ingress_arrays);
    commit_tables        = FrameCommit::bind(host_commit_region, commit_arrays);
    egress_tables        = FrameEgress::bind(host_egress_region, egress_arrays);
    host_ingress         = &ingress_tables;
    host_commit          = &commit_tables;
    host_egress          = &egress_tables;
    clear_host_ingress();
    clear_host_commit();
    clear_host_egress();
    host_input_ids       = layout.host.input_ids.bind(host);
    host_cache_positions = layout.host.cache_positions.bind(host);
    host_rope_positions  = layout.host.rope_positions.bind(host);
    if (layout.host.visual_indices) host_visual_indices = layout.host.visual_indices->bind(host);
    if (layout.host.control_tokens) host_control_tokens = layout.host.control_tokens->bind(host);
    if (layout.control_ids) control_ids = layout.control_ids->bind(backing);
    if (layout.host.masks) host_masks = layout.host.masks->bind(host);
    if (layout.host.drafts) host_drafts = layout.host.drafts->bind(host);
    if (mtp) {
        const auto bind_steps = [&](MemorySpan span, std::size_t offset, int axes) {
            return Tensor(span.subspan(offset), DType::I32,
                          {axes * batch, static_cast<int>(spec.draft_window)});
        };
        mtp->step_positions      = bind_steps(ingress, ingress_arrays.step_positions, 1);
        mtp->step_rope_positions = bind_steps(ingress, ingress_arrays.step_rope, 3);
        mtp->step_valid_columns  = bind_steps(ingress, ingress_arrays.step_valid, 1);
        const auto staging       = host_ingress_region;
        host_step_positions      = bind_steps(staging, ingress_arrays.step_positions, 1);
        host_step_rope_positions = bind_steps(staging, ingress_arrays.step_rope, 3);
        host_step_valid_columns  = bind_steps(staging, ingress_arrays.step_valid, 1);
    }
    if (layout.host.score_logprobs) host_score_logprobs = layout.host.score_logprobs->bind(host);
}

void ExecutionFrame::clear_host_ingress() noexcept {
    std::memset(host_ingress_region.data, 0, host_ingress_region.bytes);
}

void ExecutionFrame::clear_host_commit() noexcept {
    std::memset(host_commit_region.data, 0, host_commit_region.bytes);
}

void ExecutionFrame::clear_host_egress() noexcept {
    std::memset(host_egress_region.data, 0, host_egress_region.bytes);
}

void ExecutionFrame::validate_input_shape(std::int32_t width, std::int32_t batch) const {
    if (width <= 0 || batch <= 0 || batch > static_cast<std::int32_t>(spec.batch_capacity) ||
        std::int64_t(width) * batch > input_capacity) {
        throw std::invalid_argument("Qwen3.5 execution input exceeds its frame capacity");
    }
}

FrameInputView ExecutionFrame::input_view(std::int32_t width, std::int32_t batch) const {
    validate_input_shape(width, batch);
    return {.ids             = prefix(input_ids, {width, batch}),
            .cache_positions = prefix(cache_positions, {width, batch}),
            .rope_positions  = prefix(rope_positions, {width * batch, 3}),
            .hidden          = prefix(hidden, {spec.hidden, width, batch})};
}

FrameInputView ExecutionFrame::decode_view(std::int32_t width, std::int32_t batch) const {
    validate_input_shape(width, batch);
    if (width > 16) throw std::invalid_argument("Qwen3.5 decode exceeds its token capacity");
    return {prefix(width == 1 ? anchors : input_ids, {width, batch}),
            prefix(decode_positions, {width, batch}), prefix(decode_rope, {width * batch, 3}),
            prefix(hidden, {spec.hidden, width, batch})};
}

FrameInputView ExecutionFrame::verify_view(std::int32_t column, std::int32_t width,
                                           std::int32_t batch) const {
    validate_input_shape(width, batch);
    if (width > 16) throw std::invalid_argument("Qwen3.5 decode exceeds its token capacity");
    const auto columns = width * batch;
    if (column < 0 || spec.mixed_tokens == 0 ||
        std::int64_t(column) + columns > std::int64_t(spec.mixed_tokens))
        throw std::invalid_argument("Qwen3.5 verify block exceeds the mixed token axis");
    return {input_ids.slice(0, column, columns).view({width, batch}),
            prefix(decode_positions, {width, batch}), prefix(decode_rope, {columns, 3}),
            hidden.slice(1, column, columns).view({spec.hidden, width, batch})};
}

FrameInputView ExecutionFrame::host_decode_view(std::int32_t width, std::int32_t batch) const {
    validate_input_shape(width, batch);
    if (width > 16) throw std::invalid_argument("Qwen3.5 decode exceeds its token capacity");
    const auto view = [&](auto& storage, std::initializer_list<std::int32_t> shape) {
        return Tensor({storage.data(), storage.size_bytes(), Device::cpu(), true}, DType::I32,
                       shape);
    };
    return {width == 1 ? view(host_ingress->anchors, {1, batch}) : Tensor{},
            view(host_ingress->decode_positions, {width, batch}),
            view(host_ingress->decode_rope, {width * batch, 3}),
            {}};
}

FrameInputView ExecutionFrame::host_input_view(std::int32_t width, std::int32_t batch) const {
    validate_input_shape(width, batch);
    return {.ids             = prefix(host_input_ids, {width, batch}),
            .cache_positions = prefix(host_cache_positions, {width, batch}),
            .rope_positions  = prefix(host_rope_positions, {width * batch, 3}),
            .hidden          = {}};
}

} // namespace ninfer::models::qwen3_5
