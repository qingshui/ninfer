#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/text.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

execution::DraftExecutionView draft_view(RankProgramImpl& program) {
    auto& state = *program.resources.dflash;
    return {program.parameters,
            program.resources.work(),
            program.device.execution_view(),
            state.local,
            state.full ? &*state.full : nullptr,
            program.proposal_head};
}

Tensor dense_prefix(const Tensor& tensor, std::initializer_list<std::int32_t> shape) {
    return Tensor({tensor.data, tensor.bytes(), tensor.device}, tensor.dtype, shape);
}

} // namespace

void RankProgramImpl::append_draft_history(ComputeOperation& operation) {
    if (!resources.dflash) return;
    const bool forward    = std::holds_alternative<ForwardWork>(operation.request.work.operation);
    const auto batch      = forward ? 1 : static_cast<int>(operation.batch);
    std::uint32_t minimum = std::numeric_limits<std::uint32_t>::max(), maximum = 0;
    const auto rows = forward ? std::span(operation.rows).subspan(operation.forward_row, 1)
                              : std::span(operation.rows);
    for (const auto& row : rows) {
        const auto count = row.main_frontier - row.backend_frontier;
        minimum          = std::min(minimum, count);
        maximum          = std::max(maximum, count);
    }
    if (!maximum) return;
    append_draft_history(batch, {minimum, maximum});
}

void RankProgramImpl::append_draft_history(std::uint32_t batch_size,
                                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    auto& frame      = resources.frame();
    auto& state      = *resources.dflash;
    auto& draft      = *frame.dflash;
    const auto batch = static_cast<std::int32_t>(batch_size);
    const auto width = static_cast<std::int32_t>(draft_window + 1);
    auto scope       = resources.work().scope();
    auto features =
        resources.work().alloc(DType::BF16, {state.pending_features.ne[0], width, batch});
    auto positions = dense_prefix(draft.append_positions, {width, batch});
    auto counts    = draft.append_counts.slice(0, 0, batch);
    ops::prepare_ragged_prefix(state.pending_features, frame.residency_slots.slice(0, 0, batch),
                               frame.backend_frontiers.slice(0, 0, batch),
                               frame.frontiers.slice(0, 0, batch), features, positions, counts,
                               device.execution_view());
    execution::draft_append_context(draft_view(*this), features, positions, counts,
                                    frame.state_destination_offsets.slice(0, 0, batch),
                                    frame.backend_table_rows.slice(0, 0, batch), envelope);
}

void RankProgramImpl::run_dflash_draft(std::uint32_t batch_value, std::uint32_t minimum,
                                       std::uint32_t maximum) {
    auto& frame      = resources.frame();
    auto& draft      = *frame.dflash;
    const int batch  = static_cast<int>(batch_value);
    const int drafts = static_cast<int>(draft_window), width = drafts + 1;
    execution::DraftInputs inputs{frame.anchors.slice(0, 0, batch),
                                  frame.frontiers.slice(0, 0, batch),
                                  (backend == SpeculativeBackend::DFlash2
                                       ? frame.proposal_valid_columns
                                       : frame.valid_columns)
                                      .slice(0, 0, batch),
                                  frame.state_destination_offsets.slice(0, 0, batch),
                                  frame.backend_table_rows.slice(0, 0, batch),
                                  frame.sampling};
    execution::DraftOutputs outputs{
        dense_prefix(draft.proposal_ids, {width, batch}),
        dense_prefix(draft.proposal_positions, {width, batch}),
        dense_prefix(draft.drafts, {drafts, batch}),
        draft.candidate_ids.data
            ? dense_prefix(draft.candidate_ids, {draft.candidate_ids.ne[0], drafts, batch})
            : Tensor{},
        draft.proposal_q.data
            ? dense_prefix(draft.proposal_q, {draft.proposal_q.ne[0], drafts, batch})
            : Tensor{}};
    execution::DraftEnvelopes envelopes{{minimum, maximum}, {minimum, maximum}};
    execution::draft_propose(draft_view(*this), inputs, outputs, envelopes);
}

void RankProgramImpl::begin_draft(ComputeOperation& operation) {
    auto& frame           = resources.frame();
    const int batch       = static_cast<int>(operation.batch);
    const int drafts      = static_cast<int>(draft_window);
    const int width       = drafts + 1;
    std::uint32_t minimum = capacity, maximum = 0;
    // Only the decode rows draft; a mixed execution lists its prefill segments after them.
    for (std::uint32_t row = 0; row < operation.batch; ++row) {
        minimum = std::min(minimum, operation.rows[row].main_frontier);
        maximum = std::max(maximum, operation.rows[row].main_frontier);
    }
    if (!slot().graphs) slot().gpu_timer.begin_draft();
    // A draft bucket pads its batch through the zeroed host row state; an uncovered batch drafts
    // through the eager path below.
    const bool draft_bucket =
        slot().graphs && slot().graphs->captured_batch(GraphPhase::Draft, operation.batch);
    if (draft_bucket) {
        (void)slot().graphs->start(GraphPhase::Draft, operation.batch, maximum,
                                   device.execution_view());
    } else if (backend == SpeculativeBackend::Mtp) {
        begin_mtp_draft(operation);
    } else {
        run_dflash_draft(operation.batch, minimum, maximum);
    }
    if (!slot().graphs) slot().gpu_timer.end_draft();
    Tensor proposals = backend == SpeculativeBackend::Mtp
                           ? frame.mtp->proposals(batch)
                           : dense_prefix(frame.dflash->drafts, {drafts, batch});
    auto input = operation.mixed
                     ? frame.verify_view(static_cast<int>(operation.prefill_tokens), width, batch)
                     : frame.decode_view(width, batch);
    ops::prepare_verify_ids(frame.anchors.slice(0, 0, batch), proposals,
                            frame.extents.slice(0, 0, batch), input.ids, device.execution_view());
    const bool constrained = std::any_of(operation.sampling.begin(), operation.sampling.end(),
                                         [](const auto& row) { return row.constrained; });
    if (constrained) {
        CUDA_CHECK(cudaMemcpyAsync(frame.host_drafts.data, proposals.data,
                                   backend == SpeculativeBackend::Mtp ? frame.mtp->drafts.bytes()
                                                                      : proposals.bytes(),
                                   cudaMemcpyDeviceToHost, device.stream()));
        slot().drafts_ready.record(device.stream());
    }
}

// One traversal over [prefill segments | verify block]. Compact rows [0, D) are the decode rows and
// [D, D + S) the segments; the traversal reads the row metadata of all of them.
void RankProgramImpl::begin_mixed_target(ComputeOperation& operation) {
    auto& frame      = resources.frame();
    const auto rows  = static_cast<int>(operation.batch);
    const auto count = static_cast<int>(operation.segments.size());
    const auto width = static_cast<int>(operation.width);
    const auto axis  = static_cast<int>(operation.axis_tokens);
    ++operation.attention.forwards;
    operation.attention.rows += rows + count;
    operation.attention.scheduled_positions += operation.axis_tokens;
    // Each part bounds the keys its own rows can see.
    const auto envelope = [&](int first, int last) {
        std::uint32_t minimum = capacity, maximum = 0;
        for (int row = first; row < last; ++row) {
            const auto end = static_cast<std::uint32_t>(frame.host_ingress->frontiers[row] +
                                                        frame.host_ingress->valid_columns[row]);
            minimum        = std::min(minimum, end);
            maximum        = std::max(maximum, end);
            operation.attention.valid_positions += frame.host_ingress->valid_columns[row];
        }
        return ops::CausalAttentionExecutionEnvelope{minimum, maximum};
    };
    // A mixed traversal always runs eagerly: its decode tail is outside every prefill plan.
    if (slot().graphs) {
        ++prefill_plan.eager_fallbacks;
    }
    const auto steps_envelope   = envelope(0, rows);
    const auto prefill_envelope = envelope(rows, rows + count);
    const auto input            = frame.input_view(axis, 1);
    execution::TextInputs inputs{input.ids,
                                 input.cache_positions,
                                 input.rope_positions,
                                 frame.valid_columns.slice(0, 0, rows + count),
                                 frame.main_table_rows.slice(0, 0, rows + count),
                                 frame.state_source_offsets.slice(0, 0, rows),
                                 frame.state_destination_offsets.slice(0, 0, rows),
                                 std::nullopt};
    const execution::MixedUpdate update{
        operation.segments,
        {operation.prefill_tokens, static_cast<std::uint32_t>(width),
         static_cast<std::uint32_t>(rows), 0, steps_envelope},
        operation.speculative ? &*resources.replay_records : nullptr,
        segment_table(static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(count),
                      operation.boundaries)};
    resources.work().reset();
    slot().gpu_timer.begin_target();
    operation.text = std::make_unique<execution::TextRun>(
        execution::TextExecutionView{parameters, resources.work(), device.execution_view(),
                                     resources.decoder.text_kv, resources.state_images.linear(),
                                     resources.moe.get()},
        std::move(inputs), update, prefill_envelope, input.hidden);
    operation.row_finished = operation.text->start();
}

void RankProgramImpl::begin_target(ComputeOperation& operation) {
    if (!operation.width) return;
    if (operation.mixed) return begin_mixed_target(operation);
    auto& frame        = resources.frame();
    const bool forward = std::holds_alternative<ForwardWork>(operation.request.work.operation) ||
                         std::holds_alternative<ScoreWork>(operation.request.work.operation);
    const int batch = forward ? 1 : static_cast<int>(operation.batch);
    const int width = static_cast<int>(operation.width);
    // A packed traversal is one B=1 token axis whose metadata has one entry per segment.
    const int rows = operation.packed ? static_cast<int>(operation.segments.size()) : batch;
    ++operation.attention.forwards;
    operation.attention.rows += rows;
    operation.attention.scheduled_positions += std::uint64_t(width) * batch;
    if (forward)
        operation.attention.valid_positions += width;
    else
        for (int row = 0; row < batch; ++row) {
            operation.attention.valid_positions += frame.host_ingress->valid_columns[row];
        }
    auto input = forward ? frame.input_view(width, batch) : frame.decode_view(width, batch);
    if (std::holds_alternative<ForwardWork>(operation.request.work.operation) && slot().graphs &&
        replay_prefill_plan(operation))
        return;
    if (!forward && slot().graphs) {
        const auto phase = operation.speculative ? GraphPhase::Verify : GraphPhase::Main;
        // The target bucket pads its batch through the zeroed host row state; a padding row's
        // GDN State destination points at the scratch State row, so the replay never writes a
        // pooled record. An uncovered batch runs the eager traversal below, which computes the
        // same rows without graph replay.
        if (slot().graphs->captured_batch(phase, operation.batch)) {
            std::uint32_t frontier = 0;
            for (const auto& row : operation.rows) {
                frontier = std::max(frontier, row.main_frontier);
            }
            operation.graph_started = true;
            operation.projected     = true;
            operation.row_finished =
                slot().graphs->start(phase, operation.batch, frontier, device.execution_view());
            return;
        }
    }
    execution::TextInputs inputs{input.ids,
                                 input.cache_positions,
                                 input.rope_positions,
                                 frame.valid_columns.slice(0, 0, rows),
                                 frame.main_table_rows.slice(0, 0, rows),
                                 frame.state_source_offsets.slice(0, 0, rows),
                                 frame.state_destination_offsets.slice(0, 0, rows),
                                 operation.input_embeddings};
    execution::TextStateUpdate update = execution::StepUpdate{};
    if (operation.packed) {
        update = execution::PackedSequenceUpdate{
            operation.segments,
            segment_table(0, static_cast<std::uint32_t>(operation.segments.size()),
                          operation.boundaries)};
    } else if (forward) {
        const auto& row = operation.rows[operation.forward_row];
        update = execution::SequenceUpdate{row.state_offsets.source, row.state_offsets.destination};
    } else if (operation.speculative) {
        update.emplace<execution::ReplayRecords>(*resources.replay_records);
    }
    std::uint32_t minimum = capacity, maximum = 0;
    for (int row = 0; row < rows; ++row) {
        const auto end = static_cast<std::uint32_t>(frame.host_ingress->frontiers[row] +
                                                    frame.host_ingress->valid_columns[row]);
        minimum        = std::min(minimum, end);
        maximum        = std::max(maximum, end);
    }
    std::optional<execution::DFlashFeatureSink> features;
    if (resources.dflash) {
        auto& state = *resources.dflash;
        features.emplace();
        features->layers = parameters.profile->config.draft->target_layer_ids;
        if (forward) {
            features->features  = state.prefill_features.slice(1, 0, width);
            features->positions = state.prefill_positions.slice(0, 0, width);
        } else {
            features->features        = state.pending_features.slice(1, 0, width);
            features->residency_slots = frame.residency_slots.slice(0, 0, batch);
        }
    }
    resources.work().reset();
    if (forward) slot().prefill_timer.start();
    if (!forward) slot().gpu_timer.begin_target();
    operation.text = std::make_unique<execution::TextRun>(
        execution::TextExecutionView{parameters, resources.work(), device.execution_view(),
                                     resources.decoder.text_kv, resources.state_images.linear(),
                                     resources.moe.get()},
        std::move(inputs), update, ops::CausalAttentionExecutionEnvelope{minimum, maximum},
        input.hidden, features ? &*features : nullptr);
    operation.row_finished = operation.text->start();
}

bool RankProgramImpl::replay_prefill_plan(ComputeOperation& operation) {
    auto& frame            = resources.frame();
    const bool packed      = operation.packed;
    const auto segments    = packed ? static_cast<std::uint32_t>(operation.segments.size()) : 1U;
    // The shape a plan is captured for: every segment is one complete chunk window of text
    // tokens. A shorter or split window, an embedding input or draft feature state is outside
    // every bucket and takes the eager traversal.
    // A plan bakes its launch geometry, so a flight with traversal boundaries takes the eager
    // path as well.
    bool bucket_shaped = !operation.input_embeddings && !resources.dflash && segments != 0 &&
                         operation.boundaries == 0;
    if (packed) {
        for (const auto& segment : operation.segments) {
            bucket_shaped = bucket_shaped && segment.tokens == prefill_chunk;
        }
    } else {
        bucket_shaped = bucket_shaped && operation.width == prefill_chunk;
    }
    const auto width = bucket_shaped ? segments * prefill_chunk : 0U;
    if (!width || slot().graphs->captured_prefill_width(width) != width) {
        if (bucket_shaped) ++prefill_plan.misses;
        ++prefill_plan.eager_fallbacks;
        return false;
    }
    ++prefill_plan.hits;
    // Per-flight patch: the plan's segments run on their staging State rows, so each sequence's
    // State is copied in before the replay and its published State copied out after it. The rest
    // of the traversal inputs (ids, positions, row tables) are the frame the eager path fills.
    const auto cuda = device.execution_view();
    auto& images    = resources.state_images;
    const auto state = [&](std::uint32_t index) {
        return packed ? std::pair{operation.segments[index].source_offset,
                                  operation.segments[index].destination_offset}
                      : std::pair{operation.rows[operation.forward_row].state_offsets.source,
                                  operation.rows[operation.forward_row].state_offsets.destination};
    };
    std::int32_t frontier = 0;
    for (std::uint32_t index = 0; index < segments; ++index) {
        images.storage().copy(state(index).first, images.staging_row(index), cuda);
        frontier = std::max(frontier, frame.host_ingress->frontiers[index]);
    }
    slot().prefill_timer.start();
    operation.graph_started = true;
    operation.row_finished =
        slot().graphs->start_prefill(width, static_cast<std::uint32_t>(frontier), cuda);
    if (!operation.row_finished) throw std::logic_error("resident prefill plan did not submit");
    for (std::uint32_t index = 0; index < segments; ++index) {
        images.storage().copy(images.staging_row(index), state(index).second, cuda);
    }
    return true;
}

bool RankProgramImpl::advance_target(ComputeOperation& operation) {
    if (!operation.width) return true;
    if (!operation.row_finished)
        operation.row_finished = operation.text ? operation.text->poll() : slot().graphs->poll();
    if (!operation.row_finished) return false;
    workspace_peak = std::max(workspace_peak, resources.work().peak_used());
    operation.text.reset();
    return true;
}

} // namespace ninfer::models::qwen3_5::detail
