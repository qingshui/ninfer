#include "models/qwen3_5/program/resources.h"

#include "core/startup.h"
#include "models/load_options.h"

#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::size_t kAlignment = 256;

std::int32_t dimension(std::uint64_t value) {
    if (value == 0 || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("Qwen3.5 resource dimension is outside the Tensor domain");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

ResourceLayout plan_resources(const ResourceSpec& spec) {
    if (spec.compute_frames == 0) {
        throw std::invalid_argument("Qwen3.5 resources need at least one compute frame");
    }
    if (spec.decoder.enable_mtp != (spec.frame.backend == SpeculativeBackend::Mtp) ||
        spec.dflash.has_value() != is_masked_draft_backend(spec.frame.backend) ||
        spec.decoder.kv_table_rows < static_cast<std::int32_t>(spec.frame.batch_capacity)) {
        throw std::invalid_argument("Qwen3.5 context and frame capacities do not agree");
    }
    LayoutBuilder builder;
    ResourceLayout out;
    out.decoder      = plan_decoder_state(builder, spec.decoder);
    out.state_images = plan_state_image_layout(spec.state_images);
    if (spec.replay_records) {
        out.replay_records = plan_gdn_replay_records(builder, *spec.replay_records);
    }
    if (spec.dflash) {
        const auto& draft = *spec.dflash;
        auto& layout      = out.dflash.emplace();
        if (draft.feature_rows <= 0 || !spec.state_images.dflash_local) {
            throw std::invalid_argument("DFlash resources require features and local state");
        }
        if (draft.full_layers) {
            layout.full = plan_paged_kv_cache(builder, draft.full_layers, spec.decoder.capacity,
                                              draft.kv_heads, draft.head_dim,
                                              KvCacheStorage::BFloat16, spec.decoder.kv_table_rows);
        }
        layout.prefill_features = builder.add_tensor(
            DType::BF16, {draft.feature_rows, dimension(spec.frame.prefill_chunk)}, kAlignment,
            "DFlash prefill target features");
        layout.prefill_positions =
            builder.add_tensor(DType::I32, {dimension(spec.frame.prefill_chunk)}, kAlignment,
                               "DFlash prefill target positions");
        layout.pending_features = builder.add_tensor(
            DType::BF16,
            {draft.feature_rows, dimension(std::uint64_t(spec.frame.draft_window) + 1),
             dimension(spec.frame.batch_capacity)},
            kAlignment, "DFlash residency target features");
    }
    for (std::uint32_t frame = 0; frame < spec.compute_frames; ++frame) {
        out.frames.push_back(plan_frames(builder, spec.frame));
    }
    if (!spec.frame.causal_scoring) {
        out.token_counts = builder.add_tensor(
            DType::I32, {spec.frame.public_tokens, dimension(spec.frame.batch_capacity)},
            kAlignment, "residency sampling token counts");
    }
    out.context_bytes      = spec.context_bytes;
    out.host_context_bytes = spec.host_context_bytes;
    out.staging_states     = spec.staging_states;
    out.bytes              = builder.finish(kAlignment, "Qwen3.5 execution resources");
    return out;
}

std::vector<RecordSpec> ResourceLayout::context_records() const {
    std::vector<RecordSpec> records{
        {state_images.images.image_bytes, state_images.images.alignment},
        {decoder.text_kv.pages.page_bytes, decoder.text_kv.pages.alignment}};
    const auto* backend = decoder.mtp_kv           ? &*decoder.mtp_kv
                          : dflash && dflash->full ? &*dflash->full
                                                   : nullptr;
    if (backend) records.push_back({backend->pages.page_bytes, backend->pages.alignment});
    return records;
}

std::size_t ResourceLayout::pinned_bytes() const {
    const auto mapping_bytes = [](const PagedKVCacheLayout& cache) {
        const auto& table = cache.execution_tables.spec;
        return std::size_t(table.logical_page_capacity) * table.table_rows * sizeof(std::int64_t);
    };
    auto mapping = mapping_bytes(decoder.text_kv);
    if (decoder.mtp_kv) mapping += mapping_bytes(*decoder.mtp_kv);
    if (dflash && dflash->full) mapping += mapping_bytes(*dflash->full);
    const auto& spec  = frames.front().spec;
    const auto counts = token_counts ? std::size_t(spec.public_tokens) * spec.batch_capacity *
                                           sizeof(std::int32_t)
                                     : 0;
    std::size_t frame_bytes = 0;
    for (const auto& frame : frames) {
        frame_bytes += frame.host.bytes;
    }
    return frame_bytes + 2 * mapping + counts;
}

DFlashPersistentState::DFlashPersistentState(MemorySpan arena, MemorySpan backing,
                                             const DFlashPersistentLayout& layout,
                                             CyclicKVCacheView& local_state)
    : local(local_state), prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)) {
    if (layout.full) full.emplace(arena, backing, *layout.full);
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer,
                                                          const Tensor& offsets) const {
    return local.layer_view(layer, offsets);
}

PagedKVBatchLayerView DFlashPersistentState::full_batch_layer(std::uint32_t layer) const {
    if (!full) throw std::logic_error("DFlash has no full-attention context");
    return full->batch_layer_view(layer);
}

ComputeFrameResources::ComputeFrameResources(DeviceContext& device, MemorySpan backing,
                                             const FrameLayout& layout,
                                             const WorkspaceSpec& workspace,
                                             std::int64_t padding_state_offset)
    : workspace_storage(device.device(), workspace.capacity),
      work(workspace_storage.span().subspan(0, workspace.general_capacity)),
      frame(backing, layout, padding_state_offset) {}

Resources::Resources(DeviceContext& device, const ResourceLayout& layout,
                     const WorkspaceSpec& workspace, const StartupObserver& observer)
    : context_records(layout.context_records()),
      // The context arena carries one scratch State image and layout.staging_states staging
      // images beyond the pool-managed records; the session's state pool still sees exactly
      // layout.context_bytes (see context_spec). StateImageViews exposes the first tail image as
      // the scratch row decode graph padding rows write their GDN State into, and the rest as the
      // staging rows captured prefill plans run their segments against.
      context(device.device(),
              layout.context_bytes +
                  (std::size_t{1} + layout.staging_states) * layout.state_images.images.image_bytes),
      persistent(device.device(), layout.bytes),
      decoder(context.span(), persistent.span(), layout.decoder),
      state_images(context.span(), layout.state_images, layout.staging_states) {
    if (workspace.general_capacity == 0 || workspace.capacity < workspace.general_capacity) {
        throw std::invalid_argument("Qwen3.5 execution workspace layout is inconsistent");
    }
    if (layout.frames.empty()) {
        throw std::invalid_argument("Qwen3.5 resources have no compute frame layout");
    }
    // Every compute frame gets its own frame region and workspace: a flight reads and writes only
    // the frame of its ticket, so no table or plane is shared between flights in flight.
    for (const auto& frame_layout : layout.frames) {
        frames.push_back(std::make_unique<ComputeFrameResources>(
            device, persistent.span(), frame_layout, workspace, state_images.scratch_row()));
    }
    if (layout.host_context_bytes) {
        StartupPhaseScope phase(observer, StartupPhase::HostContextPin, StartupProgressUnit::Bytes,
                                layout.host_context_bytes);
        host_context = PinnedHostBuffer(layout.host_context_bytes);
        phase.complete(layout.host_context_bytes, layout.host_context_bytes);
    }
    if (workspace.vision) vision.emplace(device, *workspace.vision);
    for (std::uint32_t row = 0; row < state_images.staging_rows(); ++row) {
        state_images.storage().clear(state_images.staging_row(row), device.execution_view());
    }
    const auto backing = persistent.span();
    if (layout.replay_records) {
        replay_records.emplace(backing, *layout.replay_records);
        replay_fold.emplace(*replay_records, state_images.linear().all_layers_view());
    }
    if (layout.dflash) {
        auto* local = state_images.dflash_local();
        if (!local) throw std::logic_error("DFlash StateImage has no local state");
        dflash.emplace(context.span(), backing, *layout.dflash, *local);
    }
    if (layout.token_counts) {
        token_counts = layout.token_counts->bind(backing);
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream()));
    }
}

void Resources::select(std::uint32_t index) {
    if (index >= frames.size()) throw std::out_of_range("Qwen3.5 compute frame index is invalid");
    selected_ = index;
}

ComputeFrameResources& Resources::compute_frame(std::uint32_t index) {
    if (index >= frames.size()) throw std::out_of_range("Qwen3.5 compute frame index is invalid");
    return *frames[index];
}

const ComputeFrameResources& Resources::compute_frame(std::uint32_t index) const {
    if (index >= frames.size()) throw std::out_of_range("Qwen3.5 compute frame index is invalid");
    return *frames[index];
}

PagedKVCache* Resources::backend_kv() noexcept {
    if (auto* mtp = decoder.mtp_cache()) return mtp;
    return dflash && dflash->full ? &*dflash->full : nullptr;
}

const PagedKVCache* Resources::backend_kv() const noexcept {
    if (const auto* mtp = decoder.mtp_cache()) return mtp;
    return dflash && dflash->full ? &*dflash->full : nullptr;
}

} // namespace ninfer::models::qwen3_5
