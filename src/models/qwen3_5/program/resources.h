#pragma once

#include "core/arena.h"
#include "core/buffer.h"
#include "core/context_pool.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "models/common/moe/schedule.h"
#include "models/common/vision/resources.h"
#include "models/qwen3_5/program/frames.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/state/state_image.h"
#include "ninfer/ops/gdn_replay.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct DFlashPersistentSpec {
    std::uint32_t full_layers = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
    std::int32_t feature_rows = 0;
};

struct ResourceSpec {
    DecoderStateSpec decoder;
    StateImageSpec state_images;
    std::optional<GdnReplayRecordSpec> replay_records;
    std::optional<DFlashPersistentSpec> dflash;
    FrameSpec frame;
    // Compute frames the execution storage is replicated for: each owns an ExecutionFrame (lane
    // tables, token axis planes, pinned mirrors) and a workspace arena (see Resources).
    std::uint32_t compute_frames   = 1;
    std::size_t context_bytes      = 0;
    std::size_t host_context_bytes = 0;
    // Staging State rows behind the scratch row of the context arena: the fixed State targets of
    // captured prefill plans (one per segment of the largest plan).
    std::uint32_t staging_states = 0;
};

struct DFlashPersistentLayout {
    std::optional<PagedKVCacheLayout> full;
    TensorRegion prefill_features;
    TensorRegion prefill_positions;
    TensorRegion pending_features;
};

struct ResourceLayout {
    DecoderStateLayout decoder;
    StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    // One frame layout per compute frame, all of the same spec.
    std::vector<FrameLayout> frames;
    std::optional<TensorRegion> token_counts;
    std::size_t context_bytes      = 0;
    std::size_t host_context_bytes = 0;
    std::uint32_t staging_states   = 0;
    std::size_t bytes              = 0;

    [[nodiscard]] std::size_t pinned_bytes() const;
    [[nodiscard]] std::vector<RecordSpec> context_records() const;
};

struct WorkspaceSpec {
    std::size_t text_prefill     = 0;
    std::size_t text_mixed       = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<vision::Capacity> vision;
    std::size_t capacity = 0;
};

[[nodiscard]] ResourceLayout plan_resources(const ResourceSpec& spec);

struct DFlashPersistentState {
    CyclicKVCacheView& local;
    std::optional<PagedKVCache> full;
    Tensor prefill_features, prefill_positions, pending_features;

    DFlashPersistentState(MemorySpan arena, MemorySpan backing,
                          const DFlashPersistentLayout& layout, CyclicKVCacheView& local_state);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer,
                                                     const Tensor& offsets) const;
    [[nodiscard]] PagedKVBatchLayerView full_batch_layer(std::uint32_t layer) const;
};

// The storage one compute frame owns: its ExecutionFrame (lane tables and token-axis planes with
// their pinned mirrors) and its workspace. A flight on the frame writes and reads only these, so
// the host may fill the next frame while the device still reads the previous one.
struct ComputeFrameResources {
    DeviceBuffer workspace_storage;
    MemoryArena work;
    ExecutionFrame frame;

    ComputeFrameResources(DeviceContext& device, MemorySpan backing, const FrameLayout& layout,
                          const WorkspaceSpec& workspace, std::int64_t padding_state_offset);
};

// Rank-owned arenas and execution storage; the Directory assigns each Context record. Everything
// except `frames` is shared by all compute frames; `select` names the frame that frame(), work()
// and workspace_storage() address.
class Resources {
public:
    Resources(DeviceContext& device, const ResourceLayout& layout, const WorkspaceSpec& workspace,
              const StartupObserver& observer = {});
    Resources(const Resources&)            = delete;
    Resources& operator=(const Resources&) = delete;

    [[nodiscard]] PagedKVCache* backend_kv() noexcept;
    [[nodiscard]] const PagedKVCache* backend_kv() const noexcept;

    [[nodiscard]] std::uint32_t frame_count() const noexcept {
        return static_cast<std::uint32_t>(frames.size());
    }
    [[nodiscard]] std::uint32_t selected() const noexcept { return selected_; }
    void select(std::uint32_t index);
    [[nodiscard]] ComputeFrameResources& compute_frame(std::uint32_t index);
    [[nodiscard]] const ComputeFrameResources& compute_frame(std::uint32_t index) const;
    [[nodiscard]] ExecutionFrame& frame() { return frames[selected_]->frame; }
    [[nodiscard]] const ExecutionFrame& frame() const { return frames[selected_]->frame; }
    [[nodiscard]] MemoryArena& work() { return frames[selected_]->work; }
    [[nodiscard]] const MemoryArena& work() const { return frames[selected_]->work; }
    [[nodiscard]] DeviceBuffer& workspace_storage() { return frames[selected_]->workspace_storage; }

    const std::vector<RecordSpec> context_records;
    DeviceBuffer context;
    DeviceBuffer persistent;
    PinnedHostBuffer host_context;
    DecoderState decoder;
    StateImageViews state_images;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    std::optional<DFlashPersistentState> dflash;
    Tensor token_counts;
    std::optional<vision::Storage> vision;
    std::unique_ptr<moe::Schedule> moe;
    std::vector<std::unique_ptr<ComputeFrameResources>> frames;

private:
    std::uint32_t selected_ = 0;
};

} // namespace ninfer::models::qwen3_5
