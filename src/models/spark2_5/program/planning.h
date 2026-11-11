#pragma once

#include "models/common/graph_preparation.h"
#include "models/spark2_5/execution/workspace.h"
#include "models/spark2_5/execution_shape.h"
#include "models/spark2_5/profile.h"
#include "models/spark2_5/program/buffers.h"
#include "models/spark2_5/session/planning.h"
#include "models/spark2_5/state/kv_cache.h"
#include "models/spark2_5/state/state_image.h"
#include "runtime/engine/context_capacity.h"

#include "ninfer/ops/softmax_attention.h"

#include <optional>
#include <vector>

namespace ninfer::models::spark2_5 {

// Columns of one scored window whose logits are projected together.
inline constexpr std::uint32_t kScoreColumns = 512;

struct PersistentLayout {
    StateLayout state;
    FullKVLayout kv;
    BufferLayout buffers;
    std::size_t bytes = 0;
};

struct ExecutionPlan {
    std::shared_ptr<const ModelProfile> profile;
    ExecutionShape shape;
    PersistentLayout persistent;
    // Peak transient bytes of any traversal the Program issues, including graph capture.
    std::size_t workspace_bytes       = 0;
    std::size_t host_context_bytes    = 0;
    std::size_t pinned_io_bytes       = 0;
    std::size_t pinned_output_bytes   = 0;
    std::size_t pinned_sampling_bytes = 0;
    runtime::ContextCapacityResolution capacity;
    std::uint32_t checkpoint_interval = 8192;
    bool graph_probe                  = false;

    [[nodiscard]] std::size_t device_bytes() const;
    [[nodiscard]] std::size_t device_reservation_bytes() const;
    [[nodiscard]] std::size_t host_reservation_bytes() const;
    [[nodiscard]] std::size_t pinned_context_bytes() const;
    [[nodiscard]] std::size_t pinned_aux_bytes() const;
    [[nodiscard]] std::size_t pinned_context_counts_bytes() const;

    [[nodiscard]] std::size_t host_capacity_bytes() const { return host_context_bytes; }

    [[nodiscard]] std::size_t context_capacity() const { return capacity.device_bytes; }
};

[[nodiscard]] ExecutionShape describe_execution_shape(const EngineOptions&);

// Upper bounds of the context ranges a decode round is specialized for; the last equals
// `capacity`. The envelope of range i is (bound[i-1], bound[i]].
[[nodiscard]] std::vector<std::uint32_t> decode_context_bounds(std::uint32_t capacity);
[[nodiscard]] std::vector<ops::CausalAttentionExecutionEnvelope>
decode_envelopes(std::uint32_t capacity);

// Available bytes exclude immutable GPU weights and the caller's startup headroom. Shared-memory
// targets include planned Host preparation storage in the physical reservation.
[[nodiscard]] std::optional<ExecutionPlan>
plan_execution(std::shared_ptr<const ModelProfile>, const EngineOptions&, DeviceContext&,
               std::size_t available_bytes, std::size_t available_host_bytes);

// The configuration a graph calibration probe runs, backed by the smallest Context that holds
// one record per lane (Inputs) or the startup working set.
[[nodiscard]] ExecutionPlan plan_graph_probe(std::shared_ptr<const ModelProfile>,
                                             const EngineOptions&, DeviceContext&,
                                             GraphProbeBacking = GraphProbeBacking::Inputs);

} // namespace ninfer::models::spark2_5
