#pragma once

#include "models/common/context/storage.h"
#include "models/common/context/transfer.h"
#include "models/spark2_5/program/planning.h"

#include "core/arena.h"
#include "core/buffer.h"
#include "core/startup.h"

#include <memory>

namespace ninfer::models::spark2_5 {

// Fixed rank-local backing. Directory identities and allocation belong to the Session.
// DeviceContext outlives this owner and every submitted consumer retires before it is destroyed.
class Resources {
public:
    Resources(ExecutionPlan, DeviceContext&, const StartupObserver& observer = {});
    ~Resources();
    Resources(const Resources&)            = delete;
    Resources& operator=(const Resources&) = delete;

    void synchronize() const;

    [[nodiscard]] MemorySpan context_backing() { return context_.span(); }

    [[nodiscard]] MemorySpan persistent() { return persistent_.span(); }

    // Host mirror of plan.persistent.buffers.ingress, uploaded as one block.
    [[nodiscard]] MemorySpan ingress() { return ingress_.span(); }

    [[nodiscard]] MemorySpan egress() { return egress_.span(); }

    // Host staging of the grammar masks, laid out like the device tensor.
    [[nodiscard]] MemorySpan sampling_ingress() { return sampling_.span(); }

    // Host scratch for token IDs that are staged to the device or counted.
    [[nodiscard]] MemorySpan compute_aux() { return compute_aux_.span(); }

    [[nodiscard]] MemorySpan context_counts() { return context_counts_.span(); }

    [[nodiscard]] Tensor tensor(const TensorRegion&);
    [[nodiscard]] Tensor packed(const TensorRegion&, std::initializer_list<std::int32_t>);

    const ExecutionPlan plan;

private:
    DeviceContext& device_;
    DeviceBuffer context_, persistent_, workspace_;
    PinnedHostBuffer ingress_, egress_, sampling_, compute_aux_, context_counts_, host_context_;

public:
    MemoryArena workspace;
    StateViews states;
    FullKVCache kv;
    std::unique_ptr<context::ContextStorage> context;
    std::unique_ptr<context::ContextMigration> compute_actions, context_actions;
};

} // namespace ninfer::models::spark2_5
