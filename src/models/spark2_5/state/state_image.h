#pragma once

#include "core/cyclic_kv_cache.h"
#include "models/spark2_5/config.h"
#include "models/spark2_5/state/window_state.h"

#include <cstdint>

namespace ninfer::models::spark2_5 {

// One StateImage record per resident sequence: the sliding-layer rings and the final-norm hidden
// of the newest processed token, from which an exact-frontier restore samples without recomputing.
struct StateLayout {
    StateImageLayout images;
    CyclicKVCacheLayout window;
    StateField sample_hidden;
};

[[nodiscard]] StateLayout plan_state_image(const TextConfig&);

class StateViews {
public:
    StateViews(MemorySpan, const StateLayout&);

    [[nodiscard]] StateImageStorage& storage() { return storage_; }

    [[nodiscard]] const StateImageStorage& storage() const { return storage_; }

    [[nodiscard]] const CyclicKVCacheView& window() const { return window_; }

    [[nodiscard]] CyclicKVCacheView& window() { return window_; }

    [[nodiscard]] StateFieldView sample_hidden(Tensor offsets) const;
    [[nodiscard]] Tensor sample_hidden(std::int64_t offset) const;

private:
    StateLayout layout_;
    StateImageStorage storage_;
    CyclicKVCacheView window_;
};

} // namespace ninfer::models::spark2_5
