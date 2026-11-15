#include "models/qwen3_5/state/state_image.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5 {

StateImageDeviceLayout plan_state_image_layout(const StateImageSpec& spec) {
    StateImageBuilder builder;
    StateImageDeviceLayout layout;
    layout.linear              = plan_linear_attention_state(builder, spec.linear);
    layout.continuation_hidden = builder.add(DType::BF16, {spec.hidden}, 1, "Continuation hidden");
    if (spec.dflash_local) {
        const auto& local   = *spec.dflash_local;
        layout.dflash_local = plan_cyclic_kv_cache(builder, local.layers, local.capacity,
                                                   local.kv_heads, local.head_dim);
    }
    layout.images = builder.finish();
    return layout;
}

StateImageViews::StateImageViews(MemorySpan backing, const StateImageDeviceLayout& layout,
                                 std::uint32_t staging_images)
    : storage_(backing, layout.images), linear_(storage_, layout.linear),
      hidden_(layout.continuation_hidden), staging_rows_(staging_images) {
    if (layout.dflash_local) {
        dflash_local_.emplace(storage_, *layout.dflash_local);
        dflash_group_ = layout.dflash_local->group;
    }
    // The arena tail is the one scratch State row followed by the staging rows: pool records
    // never reach them, and the scratch offset doubles as the pool-managed extent the session's
    // state pool and graph preparation allocate below (see context_spec and Resources).
    const auto tail = (std::size_t{1} + staging_images) * storage_.image_bytes();
    if (backing.bytes < tail) {
        throw std::invalid_argument("StateImage arena must hold its scratch and staging rows");
    }
    pool_bytes_ = backing.bytes - tail;
    // Validates the scratch row and every staging row are aligned, in-bounds images.
    (void)storage_.image(static_cast<std::int64_t>(pool_bytes_));
    for (std::uint32_t row = 0; row < staging_images; ++row) (void)storage_.image(staging_row(row));
}

std::int64_t StateImageViews::staging_row(std::uint32_t index) const {
    if (index >= staging_rows_) throw std::out_of_range("StateImage staging row is out of range");
    return static_cast<std::int64_t>(pool_bytes_ + (std::size_t{1} + index) * storage_.image_bytes());
}

Tensor StateImageViews::continuation_hidden(std::int64_t offset) const {
    return storage_.tensor(hidden_, offset);
}

StateFieldView StateImageViews::continuation_hidden(const Tensor& offsets) const {
    return storage_.field(hidden_, offsets);
}

CyclicKVCacheView* StateImageViews::dflash_local() noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

const CyclicKVCacheView* StateImageViews::dflash_local() const noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

TransferWork StateImageViews::dflash_local_transfer_work() const {
    if (!dflash_group_) { throw std::logic_error("StateImage has no DFlash local state"); }
    return {.payload_bytes = dflash_group_->bytes, .copy_operations = 1};
}

} // namespace ninfer::models::qwen3_5
