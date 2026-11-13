#include "models/spark2_5/state/state_image.h"

namespace ninfer::models::spark2_5 {

StateLayout plan_state_image(const TextConfig& config) {
    StateImageBuilder builder;
    StateLayout out;
    out.window        = plan_window_state(builder, config.sliding_layers);
    out.sample_hidden = builder.add(DType::BF16, {dimension(config.hidden)}, 1, "Main sample hidden");
    out.images        = builder.finish();
    return out;
}

StateViews::StateViews(MemorySpan backing, const StateLayout& layout)
    : layout_(layout), storage_(backing, layout.images), window_(storage_, layout.window) {}

StateFieldView StateViews::sample_hidden(Tensor offsets) const {
    return storage_.field(layout_.sample_hidden, offsets);
}

Tensor StateViews::sample_hidden(std::int64_t offset) const {
    return storage_.tensor(layout_.sample_hidden, offset);
}

} // namespace ninfer::models::spark2_5
