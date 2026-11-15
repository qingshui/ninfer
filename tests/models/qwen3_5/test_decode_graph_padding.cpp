// T02 unit coverage of the decode graph padding row mask: the scratch State row a bucket replay
// redirects padding-row GDN State destinations to, and the frame ingress fill that carries it.
// The covering-bucket replay itself runs end to end over a real artifact in
// bench/models/qwen3_5/decode_step_bench.cpp (--mode both compares graph replay and eager).

#include "core/buffer.h"
#include "core/context_pool.h"
#include "core/device.h"
#include "core/layout.h"
#include "models/qwen3_5/program/frames.h"
#include "models/qwen3_5/program/graphs.h"
#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

q36::StateImageDeviceLayout plan_image() {
    const q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden       = 7,
        .dflash_local = std::nullopt,
    };
    return q36::plan_state_image_layout(spec);
}

// The arena is the pool extent plus exactly one scratch image; the pool allocates strictly below
// the scratch row, so no pooled record can overlap what a padding row writes.
void test_scratch_row(ninfer::DeviceContext& device) {
    const auto layout      = plan_image();
    const auto image_bytes = layout.images.image_bytes;
    const auto alignment   = layout.images.alignment;
    expect(image_bytes % alignment == 0, "scratch image is alignment-aligned");

    const auto pool_bytes = 3 * image_bytes;
    ninfer::DeviceBuffer arena(device.device(), pool_bytes + image_bytes);
    q36::StateImageViews views(arena.span(), layout);
    expect(views.pool_bytes() == pool_bytes, "scratch row follows the pool extent");
    expect(views.scratch_row() == static_cast<std::int64_t>(pool_bytes),
           "scratch row offset is the pool extent");

    const std::array specs{ninfer::RecordSpec{image_bytes, alignment}};
    ninfer::ContextPool pool(0, views.pool_bytes(), specs);
    const auto state_class = pool.record_class(0);
    const std::array demand{ninfer::RecordRequest{state_class, 3}};
    auto reserved          = pool.reserve(demand);
    expect(reserved.reservation && reserved.reservation->ready(),
           "pool below the scratch row holds its records");
    for (std::uint32_t record = 0; record < 3; ++record) {
        const auto offset = pool.resolve(reserved.reservation->materialize(state_class)).offset;
        expect(offset + image_bytes <= views.pool_bytes(),
               "pooled record stays below the scratch row");
    }
}

// An arena that cannot hold an aligned scratch image is rejected at construction.
void test_scratch_row_boundaries(ninfer::DeviceContext& device) {
    const auto layout      = plan_image();
    const auto image_bytes = layout.images.image_bytes;
    const auto alignment   = layout.images.alignment;

    ninfer::DeviceBuffer short_arena(device.device(), image_bytes / 2);
    bool rejected = false;
    try {
        q36::StateImageViews views(short_arena.span(), layout);
        (void)views;
    } catch (const std::exception&) {
        rejected = true;
    }
    expect(rejected, "arena without the scratch image is rejected");

    ninfer::DeviceBuffer misaligned_arena(device.device(), 3 * image_bytes + 1);
    rejected = false;
    try {
        q36::StateImageViews views(misaligned_arena.span(), layout);
        (void)views;
    } catch (const std::exception&) {
        rejected = true;
    }
    expect(rejected, "misaligned scratch row is rejected");
}

// clear_host_ingress leaves every unbound row's GDN State destination at the scratch State row
// while the rest of its host row state stays zeroed; a bound row overrides its own destination.
void test_frame_padding_fill(ninfer::DeviceContext& device) {
    const q36::FrameSpec spec{
        .hidden             = 8,
        .output_rows        = 32,
        .public_tokens      = 32,
        .prefill_chunk      = 128,
        .pack_tokens        = 128,
        .pack_segments      = 1,
        .mixed_decode_rows  = 0,
        .mixed_tokens       = 0,
        .batch_capacity     = 8,
        .draft_window       = 0,
        .backend            = ninfer::SpeculativeBackend::None,
        .causal_scoring     = false,
        .vision             = false,
        .control_token_limit = 0,
    };
    ninfer::LayoutBuilder builder;
    const auto layout = q36::plan_frames(builder, spec);
    const auto bytes  = builder.finish(256, "padding test frame");
    ninfer::DeviceBuffer backing(device.device(), bytes);
    const std::int64_t scratch = 5 * static_cast<std::int64_t>(layout.ingress_arrays.bytes);
    q36::ExecutionFrame frame(backing.span(), layout, scratch);

    const auto lanes = layout.ingress_arrays.lanes;
    frame.clear_host_ingress();
    for (std::uint32_t lane = 0; lane < lanes; ++lane) {
        expect(frame.host_ingress->state_destination_offsets[lane] == scratch,
               "unbound row carries the scratch State destination");
        expect(frame.host_ingress->state_source_offsets[lane] == 0,
               "unbound row keeps its zeroed host row state");
        expect(frame.host_ingress->frontiers[lane] == 0 && frame.host_ingress->anchors[lane] == 0 &&
                   frame.host_ingress->main_table_rows[lane] == 0,
               "unbound row keeps its zeroed host row state");
    }

    frame.host_ingress->state_destination_offsets[0] = 42;
    frame.host_ingress->state_destination_offsets[lanes - 1] = 43;
    frame.clear_host_ingress();
    expect(frame.host_ingress->state_destination_offsets[0] == scratch &&
               frame.host_ingress->state_destination_offsets[lanes - 1] == scratch,
           "re-cleared ingress restores the scratch State destination");
}

// Unprepared graphs replay nothing, and the capture buckets cover every batch of the capacity.
void test_capture_buckets() {
    const q36::Graphs graphs({512, 130, 0, ninfer::SpeculativeBackend::None});
    expect(graphs.captured_batch(q36::GraphPhase::Main, 130) == 0,
           "unprepared graphs replay no batch");

    const auto buckets = q36::graph_capture_buckets(130);
    expect(buckets.size() == 9 && buckets.front() == 1 && buckets.back() == 130,
           "capture buckets are the powers of two up to the capacity");
    for (std::uint32_t batch = 1; batch <= 130; ++batch) {
        const auto found = std::find_if(buckets.begin(), buckets.end(),
                                        [&](auto bucket) { return bucket >= batch; });
        expect(found != buckets.end(),
               "every batch of the capacity has a covering capture bucket");
    }
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    ninfer::DeviceContext device(0);
    test_scratch_row(device);
    test_scratch_row_boundaries(device);
    test_frame_padding_fill(device);
    test_capture_buckets();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
