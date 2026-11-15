// T05 unit coverage of the per-compute-frame execution storage behind the double-buffered
// pipeline: the layout replicates the frame region once per compute frame, and the storage of one
// frame (pinned lane tables, device regions, workspace) shares no byte with another frame's, so
// the host can fill frame N+1 while the device still reads frame N. The pipeline itself (two
// flights in flight, per-ticket routing, results) runs end to end over a real artifact in the
// engine tests and the serve benchmark.

#include "core/buffer.h"
#include "core/device.h"
#include "core/layout.h"
#include "models/qwen3_5/program/frames.h"
#include "models/qwen3_5/program/resources.h"
#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>

namespace {

namespace q35 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

q35::FrameSpec frame_spec() {
    return q35::FrameSpec{
        .hidden              = 8,
        .output_rows         = 32,
        .public_tokens       = 32,
        .prefill_chunk       = 128,
        .pack_tokens         = 256,
        .pack_segments       = 2,
        .mixed_decode_rows   = 0,
        .mixed_tokens        = 0,
        .batch_capacity      = 8,
        .draft_window        = 0,
        .backend             = ninfer::SpeculativeBackend::None,
        .causal_scoring      = false,
        .vision              = false,
        .control_token_limit = 0,
    };
}

q35::ResourceSpec resource_spec(std::uint32_t compute_frames) {
    q35::ResourceSpec spec;
    spec.decoder = {
        .full_attention_layers = 2,
        .mtp_layers            = 1,
        .capacity              = 129,
        .kv_heads              = 2,
        .attention_head_dim    = 256,
        .kv_storage            = ninfer::KvCacheStorage::BFloat16,
        .enable_mtp            = false,
        .kv_table_rows         = 8,
    };
    spec.state_images = {
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
        .hidden = 8,
    };
    spec.frame          = frame_spec();
    spec.compute_frames = compute_frames;
    return spec;
}

[[nodiscard]] bool disjoint(const ninfer::LayoutRegion& a, const ninfer::LayoutRegion& b) {
    return a.offset + a.bytes <= b.offset || b.offset + b.bytes <= a.offset;
}

// One persistent layout carries one frame region per compute frame, none overlapping another, and
// the pinned host demand grows by exactly one frame mirror per frame.
void test_layout_replicates_frames() {
    const auto single = q35::plan_resources(resource_spec(1));
    const auto pair   = q35::plan_resources(resource_spec(2));
    expect(single.frames.size() == 1 && pair.frames.size() == 2,
           "the layout holds one frame layout per compute frame");
    const auto& a = pair.frames[0];
    const auto& b = pair.frames[1];
    expect(disjoint(a.ingress, b.ingress) && disjoint(a.egress, b.egress) &&
               disjoint(a.commit, b.commit),
           "lane tables of two frames do not overlap");
    expect(disjoint(a.hidden.region, b.hidden.region) &&
               disjoint(a.input_ids.region, b.input_ids.region) &&
               disjoint(a.cache_positions.region, b.cache_positions.region),
           "token-axis planes of two frames do not overlap");
    expect(pair.bytes > single.bytes, "a second frame region extends the persistent layout");
    expect(pair.pinned_bytes() == single.pinned_bytes() + pair.frames[1].host.bytes,
           "pinned host demand grows by one frame mirror per frame");
    expect(pair.frames[0].host.bytes == pair.frames[1].host.bytes,
           "every frame has the same pinned footprint");
}

// A layout without a compute frame has no execution storage to run on.
void test_layout_rejects_zero_frames() {
    bool rejected = false;
    try {
        (void)q35::plan_resources(resource_spec(0));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "zero compute frames is rejected");
}

// Two frames built over one persistent backing: host tables written on frame 0 never show up on
// frame 1, the device regions and the workspaces are separate allocations.
void test_frame_storage_is_independent(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    const auto layout0 = q35::plan_frames(builder, frame_spec());
    const auto layout1 = q35::plan_frames(builder, frame_spec());
    const auto bytes   = builder.finish(256, "compute frame test backing");
    ninfer::DeviceBuffer backing(device.device(), bytes);
    q35::WorkspaceSpec workspace;
    workspace.general_capacity = 1 << 16;
    workspace.capacity         = 1 << 16;
    q35::ComputeFrameResources first(device, backing.span(), layout0, workspace, 0);
    q35::ComputeFrameResources second(device, backing.span(), layout1, workspace, 0);

    first.frame.clear_host_ingress();
    second.frame.clear_host_ingress();
    first.frame.host_ingress->anchors[0]    = 11;
    first.frame.host_ingress->frontiers[3]  = 17;
    second.frame.host_ingress->anchors[0]   = 22;
    expect(first.frame.host_ingress->anchors[0] == 11 && second.frame.host_ingress->anchors[0] == 22,
           "the pinned ingress mirrors of two frames are separate tables");
    expect(second.frame.host_ingress->frontiers[3] == 0,
           "a write to one frame's host table leaves the other frame's table untouched");
    expect(first.frame.host_ingress_region.data != second.frame.host_ingress_region.data,
           "the pinned ingress regions are separate allocations");
    expect(first.frame.anchors.data != second.frame.anchors.data &&
               first.frame.hidden.data != second.frame.hidden.data &&
               first.frame.output_tokens.data != second.frame.output_tokens.data,
           "device tables and planes of two frames sit at distinct addresses");

    first.work.reset();
    second.work.reset();
    auto from_first  = first.work.alloc(ninfer::DType::BF16, {64, 4});
    auto from_second = second.work.alloc(ninfer::DType::BF16, {64, 4});
    expect(from_first.data != from_second.data &&
               first.workspace_storage.span().data != second.workspace_storage.span().data,
           "each frame allocates its workspace from its own storage");
    first.work.reset();
    expect(second.work.used() > 0 && first.work.used() == 0,
           "resetting one frame's workspace leaves the other frame's arena intact");
}

} // namespace

int main() {
    try {
        test_layout_replicates_frames();
        test_layout_rejects_zero_frames();

        int count                   = 0;
        const cudaError_t count_err = cudaGetDeviceCount(&count);
        if (cuda_unavailable(count_err) || count == 0) {
            std::cout << (failures == 0 ? "SKIP device part: no usable CUDA device\n" : "");
        } else {
            CUDA_CHECK(count_err);
            ninfer::DeviceContext device(0);
            test_frame_storage_is_independent(device);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
