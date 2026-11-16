// T2 unit coverage of the packed segment table contract: the work descriptors (checkpoints and
// installs), the frame ingress sized for the decode rows plus the prefill segments, and the host
// and device mirrors of the varlen tables. The execution of the table runs over a real artifact
// in test_packed_prefill_real.

#include "core/buffer.h"
#include "core/device.h"
#include "core/layout.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/prefill_chunk.h"
#include "models/qwen3_5/program/frames.h"
#include "models/qwen3_5/program/resources.h"
#include "models/qwen3_5/work.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

namespace q35 = ninfer::models::qwen3_5;

static_assert(std::is_same_v<decltype(q35::WindowCheckpoint::column), std::uint32_t>);
static_assert(
    std::is_same_v<decltype(q35::WindowCheckpoint::state), ninfer::models::context::StateId>);
static_assert(std::is_same_v<decltype(q35::WindowCheckpoint::state_offset), std::int64_t>);
static_assert(std::is_same_v<decltype(q35::InputWindow::checkpoints),
                             std::vector<q35::WindowCheckpoint>>);
static_assert(std::is_same_v<decltype(q35::Work::installs), std::vector<q35::ResidencyBinding>>);

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q35::FrameSpec mixed_spec(std::uint32_t lanes, std::uint32_t segments) {
    return q35::FrameSpec{
        .hidden              = 8,
        .output_rows         = 32,
        .public_tokens       = 32,
        .prefill_chunk       = 128,
        .pack_tokens         = 256,
        .pack_segments       = segments,
        .mixed_decode_rows   = lanes,
        .mixed_tokens        = q35::resolve_mixed_tokens(256, lanes, 0),
        .batch_capacity      = lanes,
        .draft_window        = 0,
        .backend             = ninfer::SpeculativeBackend::None,
        .causal_scoring      = false,
        .vision              = false,
        .control_token_limit = 0,
    };
}

template <class F> bool throws_invalid(F&& action) {
    try {
        action();
    } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

// The decode rows of a mixed execution are bounded by the lane count, not by the pack segment
// budget, and the ingress row metadata holds the decode rows and the segments side by side.
void test_mixed_rows_follow_the_lane_count() {
    const auto spec = mixed_spec(8, 2);
    expect(spec.ingress_rows() == 10, "ingress rows are the decode rows plus the segments");
    ninfer::LayoutBuilder builder;
    const auto layout = q35::plan_frames(builder, spec);
    expect(layout.ingress_arrays.lanes == spec.pack_segments + spec.mixed_decode_rows,
           "the frame ingress holds pack_segments + mixed_decode_rows rows");
    expect(layout.ingress_arrays.boundaries == q35::kPrefillPackBoundaryLimit,
           "the frame ingress holds the boundary capacity of one packed execution");

    auto wide              = mixed_spec(8, 2);
    wide.mixed_decode_rows = 9;
    wide.mixed_tokens      = q35::resolve_mixed_tokens(256, 9, 0);
    ninfer::LayoutBuilder other;
    expect(throws_invalid([&] { (void)q35::plan_frames(other, wide); }),
           "more decode rows than lanes is rejected");

    // A frame that does not mix keeps one ingress row per lane.
    auto decode_only              = mixed_spec(8, 1);
    decode_only.mixed_decode_rows = 0;
    decode_only.mixed_tokens      = 0;
    expect(decode_only.ingress_rows() == 8, "a frame that does not mix keeps the lane count");
}

// The host mirror and the device region bind the same tables at the same offsets.
void test_tables_mirror(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    const auto layout = q35::plan_frames(builder, mixed_spec(4, 3));
    const auto bytes  = builder.finish(256, "segment table test backing");
    ninfer::DeviceBuffer backing(device.device(), bytes);
    q35::WorkspaceSpec workspace;
    workspace.general_capacity = 1 << 16;
    workspace.capacity         = 1 << 16;
    q35::ComputeFrameResources frame(device, backing.span(), layout, workspace, 0);
    auto& host      = *frame.frame.host_ingress;
    const auto rows = layout.ingress_arrays.lanes;
    expect(host.segment_offsets.size() == rows + 1 &&
               host.boundary_rows.size() == q35::kPrefillPackBoundaryLimit &&
               host.boundary_columns.size() == q35::kPrefillPackBoundaryLimit &&
               host.boundary_destinations.size() == q35::kPrefillPackBoundaryLimit,
           "host spans cover the segment table");
    expect(frame.frame.segment_offsets.ne[0] == static_cast<std::int32_t>(rows) + 1 &&
               frame.frame.boundary_destinations.ne[0] ==
                   static_cast<std::int32_t>(q35::kPrefillPackBoundaryLimit),
           "device tensors cover the segment table");

    frame.frame.clear_host_ingress();
    host.segment_offsets[2]       = 300;
    host.boundary_rows[5]         = 1;
    host.boundary_columns[5]      = 77;
    host.boundary_destinations[5] = 123456789012LL;
    host.boundary_destinations[6] = -1;
    CUDA_CHECK(cudaMemcpy(frame.frame.ingress.data, frame.frame.host_ingress_region.data,
                          frame.frame.ingress.bytes, cudaMemcpyHostToDevice));
    std::int32_t offset = 0, row = 0, column = 0;
    std::int64_t destination = 0, none = 0;
    const auto word = [](const ninfer::Tensor& tensor, std::size_t index) {
        return static_cast<const std::byte*>(tensor.data) + index * ninfer::dtype_size(tensor.dtype);
    };
    CUDA_CHECK(cudaMemcpy(&offset, word(frame.frame.segment_offsets, 2), sizeof(offset),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&row, word(frame.frame.boundary_rows, 5), sizeof(row),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&column, word(frame.frame.boundary_columns, 5), sizeof(column),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&destination, word(frame.frame.boundary_destinations, 5),
                          sizeof(destination), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&none, word(frame.frame.boundary_destinations, 6), sizeof(none),
                          cudaMemcpyDeviceToHost));
    expect(offset == 300 && row == 1 && column == 77 && destination == 123456789012LL &&
               none == -1,
           "the uploaded ingress carries the segment table to the device tensors");
    expect(reinterpret_cast<std::uintptr_t>(frame.frame.boundary_destinations.data) %
                   alignof(std::int64_t) ==
               0,
           "the boundary destinations are 8-byte aligned");
}

} // namespace

int main() {
    try {
        test_mixed_rows_follow_the_lane_count();
        int count                   = 0;
        const cudaError_t count_err = cudaGetDeviceCount(&count);
        if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver ||
            count == 0) {
            std::cout << "SKIP device part: no usable CUDA device\n";
        } else {
            CUDA_CHECK(count_err);
            ninfer::DeviceContext device(0);
            test_tables_mirror(device);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) std::cout << "ok\n";
    return failures == 0 ? 0 : 1;
}
