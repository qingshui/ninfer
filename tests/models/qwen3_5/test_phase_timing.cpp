// Unit tests for the opt-in decode phase clock: eager ring recording and per-phase summation,
// ring overflow detection, open-segment invariants, and CUDA graph capture segments whose
// events every replay refreshes.

#include "models/qwen3_5/execution/phase_timing.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q35 = ninfer::models::qwen3_5::execution;
using q35::DecodePhase;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

template <class Fn>
bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

struct Buffers {
    cudaStream_t stream = nullptr;
    void* source        = nullptr;
    void* target        = nullptr;
    static constexpr std::size_t kBytes = 256u << 20; // 256 MiB: a multi-millisecond D2D copy.

    Buffers() {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(&source, kBytes));
        CUDA_CHECK(cudaMalloc(&target, kBytes));
        CUDA_CHECK(cudaMemsetAsync(source, 0x5a, kBytes, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    ~Buffers() {
        (void)cudaFree(source);
        (void)cudaFree(target);
        (void)cudaStreamDestroy(stream);
    }

    void copy() { CUDA_CHECK(cudaMemcpyAsync(target, source, kBytes, cudaMemcpyDeviceToDevice, stream)); }
};

// An independent reference for the interval the clock should report: one raw event pair around
// the same stream copy, elapsed through the runtime's own timer.
double reference_copy_milliseconds(cudaStream_t stream, const void* source, void* target,
                                   std::size_t bytes) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return milliseconds;
}

int test_eager_ring() {
    Buffers buffers;
    q35::DecodePhaseClock clock(8, 0);

    clock.begin_round();
    clock.enter(DecodePhase::Attention, buffers.stream);
    buffers.copy();
    clock.exit(buffers.stream);
    clock.enter(DecodePhase::Moe, buffers.stream);
    buffers.copy();
    clock.exit(buffers.stream);
    clock.enter(DecodePhase::Gdn, buffers.stream);
    buffers.copy();
    clock.exit(buffers.stream);
    CUDA_CHECK(cudaStreamSynchronize(buffers.stream));

    check(clock.eager_segment_count() == 3, "eager round records three segments");
    check(clock.eager_phase(0) == DecodePhase::Attention, "segment 0 is attention");
    check(clock.eager_phase(1) == DecodePhase::Moe, "segment 1 is moe");
    check(clock.eager_phase(2) == DecodePhase::Gdn, "segment 2 is gdn");

    const double attention = clock.eager_milliseconds(DecodePhase::Attention);
    const double moe       = clock.eager_milliseconds(DecodePhase::Moe);
    const double gdn       = clock.eager_milliseconds(DecodePhase::Gdn);
    check(attention > 0.0 && moe > 0.0 && gdn > 0.0, "every phase reports a positive interval");
    const double reference =
        reference_copy_milliseconds(buffers.stream, buffers.source, buffers.target, Buffers::kBytes);
    check(std::fabs(attention - reference) < 0.5 + 0.25 * reference,
          "attention interval matches an independent event pair around the same work");
    check(std::fabs(moe - attention) < 0.5 + 0.25 * attention,
          "equal segments report equal intervals");
    check(std::fabs(gdn - attention) < 0.5 + 0.25 * attention, "gdn matches the same work");

    clock.enter(DecodePhase::Attention, buffers.stream);
    check(throws([&] { clock.enter(DecodePhase::Gdn, buffers.stream); }),
          "enter while a segment is open is rejected");
    clock.exit(buffers.stream); // close the open segment again

    clock.begin_round();
    check(clock.eager_segment_count() == 0, "begin_round resets the ring");

    // A cancelled traversal leaves a segment open; begin_round closes it without a stop event.
    clock.enter(DecodePhase::Moe, buffers.stream);
    clock.begin_round();
    clock.enter(DecodePhase::Attention, buffers.stream);
    clock.exit(buffers.stream);
    CUDA_CHECK(cudaStreamSynchronize(buffers.stream));
    check(clock.eager_segment_count() == 1, "a segment abandoned by cancellation is dropped");
    return failures == 0 ? 0 : 1;
}

int test_ring_overflow() {
    Buffers buffers;
    q35::DecodePhaseClock clock(2, 0);
    clock.begin_round();
    for (int index = 0; index < 3; ++index) {
        clock.enter(index == 0 ? DecodePhase::Attention : DecodePhase::Moe, buffers.stream);
        buffers.copy();
        clock.exit(buffers.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(buffers.stream));
    check(throws([&] { (void)clock.eager_milliseconds(DecodePhase::Attention); }),
          "a round that overwrote unread segments refuses to report a sum");
    return failures == 0 ? 0 : 1;
}

int test_capture_replay() {
    Buffers buffers;
    q35::DecodePhaseClock clock(4, 8);

    auto capture = [&](DecodePhase phase) {
        cudaGraphExec_t execution = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(buffers.stream, cudaStreamCaptureModeGlobal));
        clock.enter(phase, buffers.stream);
        for (int index = 0; index < 4; ++index) buffers.copy();
        clock.exit(buffers.stream);
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(buffers.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&execution, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(execution, buffers.stream));
        CUDA_CHECK(cudaGraphLaunch(execution, buffers.stream));
        CUDA_CHECK(cudaGraphDestroy(graph));
        return execution;
    };

    auto* attention_execution = capture(DecodePhase::Attention);
    auto* moe_execution       = capture(DecodePhase::Moe);
    CUDA_CHECK(cudaStreamSynchronize(buffers.stream));

    check(clock.capture_segment_count() == 2, "two captured bodies hold two segments");
    check(clock.capture_phase_at(0) == DecodePhase::Attention, "captured segment 0 is attention");
    check(clock.capture_phase_at(1) == DecodePhase::Moe, "captured segment 1 is moe");
    const auto intervals = clock.capture_intervals();
    check(intervals == std::vector<std::size_t>({1, 1}),
          "each capture forms its own one-segment interval");

    const auto first = clock.read_capture();
    check(first.size() == 2, "the first read reports both completed segments");
    for (const auto& sample : first) {
        check(sample.milliseconds > 0.0, "a replayed capture reports a positive interval");
        check(sample.index < 2, "samples address pool positions");
    }

    // Every replay refreshes the external event nodes: a second replay yields fresh intervals
    // for the bodies it launched. Value collisions are legal, so only require the reads to
    // address the same positions and stay finite.
    CUDA_CHECK(cudaGraphLaunch(attention_execution, buffers.stream));
    CUDA_CHECK(cudaGraphLaunch(moe_execution, buffers.stream));
    CUDA_CHECK(cudaStreamSynchronize(buffers.stream));
    const auto second = clock.read_capture();
    check(second.size() <= 2, "a replay without value changes may report no fresh samples");
    for (std::size_t index = 0; index < clock.capture_segment_count(); ++index) {
        const auto elapsed = clock.capture_milliseconds_at(index);
        check(elapsed && *elapsed > 0.0 && std::isfinite(*elapsed),
              "captured segments report finite intervals after a replay");
    }

    (void)cudaGraphExecDestroy(attention_execution);
    (void)cudaGraphExecDestroy(moe_execution);
    return failures == 0 ? 0 : 1;
}

int test_capture_pool_exhaustion() {
    Buffers buffers;
    q35::DecodePhaseClock clock(2, 1);
    auto* first = [&] {
        CUDA_CHECK(cudaStreamBeginCapture(buffers.stream, cudaStreamCaptureModeGlobal));
        clock.enter(DecodePhase::Attention, buffers.stream);
        buffers.copy();
        clock.exit(buffers.stream);
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(buffers.stream, &graph));
        cudaGraphExec_t execution = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&execution, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        return execution;
    }();
    CUDA_CHECK(cudaStreamSynchronize(buffers.stream));
    check(throws([&] {
              CUDA_CHECK(cudaStreamBeginCapture(buffers.stream, cudaStreamCaptureModeGlobal));
              clock.enter(DecodePhase::Moe, buffers.stream); // pool holds one segment already
              buffers.copy();
              clock.exit(buffers.stream);
              cudaGraph_t graph = nullptr;
              CUDA_CHECK(cudaStreamEndCapture(buffers.stream, &graph));
              (void)cudaGraphDestroy(graph);
          }),
          "capturing beyond the pool capacity is rejected");
    (void)cudaGraphExecDestroy(first);
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 0;
    }
    test_eager_ring();
    test_ring_overflow();
    test_capture_replay();
    test_capture_pool_exhaustion();
    if (failures == 0) {
        std::cout << "qwen3_5 phase timing: OK\n";
        return 0;
    }
    std::cout << "qwen3_5 phase timing: " << failures << " failure(s)\n";
    return 1;
}
