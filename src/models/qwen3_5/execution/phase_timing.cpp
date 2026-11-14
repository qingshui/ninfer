#include "models/qwen3_5/execution/phase_timing.h"

#include <atomic>
#include <memory>

namespace ninfer::models::qwen3_5::execution {
namespace {

DecodePhaseClock*& armed_clock() noexcept {
    static DecodePhaseClock* value = nullptr;
    return value;
}

bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

} // namespace

DecodePhaseClock::DecodePhaseClock(std::size_t ring_capacity, std::size_t capture_capacity) {
    if (!ring_capacity) throw std::invalid_argument("decode phase ring capacity must be positive");
    ring_.resize(ring_capacity);
    capture_pool_.resize(capture_capacity);
    capture_baseline_.resize(capture_capacity);
    for (auto* pool : {&ring_, &capture_pool_}) {
        for (auto& segment : *pool) {
            CUDA_CHECK(cudaEventCreate(&segment.start));
            CUDA_CHECK(cudaEventCreate(&segment.stop));
        }
    }
}

DecodePhaseClock::~DecodePhaseClock() {
    if (armed_clock() == this) armed_clock() = nullptr;
    for (auto* pool : {&ring_, &capture_pool_}) {
        for (auto& segment : *pool) {
            (void)cudaEventDestroy(segment.start);
            (void)cudaEventDestroy(segment.stop);
        }
    }
}

void DecodePhaseClock::arm(std::size_t ring_capacity, std::size_t capture_capacity) {
    auto& slot = armed_clock();
    if (slot) throw std::logic_error("a decode phase clock is already armed");
    // The slot owns the clock: captured graphs embed its events for the engine lifetime and
    // only the benchmark process ever arms one, so the instance lives until process exit.
    slot = new DecodePhaseClock(ring_capacity, capture_capacity);
}

DecodePhaseClock* DecodePhaseClock::current() noexcept { return armed_clock(); }

void DecodePhaseClock::begin_round() {
    open_           = nullptr;
    open_capturing_ = false;
    ring_used_      = 0;
    ring_overflowed_ = false;
}

void DecodePhaseClock::enter(DecodePhase phase, cudaStream_t stream) {
    if (open_) throw std::logic_error("decode phase segment is already open");
    const bool capturing = stream_is_capturing(stream);
    Segment* segment      = nullptr;
    if (capturing) {
        if (captured_ == capture_pool_.size()) {
            throw std::runtime_error("decode phase capture pool is exhausted");
        }
        // Consecutive DecodeGraphDefinition captures share one stream; the graph under capture
        // identifies the interval each segment belongs to. The status out-parameter is required
        // (CUDA 13 runtime rejects nullptr with cudaErrorInvalidValue).
        cudaStreamCaptureStatus interval_status = cudaStreamCaptureStatusNone;
        cudaGraph_t graph                       = nullptr;
        CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &interval_status, nullptr, &graph, nullptr,
                                            nullptr, nullptr));
        if (graph != interval_graph_) {
            if (interval_graph_) capture_intervals_.push_back(captured_ - interval_start_);
            interval_graph_ = graph;
            interval_start_ = captured_;
        }
        segment = &capture_pool_[captured_++];
    } else {
        if (ring_used_ == ring_.size()) {
            ring_overflowed_ = true;
            ring_used_       = 0;
        }
        segment = &ring_[ring_used_++];
    }
    segment->phase = phase;
    record(segment->start, stream, capturing);
    open_            = segment;
    open_capturing_  = capturing;
}

void DecodePhaseClock::exit(cudaStream_t stream) {
    if (!open_) return;
    record(open_->stop, stream, open_capturing_);
    open_           = nullptr;
    open_capturing_ = false;
}

DecodePhase DecodePhaseClock::eager_phase(std::size_t index) const {
    if (index >= ring_used_) throw std::out_of_range("eager decode phase segment is not recorded");
    return ring_[index].phase;
}

double DecodePhaseClock::eager_milliseconds(DecodePhase phase) const {
    if (ring_overflowed_) {
        throw std::runtime_error("decode phase ring overflowed within one round");
    }
    double total = 0;
    for (std::size_t index = 0; index < ring_used_; ++index) {
        const auto& segment = ring_[index];
        if (segment.phase != phase) continue;
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, segment.start, segment.stop));
        total += milliseconds;
    }
    return total;
}

std::vector<DecodePhaseClock::CaptureSample> DecodePhaseClock::read_capture() {
    std::vector<CaptureSample> fresh;
    for (std::size_t index = 0; index < captured_; ++index) {
        const auto& segment = capture_pool_[index];
        if (cudaEventQuery(segment.stop) != cudaSuccess) continue;
        float milliseconds = 0;
        if (cudaEventElapsedTime(&milliseconds, segment.start, segment.stop) != cudaSuccess)
            continue;
        const double value = milliseconds;
        const auto& baseline = capture_baseline_[index];
        if (baseline && *baseline == value) continue;
        capture_baseline_[index] = value;
        fresh.push_back({index, segment.phase, value});
    }
    return fresh;
}

std::optional<double> DecodePhaseClock::capture_milliseconds_at(std::size_t index) const {
    if (index >= captured_) {
        throw std::out_of_range("captured decode phase segment is not recorded");
    }
    const auto& segment = capture_pool_[index];
    float milliseconds = 0;
    if (cudaEventQuery(segment.stop) != cudaSuccess) return std::nullopt;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, segment.start, segment.stop));
    return static_cast<double>(milliseconds);
}

DecodePhase DecodePhaseClock::capture_phase_at(std::size_t index) const {
    if (index >= captured_) {
        throw std::out_of_range("captured decode phase segment is not recorded");
    }
    return capture_pool_[index].phase;
}

void DecodePhaseClock::record(cudaEvent_t event, cudaStream_t stream, bool capturing) const {
    if (capturing) {
        CUDA_CHECK(cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal));
        return;
    }
    CUDA_CHECK(cudaEventRecord(event, stream));
}

} // namespace ninfer::models::qwen3_5::execution
