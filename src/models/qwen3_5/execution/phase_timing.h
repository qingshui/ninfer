#pragma once

// Opt-in decode phase interval clock for the Qwen3.5 Main traversal.
//
// The decode-step benchmark arms the process-wide instance before it constructs an Engine.
// Eager rounds record one CUDA event pair per phase segment on the execution stream; CUDA
// graph capture records the same boundaries as external nodes, so every replay of a captured
// bucket re-records them. Until armed, the dispatch hooks only perform one pointer load.

#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

enum class DecodePhase : std::uint8_t { Attention, Gdn, Moe };

inline constexpr std::size_t kDecodePhaseCount = 3;

class DecodePhaseClock {
public:
    // Both capacities count phase segments, not events. The eager ring holds the segments of
    // one measured round (plus headroom); the capture pool holds every segment embedded into
    // captured graphs and must cover every bucket and profile capture of every armed engine.
    DecodePhaseClock(std::size_t ring_capacity, std::size_t capture_capacity);
    ~DecodePhaseClock();
    DecodePhaseClock(const DecodePhaseClock&)            = delete;
    DecodePhaseClock& operator=(const DecodePhaseClock&) = delete;

    // Installs the process-wide clock and retains it: captured graphs embed its events, so the
    // slot — not the caller — is the owner for the remaining process lifetime.
    static void arm(std::size_t ring_capacity, std::size_t capture_capacity);
    [[nodiscard]] static DecodePhaseClock* current() noexcept;

    // Opens one measured eager round: closes any segment a cancelled traversal left open and
    // resets the ring cursor.
    void begin_round();

    // One segment spans enter/exit on `stream`. Inside a stream capture the boundaries become
    // external nodes of the captured graph, so each replay refreshes the segment's events.
    void enter(DecodePhase phase, cudaStream_t stream);
    void exit(cudaStream_t stream);

    // Segments the current eager round recorded; eager_milliseconds sums one phase over them
    // and throws when the round overflowed the ring (a segment was overwritten before reading).
    [[nodiscard]] std::size_t eager_segment_count() const noexcept { return ring_used_; }
    [[nodiscard]] DecodePhase eager_phase(std::size_t index) const;
    [[nodiscard]] double eager_milliseconds(DecodePhase phase) const;

    // Capture pool reading. A replay refreshes the events of the captured bodies it launches;
    // since a bucket's MoE layer graphs pick their subgraph per round by routing state, the
    // reader cannot know a fresh position statically. Instead it reports the segments whose
    // interval changed since the previous read: a first read only establishes the baseline.
    [[nodiscard]] std::size_t capture_segment_count() const noexcept { return captured_; }
    // Segment count of each contiguous stream-capture interval, in capture order. The interval
    // under capture is included; it closes when a different graph's capture begins.
    [[nodiscard]] std::vector<std::size_t> capture_intervals() const {
        auto intervals = capture_intervals_;
        if (interval_graph_) intervals.push_back(captured_ - interval_start_);
        return intervals;
    }
    struct CaptureSample {
        std::size_t index;
        DecodePhase phase;
        double milliseconds;
    };
    [[nodiscard]] std::vector<CaptureSample> read_capture();
    // Interval of one capture segment by pool position; absent until its events completed.
    [[nodiscard]] std::optional<double> capture_milliseconds_at(std::size_t index) const;
    [[nodiscard]] DecodePhase capture_phase_at(std::size_t index) const;

private:
    struct Segment {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop  = nullptr;
        DecodePhase phase = DecodePhase::Attention;
    };

    void record(cudaEvent_t event, cudaStream_t stream, bool capturing) const;

    std::vector<Segment> ring_;
    std::vector<Segment> capture_pool_;
    // Interval each capture segment reported at the previous read; absent before the first one.
    std::vector<std::optional<double>> capture_baseline_;
    std::vector<std::size_t> capture_intervals_;
    std::size_t interval_start_  = 0;
    cudaGraph_t interval_graph_  = nullptr;
    std::size_t ring_used_     = 0;
    bool ring_overflowed_      = false;
    std::size_t captured_      = 0;
    Segment* open_             = nullptr;
    bool open_capturing_       = false;
};

} // namespace ninfer::models::qwen3_5::execution
