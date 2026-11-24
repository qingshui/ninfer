#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>

namespace ninfer::runtime {

// Bounds the resident lanes of an automatically fitted Engine by the cadence of its decode lanes.
//
// Automatic lane fitting sizes the lane table to the memory that holds it. Under saturation every
// resident lane then waits for one flight of all the others between two tokens, so the token
// interval grows with the number of resident lanes much faster than the work per second does. The
// governor holds the resident-lane quota where the mean token interval sits at the target: it
// admits more while the interval is below the target and the quota holds admissions back, and
// fewer once the interval is above it. A lowered quota never evicts a resident; it holds new
// admissions back until the residents finish.
//
// A load that misses the target by a wide margin with the residents held to the floor while
// admissions wait is bound by its prefill work rather than by its decode lanes (a single-sample
// load whose flights all carry long prefills, for example). The floor still shortens its token
// interval, but costs it much of its throughput and time to first token while the target stays
// out of reach. Once that has lasted `release_after`, the governor releases every lane for
// `release_for` (a window that meets the target ends the release early) and then governs again.
// An Engine that runs out of residents calls `idle`, so that the next load starts from the floor
// rather than from the quota or the release the previous load left.
//
// The owner reports the token intervals and the admissions the quota held back, closes the
// observation window with `update` at a boundary of its own choosing and reads the quota. A window
// that held an admission back asks the owner to check its queue again, so that demand the quota
// keeps waiting reports itself in the next window too. The governor reads no clock.
class LaneGovernor {
public:
    using Clock    = std::chrono::steady_clock;
    using Duration = Clock::duration;

    struct Policy {
        // Mean token interval of a decode lane to hold.
        Duration target;
        // Observation time per adjustment, and the tokens it needs to adjust.
        Duration window;
        std::uint32_t min_samples;
        // Relative dead band around the target.
        double band;
        // Share of the proportional correction one window applies, and the largest relative
        // change of the quota one window may make.
        double gain;
        double max_step;
        // How long an unreachable target lasts before every lane is released, and for how long.
        Duration release_after;
        Duration release_for;
    };

    // The quota starts at `floor` and moves within [floor, capacity]. A floor at the capacity
    // leaves nothing to govern: the quota stays at the capacity.
    LaneGovernor(std::uint32_t capacity, std::uint32_t floor, Policy policy)
        : capacity_(std::max<std::uint32_t>(capacity, 1)),
          floor_(std::clamp<std::uint32_t>(floor, 1, capacity_)), quota_(floor_), policy_(policy) {}

    [[nodiscard]] std::uint32_t quota() const noexcept { return quota_; }

    // A decode lane committed `tokens` tokens `interval` after its previous commit.
    void observe(Duration interval, std::uint32_t tokens) noexcept {
        if (tokens == 0) return;
        // A lane that sat out a pause or a preemption must not dominate the mean.
        sum_ += std::min(interval, policy_.target * (kOutlierTargets * tokens));
        count_ += tokens;
    }

    // An admission waited for the quota: a free lane existed but the quota did not grant it.
    void hold() noexcept { held_ = true; }

    // The Engine has no lane in use. The next load starts from the floor, as on a fresh Engine:
    // a quota or a release that suited the previous load must not admit the next one's burst.
    void idle() noexcept {
        quota_ = floor_ >= capacity_ ? capacity_ : floor_;
        unreachable_since_.reset();
        released_until_.reset();
        window_start_.reset();
        sum_   = Duration::zero();
        count_ = 0;
        held_  = false;
    }

    // Closes the observation window once it has run for `policy.window` and adjusts the quota from
    // it. `occupied` is the number of lanes in use. Returns whether the closed window held an
    // admission back.
    [[nodiscard]] bool update(Clock::time_point now, std::uint32_t occupied) noexcept {
        if (floor_ >= capacity_) return false;
        if (!window_start_) {
            restart(now);
            return false;
        }
        const auto started = *window_start_;
        const auto elapsed = now - started;
        if (elapsed < policy_.window) return false;
        if (count_ < policy_.min_samples) {
            // Samples that straggle in across a long idle gap describe an earlier load.
            if (elapsed > policy_.window * kStaleWindows) restart(now);
            return false;
        }
        const double mean = static_cast<double>(sum_.count()) / static_cast<double>(count_);
        const bool held   = held_;
        restart(now);
        const double target = static_cast<double>(policy_.target.count());
        const bool slow     = mean > target * (1.0 + policy_.band);
        const bool fast     = mean < target * (1.0 - policy_.band);
        if (released_until_) {
            if (!fast && now < *released_until_) return held;
            released_until_.reset();
        }
        if (quota_ == floor_ && held && occupied <= floor_ && mean > target * kUnreachable) {
            if (!unreachable_since_) unreachable_since_ = started;
            if (now - *unreachable_since_ >= policy_.release_after) {
                unreachable_since_.reset();
                released_until_ = now + policy_.release_for;
                quota_          = capacity_;
                return held;
            }
        } else {
            unreachable_since_.reset();
        }
        if (slow) {
            // Too slow. A quota below the residents is still draining toward itself; only a
            // quota the load already fills has to come down.
            if (occupied <= quota_) {
                const double step =
                    std::min(policy_.max_step, policy_.gain * (1.0 - target / mean));
                const double base = static_cast<double>(std::max(occupied, floor_));
                const auto lanes  = static_cast<std::uint32_t>(base - std::max(1.0, step * base));
                quota_            = std::max(floor_, std::min(quota_, lanes));
            }
        } else if (fast && held) {
            // Fast enough while the quota held admissions back: those lanes could ride too.
            const double step  = std::min(policy_.max_step, policy_.gain * (target / mean - 1.0));
            const double grown = static_cast<double>(quota_) +
                                 std::max(1.0, step * static_cast<double>(quota_));
            quota_ = std::min<std::uint32_t>(capacity_, static_cast<std::uint32_t>(grown));
        }
        return held;
    }

private:
    // A token interval counts at most this many targets.
    static constexpr std::int64_t kOutlierTargets = 4;
    // A window that has run this many windows without enough samples starts over.
    static constexpr std::int64_t kStaleWindows = 8;
    // A mean interval this many targets long at the floor is out of reach, not a band excursion.
    static constexpr double kUnreachable = 1.25;

    void restart(Clock::time_point now) noexcept {
        window_start_ = now;
        sum_          = Duration::zero();
        count_        = 0;
        held_         = false;
    }

    std::uint32_t capacity_;
    std::uint32_t floor_;
    std::uint32_t quota_;
    Policy policy_;
    std::optional<Clock::time_point> window_start_;
    Duration sum_{};
    std::uint64_t count_ = 0;
    bool held_           = false;
    std::optional<Clock::time_point> unreachable_since_;
    std::optional<Clock::time_point> released_until_;
};

} // namespace ninfer::runtime
