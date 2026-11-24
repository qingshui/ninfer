#include "runtime/engine/lane_governor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::runtime::LaneGovernor;
using std::chrono::microseconds;
using std::chrono::milliseconds;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr std::uint32_t kCapacity = 358;
constexpr std::uint32_t kFloor    = 44;

LaneGovernor::Policy policy() {
    return {.target        = milliseconds(23),
            .window        = milliseconds(150),
            .min_samples   = 32,
            .band          = 0.04,
            .gain          = 0.6,
            .max_step      = 0.25,
            .release_after = std::chrono::seconds(2),
            .release_for   = std::chrono::seconds(20)};
}

struct Window {
    microseconds interval;
    std::uint32_t occupied;
    bool held           = true;
    std::uint32_t count = 64;
};

// Feeds one window of `count` single-token intervals and closes it; returns what update returns.
bool run_window(LaneGovernor& governor, LaneGovernor::Clock::time_point& now, const Window& w) {
    for (std::uint32_t i = 0; i < w.count; ++i) governor.observe(w.interval, 1);
    if (w.held) governor.hold();
    now += milliseconds(150);
    return governor.update(now, w.occupied);
}

// Opens the first window; the governor adjusts from the next one on.
LaneGovernor::Clock::time_point open(LaneGovernor& governor) {
    const auto start = LaneGovernor::Clock::time_point{};
    require(!governor.update(start, 0), "the first update only opens a window");
    return start;
}

void grow_to(LaneGovernor& governor, LaneGovernor::Clock::time_point& now, std::uint32_t lanes) {
    for (int i = 0; i < 64 && governor.quota() < lanes; ++i)
        (void)run_window(governor, now, {microseconds(5000), governor.quota()});
    require(governor.quota() >= lanes, "the quota did not grow to the requested lanes");
}

void test_starts_at_the_floor_and_waits_for_a_window() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    require(governor.quota() == kFloor, "the quota starts at the floor");
    auto now = open(governor);
    for (int i = 0; i < 64; ++i) governor.observe(milliseconds(10), 1);
    governor.hold();
    require(!governor.update(now + milliseconds(100), kFloor), "an open window does not close");
    require(governor.quota() == kFloor, "the quota holds while the window is open");
    require(governor.update(now + milliseconds(150), kFloor), "a closed window reports its hold");
    require(governor.quota() > kFloor, "a fast interval that held admissions back grows the quota");
}

void test_too_few_samples_keep_the_window_open() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    require(!run_window(governor, now, {microseconds(5000), kFloor, true, 8}),
            "a window under the sample minimum does not close");
    require(governor.quota() == kFloor, "the quota holds without samples");
    // The samples and the hold stay for the next window: 8 + 64 reach the minimum.
    require(run_window(governor, now, {microseconds(5000), kFloor, false}),
            "the samples and the hold of both windows count");
    require(governor.quota() > kFloor, "the open window's hold grew the quota");
}

void test_growth_needs_held_admissions_and_is_bounded() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    // Fast, but nothing waits for a lane: there is nothing to release.
    require(!run_window(governor, now, {microseconds(10000), kFloor, false}),
            "a window without a hold reports none");
    require(governor.quota() == kFloor, "a quota nothing waits for does not grow");
    auto last = governor.quota();
    for (int i = 0; i < 40 && governor.quota() < kCapacity; ++i) {
        require(run_window(governor, now, {microseconds(10000), governor.quota()}),
                "a window with a hold reports it");
        require(governor.quota() > last, "growth is monotone");
        require(governor.quota() <= last + last / 4 + 1, "one window grows by a bounded step");
        last = governor.quota();
    }
    require(governor.quota() == kCapacity, "growth stops at the capacity");
    (void)run_window(governor, now, {microseconds(10000), kCapacity});
    require(governor.quota() == kCapacity, "the capacity does not grow");
}

void test_shrink_follows_the_load_and_stops_at_the_floor() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    grow_to(governor, now, 200);
    const auto grown = governor.quota();
    // 30 ms against a 23 ms target with the quota in use: shrink from what is resident.
    (void)run_window(governor, now, {microseconds(30000), grown});
    require(governor.quota() < grown, "a slow interval with the quota in use shrinks it");
    require(governor.quota() >= grown - grown / 4 - 1, "one window shrinks by a bounded step");
    // Residents above the lowered quota are still draining: the quota waits for them.
    const auto lowered = governor.quota();
    (void)run_window(governor, now, {microseconds(30000), grown});
    require(governor.quota() == lowered, "a quota below the residents is not lowered again");
    // Nothing waits for a lane, so the floor is where a persistent overload settles.
    for (int i = 0; i < 80; ++i)
        (void)run_window(governor, now, {microseconds(60000), governor.quota(), false});
    require(governor.quota() == kFloor, "a persistent overload settles at the floor");
    (void)run_window(governor, now, {microseconds(60000), kFloor, false});
    require(governor.quota() == kFloor, "the floor bounds the quota");
}

// A target out of reach at the floor while admissions wait releases every lane for a while.
void test_an_unreachable_target_releases_every_lane() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now    = open(governor);
    int windows = 0;
    // 40 ms is past reach (1.25 targets is 28.75 ms).
    while (governor.quota() == kFloor && windows < 40) {
        (void)run_window(governor, now, {microseconds(40000), kFloor});
        ++windows;
    }
    require(governor.quota() == kCapacity, "an unreachable target released every lane");
    require(windows == 14, "the release waited for release_after");
    for (int i = 0; i < 100; ++i) (void)run_window(governor, now, {microseconds(120000), kCapacity});
    require(governor.quota() == kCapacity, "the release keeps every lane for release_for");
    for (int i = 0; i < 40; ++i)
        (void)run_window(governor, now, {microseconds(120000), governor.quota()});
    require(governor.quota() < kCapacity, "the governor governs again after the release");
}

void test_a_slow_floor_releases_only_past_reach_with_demand() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    for (int i = 0; i < 40; ++i) (void)run_window(governor, now, {microseconds(40000), kFloor, false});
    require(governor.quota() == kFloor, "without waiting demand the floor holds");
    // Slow but within reach: the lanes held back are what keeps the target in sight.
    for (int i = 0; i < 40; ++i) (void)run_window(governor, now, {microseconds(27000), kFloor});
    require(governor.quota() == kFloor, "a reachable target does not release");
}

// An idle Engine starts over from the floor, whatever the previous load left.
void test_idle_starts_over_from_the_floor() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    grow_to(governor, now, 200);
    governor.idle();
    require(governor.quota() == kFloor, "idle returned a grown quota to the floor");
    // A release ends too, and the next load must miss the target for release_after again.
    now = open(governor);
    for (int i = 0; i < 15; ++i) (void)run_window(governor, now, {microseconds(40000), kFloor});
    require(governor.quota() == kCapacity, "the release started");
    governor.idle();
    require(governor.quota() == kFloor, "idle ended the release");
    now = open(governor);
    for (int i = 0; i < 10; ++i) (void)run_window(governor, now, {microseconds(40000), kFloor});
    require(governor.quota() == kFloor, "the next load waits release_after before a release");
    // Samples from before the idle gap do not close the next load's first window.
    for (int i = 0; i < 64; ++i) governor.observe(microseconds(5000), 1);
    governor.idle();
    now = open(governor);
    require(!run_window(governor, now, {microseconds(23000), kFloor, true, 8}),
            "the previous load's samples were dropped");
    require(governor.quota() == kFloor, "stale fast samples did not grow the quota");
}

void test_a_fast_window_ends_the_release() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    for (int i = 0; i < 15; ++i) (void)run_window(governor, now, {microseconds(40000), kFloor});
    require(governor.quota() == kCapacity, "the release started");
    // The load eased: a fast window ends the release, and the next slow one governs.
    (void)run_window(governor, now, {microseconds(15000), 100, false});
    (void)run_window(governor, now, {microseconds(30000), 100});
    require(governor.quota() < kCapacity, "the release ended early");
}

void test_dead_band_holds_the_quota() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    for (const auto interval : {22300, 23000, 23800})
        (void)run_window(governor, now, {microseconds(interval), kFloor});
    require(governor.quota() == kFloor, "an interval inside the band does not adjust");
}

void test_one_outlier_does_not_dominate() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    grow_to(governor, now, 200);
    const auto grown = governor.quota();
    for (int i = 0; i < 63; ++i) governor.observe(milliseconds(23), 1);
    // A lane that sat out a ten second preemption: capped at four targets it lifts the mean by 6%.
    governor.observe(std::chrono::seconds(10), 1);
    now += milliseconds(150);
    (void)governor.update(now, grown);
    require(governor.quota() >= grown - grown / 10, "an outlier cut the quota by more than a tenth");
}

void test_samples_across_an_idle_gap_are_dropped() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    for (int i = 0; i < 8; ++i) governor.observe(milliseconds(90), 1);
    now += std::chrono::seconds(2);
    require(!governor.update(now, kFloor), "stale samples under the minimum do not close a window");
    // The window restarted: only the fresh fast samples count now.
    require(run_window(governor, now, {microseconds(10000), kFloor}),
            "fresh samples close the restarted window");
    require(governor.quota() > kFloor, "the stale slow samples did not hold the quota down");
}

void test_a_floor_at_the_capacity_leaves_nothing_to_govern() {
    LaneGovernor governor(16, 32, policy());
    require(governor.quota() == 16, "a floor above the capacity pins the quota to the capacity");
    auto now = open(governor);
    require(!run_window(governor, now, {microseconds(90000), 16}),
            "an ungoverned quota reports no hold");
    require(governor.quota() == 16, "the quota stayed at the capacity");
}

// A lane that commits several tokens at once advances by the interval per token.
void test_multi_token_commits_count_per_token() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    // 30 ms per commit is slow for one token and fast for two (15 ms per token).
    for (int i = 0; i < 32; ++i) governor.observe(milliseconds(30), 2);
    governor.hold();
    now += milliseconds(150);
    require(governor.update(now, kFloor), "two-token commits close the window");
    require(governor.quota() > kFloor, "15 ms per token grows the quota");
    // Commits without tokens carry no cadence.
    const auto quota = governor.quota();
    for (int i = 0; i < 64; ++i) governor.observe(milliseconds(90), 0);
    governor.hold();
    now += milliseconds(150);
    require(!governor.update(now, quota), "commits without tokens are not samples");
    require(governor.quota() == quota, "commits without tokens do not move the quota");
}

// The hold of a window reaches only that window, which reports it so its owner checks again.
void test_holds_are_reported_per_window() {
    LaneGovernor governor(kCapacity, kFloor, policy());
    auto now = open(governor);
    require(run_window(governor, now, {microseconds(23000), kFloor}), "a held window reports it");
    require(!run_window(governor, now, {microseconds(23000), kFloor, false}),
            "the next window starts without a hold");
    // A slow window that held admissions back does not grow the quota.
    grow_to(governor, now, 100);
    const auto quota = governor.quota();
    require(run_window(governor, now, {microseconds(30000), quota}), "a slow held window reports it");
    require(governor.quota() < quota, "a slow window shrinks the quota despite its hold");
}

// A saturated Engine with demand always waiting: a lane's token interval is a fixed cost plus a
// cost per resident lane, and the residents follow the quota with a lag.
void simulate(LaneGovernor& governor, LaneGovernor::Clock::time_point& now, double residents,
              int windows, int settled, std::uint32_t& low, std::uint32_t& high) {
    low  = kCapacity;
    high = 0;
    for (int window = 0; window < windows; ++window) {
        const auto interval =
            microseconds(static_cast<std::int64_t>((9.0 + 0.19 * residents) * 1000.0));
        (void)run_window(governor, now, {interval, static_cast<std::uint32_t>(residents + 0.5)});
        residents += (governor.quota() - residents) * 0.5;
        if (window >= settled) {
            low  = std::min(low, governor.quota());
            high = std::max(high, governor.quota());
        }
    }
}

void test_the_loop_settles_where_the_interval_meets_the_target() {
    const double equilibrium = (23.0 - 9.0) / 0.19;
    std::uint32_t low = 0, high = 0;
    LaneGovernor from_below(kCapacity, kFloor, policy());
    auto now = open(from_below);
    simulate(from_below, now, kFloor, 120, 60, low, high);
    require(low + 8 >= equilibrium && high <= equilibrium + 8,
            "the quota holds within eight lanes of the equilibrium from below");
    LaneGovernor from_above(kCapacity, kFloor, policy());
    now = open(from_above);
    grow_to(from_above, now, kCapacity);
    simulate(from_above, now, kCapacity, 200, 120, low, high);
    require(low + 8 >= equilibrium && high <= equilibrium + 8,
            "the quota holds within eight lanes of the equilibrium from above");
}

} // namespace

int main() {
    try {
        test_starts_at_the_floor_and_waits_for_a_window();
        test_too_few_samples_keep_the_window_open();
        test_growth_needs_held_admissions_and_is_bounded();
        test_shrink_follows_the_load_and_stops_at_the_floor();
        test_dead_band_holds_the_quota();
        test_one_outlier_does_not_dominate();
        test_samples_across_an_idle_gap_are_dropped();
        test_a_floor_at_the_capacity_leaves_nothing_to_govern();
        test_multi_token_commits_count_per_token();
        test_holds_are_reported_per_window();
        test_an_unreachable_target_releases_every_lane();
        test_a_slow_floor_releases_only_past_reach_with_demand();
        test_a_fast_window_ends_the_release();
        test_idle_starts_over_from_the_floor();
        test_the_loop_settles_where_the_interval_meets_the_target();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}