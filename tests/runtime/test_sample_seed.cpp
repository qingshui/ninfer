// The branch seed derivation of a sample group: branch i of a request seeds its sampling with
// derive_sample_seed(seed, i), so the counter-based sampling streams of the branches of one
// request decorrelate while the same submission repeats the same branch seeds.
#include "ninfer/types.h"

#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void test_derived_seeds_decorrelate_and_repeat() {
    const std::uint64_t seed = 0x1234'5678'9ABC'DEF0ULL;

    // Deterministic in (seed, branch): the same submission repeats the same branch seeds.
    for (const std::uint32_t branch : {0U, 1U, 3U, 255U}) {
        require(ninfer::derive_sample_seed(seed, branch) ==
                    ninfer::derive_sample_seed(seed, branch),
                "the same (seed, branch) derived different seeds");
    }

    // Every branch of one request samples from its own stream, and no branch repeats the raw
    // request seed of another branch's group.
    std::vector<std::uint64_t> branches;
    for (std::uint32_t branch = 0; branch < 64; ++branch) {
        const auto derived = ninfer::derive_sample_seed(seed, branch);
        require(derived != seed, "a branch seed repeated the raw request seed");
        branches.push_back(derived);
    }
    require(std::set<uint64_t>(branches.begin(), branches.end()).size() == branches.size(),
            "two branches of one request derived the same seed");

    // The derivation is a function of both terms: moving the request or the branch moves the
    // seed, so two different requests sample different streams even on the same branch.
    require(ninfer::derive_sample_seed(seed, 0) != ninfer::derive_sample_seed(seed, 1),
            "two branches derived the same seed");
    require(ninfer::derive_sample_seed(seed, 2) != ninfer::derive_sample_seed(seed + 1, 2),
            "two requests derived the same branch seed");
    require(ninfer::derive_sample_seed(0, 0) == ninfer::derive_sample_seed(0, 0),
            "a zero seed broke determinism");
}

} // namespace

int main() {
    try {
        test_derived_seeds_decorrelate_and_repeat();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
