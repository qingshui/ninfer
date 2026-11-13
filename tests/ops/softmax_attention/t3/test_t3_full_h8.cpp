// T3 / F3 (full layer): causal_softmax_attention for D256/Hq8/Hkv2 at the maximum visible-key
// count kCausalAttentionMaximumVisibleKeys == 1048576, for the BF16 and FP8 cache formats.
// Reuses the shared causal fixture and its independent FP64 oracle (cache.cpp, oracle.cpp).
#include "../causal/fixture.h"
#include "kv_cache_storage.h"

#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::causal_attention;

namespace {
constexpr Geometry kH8Kv2{"d256-h8-kv2", 8, 2};
constexpr std::uint32_t kMaxKeys = 1048576;

// Append one token at position 1048575, so exactly 1048576 keys are visible, and compare the
// output against the FP64 oracle for a few checked queries.
int visible_keys_at_maximum(KvCacheStorage storage) {
    int failures = 0;
    if (ops::kCausalAttentionMaximumVisibleKeys != kMaxKeys) {
        std::cerr << "kCausalAttentionMaximumVisibleKeys != 1048576\n";
        ++failures;
    }
    const AttentionCase c{/*tokens*/ 1, /*base*/ static_cast<std::int32_t>(kMaxKeys) - 1, kMaxKeys,
                          /*seed*/ 7301u};
    failures += run_append_case(cuda_environment().context.execution_view(), kH8Kv2, storage, c,
                                MappingPattern::Fragmented);
    return failures;
}

// A plan whose envelope reaches 1048576 visible keys is accepted for the H8 geometry.
int plan_accepts_maximum(KvCacheStorage storage) {
    const auto execution = cuda_environment().context.execution_view();
    try {
        (void)ops::causal_softmax_attention_workspace_capacity_bytes(
            op_geometry(kH8Kv2), storage, {1, kMaxKeys}, 1, 1, 1,
            execution.multiprocessor_count());
        (void)ops::causal_softmax_attention_workspace_capacity_bytes(
            op_geometry(kH8Kv2), storage, {kMaxKeys, kMaxKeys}, 1, 1, 1,
            execution.multiprocessor_count());
    } catch (const std::exception& e) {
        std::cerr << "H8 plan rejected 1048576 visible keys: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0}) {
        const std::string name = cache_name(storage);
        const std::vector<std::pair<std::string, std::string>> tags{
            {"format", kv_cache_storage_name(storage)}, {"execution", "eager"}};
        suite.add("t3/full/h8kv2/" + name + "/plan_1m", [=] { return plan_accepts_maximum(storage); },
                  tags);
        suite.add("t3/full/h8kv2/" + name + "/visible_keys_1m",
                  [=] { return visible_keys_at_maximum(storage); }, tags);
    }
    return suite.main(argc, argv);
}
