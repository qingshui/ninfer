// Registration of the Qwen3.5-122B Linear geometries. Registered (n,k) pairs and split
// segmentations must resolve on every target for the full token range; an unregistered geometry
// must throw std::invalid_argument naming its parameters. The numerical contract of each
// registered shape is covered by the projection and split suites.
#include "core/platform.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_split.h"
#include "ops/suite.h"

#include <array>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using ninfer::ops::LinearPolicy;
using ninfer::ops::LinearProfile;
using ninfer::ops::ProjectionProfile;

constexpr std::int32_t kMaxTokens = 8192;

int check_message(const char* label, const std::string& message,
                  std::initializer_list<const char*> tokens) {
    int failures = 0;
    for (const char* token : tokens) {
        if (message.find(token) == std::string::npos) {
            std::cerr << label << ": message lacks '" << token << "': " << message << '\n';
            ++failures;
        }
    }
    return failures;
}

template <class Call>
int accepted(const char* label, Call call) {
    try {
        call();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << label << ": registered geometry rejected: " << error.what() << '\n';
        return 1;
    }
}

template <class Call>
int rejected(const char* label, Call call, std::initializer_list<const char*> tokens) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        return check_message(label, error.what(), tokens);
    }
    std::cerr << label << ": unregistered geometry was accepted\n";
    return 1;
}

std::size_t projection_capacity(QType type, int n, int k, LinearPolicy policy) {
    return ops::linear_workspace_capacity_bytes(type, n, k, policy, 1, kMaxTokens);
}

std::size_t split_capacity(const ProjectionProfile& profile, std::span<const std::int32_t> rows) {
    return ops::linear_split_workspace_capacity_bytes(profile, rows, 1, kMaxTokens);
}

LinearProfile nvfp4_parent(int n, int k) {
    return {QType::NVFP4, n, k, LinearPolicy::AllowA4, QuantLayout::BlockScaleK16M128x4};
}

LinearProfile bf16_parent(int n, int k) {
    return {QType::BF16, n, k, LinearPolicy::A16Only, QuantLayout::Contiguous};
}

#if NINFER_TARGET_B200
LinearProfile fp8_parent(int n, int k, LinearPolicy policy) {
    return {QType::FP8_E4M3FN_ROW_FP32, n, k, policy, QuantLayout::RowScale};
}

// The tcgen05 FP8 route quantizes its activation, so only the A8-permitting policies resolve.
int fp8_geometries() {
    int failures = 0;
    constexpr std::array<std::int32_t, 2> gdn{12288, 8192};
    for (const auto policy : {LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
        failures += accepted("split fp8 n20480_k3072", [&] {
            if (split_capacity({fp8_parent(20480, 3072, policy), std::nullopt}, gdn) == 0)
                throw std::logic_error("A8 route reported no workspace");
        });
        failures += accepted("linear_add fp8 n3072_k8192", [&] {
            if (ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_FP32, 3072, 8192,
                                                         policy, 1, kMaxTokens) == 0)
                throw std::logic_error("A8 route reported no workspace");
        });
    }
    failures += rejected("split fp8 n20480_k3072 A16Only",
                         [&] {
                             (void)split_capacity(
                                 {fp8_parent(20480, 3072, LinearPolicy::A16Only), std::nullopt},
                                 gdn);
                         },
                         {"20480", "3072", "A8"});
    failures += rejected("linear_add fp8 n3072_k8192 A16Only",
                         [] {
                             (void)ops::linear_add_workspace_capacity_bytes(
                                 QType::FP8_E4M3FN_ROW_FP32, 3072, 8192, LinearPolicy::A16Only, 1,
                                 kMaxTokens);
                         },
                         {"3072", "8192", "A8"});
    failures += rejected("linear_add fp8 n3072_k4096",
                         [] {
                             (void)ops::linear_add_workspace_capacity_bytes(
                                 QType::FP8_E4M3FN_ROW_FP32, 3072, 4096, LinearPolicy::AllowA8, 1,
                                 kMaxTokens);
                         },
                         {"unsupported"});
    return failures;
}
#endif

int projection_registered() {
    int failures = 0;
    for (const auto policy : {LinearPolicy::A16Only, LinearPolicy::AllowA4})
        failures += accepted("nvfp4 (3072,8192)",
                             [&] { (void)projection_capacity(QType::NVFP4, 3072, 8192, policy); });
    failures += accepted("bf16 (3072,8192)", [] {
        (void)projection_capacity(QType::BF16, 3072, 8192, LinearPolicy::A16Only);
    });
    failures += accepted("q8 (248320,3072)", [] {
        (void)projection_capacity(QType::Q8_G32_FP16, 248320, 3072, LinearPolicy::A16Only);
    });
    return failures;
}

int projection_unregistered() {
    int failures = 0;
    // The Spec text lists nvfp4 (3072,16384); the checkpoint has no such tensor, so it stays
    // unregistered and must be named in the error.
    failures += rejected("nvfp4 (3072,16384)",
                         [] {
                             (void)projection_capacity(QType::NVFP4, 3072, 16384,
                                                       LinearPolicy::AllowA4);
                         },
                         {"3072", "16384"});
    failures += rejected("bf16 (3072,16384)",
                         [] {
                             (void)projection_capacity(QType::BF16, 3072, 16384,
                                                       LinearPolicy::A16Only);
                         },
                         {"3072", "16384"});
    failures += rejected("q8 (248320,4096)",
                         [] {
                             (void)projection_capacity(QType::Q8_G32_FP16, 248320, 4096,
                                                       LinearPolicy::A16Only);
                         },
                         {"248320", "4096"});
    return failures;
}

int split_registered() {
    int failures                                  = 0;
    constexpr std::array<std::int32_t, 4> qkvz{8192, 512, 8192, 512};
    constexpr std::array<std::int32_t, 2> gdn{12288, 8192};
    failures += accepted("split nvfp4 n17408_k3072", [&] {
        (void)split_capacity({nvfp4_parent(17408, 3072), std::nullopt}, qkvz);
    });
    failures += accepted("split bf16 n20480_k3072", [&] {
        (void)split_capacity({bf16_parent(20480, 3072), std::nullopt}, gdn);
    });
    return failures;
}

int split_unregistered() {
    int failures                                    = 0;
    constexpr std::array<std::int32_t, 4> reordered{8192, 8192, 512, 512};
    constexpr std::array<std::int32_t, 2> swapped{8192, 12288};
    failures += rejected("split nvfp4 reordered segments",
                         [&] {
                             (void)split_capacity({nvfp4_parent(17408, 3072), std::nullopt},
                                                  reordered);
                         },
                         {"17408", "3072", "8192,8192,512,512"});
    failures += rejected("split bf16 swapped segments",
                         [&] {
                             (void)split_capacity({bf16_parent(20480, 3072), std::nullopt}, swapped);
                         },
                         {"20480", "3072", "8192,12288"});
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    using namespace ninfer::test;
    OpSuite suite(false);
    suite.add("linear/122b/projection_registered", projection_registered);
    suite.add("linear/122b/projection_unregistered", projection_unregistered);
    suite.add("linear/122b/split_registered", split_registered);
    suite.add("linear/122b/split_unregistered", split_unregistered);
#if NINFER_TARGET_B200
    suite.add("linear/122b/fp8_geometries", fp8_geometries);
#endif
    return suite.main(argc, argv);
}
