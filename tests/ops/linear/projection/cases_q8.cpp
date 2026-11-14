#include "ops/linear/projection/runner.h"
#include "core/platform.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace ninfer::test::linear_weights;
using namespace ninfer::test::linear;
namespace linear_suite = ninfer::test::linear_suite;

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

const std::array kGeometries{
#if NINFER_TARGET_SPARK
    Geometry{4096, 16384, 1009U}, Geometry{4096, 1024, 983U},  Geometry{4096, 4096, 947U},
    Geometry{12288, 4096, 911U},  Geometry{2560, 3072, 383U},  Geometry{8192, 2560, 373U},
#endif
    Geometry{1024, 2048, 257U},   Geometry{1024, 5120, 223U},  Geometry{2048, 4096, 251U},
    Geometry{2048, 4608, 271U},   Geometry{2048, 16384, 283U}, Geometry{2560, 6144, 307U},
    Geometry{4608, 4608, 277U},   Geometry{5120, 4608, 281U},  Geometry{5120, 6144, 239U},
    Geometry{5120, 10240, 211U},  Geometry{5120, 17408, 241U}, Geometry{5120, 25600, 293U},
    Geometry{6144, 2560, 347U},   Geometry{6144, 5120, 227U},  Geometry{9216, 2048, 263U},
    Geometry{10240, 2560, 331U},  Geometry{12288, 2048, 269U}, Geometry{12288, 2560, 313U},
    Geometry{14336, 5120, 229U},  Geometry{16384, 2560, 359U}, Geometry{34816, 5120, 233U},
    Geometry{248320, 5120, 197U}, Geometry{248320, 3072, 349U}};

int q8_workspace_domain(std::int32_t n, std::int32_t k) {
    int failures = 0;
    for (auto policy : {ninfer::ops::LinearPolicy::A16Only, ninfer::ops::LinearPolicy::AllowA8,
                        ninfer::ops::LinearPolicy::AllowA4}) {
        if (ninfer::ops::linear_workspace_capacity_bytes(
                ninfer::QType::Q8_G32_FP16, n, k, policy, 1,
                std::numeric_limits<std::int32_t>::max()) != 0) {
            std::cerr << "Q8 [" << n << ',' << k
                      << "]: expected zero workspace across the positive T domain\n";
            ++failures;
        }
        for (auto [first, last] : {std::pair{0, 1}, std::pair{-1, 8}, std::pair{8, 7}}) {
            try {
                (void)ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::Q8_G32_FP16, n, k,
                                                                   policy, first, last);
                std::cerr << "Q8 [" << n << ',' << k << "]: accepted invalid workspace interval\n";
                ++failures;
            } catch (const std::invalid_argument&) {}
        }
    }
    return failures;
}

// The lm_head route must reproduce the reference argmax exactly whenever the reference top-2
// margin clears the BF16 output roundoff. Near-tied columns are not decisive and are skipped;
// the fixture has to stay mostly decisive or the check would no longer measure anything.
int q8_lm_head_argmax(std::int32_t n, std::int32_t k) {
    const auto host      = make_q8_g32_fp16_weight(n, k, 205U);
    const auto& context  = ninfer::test::cuda_environment().context;
    const auto execution = ninfer::test::cuda_environment().default_execution;
    ninfer::DeviceBuffer device_weight(context.device(), host.payload.size());
    device_weight.copy_from_host(host.payload.data(), device_weight.size());
    const auto weight = host.device_weight(device_weight.span());
    int failures      = 0;
    for (const std::int32_t t : {1, 130, 520}) {
        std::vector<std::int32_t> checked;
        if (t == 520) {
            for (std::int32_t token = 0; token < t; token += 40) checked.push_back(token);
        } else {
            checked.resize(t);
            std::iota(checked.begin(), checked.end(), 0);
        }
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
        std::vector<float> activation(static_cast<std::size_t>(k) * t);
        for (std::int32_t token = 0; token < t; ++token) {
            for (std::int32_t column = 0; column < k; ++column) {
                std::uint32_t mixed = 251U + static_cast<std::uint32_t>(t) * 0x9e3779b9U +
                                      static_cast<std::uint32_t>(column) * 0x85ebca6bU +
                                      static_cast<std::uint32_t>(token) * 0xc2b2ae35U;
                mixed ^= mixed >> 15;
                mixed *= 0x7feb352dU;
                mixed ^= mixed >> 13;
                const float value = static_cast<float>(static_cast<int>(mixed & 0xffU) - 128) *
                                    (1.0F / 256.0F);
                bits[static_cast<std::size_t>(token) * k + column] =
                    ninfer::test::f32_to_bf16(value);
                activation[static_cast<std::size_t>(token) * k + column] = value;
            }
        }
        ninfer::DeviceBuffer device_activation(context.device(), bits.size() * sizeof(std::uint16_t));
        device_activation.copy_from_host(bits.data(), device_activation.size());
        ninfer::Tensor x(device_activation.span(), ninfer::DType::BF16, {k, t});
        ninfer::DeviceBuffer output(context.device(),
                                    static_cast<std::size_t>(n) * t * sizeof(std::uint16_t));
        ninfer::Tensor y(output.span(), ninfer::DType::BF16, {n, t});
        ninfer::ops::linear(x, weight, y, execution);
        ninfer::test::cuda_check(cudaStreamSynchronize(execution.stream()), "lm_head argmax sync");
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(n) * t);
        ninfer::test::cuda_check(
            cudaMemcpy(logits.data(), output.data(), logits.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost),
            "lm_head argmax readback");

        struct Top {
            double first = -std::numeric_limits<double>::infinity(), second = -std::numeric_limits<double>::infinity();
            std::int32_t first_row = -1;
        };
        std::mutex merge_mutex;
        std::vector<Top> reference(checked.size());
        ninfer::test::parallel_rows(
            n, static_cast<std::uint64_t>(n) * k * checked.size(), [&](int begin, int end) {
                std::vector<Top> local(checked.size());
                std::vector<double> row(k);
                for (int r = begin; r < end; ++r) {
                    for (std::int32_t column = 0; column < k; ++column)
                        row[column] = ninfer::test::quantized_weight::logical_weight_fp64(
                            host, r, column);
                    for (std::size_t index = 0; index < checked.size(); ++index) {
                        const float* values =
                            activation.data() + static_cast<std::size_t>(checked[index]) * k;
                        double sum = 0.0;
                        for (std::int32_t column = 0; column < k; ++column)
                            sum += row[column] * static_cast<double>(values[column]);
                        auto& top = local[index];
                        if (sum > top.first) {
                            top.second    = top.first;
                            top.first     = sum;
                            top.first_row = r;
                        } else if (sum > top.second) {
                            top.second = sum;
                        }
                    }
                }
                std::lock_guard lock(merge_mutex);
                for (std::size_t index = 0; index < checked.size(); ++index) {
                    auto& top = reference[index];
                    if (local[index].first > top.first) {
                        if (local[index].first_row != top.first_row) top.second = top.first;
                        top.first     = local[index].first;
                        top.first_row = local[index].first_row;
                        top.second    = std::max(top.second, local[index].second);
                    } else {
                        top.second = std::max(top.second, local[index].first);
                    }
                }
            });

        std::int32_t decisive = 0;
        for (std::size_t index = 0; index < checked.size(); ++index) {
            const auto& top = reference[index];
            if (top.first_row < 0 || !std::isfinite(top.first) || !std::isfinite(top.second)) {
                std::cerr << "lm_head argmax t=" << t << ": reference row "
                          << checked[index] << " has no finite logits\n";
                ++failures;
                continue;
            }
            // BF16 output rounding moves each logit by at most |v|/256, so a smaller top-2
            // margin cannot decide the argmax; those tokens are ties by construction.
            if (top.first - top.second <= (std::abs(top.first) + std::abs(top.second)) / 256.0 +
                                              5.0e-3)
                continue;
            ++decisive;
            const std::int32_t token = checked[index];
            std::int32_t actual      = 0;
            float best = ninfer::test::bf16_to_f32(logits[static_cast<std::size_t>(token) * n]);
            for (std::int32_t r = 1; r < n; ++r) {
                const float value =
                    ninfer::test::bf16_to_f32(logits[static_cast<std::size_t>(token) * n + r]);
                if (value > best) {
                    actual = r;
                    best   = value;
                }
            }
            if (actual != top.first_row) {
                std::cerr << "lm_head argmax t=" << t << " token " << token
                          << ": reference row " << top.first_row << " but kernel row " << actual
                          << '\n';
                ++failures;
            }
        }
        if (decisive * 2 < static_cast<std::int32_t>(checked.size())) {
            std::cerr << "lm_head argmax t=" << t << ": only " << decisive << " of "
                      << checked.size() << " tokens are decisive\n";
            ++failures;
        }
    }
    return failures;
}

void q8_a16_conformance(Suite& suite) {
    for (const auto& shape : kGeometries) {
        std::vector<Invocation> calls;
        // Cover live-column tails and the transitions from K-split to tiled contractions.
        for (int t : {1,  2,  3,  4,  5,  7,  8,  9,  15,  16,  17,  23,  24,  25,
                      31, 32, 33, 39, 40, 41, 44, 47, 48,  49,  55,  56,  57,  63,
                      64, 65, 79, 80, 81, 95, 96, 97, 127, 128, 129, 256, 1024}) {
            calls.push_back({t});
        }
        if (shape.n == 2048 && shape.k == 4096) {
            for (int t : {895, 896, 897}) calls.push_back({t});
            for (int t : {191, 192, 193, 575, 576, 577, 2047, 2048, 2049, 4096, 4097})
                calls.push_back({t});
            for (int t : {49, 65, 129, 577, 2049})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            add_workspace_cases(suite, ninfer::QType::Q8_G32_FP16, shape.n, shape.k);
        }
        if (shape.n == 4096 && (shape.k == 1024 || shape.k == 16384)) {
            for (int t : {191, 192, 193, 287, 288, 289, 511, 512, 513, 1023, 1025, 4096, 4097})
                calls.push_back({t});
            if (shape.k == 1024)
                for (int t : {575, 576, 577}) calls.push_back({t});
            for (int t : {4, 17, 49, 129, 577, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            add_workspace_cases(suite, ninfer::QType::Q8_G32_FP16, shape.n, shape.k);
        }
        if (shape.n == 4096 && shape.k == 4096) {
            for (int t : {255, 257}) calls.push_back({t});
            calls.push_back({256, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        if ((shape.n == 12288 || shape.n == 4096) && shape.k == 4096) {
            for (int t :
                 {6, 191, 192, 193, 511, 512, 513, 1023, 1025, 2047, 2048, 2049, 4095, 4096, 4097})
                calls.push_back({t});
            for (int t : {1,   4,   6,   8,   9,   24,   25,   40,   41,   96,  97,
                          128, 129, 192, 193, 512, 1023, 1024, 1025, 2048, 4096})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        if (shape.n == 8192 && shape.k == 2560) {
            for (int t : {6, 191, 192, 193, 511, 512, 513, 1023, 1025, 2048, 4096, 4097})
                calls.push_back({t});
            for (int t : {1, 6, 16, 17, 32, 33, 96, 97, 128, 129, 192, 193, 512, 1024, 4096})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        if (shape.n == 2560 && shape.k == 3072) {
            for (int t : {6, 191, 192, 193, 511, 512, 513, 575, 576, 577, 1023, 1025, 2047, 2048,
                          2049, 4096, 4097})
                calls.push_back({t});
            for (int t :
                 {1, 6, 16, 17, 32, 33, 64, 65, 96, 97, 128, 129, 192, 193, 512, 1024, 4096})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        if (shape.n == 2560 && shape.k == 6144) {
            for (int t : {50, 51, 52, 511, 512, 513, 1025, 2047, 2048, 2049, 4096})
                calls.push_back({t});
            for (int t : {4, 33, 49, 50, 51, 129, 512, 1024, 1025, 2049})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {40, 512, 1024, 1025}) calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 12288 && shape.k == 2560) {
            for (int t : {191, 192, 193, 1023, 1151, 1152, 1153, 2048, 4096}) calls.push_back({t});
            for (int t : {192, 1152})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {34, 35, 36, 72, 104, 105, 110, 111, 112, 113, 512, 1025})
                calls.push_back({t});
            for (int t : {17, 32,  34,  35,  40,  41,  64,  79,  80,   81,  96,
                          97, 104, 110, 111, 112, 113, 129, 512, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {34, 80, 112, 512, 1024, 1025})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 10240 && shape.k == 2560) {
            for (int t : {66, 67, 68, 512, 1025}) calls.push_back({t});
            for (int t : {17, 24, 25, 32, 33,  40,  41,  48,  49,   64,  65,
                          67, 68, 96, 97, 127, 128, 129, 512, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {24, 32, 40, 48, 67, 96, 127, 128, 512, 1024, 1025})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 6144 && shape.k == 2560) {
            for (int t : {6, 94, 112, 113, 192, 193, 512, 513, 1023, 1025, 2048, 4096, 4097})
                calls.push_back({t});
            calls.push_back({6, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {17, 24, 25,  32,  33,  40,  41,  64,  65,  80,   81,  95,
                          96, 97, 112, 113, 127, 128, 129, 512, 513, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {24, 32, 40, 64, 80, 96, 112, 127, 128, 512, 1024, 1025})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 16384 && shape.k == 2560) {
            for (int t : {191, 192, 193, 1023, 1151, 1152, 1153, 2048, 4096}) calls.push_back({t});
            for (int t : {192, 1152})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {512, 513, 1025}) calls.push_back({t});
            for (int t : {1,  8,  9,  16, 17, 24, 25,  32,  33,  40,  41,   48,
                          49, 63, 64, 65, 96, 97, 128, 129, 512, 513, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {8, 16, 24, 32, 40, 48, 63, 64, 96, 128, 512, 1024})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if (((shape.n == 12288 || shape.n == 4096) && shape.k == 4096) ||
            (shape.n == 2560 && shape.k == 3072) || (shape.n == 8192 && shape.k == 2560) ||
            (shape.n == 2560 && shape.k == 6144) || (shape.n == 12288 && shape.k == 2560) ||
            (shape.n == 10240 && shape.k == 2560) || (shape.n == 6144 && shape.k == 2560) ||
            (shape.n == 16384 && shape.k == 2560)) {
            auto test     = linear_suite::make_case("q8", shape.n, shape.k, 0,
                                                    ninfer::ops::LinearPolicy::A16Only,
                                                    linear_suite::Execution::Eager, "workspace-domain");
            test.policies = 7;
            suite.add(std::move(test), [shape] { return q8_workspace_domain(shape.n, shape.k); });
            add_workspace_cases(suite, ninfer::QType::Q8_G32_FP16, shape.n, shape.k);
        }
        if (shape.n == 6144 && shape.k == 5120) {
            for (int t : {191, 192, 193}) calls.push_back({t});
        }
        if (shape.k == 4608) {
            for (int t : {6, 11, 12, 13, 14, 19, 20, 21, 27, 28, 29}) calls.push_back({t});
            if (shape.n == 2048) {
                for (int t : {870, 871, 872}) calls.push_back({t});
            }
            if (shape.n == 4608) {
                for (int t : {255, 257}) calls.push_back({t});
            }
        }
        if (shape.n == 9216 && shape.k == 2048) {
            for (int t : {12, 13, 14}) calls.push_back({t});
        }
        if (shape.n == 248320) {
            calls.push_back({34});
            if (shape.k == 3072) {
                // lm_head of the 122B geometry: decode 130, boundary 519/520/521, prefill tail
                // through the single-flight prefill budget (8192).
                for (int t : {130, 519, 520, 521, 2048, 2049, 4097, 8191, 8192})
                    calls.push_back({t});
                for (int t : {130, 520, 2048, 8192})
                    calls.push_back(
                        {t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
                add_workspace_cases(suite, ninfer::QType::Q8_G32_FP16, shape.n, shape.k);
            }
        }
        if (shape.n == 2048 && shape.k == 16384) {
            for (int t : {383,  384,  385,  479,  480,  481,  639,  640,  641,  703,
                          704,  705,  959,  960,  961,  1343, 1344, 1345, 1679, 1680,
                          1681, 2015, 2016, 2017, 2111, 2112, 2113, 4096}) {
                calls.push_back({t});
            }
        }
        for (int t : {1, 8, 16, 32, 48, 64, 65, 128}) {
            calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        for (int t : {1, 16, 64, 128}) calls.push_back({t, CallForm::A16Convenience});
        calls.push_back({17, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA8});
        calls.push_back({64, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA4});
        add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::TokenBlocks,
                  make_q8_g32_fp16_weight,
                  {shape.n, shape.k, shape.seed, Comparison::Sampled, true, calls});
    }
#if NINFER_TARGET_SPARK
    const std::array stress_calls{Invocation{1}, Invocation{6}, Invocation{41}, Invocation{128},
                                  Invocation{1024}};
    for (const Geometry geometry : {Geometry{12288, 4096, 929U}, Geometry{4096, 4096, 967U}}) {
        for (const auto pattern : {ActivationPattern::KTail, ActivationPattern::Cancellation}) {
            ShapeCase shape{geometry.n,          geometry.k, geometry.seed,
                            Comparison::Sampled, true,       stress_calls};
            shape.activation_pattern = pattern;
            add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::Hashed,
                      make_q8_g32_fp16_weight, shape);
        }
    }
#endif
    const std::array full_calls{Invocation{1}, Invocation{4}, Invocation{8}};
    for (const Geometry shape : {Geometry{2048, 4096, 977U}, Geometry{2560, 6144, 311U},
                                 Geometry{12288, 2560, 317U}, Geometry{10240, 2560, 337U},
#if NINFER_TARGET_SPARK
                                 Geometry{2560, 3072, 389U}, Geometry{8192, 2560, 379U},
                                 Geometry{12288, 4096, 919U}, Geometry{4096, 4096, 953U},
#endif
                                 Geometry{6144, 2560, 349U}, Geometry{16384, 2560, 367U}}) {
        add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::TokenBlocks,
                  make_q8_g32_fp16_weight,
                  {shape.n, shape.k, shape.seed, Comparison::Full, true, full_calls});
    }

    {
        const std::array calls{Invocation{1}, Invocation{17}, Invocation{49}, Invocation{65},
                               Invocation{2049}};
        for (auto pattern : {ActivationPattern::KTail, ActivationPattern::Cancellation}) {
            add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::Hashed,
                      make_q8_g32_fp16_weight,
                      {2048, 4096, 1017U, Comparison::Sampled, true, calls, pattern});
        }
    }
#if NINFER_TARGET_SPARK
    for (const Geometry shape : {Geometry{4096, 1024, 1031U}, Geometry{4096, 16384, 1043U}}) {
        const std::array calls{Invocation{1},  Invocation{16},  Invocation{17},  Invocation{49},
                               Invocation{65}, Invocation{577}, Invocation{1025}};
        for (auto pattern : {ActivationPattern::KTail, ActivationPattern::Cancellation}) {
            add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::Hashed,
                      make_q8_g32_fp16_weight,
                      {shape.n, shape.k, shape.seed, Comparison::Sampled, true, calls, pattern});
        }
        add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::TokenBlocks,
                  make_q8_g32_fp16_weight,
                  {shape.n, shape.k, shape.seed + 1, Comparison::Full, true, full_calls});
    }
    {
        const std::array calls{Invocation{41}, Invocation{49}, Invocation{63}};
        add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::Hashed,
                  make_q8_g32_fp16_weight, {4096, 1024, 1021U, Comparison::Full, true, calls});
    }
#endif
}
} // namespace

void ninfer::test::linear::add_q8_cases(Suite& suite) {
    q8_a16_conformance(suite);
    auto test = linear_suite::make_case("q8", 248320, 3072, 0,
                                        ninfer::ops::LinearPolicy::A16Only,
                                        linear_suite::Execution::Eager, "lm-head-argmax");
    test.policies = 7;
    suite.add(std::move(test), [] { return q8_lm_head_argmax(248320, 3072); });
}
