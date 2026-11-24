#include "ops/suite.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/types.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"

#include "gdn_ref.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStateDim = 128;

// One error budget for the complete recurrence over represented public inputs, including
// continuation. FP32 state storage does not prescribe the precision of private arithmetic.
constexpr ReductionCriterion kOutputCriterion{6.0e-3, 5.0e-6, 8.0e-3};
constexpr ReductionCriterion kStateCriterion{4.5e-3, 1.0e-5, 9.0e-3};
// A BF16 State is published rounded to nearest even and its reference rounds the exact state at
// the same publications. Each rounding moves an element by at most 2^-8 of it, so a published
// state lies within one BF16 unit, 2^-7, of the reference's beyond the FP32 arithmetic error.
// A published state and the outputs that read it carry that unit in both terms.
constexpr double kBf16Unit = 1.0 / 128.0;
constexpr ReductionCriterion kBf16OutputCriterion{6.0e-3 + kBf16Unit, 5.0e-6, 8.0e-3 + kBf16Unit};
constexpr ReductionCriterion kBf16StateCriterion{4.5e-3 + kBf16Unit, 1.0e-5, 9.0e-3 + kBf16Unit};

enum class StateWord { Fp32, Bf16 };

DType state_dtype(StateWord word) { return word == StateWord::Bf16 ? DType::BF16 : DType::FP32; }

std::size_t state_word_bytes(StateWord word) { return word == StateWord::Bf16 ? 2 : 4; }

const char* state_word_name(StateWord word) { return word == StateWord::Bf16 ? "bf16" : "fp32"; }

enum class InputPattern {
    Ordinary,
    WeakDecay,
    ZeroDecay,
    SmallBeta,
    StrongDecay,
    NearCollinear,
    ZeroState,
    ZeroQuery,
    ZeroSignal,
    ZeroBeta,
    UnitBeta,
    ExactRounding,
};

struct Case {
    const char* name;
    int qk_heads;
    int value_heads;
    int tokens;
    bool normalize_qk;
    bool near_zero_qk = false;
};

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

void normalize_rows(std::vector<float>& values, int width) {
    const std::size_t rows = values.size() / static_cast<std::size_t>(width);
    for (std::size_t row = 0; row < rows; ++row) {
        float* base  = values.data() + row * static_cast<std::size_t>(width);
        double sumsq = 0.0;
        for (int d = 0; d < width; ++d) {
            const double value = static_cast<double>(base[d]);
            sumsq += value * value;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (int d = 0; d < width; ++d) {
            base[d] = static_cast<float>(static_cast<double>(base[d]) * inv);
        }
    }
}

gdn_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed,
                            InputPattern pattern = InputPattern::Ordinary) {
    gdn_ref::Inputs in;
    in.head_dim    = kStateDim;
    in.qk_heads    = test_case.qk_heads;
    in.value_heads = test_case.value_heads;
    in.tokens      = test_case.tokens;

    const std::size_t qk_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * test_case.tokens);
    const std::size_t value_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * test_case.tokens);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);
    in.q.resize(qk_size);
    in.k.resize(qk_size);
    in.v.resize(value_size);
    in.g.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.beta.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0f, 1.0f);
    fill_uniform(in.k, generator, -1.0f, 1.0f);
    fill_uniform(in.v, generator, -0.5f, 0.5f);
    fill_uniform(in.g, generator, -0.10f, -0.005f);
    fill_uniform(in.beta, generator, 0.05f, 0.95f);
    fill_uniform(in.state, generator, -0.02f, 0.02f);

    if (test_case.near_zero_qk) {
        for (float& value : in.q) { value *= 1.0e-4f; }
        for (float& value : in.k) { value *= 1.0e-4f; }
    } else if (!test_case.normalize_qk) {
        // Raw-Q/K mode still receives a stable, entirely valid public input. This host-side
        // generation choice is not part of the oracle.
        normalize_rows(in.q, kStateDim);
        normalize_rows(in.k, kStateDim);
    }

    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    switch (pattern) {
    case InputPattern::Ordinary:
        break;
    case InputPattern::WeakDecay:
        std::fill(in.g.begin(), in.g.end(), -1.0e-6F);
        break;
    case InputPattern::ZeroDecay:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        break;
    case InputPattern::SmallBeta:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.002F);
        break;
    case InputPattern::StrongDecay:
        std::fill(in.g.begin(), in.g.end(), -80.0F);
        break;
    case InputPattern::NearCollinear: {
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.98F);
        for (float& value : in.state) value *= 50.0F;
        const std::vector<float> base(in.k.begin(), in.k.begin() + test_case.qk_heads * kStateDim);
        for (int t = 0; t < test_case.tokens; ++t) {
            for (int qh = 0; qh < test_case.qk_heads; ++qh) {
                for (int d = 0; d < kStateDim; ++d) {
                    const auto offset = (t * test_case.qk_heads + qh) * kStateDim + d;
                    in.k[offset]      = base[qh * kStateDim + d] + 0.002F * in.k[offset];
                }
            }
        }
        round_to_bf16(in.k);
        // Make V close to the prediction from the initial state. This stresses cancellation
        // in the update; it does not change the independent mathematical oracle.
        for (int t = 0; t < test_case.tokens; ++t) {
            for (int h = 0; h < test_case.value_heads; ++h) {
                const int qh      = h / (test_case.value_heads / test_case.qk_heads);
                const auto offset = (t * test_case.qk_heads + qh) * kStateDim;
                double sum        = 1.0e-6;
                for (int d = 0; d < kStateDim; ++d)
                    sum += double(in.k[offset + d]) * in.k[offset + d];
                const double inv = test_case.normalize_qk ? 1.0 / std::sqrt(sum) : 1.0;
                for (int row = 0; row < kStateDim; ++row) {
                    double prediction = 0.0;
                    for (int d = 0; d < kStateDim; ++d)
                        prediction += double(in.state[(h * kStateDim + row) * kStateDim + d]) *
                                      in.k[offset + d] * inv;
                    in.v[(t * test_case.value_heads + h) * kStateDim + row] =
                        static_cast<float>(prediction);
                }
            }
        }
        round_to_bf16(in.v);
        break;
    }
    case InputPattern::ZeroState:
        std::fill(in.state.begin(), in.state.end(), 0.0F);
        break;
    case InputPattern::ZeroQuery:
        std::fill(in.q.begin(), in.q.end(), 0.0F);
        break;
    case InputPattern::ZeroSignal:
        std::fill(in.state.begin(), in.state.end(), 0.0F);
        std::fill(in.v.begin(), in.v.end(), 0.0F);
        break;
    case InputPattern::ZeroBeta:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.0F);
        break;
    case InputPattern::UnitBeta:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 1.0F);
        break;
    case InputPattern::ExactRounding: {
        // A raw unit key on coordinate 0 with g = 0 and beta = 1/2 replaces column 0 of the
        // state by the mean of it and V at every token and leaves the other columns unchanged.
        // Column 0 starts on, and V lies on, BF16 grids in binades 2^-7 and 2^-6. FP32 then
        // forms the states of a few tokens exactly, and a published state must equal the
        // reference's rounding bit for bit; the means hit many rounding ties.
        if (test_case.normalize_qk) throw std::logic_error("exact rounding inputs use raw Q/K");
        std::fill(in.k.begin(), in.k.end(), 0.0F);
        for (std::size_t i = 0; i < in.k.size(); i += kStateDim) in.k[i] = 1.0F;
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.5F);
        std::uniform_int_distribution<int> mantissa(0, 127), binade(6, 7), sign(0, 1);
        const auto grid_value = [&](int exponent) {
            const float magnitude =
                std::ldexp(1.0F + static_cast<float>(mantissa(generator)) / 128.0F, -exponent);
            return sign(generator) != 0 ? -magnitude : magnitude;
        };
        for (std::size_t i = 0; i < in.state.size(); i += kStateDim) in.state[i] = grid_value(6);
        for (float& value : in.v) value = grid_value(binade(generator));
        break;
    }
    }
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

// The bytes of a State field holding `values`, which lie on the BF16 grid for a BF16 State.
std::vector<std::uint8_t> state_bytes(const std::vector<float>& values, StateWord word) {
    std::vector<std::uint8_t> bytes(values.size() * state_word_bytes(word));
    if (word == StateWord::Bf16) {
        const auto bits = bf16_bits(values);
        std::memcpy(bytes.data(), bits.data(), bytes.size());
    } else {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

std::vector<double> read_state(const void* device, std::size_t count, StateWord word) {
    if (word == StateWord::Bf16) return from_device_bf16(device, count);
    return doubles(from_device<float>(device, count));
}

template <typename T>
int verify_exact(const std::string& label, const std::vector<T>& got,
                 const std::vector<T>& expected) {
    return ninfer::test::verify_exact(label.c_str(), got, expected);
}

// `published` marks a state the case published as BF16, or outputs that read one.
int verify_heads(const std::string& label, std::span<const double> got,
                 std::span<const double> expected, int value_heads, int tokens, bool state,
                 bool published = false) {
    const auto& criterion = state ? (published ? kBf16StateCriterion : kStateCriterion)
                                  : (published ? kBf16OutputCriterion : kOutputCriterion);
    const std::size_t head_size =
        static_cast<std::size_t>(kStateDim) * (state ? kStateDim : tokens);
    if (got.size() != head_size * value_heads || got.size() != expected.size()) {
        throw std::logic_error("GDN per-head comparison shape mismatch");
    }
    std::vector<double> head_got(head_size), head_ref(head_size);
    int failures = 0;
    for (int h = 0; h < value_heads; ++h) {
        if (state) {
            std::copy_n(got.data() + h * head_size, head_size, head_got.data());
            std::copy_n(expected.data() + h * head_size, head_size, head_ref.data());
        } else {
            for (int t = 0; t < tokens; ++t) {
                const auto source = (static_cast<std::size_t>(t) * value_heads + h) * kStateDim;
                std::copy_n(got.data() + source, kStateDim, head_got.data() + t * kStateDim);
                std::copy_n(expected.data() + source, kStateDim, head_ref.data() + t * kStateDim);
            }
        }
        const std::string head_label = label + " head=" + std::to_string(h);
        const auto stats = compute_reduction_stats(head_got.data(), head_ref.data(), head_size);
        const double gross_limit = gross_error_limit(stats, criterion);
        report_reduction_stats(head_label, head_size, stats, criterion);
        if (!reduction_passes(stats, head_size, criterion)) {
            std::cerr << head_label << ": rel_l2=" << stats.relative_l2
                      << " limit=" << criterion.relative_l2
                      << " max_abs=" << stats.maximum_absolute_error
                      << " gross_limit=" << gross_limit << " non_finite=" << stats.first_non_finite
                      << '\n';
            ++failures;
        }
    }
    return failures;
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    return doubles(from_device<float>(device, count));
}

int verify_common_inputs_unchanged(const std::string& label, const gdn_ref::Inputs& in,
                                   const DeviceBuffer& q, const DeviceBuffer& k,
                                   const DeviceBuffer& v, const DeviceBuffer& g,
                                   const DeviceBuffer& beta) {
    int failures = 0;
    failures += verify_exact(label + " q unchanged", from_device<std::uint16_t>(q, in.q.size()),
                             bf16_bits(in.q));
    failures += verify_exact(label + " k unchanged", from_device<std::uint16_t>(k, in.k.size()),
                             bf16_bits(in.k));
    failures += verify_exact(label + " v unchanged", from_device<std::uint16_t>(v, in.v.size()),
                             bf16_bits(in.v));
    failures += verify_exact(label + " g unchanged", from_device<float>(g, in.g.size()), in.g);
    failures +=
        verify_exact(label + " beta unchanged", from_device<float>(beta, in.beta.size()), in.beta);
    return failures;
}

struct DeviceInputs {
    explicit DeviceInputs(const gdn_ref::Inputs& in)
        : q(to_device_bf16(in.q)), k(to_device_bf16(in.k)), v(to_device_bf16(in.v)),
          g(to_device_f32(in.g)), beta(to_device_f32(in.beta)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
};

gdn_ref::Result reference_prefix(const gdn_ref::Inputs& in, int tokens, float scale,
                                 bool normalize_qk,
                                 const std::vector<std::int64_t>& publications = {}) {
    if (tokens == in.tokens) return gdn_ref::evaluate(in, scale, normalize_qk, false, publications);
    gdn_ref::Inputs prefix;
    prefix.head_dim    = in.head_dim;
    prefix.qk_heads    = in.qk_heads;
    prefix.value_heads = in.value_heads;
    prefix.tokens      = tokens;
    prefix.q.assign(in.q.begin(), in.q.begin() + tokens * in.qk_heads * kStateDim);
    prefix.k.assign(in.k.begin(), in.k.begin() + tokens * in.qk_heads * kStateDim);
    prefix.v.assign(in.v.begin(), in.v.begin() + tokens * in.value_heads * kStateDim);
    prefix.g.assign(in.g.begin(), in.g.begin() + tokens * in.value_heads);
    prefix.beta.assign(in.beta.begin(), in.beta.begin() + tokens * in.value_heads);
    prefix.state = in.state;
    // Re-evaluate a few logical prefixes from the original public state. Neither GPU state nor
    // a cast of an earlier FP64 result is fed back into the oracle at a call boundary; a BF16
    // State rounds the exact state at the same publications.
    return gdn_ref::evaluate(prefix, scale, normalize_qk, false, publications);
}

int run_case(const Case& test_case, std::uint32_t seed,
             InputPattern pattern = InputPattern::Ordinary, bool inplace = false,
             std::vector<int> partitions = {}, StateWord word = StateWord::Fp32) {
    gdn_ref::Inputs in = make_inputs(test_case, seed, pattern);
    if (word == StateWord::Bf16) round_to_bf16(in.state);
    if (partitions.empty()) partitions.push_back(test_case.tokens);
    if (std::accumulate(partitions.begin(), partitions.end(), 0) != test_case.tokens ||
        *std::min_element(partitions.begin(), partitions.end()) <= 0) {
        throw std::logic_error("invalid GDN continuation partition");
    }
    const std::string label =
        std::string(test_case.name) + " H=" + std::to_string(test_case.qk_heads) + "/" +
        std::to_string(test_case.value_heads) + " T=" + std::to_string(test_case.tokens) +
        (test_case.normalize_qk ? " normalized" : " raw") +
        (inplace ? " inplace" : " distinct-state") + " state=" + state_word_name(word);
    std::cout << "GDN_CASE " << label << " calls=" << partitions.size() << std::endl;
    const float scale = 1.0F / std::sqrt(static_cast<float>(kStateDim));
    DeviceInputs device(in);
    const std::size_t state_bytes_total = in.state.size() * state_word_bytes(word);
    GuardedDeviceBuffer state_a(state_bytes_total);
    GuardedDeviceBuffer state_b(state_bytes_total);
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_a.copy_from_host(state_bytes(in.state, word).data(), state_a.bytes());
    out.fill(0xff);
    Tensor q(device.q.span(), DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.span(), DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.span(), DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.span(), DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.span(), DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor output(out.span(), DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});

    const int maximum_call     = *std::max_element(partitions.begin(), partitions.end());
    const auto workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, 1, maximum_call);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(workspace_bytes, 256));
    scratch.fill(0xff);
    MemoryArena workspace(scratch.span());
    const auto device_execution = cuda_environment().default_execution;

    GuardedDeviceBuffer* current = &state_a;
    int failures                 = 0;
    int begin                    = 0;
    std::vector<std::int64_t> publications;
    for (int length : partitions) {
        auto* destination = inplace ? current : (current == &state_a ? &state_b : &state_a);
        std::vector<std::uint8_t> source_before;
        if (!inplace) {
            source_before = from_device<std::uint8_t>(current->data(), state_bytes_total);
            destination->fill(0xff);
        }
        Tensor input_state(current->span(), state_dtype(word),
                           {kStateDim, kStateDim, test_case.value_heads});
        Tensor output_state(destination->span(), state_dtype(word),
                            {kStateDim, kStateDim, test_case.value_heads});
        Tensor qs = q.slice(2, begin, length), ks = k.slice(2, begin, length);
        Tensor vs = v.slice(2, begin, length), gs = g.slice(1, begin, length);
        Tensor bs = beta.slice(1, begin, length), os = output.slice(2, begin, length);
        if (inplace) {
            ops::gated_delta_net(qs, ks, vs, gs, bs, scale, test_case.normalize_qk, workspace,
                                 output_state, os, device_execution);
        } else {
            ops::gated_delta_net(qs, ks, vs, gs, bs, scale, test_case.normalize_qk, workspace,
                                 input_state, output_state, os, device_execution);
        }
        cuda_synchronize();
        begin += length;
        if (word == StateWord::Bf16) publications.push_back(begin);
        const std::string checkpoint = label + " prefix=" + std::to_string(begin);
        const auto ref =
            reference_prefix(in, begin, scale, test_case.normalize_qk, publications);
        const auto output_count =
            static_cast<std::size_t>(begin) * test_case.value_heads * kStateDim;
        const auto got_output = from_device_bf16(out.data(), output_count);
        const auto got_state  = read_state(destination->data(), in.state.size(), word);
        // Calls after the first read the state the previous call published.
        const bool bf16      = word == StateWord::Bf16;
        const bool continued = bf16 && begin > partitions.front();
        failures += verify_heads(checkpoint + " out", got_output, ref.out, test_case.value_heads,
                                 begin, false, continued);
        if (partitions.size() > 1) {
            const auto offset =
                static_cast<std::size_t>(begin - length) * test_case.value_heads * kStateDim;
            failures += verify_heads(checkpoint + " current out",
                                     std::span<const double>(got_output).subspan(offset),
                                     std::span<const double>(ref.out).subspan(offset),
                                     test_case.value_heads, length, false, continued);
        }
        failures += verify_heads(checkpoint + " state", got_state, ref.final_state,
                                 test_case.value_heads, 1, true, bf16);
        if (pattern == InputPattern::ExactRounding) {
            failures += verify_exact(checkpoint + " exact state", got_state, ref.final_state);
        }
        if (pattern == InputPattern::ZeroQuery || pattern == InputPattern::ZeroSignal) {
            failures += verify_exact(checkpoint + " exact zero output", got_output,
                                     std::vector<double>(output_count, 0.0));
        }
        if (pattern == InputPattern::ZeroSignal) {
            failures += verify_exact(checkpoint + " exact zero state", got_state,
                                     std::vector<double>(in.state.size(), 0.0));
        }
        if (pattern == InputPattern::ZeroBeta) {
            failures += verify_exact(checkpoint + " identity state", got_state, doubles(in.state));
        }
        if (!inplace) {
            failures += verify_exact(checkpoint + " source state unchanged",
                                     from_device<std::uint8_t>(current->data(), state_bytes_total),
                                     source_before);
        }
        if (output_count < in.v.size()) {
            const auto* suffix = static_cast<const std::uint16_t*>(out.data()) + output_count;
            failures +=
                verify_exact(checkpoint + " unconsumed output unchanged",
                             from_device<std::uint16_t>(suffix, in.v.size() - output_count),
                             std::vector<std::uint16_t>(in.v.size() - output_count, 0xffffU));
        }
        failures += state_a.verify_guards(checkpoint + " state-a");
        failures += state_b.verify_guards(checkpoint + " state-b");
        failures += out.verify_guards(checkpoint + " out");
        failures += scratch.verify_guards(checkpoint + " workspace");
        if (workspace.used() != 0) {
            std::cerr << checkpoint << ": workspace was not released\n";
            ++failures;
        }
        current = destination;
    }
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace exceeded its interval capacity\n";
        ++failures;
    }
    return failures;
}

int batch_update_case(const Case& test_case, const std::vector<int>& source_slots,
                      const std::vector<int>& destination_slots, int slots, std::uint32_t seed,
                      StateWord word = StateWord::Fp32,
                      InputPattern pattern = InputPattern::Ordinary) {
    if (test_case.tokens != 1) { throw std::logic_error("batch_update_case requires W=1"); }
    if (source_slots.size() != destination_slots.size()) {
        throw std::logic_error("batch_update_case selector sizes differ");
    }
    const int batch   = static_cast<int>(source_slots.size());
    const int width   = 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const std::size_t qk_row_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * width);
    const std::size_t value_row_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * width);
    const std::size_t gate_row_size = static_cast<std::size_t>(test_case.value_heads * width);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);

    gdn_ref::Inputs aggregate;
    aggregate.head_dim    = kStateDim;
    aggregate.qk_heads    = test_case.qk_heads;
    aggregate.value_heads = test_case.value_heads;
    aggregate.tokens      = static_cast<std::int64_t>(width) * batch;
    aggregate.q.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.k.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.v.reserve(value_row_size * static_cast<std::size_t>(batch));
    aggregate.g.reserve(gate_row_size * static_cast<std::size_t>(batch));
    aggregate.beta.reserve(gate_row_size * static_cast<std::size_t>(batch));

    std::vector<gdn_ref::Inputs> rows;
    rows.reserve(static_cast<std::size_t>(batch));
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 0.125f);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Inputs input =
            make_inputs(test_case, seed + static_cast<std::uint32_t>(row) * 97U, pattern);
        aggregate.q.insert(aggregate.q.end(), input.q.begin(), input.q.end());
        aggregate.k.insert(aggregate.k.end(), input.k.begin(), input.k.end());
        aggregate.v.insert(aggregate.v.end(), input.v.begin(), input.v.end());
        aggregate.g.insert(aggregate.g.end(), input.g.begin(), input.g.end());
        aggregate.beta.insert(aggregate.beta.end(), input.beta.begin(), input.beta.end());
        std::copy(input.state.begin(), input.state.end(),
                  initial_states.begin() +
                      static_cast<std::size_t>(source_slots[static_cast<std::size_t>(row)]) *
                          state_size);
        rows.push_back(std::move(input));
    }
    if (word == StateWord::Bf16) round_to_bf16(initial_states);

    // Repeated source selectors share the final represented pool contents, independent of row
    // inputs.
    for (int row = 0; row < batch; ++row) {
        const auto begin = initial_states.begin() + std::size_t(source_slots[row]) * state_size;
        rows[row].state.assign(begin, begin + state_size);
    }
    std::vector<gdn_ref::Result> references;
    references.reserve(static_cast<std::size_t>(batch));
    std::vector<bool> written_slots(static_cast<std::size_t>(slots), false);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Result reference = gdn_ref::evaluate(
            rows[static_cast<std::size_t>(row)], static_cast<double>(scale),
            test_case.normalize_qk, false,
            word == StateWord::Bf16 ? std::vector<std::int64_t>{1} : std::vector<std::int64_t>{});
        written_slots[static_cast<std::size_t>(destination_slots[static_cast<std::size_t>(row)])] =
            true;
        references.push_back(std::move(reference));
    }

    DeviceInputs device(aggregate);
    GuardedDeviceBuffer states(initial_states.size() * state_word_bytes(word));
    GuardedDeviceBuffer out(aggregate.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(state_bytes(initial_states, word).data(), states.bytes());
    out.fill(0xff);

    Tensor q(device.q.span(), DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor k(device.k.span(), DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor v(device.v.span(), DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    Tensor g(device.g.span(), DType::FP32, {test_case.value_heads, width, batch});
    Tensor beta(device.beta.span(), DType::FP32, {test_case.value_heads, width, batch});
    Tensor states_tensor(states.span(), state_dtype(word),
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    RecordStateFixture history(states_tensor, 3);
    const auto source_offsets             = history.record_offsets(source_slots);
    const auto destination_offsets        = history.record_offsets(destination_slots);
    DeviceBuffer device_source_slots      = to_device(source_offsets);
    DeviceBuffer device_destination_slots = to_device(destination_offsets);
    Tensor source_slots_tensor(device_source_slots.span(), DType::I64, {batch});
    Tensor destination_slots_tensor(device_destination_slots.span(), DType::I64, {batch});
    Tensor out_tensor(out.span(), DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    ops::gated_delta_net_batch_update(q, k, v, g, beta, scale, test_case.normalize_qk,
                                      history.view(source_slots_tensor),
                                      history.view(destination_slots_tensor), out_tensor,
                                      ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " batch update B=" +
                              std::to_string(batch) + " state=" + state_word_name(word);
    int failures                         = 0;
    const std::vector<double> got_output = from_device_bf16(out.data(), aggregate.v.size());
    history.copy_back();
    failures += history.verify_guards("recurrent state");
    const std::vector<double> got_states = read_state(states.data(), initial_states.size(), word);
    const std::vector<double> initial_values = doubles(initial_states);
    for (int row = 0; row < batch; ++row) {
        const auto row_label = label + " row=" + std::to_string(row);
        const auto& reference = references[static_cast<std::size_t>(row)];
        failures += verify_heads(
            row_label + " out",
            std::span<const double>(got_output).subspan(row * value_row_size, value_row_size),
            reference.out, test_case.value_heads, 1, false);
        const std::size_t begin =
            static_cast<std::size_t>(destination_slots[static_cast<std::size_t>(row)]) * state_size;
        failures += verify_heads(
            row_label + " state", std::span<const double>(got_states).subspan(begin, state_size),
            reference.final_state, test_case.value_heads, 1, true, word == StateWord::Bf16);
        if (pattern == InputPattern::ExactRounding) {
            failures += verify_exact(
                row_label + " exact state",
                std::vector<double>(got_states.begin() + begin,
                                    got_states.begin() + begin + state_size),
                reference.final_state);
        }
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written_slots[static_cast<std::size_t>(slot)]) continue;
        const std::size_t begin = static_cast<std::size_t>(slot) * state_size;
        failures += verify_exact(
            label + " untouched slot " + std::to_string(slot),
            std::vector<double>(got_states.begin() + begin, got_states.begin() + begin + state_size),
            std::vector<double>(initial_values.begin() + begin,
                                initial_values.begin() + begin + state_size));
    }
    failures += verify_exact(label + " source selectors unchanged",
                             from_device<std::int64_t>(device_source_slots, source_offsets.size()),
                             source_offsets);
    failures += verify_exact(
        label + " destination selectors unchanged",
        from_device<std::int64_t>(device_destination_slots, destination_offsets.size()),
        destination_offsets);
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, aggregate, device.q, device.k, device.v,
                                               device.g, device.beta);
    return failures;
}

int graph_case(Case c, int batch, bool pooled, StateWord word = StateWord::Fp32) {
    const int slots              = 2 * batch;
    const std::size_t state_size = std::size_t(kStateDim) * kStateDim * c.value_heads;
    const std::size_t qk_size    = std::size_t(kStateDim) * c.qk_heads * c.tokens;
    const std::size_t value_size = std::size_t(kStateDim) * c.value_heads * c.tokens;
    const std::size_t gate_size  = std::size_t(c.value_heads) * c.tokens;
    Case aggregate_case          = c;
    aggregate_case.tokens *= batch;
    auto in = make_inputs(aggregate_case, 15100U);
    DeviceInputs device(in);
    GuardedDeviceBuffer state(state_size * slots * state_word_bytes(word));
    GuardedDeviceBuffer output(value_size * batch * sizeof(std::uint16_t));
    std::vector<std::int32_t> source(batch), destination(batch);
    DeviceBuffer source_device      = to_device(std::vector<std::int64_t>(batch)),
                 destination_device = to_device(std::vector<std::int64_t>(batch));
    Tensor q(device.q.span(), DType::BF16, {kStateDim, c.qk_heads, c.tokens, batch});
    Tensor k(device.k.span(), DType::BF16, {kStateDim, c.qk_heads, c.tokens, batch});
    Tensor v(device.v.span(), DType::BF16, {kStateDim, c.value_heads, c.tokens, batch});
    Tensor g(device.g.span(), DType::FP32, {c.value_heads, c.tokens, batch});
    Tensor beta(device.beta.span(), DType::FP32, {c.value_heads, c.tokens, batch});
    Tensor out(output.span(), DType::BF16, {kStateDim, c.value_heads, c.tokens, batch});
    Tensor states(state.span(), state_dtype(word), {kStateDim, kStateDim, c.value_heads, slots});
    RecordStateFixture history(states, 3);
    Tensor sources(source_device.span(), DType::I64, {batch});
    Tensor destinations(destination_device.span(), DType::I64, {batch});
    const auto capacity = pooled ? 0
                                 : ops::gated_delta_net_workspace_capacity_bytes(
                                       c.qk_heads, c.value_heads, c.tokens, c.tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 256));
    MemoryArena workspace(scratch.span());
    auto& context        = cuda_environment().context;
    const auto execution = context.execution_view();
    const float scale    = 1.0F / std::sqrt(float(kStateDim));
    const auto launch    = [&] {
        if (pooled) {
            ops::gated_delta_net_batch_update(q, k, v, g, beta, scale, c.normalize_qk,
                                              history.view(sources), history.view(destinations),
                                              out, execution);
        } else {
            auto si = history.tensor(0);
            auto so = history.tensor(1);
            ops::gated_delta_net(q, k, v, g, beta, scale, c.normalize_qk, workspace, si, so, out,
                                    execution);
        }
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int replay = 0; replay < 2; ++replay) {
        in                     = make_inputs(aggregate_case, 15100U + 97U * replay);
        const auto upload_bf16 = [](DeviceBuffer& buffer, const std::vector<float>& values) {
            const auto bits = bf16_bits(values);
            buffer.copy_from_host(bits.data(), buffer.size());
        };
        upload_bf16(device.q, in.q);
        upload_bf16(device.k, in.k);
        upload_bf16(device.v, in.v);
        device.g.copy_from_host(in.g.data(), device.g.size());
        device.beta.copy_from_host(in.beta.data(), device.beta.size());
        std::vector<float> initial(state_size * slots);
        for (int slot = 0; slot < slots; ++slot)
            for (std::size_t i = 0; i < state_size; ++i)
                initial[slot * state_size + i] = in.state[i] * (1.0F + 0.25F * slot);
        if (word == StateWord::Bf16) round_to_bf16(initial);
        for (int row = 0; row < batch; ++row) {
            source[row]      = pooled ? 2 * row + replay : 0;
            destination[row] = pooled ? 2 * row : 1;
        }
        state.copy_from_host(state_bytes(initial, word).data(), state.bytes());
        history.upload();
        const auto source_offsets      = history.record_offsets(source);
        const auto destination_offsets = history.record_offsets(destination);
        source_device.copy_from_host(source_offsets.data(), source_device.size());
        destination_device.copy_from_host(destination_offsets.data(), destination_device.size());
        output.fill(0xff);
        scratch.fill(0xff);
        cuda_synchronize();
        if (!graph.ready()) {
            launch(); // Initialize launch resources before capture, then restore the state.
            context.synchronize();
            history.upload();
            output.fill(0xff);
            scratch.fill(0xff);
            cuda_synchronize();
            definition.capture(context.stream(), launch);
            graph.instantiate(definition);
        }
        graph.launch(context.stream());
        context.synchronize();
        history.copy_back();
        const auto got_state    = read_state(state.data(), initial.size(), word);
        const auto initial_values = doubles(initial);
        const auto got_output   = from_device_bf16(output.data(), value_size * batch);
        const std::string label = std::string(c.name) + " replay=" + std::to_string(replay) +
                                  " state=" + state_word_name(word);
        for (int row = 0; row < batch; ++row) {
            gdn_ref::Inputs input;
            input.head_dim    = kStateDim;
            input.qk_heads    = c.qk_heads;
            input.value_heads = c.value_heads;
            input.tokens      = c.tokens;
            const auto slice  = [row](const std::vector<float>& values, std::size_t size) {
                return std::vector<float>(values.begin() + row * size,
                                           values.begin() + (row + 1) * size);
            };
            input.q    = slice(in.q, qk_size);
            input.k    = slice(in.k, qk_size);
            input.v    = slice(in.v, value_size);
            input.g    = slice(in.g, gate_size);
            input.beta = slice(in.beta, gate_size);
            input.state.assign(initial.begin() + source[row] * state_size,
                               initial.begin() + (source[row] + 1) * state_size);
            const auto reference = gdn_ref::evaluate(
                input, scale, c.normalize_qk, false,
                word == StateWord::Bf16 ? std::vector<std::int64_t>{c.tokens}
                                        : std::vector<std::int64_t>{});
            failures += verify_heads(label + " output row=" + std::to_string(row),
                                     std::span(got_output).subspan(row * value_size, value_size),
                                     reference.out, c.value_heads, c.tokens, false);
            failures += verify_heads(
                label + " state row=" + std::to_string(row),
                std::span<const double>(got_state).subspan(destination[row] * state_size,
                                                           state_size),
                reference.final_state, c.value_heads, 1, true, word == StateWord::Bf16);
        }
        for (int slot = 0; slot < slots; ++slot) {
            if (std::find(destination.begin(), destination.end(), slot) != destination.end())
                continue;
            failures +=
                verify_exact(label + " untouched state",
                             std::vector<double>(got_state.begin() + slot * state_size,
                                                 got_state.begin() + (slot + 1) * state_size),
                             std::vector<double>(initial_values.begin() + slot * state_size,
                                                 initial_values.begin() + (slot + 1) * state_size));
        }
        failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v,
                                                   device.g, device.beta);
        failures += verify_exact(label + " sources",
                                 from_device<std::int64_t>(source_device, batch), source_offsets);
        failures +=
            verify_exact(label + " destinations",
                         from_device<std::int64_t>(destination_device, batch), destination_offsets);
        failures += history.verify_guards(label);
        failures += output.verify_guards(label);
        failures += scratch.verify_guards(label);
        if (workspace.used() != 0 || workspace.peak_used() > capacity) ++failures;
    }
    return failures;
}

struct SplitRunResult {
    std::vector<double> last_token;
    std::vector<double> final_state;
};

// Prefill 128 plus one decode step must reproduce the one-shot 129-token tail of the same
// represented inputs within the shared GDN tolerance, including the final SSM state.
SplitRunResult run_split(const Case& test_case, const gdn_ref::Inputs& in, bool split) {
    const float scale    = 1.0F / std::sqrt(static_cast<float>(kStateDim));
    const int tokens     = test_case.tokens;
    const int max_call   = split ? tokens - 1 : tokens;
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    out.fill(0xff);
    Tensor q(device.q.span(), DType::BF16, {kStateDim, test_case.qk_heads, tokens});
    Tensor k(device.k.span(), DType::BF16, {kStateDim, test_case.qk_heads, tokens});
    Tensor v(device.v.span(), DType::BF16, {kStateDim, test_case.value_heads, tokens});
    Tensor g(device.g.span(), DType::FP32, {test_case.value_heads, tokens});
    Tensor beta(device.beta.span(), DType::FP32, {test_case.value_heads, tokens});
    Tensor ssm_state(state.span(), DType::FP32, {kStateDim, kStateDim, test_case.value_heads});
    Tensor output(out.span(), DType::BF16, {kStateDim, test_case.value_heads, tokens});
    const auto workspace_bytes =
        ops::gated_delta_net_workspace_capacity_bytes(test_case.qk_heads, test_case.value_heads,
                                                      1, max_call);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(workspace_bytes, 256));
    MemoryArena workspace(scratch.span());
    const auto device_execution = cuda_environment().default_execution;
    if (!split) {
        ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace, ssm_state,
                             output, device_execution);
    } else {
        int begin = 0;
        for (int length : {tokens - 1, 1}) {
            Tensor qs = q.slice(2, begin, length), ks = k.slice(2, begin, length);
            Tensor vs = v.slice(2, begin, length), gs = g.slice(1, begin, length);
            Tensor bs = beta.slice(1, begin, length), os = output.slice(2, begin, length);
            ops::gated_delta_net(qs, ks, vs, gs, bs, scale, test_case.normalize_qk, workspace,
                                 ssm_state, os, device_execution);
            begin += length;
        }
    }
    cuda_synchronize();
    if (workspace.used() != 0) throw std::logic_error("split run leaked workspace");
    return {from_device_bf16(static_cast<const std::uint16_t*>(out.data()) +
                                 static_cast<std::size_t>(tokens - 1) * test_case.value_heads *
                                     kStateDim,
                             test_case.value_heads * kStateDim),
            read_f32(state.data(), in.state.size())};
}

int split_vs_single_shot_case() {
    const Case test_case{"122b split", 16, 64, 129, true};
    const gdn_ref::Inputs in = make_inputs(test_case, 15429U);
    const auto one_shot      = run_split(test_case, in, false);
    const auto split         = run_split(test_case, in, true);
    int failures             = 0;
    const auto compare       = [&](const std::string& what, const std::vector<double>& got,
                                   const std::vector<double>& expected) {
        if (got.size() != expected.size()) throw std::logic_error("GDN split comparison size");
        const auto stats = compute_reduction_stats(got.data(), expected.data(), got.size());
        if (stats.first_non_finite >= 0 || stats.relative_l2 > 1.0e-2) {
            std::cerr << "GDN 122b split vs one-shot " << what
                      << ": rel_l2=" << stats.relative_l2 << " non_finite="
                      << stats.first_non_finite << '\n';
            ++failures;
        }
        return failures;
    };
    compare("last token", split.last_token, one_shot.last_token);
    compare("final state", split.final_state, one_shot.final_state);
    return failures;
}

int contract_rejection_cases() {
    DeviceBuffer q_buffer(ninfer::test::cuda_environment().context.device(),
                          kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer k_buffer(ninfer::test::cuda_environment().context.device(),
                          kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer v_buffer(ninfer::test::cuda_environment().context.device(),
                          kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer g_buffer(ninfer::test::cuda_environment().context.device(), 8 * sizeof(float));
    DeviceBuffer beta_buffer(ninfer::test::cuda_environment().context.device(), 8 * sizeof(float));
    DeviceBuffer state_buffer(ninfer::test::cuda_environment().context.device(),
                              kStateDim * kStateDim * 8 * sizeof(float));
    DeviceBuffer out_buffer(ninfer::test::cuda_environment().context.device(),
                            kStateDim * 8 * sizeof(std::uint16_t));
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(), 256);
    MemoryArena workspace(workspace_storage.span());
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));

    auto is_rejected = [&](int activation_dim, int state_dim, int qk_heads, int value_heads) {
        Tensor q(q_buffer.span(), DType::BF16, {activation_dim, qk_heads, 1});
        Tensor k(k_buffer.span(), DType::BF16, {activation_dim, qk_heads, 1});
        Tensor v(v_buffer.span(), DType::BF16, {activation_dim, value_heads, 1});
        Tensor g(g_buffer.span(), DType::FP32, {value_heads, 1});
        Tensor beta(beta_buffer.span(), DType::FP32, {value_heads, 1});
        Tensor state(state_buffer.span(), DType::FP32, {state_dim, state_dim, value_heads});
        Tensor out(out_buffer.span(), DType::BF16, {activation_dim, value_heads, 1});
        try {
            ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, state, out,
                                 cuda_environment().default_execution);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    if (!is_rejected(64, kStateDim, 4, 8)) {
        std::cerr << "gated_delta_net accepted Q/K/V head dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, 64, 4, 8)) {
        std::cerr << "gated_delta_net accepted state dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, kStateDim, 4, 6)) {
        std::cerr << "gated_delta_net accepted a non-divisible head map\n";
        ++failures;
    }
    return failures;
}

// State fields are FP32 or BF16, and every state field of one call shares the dtype.
int state_dtype_rejection_cases() {
    constexpr int kQkHeads = 1, kValueHeads = 3;
    auto& context          = ninfer::test::cuda_environment().context;
    const auto execution   = ninfer::test::cuda_environment().default_execution;
    const std::size_t state_elements = std::size_t(kStateDim) * kStateDim * kValueHeads;
    DeviceBuffer q(context.device(), std::size_t(kStateDim) * kQkHeads * 2);
    DeviceBuffer k(context.device(), std::size_t(kStateDim) * kQkHeads * 2);
    DeviceBuffer v(context.device(), std::size_t(kStateDim) * kValueHeads * 2);
    DeviceBuffer g(context.device(), kValueHeads * 4);
    DeviceBuffer beta(context.device(), kValueHeads * 4);
    DeviceBuffer out(context.device(), std::size_t(kStateDim) * kValueHeads * 2);
    DeviceBuffer fp32_state(context.device(), state_elements * 4);
    DeviceBuffer bf16_state(context.device(), state_elements * 2);
    DeviceBuffer offsets = to_device(std::vector<std::int64_t>{0});
    ninfer::DeviceBuffer workspace_storage(context.device(), 256);
    MemoryArena workspace(workspace_storage.span());
    const float scale = 1.0F / std::sqrt(static_cast<float>(kStateDim));
    Tensor qt(q.span(), DType::BF16, {kStateDim, kQkHeads, 1});
    Tensor kt(k.span(), DType::BF16, {kStateDim, kQkHeads, 1});
    Tensor vt(v.span(), DType::BF16, {kStateDim, kValueHeads, 1});
    Tensor gt(g.span(), DType::FP32, {kValueHeads, 1});
    Tensor bt(beta.span(), DType::FP32, {kValueHeads, 1});
    Tensor ot(out.span(), DType::BF16, {kStateDim, kValueHeads, 1});
    const auto state = [&](DeviceBuffer& buffer, DType dtype, int records = 0) {
        if (records == 0) return Tensor(buffer.span(), dtype, {kStateDim, kStateDim, kValueHeads});
        return Tensor(buffer.span(), dtype, {kStateDim, kStateDim, kValueHeads, records});
    };
    const auto rejected = [](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };
    int failures = 0;
    Tensor fp32 = state(fp32_state, DType::FP32), bf16 = state(bf16_state, DType::BF16);
    Tensor fp16 = state(bf16_state, DType::FP16);
    if (!rejected([&] {
            ops::gated_delta_net(qt, kt, vt, gt, bt, scale, true, workspace, fp32, bf16, ot,
                                 execution);
        })) {
        std::cerr << "gated_delta_net accepted FP32 input state with BF16 output state\n";
        ++failures;
    }
    if (!rejected([&] {
            ops::gated_delta_net(qt, kt, vt, gt, bt, scale, true, workspace, fp16, ot, execution);
        })) {
        std::cerr << "gated_delta_net accepted an FP16 state\n";
        ++failures;
    }
    Tensor selector(offsets.span(), DType::I64, {1});
    Tensor qb = qt.view({kStateDim, kQkHeads, 1, 1}), kb = kt.view({kStateDim, kQkHeads, 1, 1});
    Tensor vb = vt.view({kStateDim, kValueHeads, 1, 1}), ob = ot.view({kStateDim, kValueHeads, 1, 1});
    Tensor gb = gt.view({kValueHeads, 1, 1}), bb = bt.view({kValueHeads, 1, 1});
    if (!rejected([&] {
            ops::gated_delta_net_batch_update(
                qb, kb, vb, gb, bb, scale, true,
                dense_record_field(state(fp32_state, DType::FP32, 1), 3, selector),
                dense_record_field(state(bf16_state, DType::BF16, 1), 3, selector), ob, execution);
        })) {
        std::cerr << "gated_delta_net batch update accepted mixed state dtypes\n";
        ++failures;
    }
    return failures;
}

int batch_update_bound_rejection_case() {
    // The batch update admits one row per decode lane; one past the engine ceiling is rejected
    // before any state field or storage is consulted.
    constexpr int kQkHeads    = 1;
    constexpr int kValueHeads = 3;
    constexpr int kBatch      = static_cast<int>(kMaximumConcurrency) + 1;
    auto& context             = ninfer::test::cuda_environment().context;
    DeviceBuffer q(context.device(), std::size_t(kStateDim) * kQkHeads * kBatch * 2);
    DeviceBuffer k(context.device(), std::size_t(kStateDim) * kQkHeads * kBatch * 2);
    DeviceBuffer v(context.device(), std::size_t(kStateDim) * kValueHeads * kBatch * 2);
    DeviceBuffer g(context.device(), std::size_t(kValueHeads) * kBatch * 4);
    DeviceBuffer beta(context.device(), std::size_t(kValueHeads) * kBatch * 4);
    DeviceBuffer out(context.device(), std::size_t(kStateDim) * kValueHeads * kBatch * 2);
    DeviceBuffer states(context.device(), std::size_t(kStateDim) * kStateDim * kValueHeads * 4);
    DeviceBuffer offsets(context.device(), std::size_t(kBatch) * 8);
    Tensor qt(q.span(), DType::BF16, {kStateDim, kQkHeads, 1, kBatch});
    Tensor kt(k.span(), DType::BF16, {kStateDim, kQkHeads, 1, kBatch});
    Tensor vt(v.span(), DType::BF16, {kStateDim, kValueHeads, 1, kBatch});
    Tensor gt(g.span(), DType::FP32, {kValueHeads, 1, kBatch});
    Tensor bt(beta.span(), DType::FP32, {kValueHeads, 1, kBatch});
    Tensor ot(out.span(), DType::BF16, {kStateDim, kValueHeads, 1, kBatch});
    Tensor states_tensor(states.span(), DType::FP32, {kStateDim, kStateDim, kValueHeads, 1});
    RecordStateFixture history(states_tensor, 3);
    Tensor selector(offsets.span(), DType::I64, {kBatch});
    try {
        ops::gated_delta_net_batch_update(
            qt, kt, vt, gt, bt, 1.0F / std::sqrt(static_cast<float>(kStateDim)), true,
            history.view(selector), history.view(selector), ot,
            ninfer::test::cuda_environment().default_execution);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "gated_delta_net batch update accepted B=kMaximumConcurrency+1\n";
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("gdn/contract", [] {
        int failures               = 0;
        const std::size_t interval = ops::gated_delta_net_workspace_capacity_bytes(16, 48, 15, 17);
        const std::size_t witness  = ops::gated_delta_net_workspace_capacity_bytes(16, 48, 17, 17);
        if (interval != witness) {
            std::cerr << "gated_delta_net interval capacity missed the chunk boundary\n";
            ++failures;
        }
        try {
            (void)ops::gated_delta_net_workspace_capacity_bytes(16, 48, 0, 65);
            std::cerr << "gated_delta_net accepted an invalid token interval\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
        try {
            (void)ops::gated_delta_net_workspace_capacity_bytes(4, 6, 1, 65);
            std::cerr << "gated_delta_net workspace accepted a non-divisible head map\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
        failures += contract_rejection_cases();
        failures += state_dtype_rejection_cases();
        failures += batch_update_bound_rejection_case();
        return failures;
    });
    const auto add = [&](Case c, unsigned seed, InputPattern pattern = InputPattern::Ordinary,
                         bool inplace = false, std::vector<int> chunks = {},
                         StateWord word = StateWord::Fp32) {
        auto id = (std::string(c.name).starts_with("local") ? "gdn_local/" : "gdn/") +
                  std::string(c.name) + "/h" + std::to_string(c.qk_heads) + "/v" +
                  std::to_string(c.value_heads) + "/t" + std::to_string(c.tokens) +
                  (c.normalize_qk ? "/normalized" : "/raw") + (inplace ? "/alias" : "/disjoint");
        if (word == StateWord::Bf16) id += "/bf16-state";
        suite.add(id, [=] { return run_case(c, seed, pattern, inplace, chunks, word); },
                  {{"execution", "eager"}});
    };
    const auto batch = [&](Case c, std::vector<int> source, std::vector<int> destination, int slots,
                           unsigned seed, StateWord word = StateWord::Fp32,
                           InputPattern pattern = InputPattern::Ordinary) {
        std::string id =
            (std::string(c.name).starts_with("local") ? "gdn_batch_local/" : "gdn/batch/") +
            std::string(c.name) + "/b" + std::to_string(source.size()) +
            (c.normalize_qk ? "/normalized" : "/raw");
        if (word == StateWord::Bf16) id += "/bf16-state";
        for (int x : source) id += "/s" + std::to_string(x);
        for (int x : destination) id += "/d" + std::to_string(x);
        suite.add(id, [=] {
            return batch_update_case(c, source, destination, slots, seed, word, pattern);
        });
    };
    for (bool normalize : {false, true}) {
        for (int t : {1,   2,   3,   4,   5,   6,   7,    8,    9,    10,   11,  12,  13,
                      14,  15,  16,  17,  31,  32,  33,   63,   64,   65,   127, 128, 129,
                      255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2048, 4096})
            for (bool alias : {false, true})
                add({"local", 8, 24, t, normalize}, 21000U + t, InputPattern::Ordinary, alias);
        for (int t : {1, 10, 11, 12, 16, 17, 129, 1024})
            suite.add("gdn_local/graph/t" + std::to_string(t) +
                          (normalize ? "/normalized" : "/raw"),
                      [=] { return graph_case({"local-graph", 8, 24, t, normalize}, 1, false); },
                      {{"execution", "graph"}});
    }
    for (auto pattern : {InputPattern::WeakDecay, InputPattern::ZeroDecay, InputPattern::SmallBeta,
                         InputPattern::NearCollinear, InputPattern::ZeroQuery,
                         InputPattern::ZeroSignal, InputPattern::ZeroBeta, InputPattern::UnitBeta})
        for (bool alias : {false, true}) {
            const char* name = pattern == InputPattern::WeakDecay       ? "local-weak"
                               : pattern == InputPattern::ZeroDecay     ? "local-zero-decay"
                               : pattern == InputPattern::SmallBeta     ? "local-small-beta"
                               : pattern == InputPattern::NearCollinear ? "local-collinear"
                               : pattern == InputPattern::ZeroQuery     ? "local-zero-query"
                               : pattern == InputPattern::ZeroSignal    ? "local-zero-signal"
                               : pattern == InputPattern::ZeroBeta      ? "local-zero-beta"
                                                                        : "local-unit-beta";
            add({name, 8, 24, 257, true}, 23001U + static_cast<unsigned>(pattern), pattern, alias);
        }
    add({"local-continuation", 8, 24, 1025, true}, 24001U, InputPattern::WeakDecay, true,
        {15, 17, 992, 1});
    add({"27b recurrent", 16, 48, 1, true}, 12001U, InputPattern::Ordinary, true);
    add({"27b recurrent", 16, 48, 7, false}, 12007U);
    add({"35b recurrent", 16, 32, 15, true}, 12015U);
    // Qwen3.5-122B GDN geometry: 16 Q/K heads, 64 value heads, per-token normalization.
    add({"122b recurrent", 16, 64, 257, true}, 15001U, InputPattern::Ordinary, true);
    add({"122b recurrent", 16, 64, 257, false}, 15201U, InputPattern::Ordinary, false);
    add({"122b decode", 16, 64, 16, true}, 15016U, InputPattern::Ordinary, true,
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    add({"122b single-shot", 16, 64, 129, true}, 15129U, InputPattern::Ordinary, true);
    // Same seed as the single-shot case: identical inputs, so the committed decode token must
    // match the one-shot 129-token prefix within the shared reference tolerance.
    add({"122b prefill continuation", 16, 64, 129, true}, 15129U, InputPattern::Ordinary, true,
        {128, 1});
    suite.add("gdn/122b split-vs-single-shot", [] { return split_vs_single_shot_case(); });

    struct Geometry {
        const char* name;
        int qk_heads, value_heads, tokens;
    };

    for (const auto geometry :
         {Geometry{"qwen-27b", 16, 48, 1025}, Geometry{"qwen-35b-a3b", 16, 32, 1025},
          Geometry{"kimi-k3", 96, 96, 1025}, Geometry{"glm5.3-flash", 64, 64, 1025},
          Geometry{"single-head", 1, 1, 65}, Geometry{"shared-qk", 1, 3, 65},
          Geometry{"generic-grouped-map", 5, 15, 65}}) {
        for (bool normalize : {false, true}) {
            add({geometry.name, geometry.qk_heads, geometry.value_heads, geometry.tokens,
                 normalize},
                16180U + geometry.value_heads, InputPattern::Ordinary, !normalize);
        }
    }
    for (int tokens : {15, 16, 17, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025, 4095, 4096}) {
        add({"prefill boundary", 1, 3, tokens, true}, 90000U + tokens, InputPattern::Ordinary,
            tokens % 2 != 0);
    }
    for (int tokens : {15, 16, 17}) {
        add({"raw prefill boundary", 5, 15, tokens, false}, 91000U + tokens);
    }

    for (int heads : {32, 48}) {
        for (int tokens : {127, 128, 129}) {
            add({"prefill tiling boundary", 16, heads, tokens, tokens != 127},
                92000U + heads * 100U + tokens);
        }
    }

    struct Stress {
        const char* name;
        InputPattern pattern;
        int tokens;
        std::uint32_t seed;
    };

    // Retain the previously observed error cases, including long weak/zero decay. The limits
    // above are fixed before qualification, shared by every geometry and every input pattern.
    for (const auto stress :
         {Stress{"weak decay", InputPattern::WeakDecay, 16385, 332211U},
          Stress{"zero decay", InputPattern::ZeroDecay, 16385, 332212U},
          Stress{"small beta", InputPattern::SmallBeta, 16385, 332213U},
          Stress{"strong decay", InputPattern::StrongDecay, 16385, 332214U},
          Stress{"near collinear", InputPattern::NearCollinear, 4097, 332215U}}) {
        add({stress.name, 1, 3, stress.tokens, true}, stress.seed, stress.pattern);
    }
    for (const auto stress : {Stress{"raw weak decay", InputPattern::WeakDecay, 16385, 332211U},
                              Stress{"raw zero decay", InputPattern::ZeroDecay, 16385, 332212U}}) {
        add({stress.name, 1, 3, stress.tokens, false}, stress.seed, stress.pattern, true);
    }
    add({"35b zero decay", 16, 32, 4097, true}, 16032U, InputPattern::ZeroDecay);
    add({"35b weak decay", 16, 32, 4097, true}, 16032U, InputPattern::WeakDecay, true);
    add({"122b zero decay", 16, 64, 2049, true}, 16064U, InputPattern::ZeroDecay);
    add({"122b weak decay", 16, 64, 2049, true}, 16064U, InputPattern::WeakDecay, true);
    add({"122b prefill continuation", 16, 64, 2049, true}, 14264U, InputPattern::ZeroState, false,
        {1024, 1024, 1});
    add({"27b strong tail", 16, 48, 65, true}, 12065U, InputPattern::StrongDecay);
    // The first checkpoint is the aligned T=64 prefix of the identical T=65 stress input.
    add({"27b strong continuation", 16, 48, 65, true}, 12065U, InputPattern::StrongDecay, true,
        {64, 1});

    add({"27b prefill continuation", 16, 48, 2049, true}, 14249U, InputPattern::ZeroState, false,
        {1024, 1024, 1});
    add({"weak prefill continuation", 1, 3, 4097, true}, 14297U, InputPattern::WeakDecay, true,
        {1024, 1024, 1024, 1024, 1});
    add({"raw uneven continuation", 5, 15, 1025, false}, 14125U, InputPattern::ZeroDecay, false,
        {15, 17, 992, 1});
    add({"near-zero normalization", 1, 3, 17, true, true}, 14017U);
    for (bool normalize : {false, true}) {
        add({"zero query", 1, 3, 17, normalize}, 14317U, InputPattern::ZeroQuery);
        add({"zero signal", 1, 3, 17, normalize}, 14417U, InputPattern::ZeroSignal, true);
        add({"zero beta", 1, 3, 17, normalize}, 14517U, InputPattern::ZeroBeta);
        add({"unit beta", 1, 3, 257, normalize}, 14657U, InputPattern::UnitBeta, true);
    }

    for (int b = 1; b <= 8; ++b) {
        std::vector<int> source(b), alias(b), distinct(b), shared(b, 0), fork(b);
        for (int row = 0; row < b; ++row) {
            source[row]   = 2 * row + 2;
            alias[row]    = source[row];
            distinct[row] = source[row] + 1;
            fork[row]     = 2 * row + 1;
        }
        for (bool normalize : {false, true}) {
            batch({"local-ordinary", 8, 24, 1, normalize}, source, alias, 2 * b + 4, 25000U + b);
            batch({"local-ordinary", 8, 24, 1, normalize}, source, distinct, 2 * b + 4, 25000U + b);
            batch({"local-fork", 8, 24, 1, normalize}, shared, fork, 2 * b + 4, 26000U + b);
            batch({"local-near-zero", 8, 24, 1, normalize, true}, source, distinct, 2 * b + 4,
                  27000U + b);
            suite.add(
                "gdn_batch_local/graph/b" + std::to_string(b) +
                    (normalize ? "/normalized" : "/raw"),
                [=] { return graph_case({"local-batch-graph", 8, 24, 1, normalize}, b, true); },
                {{"execution", "graph"}});
        }
    }
    // The production decode path updates selected state-pool slots in place at width one.
    batch({"27b selected-slot fused-qk-norm", 16, 48, 1, true}, {7}, {7}, 8, 12101u);
    batch({"35b selected-slot near-zero", 16, 32, 1, true, true}, {6}, {6}, 8, 12201u);
    batch({"35b ordinary", 16, 32, 1, true}, {8, 9, 10, 11, 12, 13, 14, 15},
          {8, 9, 10, 11, 12, 13, 14, 15}, 16, 13001u);
    batch({"35b mixed fork destinations", 16, 32, 1, true}, {0, 2, 4, 6}, {1, 3, 5, 7}, 8, 13101u);
    // The new execution model decodes one in-place row per lane; carry the 122B head map at a
    // wide decode batch far past the old sixteen-row bound, with state continuing per row.
    {
        std::vector<int> wide(520);
        std::iota(wide.begin(), wide.end(), 0);
        batch({"122b wide decode", 16, 64, 1, true}, wide, wide, 520, 15320u);
    }

    // BF16 State storage: every publication rounds to nearest even and the FP64 reference rounds
    // the exact state at the same publications. The geometries reach every recurrent and chunked
    // kernel: value tiles 8 (1,3), 16 (5,15), 32 (8,24) and the 64-value register stripe (16,64).
    constexpr auto kBf16 = StateWord::Bf16;
    for (bool normalize : {false, true}) {
        for (int t : {1, 2, 15, 16, 17, 64, 129, 1025})
            for (bool alias : {false, true})
                add({"local", 8, 24, t, normalize}, 31000U + t, InputPattern::Ordinary, alias, {},
                    kBf16);
        for (const auto [qk, value] : {std::pair{1, 3}, std::pair{5, 15}})
            add({"bf16 tile", qk, value, 65, normalize}, 31100U + value, InputPattern::Ordinary,
                !normalize, {}, kBf16);
        add({"zero signal", 1, 3, 17, normalize}, 31417U, InputPattern::ZeroSignal, true, {}, kBf16);
        add({"zero beta", 1, 3, 17, normalize}, 31517U, InputPattern::ZeroBeta, false, {}, kBf16);
    }
    add({"local-continuation", 8, 24, 1025, true}, 34001U, InputPattern::WeakDecay, true,
        {15, 17, 992, 1}, kBf16);
    add({"122b recurrent", 16, 64, 257, true}, 35001U, InputPattern::Ordinary, true, {}, kBf16);
    add({"122b recurrent", 16, 64, 257, false}, 35201U, InputPattern::Ordinary, false, {}, kBf16);
    add({"122b decode", 16, 64, 16, true}, 35016U, InputPattern::Ordinary, true,
        std::vector<int>(16, 1), kBf16);
    add({"122b prefill continuation", 16, 64, 129, true}, 35129U, InputPattern::Ordinary, true,
        {128, 1}, kBf16);
    add({"122b prefill continuation", 16, 64, 2049, true}, 34264U, InputPattern::ZeroState, false,
        {1024, 1024, 1}, kBf16);
    add({"122b weak decay", 16, 64, 2049, true}, 36064U, InputPattern::WeakDecay, true, {}, kBf16);
    // Exactly formed states pin nearest-even rounding and its points: one call publishes only its
    // final state, a continuation publishes at every call.
    for (const auto word : {StateWord::Fp32, kBf16}) {
        for (int t : {1, 2, 3})
            add({"exact rounding", 2, 6, t, false}, 37000U + t, InputPattern::ExactRounding, false,
                {}, word);
        add({"exact rounding continuation", 2, 6, 3, false}, 37010U, InputPattern::ExactRounding,
            true, {1, 2}, word);
        batch({"exact rounding", 2, 6, 1, false}, {1, 2, 3, 0}, {1, 2, 3, 0}, 4, 37020U, word,
              InputPattern::ExactRounding);
    }
    for (int b : {1, 4, 8}) {
        std::vector<int> source(b), distinct(b), shared(b, 0), fork(b);
        for (int row = 0; row < b; ++row) {
            source[row]   = 2 * row + 2;
            distinct[row] = source[row] + 1;
            fork[row]     = 2 * row + 1;
        }
        for (bool normalize : {false, true}) {
            batch({"local-ordinary", 8, 24, 1, normalize}, source, source, 2 * b + 4, 45000U + b,
                  kBf16);
            batch({"local-ordinary", 8, 24, 1, normalize}, source, distinct, 2 * b + 4, 45000U + b,
                  kBf16);
            batch({"local-fork", 8, 24, 1, normalize}, shared, fork, 2 * b + 4, 46000U + b, kBf16);
        }
    }
    {
        std::vector<int> wide(160);
        std::iota(wide.begin(), wide.end(), 0);
        batch({"122b wide decode", 16, 64, 1, true}, wide, wide, 160, 35320u, kBf16);
    }
    // A BF16 batch update changes its passes per CTA at 48 rows: both sides, in place and into
    // other slots.
    for (int rows : {47, 48}) {
        std::vector<int> source(rows), moved(rows);
        std::iota(source.begin(), source.end(), 0);
        for (int row = 0; row < rows; ++row) moved[row] = rows + (row + 7) % rows;
        batch({"122b decode rows", 16, 64, 1, true}, source, source, 2 * rows, 35400U + rows,
              kBf16);
        batch({"122b decode rows", 16, 64, 1, false}, source, moved, 2 * rows, 35500U + rows,
              kBf16);
    }
    suite.add("gdn_batch_local/graph/b4/normalized/bf16-state",
              [] { return graph_case({"local-batch-graph", 8, 24, 1, true}, 4, true, kBf16); },
              {{"execution", "graph"}});
    suite.add("gdn/graph/running-t17/bf16-state",
              [] { return graph_case({"running-t17", 16, 48, 17, true}, 1, false, kBf16); },
              {{"execution", "graph"}});

    suite.add("gdn/graph/running-t1",
              [] { return graph_case({"running-t1", 16, 32, 1, false}, 1, false); },
              {{"execution", "graph"}});
    suite.add("gdn/graph/running-t17",
              [] { return graph_case({"running-t17", 16, 48, 17, true}, 1, false); },
              {{"execution", "graph"}});
    suite.add("gdn/graph/batch-update",
              [] { return graph_case({"batch-update", 16, 48, 1, true}, 4, true); },
              {{"execution", "graph"}});

    return suite.main(argc, argv);
}
