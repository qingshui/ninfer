// GDN projections of the Qwen3.5-122B geometry with E4M3 row-scaled weights (FP32 scales) on the
// B200 tcgen05 route: linear_split for the qkvz parent (n=20480, k=3072, segments {12288, 8192})
// and linear_add for the output projection (n=3072, k=8192). Both quantize the activation per
// token, so the AllowA8 criterion applies. The reference decodes the stored codes and scales with
// the fixture's own code and accumulates in FP64 from the exact BF16 activations. Decode token
// counts (at most 128) split the K loop of the last round's tiles; 48 to 128 tokens cover the 64-
// and 128-token tiles there.

#include "ops/linear/fp8_row_fp32_t4/helpers.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_split.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::t4;

constexpr double kBf16Roundoff = 1.0 / 256.0;
constexpr ReductionCriterion kA8Criterion{0.04, kBf16Roundoff, 0.06};
constexpr std::array kTokens{1, 2, 17, 48, 64, 96, 128, 130, 257, 520, 1025, 3200};
constexpr int kMaxTokens = 3200;

double dot(const std::vector<double>& row, const std::vector<std::uint16_t>& act, int k, int token) {
    double acc = 0.0;
    for (int c = 0; c < k; ++c)
        acc += row[c] * static_cast<double>(bf16_to_f32(act[static_cast<std::size_t>(token) * k + c]));
    return acc;
}

int count_unwritten(const std::vector<std::uint16_t>& words, std::size_t count) {
    int bad = 0;
    for (std::size_t i = 0; i < count; ++i) bad += !bf16_is_finite(words[i]);
    return bad;
}

int split_case(int tokens) {
    constexpr int kN = 20480, kK = 3072, kFirst = 12288, kSecond = 8192;
    static const Fp8RowFp32Rep rep(kN, kK);
    static const auto act = make_activation(kK, kMaxTokens);
    const Weight w        = rep.view();
    const ops::SingleProjectionWeight parent{w, ops::LinearPolicy::AllowA8};
    const std::array<std::int32_t, 2> rows_per_output{kFirst, kSecond};
    const std::size_t capacity = ops::linear_split_workspace_capacity_bytes(
        parent, rows_per_output, tokens, tokens);
    GuardedDeviceBuffer x_dev(act.size() * 2), first_dev(std::size_t{kFirst} * kMaxTokens * 2),
        second_dev(std::size_t{kSecond} * kMaxTokens * 2),
        scratch(std::max<std::size_t>(capacity + 256, 256));
    x_dev.copy_from_host(act.data(), act.size() * 2);
    first_dev.fill(0xff);
    second_dev.fill(0xff);
    MemoryArena arena(scratch.span());
    Tensor x(x_dev.span().subspan(0, std::size_t{kK} * tokens * 2), DType::BF16, {kK, tokens});
    const std::array outputs{
        Tensor(first_dev.span().subspan(0, std::size_t{kFirst} * tokens * 2), DType::BF16,
               {kFirst, tokens}),
        Tensor(second_dev.span().subspan(0, std::size_t{kSecond} * tokens * 2), DType::BF16,
               {kSecond, tokens})};
    const auto exec = cuda_environment().default_execution;
    ops::linear_split(x, parent, outputs, arena, exec);
    cuda_check(cudaStreamSynchronize(exec.stream()), "synchronize linear_split");

    std::vector<std::uint16_t> first(std::size_t{kFirst} * tokens),
        second(std::size_t{kSecond} * tokens);
    first_dev.copy_to_host(first.data(), first.size() * 2);
    second_dev.copy_to_host(second.data(), second.size() * 2);
    const std::string name = "fp8 split n20480_k3072 T=" + std::to_string(tokens);
    int failures           = 0;
    if (const int bad = count_unwritten(first, first.size()) + count_unwritten(second, second.size())) {
        std::cerr << name << ": " << bad << " non-finite or unwritten outputs\n";
        ++failures;
    }
    auto rows = sample_rows(kN);
    rows.insert(rows.end(), {kFirst - 1, kFirst, kFirst + 1});
    std::vector<double> got, ref;
    for (const std::int32_t row : rows) {
        const auto decoded = rep.decode_row(row);
        for (const std::int32_t token : sample_tokens(tokens)) {
            ref.push_back(dot(decoded, act, kK, token));
            const auto& segment = row < kFirst ? first : second;
            const int width     = row < kFirst ? kFirst : kSecond;
            const int column    = row < kFirst ? row : row - kFirst;
            got.push_back(bf16_to_f32(segment[static_cast<std::size_t>(token) * width + column]));
        }
    }
    failures += verify_reduction(name, got, ref, kA8Criterion);
    failures += first_dev.verify_guards(name + " first output");
    failures += second_dev.verify_guards(name + " second output");
    failures += x_dev.verify_guards(name + " input");
    failures += scratch.verify_guards(name + " workspace");
    failures += rep.verify_preserved(name.c_str());
    return failures;
}

int add_case(int tokens) {
    constexpr int kN = 3072, kK = 8192;
    static const Fp8RowFp32Rep rep(kN, kK);
    static const auto act = make_activation(kK, kMaxTokens);
    static const auto residual_bits = [] {
        std::mt19937 rng(91U);
        std::uniform_real_distribution<float> dist(-2.F, 2.F);
        std::vector<std::uint16_t> bits(std::size_t{kN} * kMaxTokens);
        for (auto& b : bits) b = f32_to_bf16(dist(rng));
        return bits;
    }();
    const Weight w             = rep.view();
    const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
        w.qtype, kN, kK, ops::LinearPolicy::AllowA8, tokens, tokens);
    GuardedDeviceBuffer x_dev(act.size() * 2), residual_dev(residual_bits.size() * 2),
        scratch(std::max<std::size_t>(capacity + 256, 256));
    x_dev.copy_from_host(act.data(), act.size() * 2);
    residual_dev.copy_from_host(residual_bits.data(), residual_bits.size() * 2);
    MemoryArena arena(scratch.span());
    Tensor x(x_dev.span().subspan(0, std::size_t{kK} * tokens * 2), DType::BF16, {kK, tokens});
    Tensor residual(residual_dev.span().subspan(0, std::size_t{kN} * tokens * 2), DType::BF16,
                    {kN, tokens});
    const auto exec = cuda_environment().default_execution;
    ops::linear_add(x, w, residual, ops::LinearPolicy::AllowA8, arena, exec);
    cuda_check(cudaStreamSynchronize(exec.stream()), "synchronize linear_add");

    std::vector<std::uint16_t> out(std::size_t{kN} * tokens);
    residual_dev.copy_to_host(out.data(), out.size() * 2);
    const std::string name = "fp8 add n3072_k8192 T=" + std::to_string(tokens);
    int failures           = 0;
    if (const int bad = count_unwritten(out, out.size())) {
        std::cerr << name << ": " << bad << " non-finite outputs\n";
        ++failures;
    }
    std::vector<double> got, ref;
    for (const std::int32_t row : sample_rows(kN)) {
        const auto decoded = rep.decode_row(row);
        for (const std::int32_t token : sample_tokens(tokens)) {
            const std::size_t at = static_cast<std::size_t>(token) * kN + row;
            ref.push_back(dot(decoded, act, kK, token) + bf16_to_f32(residual_bits[at]));
            got.push_back(bf16_to_f32(out[at]));
        }
    }
    failures += verify_reduction(name, got, ref, kA8Criterion);
    failures += residual_dev.verify_guards(name + " residual");
    failures += x_dev.verify_guards(name + " input");
    failures += scratch.verify_guards(name + " workspace");
    failures += rep.verify_preserved(name.c_str());
    return failures;
}

// The tcgen05 route multiplies E4M3 codes: an A16-only permission has no route and must throw.
int a16_rejected() {
    const Fp8RowFp32Rep split_rep(20480, 3072), add_rep(3072, 8192);
    const auto act = make_activation(8192, 1);
    GuardedDeviceBuffer x_dev(act.size() * 2), out_dev(20480 * 2), scratch(1 << 20);
    x_dev.copy_from_host(act.data(), act.size() * 2);
    MemoryArena arena(scratch.span());
    const auto exec = cuda_environment().default_execution;
    int failures    = 0;
    try {
        Tensor x(x_dev.span().subspan(0, 3072 * 2), DType::BF16, {3072, 1});
        const std::array outputs{Tensor(out_dev.span().subspan(0, 12288 * 2), DType::BF16, {12288, 1}),
                                 Tensor(out_dev.span().subspan(12288 * 2, 8192 * 2), DType::BF16,
                                        {8192, 1})};
        ops::linear_split(x, ops::SingleProjectionWeight{split_rep.view(), ops::LinearPolicy::A16Only},
                          outputs, arena, exec);
        std::cerr << "fp8 split n20480_k3072 accepted A16Only\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        Tensor x(x_dev.span(), DType::BF16, {8192, 1});
        Tensor residual(out_dev.span().subspan(0, 3072 * 2), DType::BF16, {3072, 1});
        ops::linear_add(x, add_rep.view(), residual, ops::LinearPolicy::A16Only, arena, exec);
        std::cerr << "fp8 add n3072_k8192 accepted A16Only\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

int run() {
    int failures = a16_rejected();
    for (const int tokens : kTokens) failures += split_case(tokens);
    for (const int tokens : kTokens) failures += add_case(tokens);
    return failures;
}

} // namespace

int main() {
    try {
        const int failures = run();
        std::cout << (failures == 0 ? "PASS" : "FAIL") << ": linear_fp8_b200\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "linear_fp8_b200: unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
