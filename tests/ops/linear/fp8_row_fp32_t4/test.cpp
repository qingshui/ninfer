// T4 / F2: FP8_E4M3FN_ROW_FP32 and the Spark-X2.6-1.7B Linear shapes.
//
// Contract under test (spec F2, include/ninfer/ops/linear.h):
//   * the artifact format name "fp8_e4m3fn_row_fp32" is a known tensor format;
//   * Y[n,t] = sum_k decode(W)[n,k] * X[k,t] for every registered Spark shape
//     (qkv 3072x2048, g 8x2048, out 2048x2048, gate_up 13312x2048, down 2048x6656,
//     head 131072x2048), decode M in 1..16 and a large prefill M, for the FP32-row-scale FP8
//     representation (AllowA8 criterion) and the BF16 representation;
//   * BF16 weights never take an FP8 branch: even under AllowA8 they must meet the A16
//     criterion, which a per-token FP8 activation quantization cannot;
//   * an unregistered shape still throws.
//
// The reference decodes the stored E4M3 codes and FP32 row scales (or BF16 words) with its own
// code (reps.h) and accumulates in FP64 from the exact BF16 activations. Only a spread of rows
// is compared per call; the spread always contains the first and last row.

#include "helpers.h"

#include "artifact/formats.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <algorithm>
#include <exception>
#include <string>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::t4;

constexpr double kBf16Roundoff = 1.0 / 256.0;

// Public numeric policy: A16 = bf16 roundoff, A8 = 4% quantization allowance.
constexpr ReductionCriterion kA16Criterion{kBf16Roundoff, kBf16Roundoff, 2.0 * kBf16Roundoff};
constexpr ReductionCriterion kA8Criterion{0.04, kBf16Roundoff, 0.06};

// One representation x one shape: decode M = 1..16 plus a large prefill M against FP64.
int check_shape(const Shape& s, const Rep& rep, ops::LinearPolicy policy,
                const ReductionCriterion& criterion, const std::string& label) {
    const Weight w = rep.view();
    const auto act = make_activation(s.k, kMaxTokens);
    GuardedDeviceBuffer x_dev(act.size() * 2),
        out_dev(static_cast<std::size_t>(s.n) * kMaxTokens * 2);
    x_dev.copy_from_host(act.data(), act.size() * 2);
    const std::size_t ws =
        ops::linear_workspace_capacity_bytes(w.qtype, s.n, s.k, policy, 1, kMaxTokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(ws, 256));

    const auto rows = sample_rows(s.n);
    std::vector<std::vector<double>> decoded;
    for (const std::int32_t r : rows) decoded.push_back(rep.decode_row(r));

    std::vector<std::int32_t> ts;
    for (std::int32_t t = 1; t <= 16; ++t) ts.push_back(t);
    ts.push_back(kPrefillTokens);

    int failures = 0;
    std::vector<std::uint16_t> host_out(static_cast<std::size_t>(s.n) * kMaxTokens);
    const auto exec = cuda_environment().default_execution;
    for (const std::int32_t t : ts) {
        const std::string name = label + " M=" + std::to_string(t);
        out_dev.fill(0xff);
        MemoryArena arena(scratch.span());
        Tensor x(x_dev.span().subspan(0, static_cast<std::size_t>(s.k) * t * 2), DType::BF16,
                 {s.k, t});
        Tensor out(out_dev.span().subspan(0, static_cast<std::size_t>(s.n) * t * 2), DType::BF16,
                   {s.n, t});
        ops::linear(x, w, out, policy, arena, exec);
        cuda_check(cudaStreamSynchronize(exec.stream()), "synchronize linear");

        const std::size_t words = static_cast<std::size_t>(s.n) * t;
        out_dev.copy_to_host(host_out.data(), words * 2);
        std::size_t bad = 0;
        for (std::size_t i = 0; i < words; ++i) bad += !bf16_is_finite(host_out[i]);
        if (bad != 0) {
            std::cerr << name << ": " << bad << " non-finite or unwritten outputs\n";
            ++failures;
        }

        std::vector<double> got, ref;
        for (const std::int32_t tok : sample_tokens(t)) {
            for (std::size_t i = 0; i < rows.size(); ++i) {
                double acc = 0.0;
                for (std::int32_t c = 0; c < s.k; ++c) {
                    acc += decoded[i][c] * static_cast<double>(bf16_to_f32(
                                               act[static_cast<std::size_t>(tok) * s.k + c]));
                }
                ref.push_back(acc);
                got.push_back(static_cast<double>(
                    bf16_to_f32(host_out[static_cast<std::size_t>(tok) * s.n + rows[i]])));
            }
        }
        failures += verify_reduction(name, got, ref, criterion);
    }
    failures += out_dev.verify_guards(label + " output");
    failures += x_dev.verify_guards(label + " input");
    failures += scratch.verify_guards(label + " workspace");
    failures += rep.verify_preserved(label.c_str());
    return failures;
}

int check_unregistered_shape_throws() {
    const Fp8RowFp32Rep rep(4096, 2048);
    const Weight w = rep.view();
    const auto act = make_activation(2048, 1);
    GuardedDeviceBuffer x_dev(act.size() * 2), out_dev(4096 * 2), scratch(256);
    x_dev.copy_from_host(act.data(), act.size() * 2);
    MemoryArena arena(scratch.span());
    Tensor x(x_dev.span(), DType::BF16, {2048, 1});
    Tensor out(out_dev.span(), DType::BF16, {4096, 1});
    const auto exec = cuda_environment().default_execution;
    try {
        ops::linear(x, w, out, ops::LinearPolicy::AllowA8, arena, exec);
        cuda_check(cudaStreamSynchronize(exec.stream()), "synchronize unregistered linear");
    } catch (const std::exception&) { return 0; }
    std::cerr << "unregistered FP8_E4M3FN_ROW_FP32 4096x2048 was accepted\n";
    return 1;
}

int run() {
    int failures = 0;

    // F2.1: the artifact format table knows the new format (no "unknown tensor format").
    try {
        if (artifact::parse_format("fp8_e4m3fn_row_fp32") != QType::FP8_E4M3FN_ROW_FP32 ||
            artifact::format_name(QType::FP8_E4M3FN_ROW_FP32) != "fp8_e4m3fn_row_fp32") {
            std::cerr << "fp8_e4m3fn_row_fp32 does not round-trip through the format table\n";
            ++failures;
        }
    } catch (const std::exception& error) {
        std::cerr << "parse_format(fp8_e4m3fn_row_fp32) threw: " << error.what() << '\n';
        ++failures;
    }

    // F2.2 / F2.3: every Spark shape and representation against the FP64 reference.
    for (const Shape& s : kShapes) {
        {
            const Fp8RowFp32Rep rep(s.n, s.k);
            failures += check_shape(s, rep, ops::LinearPolicy::AllowA8, kA8Criterion,
                                    std::string("fp8_row_fp32 ") + s.name);
        }
        {
            // BF16 under AllowA8 must still meet the A16 criterion; an FP8 activation branch
            // (4% allowance) would not.
            const Bf16Rep rep(s.n, s.k);
            failures += check_shape(s, rep, ops::LinearPolicy::AllowA8, kA16Criterion,
                                    std::string("bf16 ") + s.name);
        }
    }

    // Boundary: a shape that is not registered still fails loudly.
    failures += check_unregistered_shape_throws();
    return failures;
}

} // namespace

int main() {
    try {
        const int failures = run();
        std::cout << (failures == 0 ? "PASS" : "FAIL") << ": linear_fp8_row_fp32_t4\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "linear_fp8_row_fp32_t4: unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
