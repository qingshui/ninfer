// tcgen05 dense GEMM (B200 / sm_100a): direct operand-level checks of the four epilogues, of the
// split-K path of decode calls and its plan, and of the linear_backend() routing. The shape-level
// coverage of the registered 122B projections is in the projection / split / add suites, which run
// the same kernels through the public Ops.
//
// The reference is an independent double-precision contraction of the dequantized operands. The
// NVFP4 operands are built here in the K16M128x4 atom order from the format definition
// (offset = atom * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4), not by the
// production quantizer.

#include "ops/linear/backend.h"
#include "ops/linear/kernels/sm100/gemm.h"
#include "ops/float_codec.h"
#include "ops/op_tester.h"
#include "ops/suite.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops::detail;

constexpr std::array<float, 8> kE2m1 = {0.F, .5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};

float decode_e4m3(std::uint8_t byte) {
    const int exponent = byte >> 3, mantissa = byte & 7;
    if (exponent == 0) return static_cast<float>(mantissa) * std::ldexp(1.F, -9);
    return (1.F + static_cast<float>(mantissa) / 8.F) * std::ldexp(1.F, exponent - 7);
}

struct Fp4Matrix {
    std::vector<std::uint8_t> codes;  // [rows, k / 2], low nibble first
    std::vector<std::uint8_t> scales; // K16M128x4 atoms, rows padded to whole 128-row tiles
    std::vector<float> dequantized;   // [rows, k]
};

Fp4Matrix make_fp4(int rows, int k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int groups = k / 16, padded = (rows + 127) / 128 * 128;
    Fp4Matrix m;
    m.codes.resize(static_cast<std::size_t>(rows) * k / 2);
    for (auto& byte : m.codes) byte = static_cast<std::uint8_t>(rng());
    std::vector<std::uint8_t> logical(static_cast<std::size_t>(rows) * groups);
    for (auto& s : logical) s = static_cast<std::uint8_t>(((5 + rng() % 5) << 3) | (rng() & 7));
    m.scales.assign(static_cast<std::size_t>(padded) * groups, 0);
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < groups; ++g) {
            const std::size_t atom = static_cast<std::size_t>(r / 128) * (k / 64) + g / 4;
            m.scales[atom * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + g % 4] =
                logical[static_cast<std::size_t>(r) * groups + g];
        }
    m.dequantized.resize(static_cast<std::size_t>(rows) * k);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < k; ++c) {
            const std::uint8_t byte = m.codes[static_cast<std::size_t>(r) * k / 2 + c / 2];
            const int nibble        = (c & 1) ? byte >> 4 : byte & 15;
            const float value       = kE2m1[nibble & 7] * ((nibble & 8) ? -1.F : 1.F);
            m.dequantized[static_cast<std::size_t>(r) * k + c] =
                value * decode_e4m3(logical[static_cast<std::size_t>(r) * groups + c / 16]);
        }
    return m;
}

// acc[token][row] = sum_k w[row][k] * x[token][k] in double.
std::vector<double> contract(const std::vector<float>& w, const std::vector<float>& x, int rows,
                             int k, int tokens) {
    std::vector<double> acc(static_cast<std::size_t>(tokens) * rows);
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
            double sum = 0;
            for (int i = 0; i < k; ++i)
                sum += static_cast<double>(w[static_cast<std::size_t>(r) * k + i]) *
                       x[static_cast<std::size_t>(t) * k + i];
            acc[static_cast<std::size_t>(t) * rows + r] = sum;
        }
    return acc;
}

// Three Split segments over the weight rows; the last takes the rows the first two leave.
std::array<int, 3> segments_of(int rows) { return {512, 256, rows - 768}; }

const char* kind_name(int kind) {
    return kind == Sm100Epilogue::Identity ? "identity"
           : kind == Sm100Epilogue::Add    ? "add"
           : kind == Sm100Epilogue::Split  ? "split"
                                           : "swiglu";
}

// Runs one epilogue through `launch(epilogue, y)` and compares with the reference accumulator.
template <class Launch>
int check_epilogue(const std::string& label, int kind, const std::vector<double>& acc, int rows,
                   int tokens, double scale, const PointwiseCriterion& criterion,
                   const Launch& launch) {
    const std::array<int, 3> segments = segments_of(rows);
    std::vector<float> residual(static_cast<std::size_t>(tokens) * rows);
    fill_uniform(residual, 77U, -4.F, 4.F);
    round_to_bf16(residual);
    const int out_rows = kind == Sm100Epilogue::SwiGLU ? rows / 2 : rows;

    DeviceBuffer y = kind == Sm100Epilogue::Add ? to_device_bf16(residual)
                                                : DeviceBuffer(cuda_environment().context.device(),
                                                               static_cast<std::size_t>(tokens) *
                                                                   rows * 2);
    if (kind != Sm100Epilogue::Add) y.fill(0xff);
    std::vector<DeviceBuffer> segment_buffers;
    std::vector<__nv_bfloat16*> segment_pointers;
    Sm100Epilogue epilogue;
    epilogue.kind = static_cast<Sm100Epilogue::Kind>(kind);
    if (kind == Sm100Epilogue::Add) epilogue.residual = static_cast<const __nv_bfloat16*>(y.data());
    if (kind == Sm100Epilogue::Split) {
        for (int width : segments) {
            segment_buffers.emplace_back(cuda_environment().context.device(),
                                         static_cast<std::size_t>(tokens) * width * 2);
            segment_buffers.back().fill(0xff);
            segment_pointers.push_back(static_cast<__nv_bfloat16*>(segment_buffers.back().data()));
        }
        epilogue.segments = segments;
        epilogue.outputs  = segment_pointers;
    }
    launch(epilogue, static_cast<__nv_bfloat16*>(y.data()));
    cuda_check_last_launch("sm100 gemm");
    cuda_synchronize();

    std::vector<std::vector<double>> segment_values;
    for (std::size_t s = 0; s < segment_buffers.size(); ++s)
        segment_values.push_back(from_device_bf16(
            segment_buffers[s], static_cast<std::size_t>(tokens) * segments[s]));
    const auto all_values =
        from_device_bf16(y, static_cast<std::size_t>(tokens) * out_rows);
    std::vector<double> got, expected;
    for (int t = 0; t < tokens; ++t) {
        int segment = 0, base = 0;
        for (int r = 0; r < out_rows; ++r) {
            const double a = acc[static_cast<std::size_t>(t) * rows + r] * scale;
            double value   = a;
            if (kind == Sm100Epilogue::Add) value += residual[static_cast<std::size_t>(t) * rows + r];
            if (kind == Sm100Epilogue::SwiGLU) {
                const double up = acc[static_cast<std::size_t>(t) * rows + r + rows / 2] * scale;
                value           = a / (1.0 + std::exp(-a)) * up;
            }
            expected.push_back(value);
            if (kind == Sm100Epilogue::Split) {
                if (r == base + segments[segment]) base += segments[segment++];
                got.push_back(segment_values[segment][static_cast<std::size_t>(t) *
                                                          segments[segment] + (r - base)]);
            }
        }
        if (kind != Sm100Epilogue::Split)
            got.insert(got.end(), all_values.begin() + static_cast<std::ptrdiff_t>(t) * out_rows,
                       all_values.begin() + static_cast<std::ptrdiff_t>(t + 1) * out_rows);
    }
    return verify_pointwise(label, got, expected, criterion);
}

// Split-K scratch, poisoned with NaN before every launch: a partial that the reduction reads
// without the GEMM having written it turns the output non-finite.
struct SplitScratch {
    DeviceBuffer buffer{cuda_environment().context.device(), kSm100SplitPartialBytes};

    Sm100Partials poisoned() {
        buffer.fill(0xff);
        return {static_cast<float*>(buffer.data()), kSm100SplitPartialBytes};
    }
};

// A split case whose plan stopped cutting tiles would only repeat a whole-K case.
int require_split(const std::string& label, const Sm100GemmPlan& plan) {
    if (plan.split_tiles > 0) return 0;
    std::cerr << label << ": the plan runs every tile over the whole K loop\n";
    return 1;
}

std::string case_label(const char* format, int kind, int tokens, int rows, int k, bool split) {
    std::string label = std::string("sm100 ") + format + " " + kind_name(kind) + " t" +
                        std::to_string(tokens);
    if (split) label += " r" + std::to_string(rows) + " k" + std::to_string(k) + " split";
    return label;
}

int nvfp4_case(int kind, int tokens, int k = 1024, bool split = false) {
    constexpr int kRows    = 1536;
    constexpr float kAlpha = 1.F / 64.F;
    const std::string label = case_label("nvfp4", kind, tokens, kRows, k, split);
    if (split && require_split(label, nvfp4_sm100_plan(kRows, k, tokens, kSm100SplitPartialBytes)))
        return 1;
    const Fp4Matrix w = make_fp4(kRows, k, 11U), x = make_fp4(tokens, k, 12U);
    const auto acc    = contract(w.dequantized, x.dequantized, kRows, k, tokens);
    const auto wc = to_device(w.codes), ws = to_device(w.scales), xc = to_device(x.codes),
               xs = to_device(x.scales);
    const Nvfp4A4Operands operands{static_cast<const std::uint8_t*>(xc.data()),
                                   static_cast<const std::uint8_t*>(xs.data()),
                                   static_cast<const std::uint8_t*>(wc.data()),
                                   static_cast<const std::uint8_t*>(ws.data()),
                                   kRows,
                                   k,
                                   tokens,
                                   kAlpha,
                                   Nvfp4ScaleLayout::K16M128x4};
    std::optional<SplitScratch> scratch;
    if (split) scratch.emplace();
    return check_epilogue(label, kind, acc, kRows, tokens, kAlpha, {0.05, 0.02},
                          [&](const Sm100Epilogue& e, __nv_bfloat16* y) {
                              launch_sm100_nvfp4_gemm(operands, e, y, nullptr,
                                                      scratch ? scratch->poisoned()
                                                              : Sm100Partials{});
                          });
}

int bf16_case(int kind, int tokens, int k = 1024) {
    constexpr int kRows = 1536;
    std::vector<float> w(static_cast<std::size_t>(kRows) * k),
        x(static_cast<std::size_t>(tokens) * k);
    fill_uniform(w, 21U, -1.F, 1.F);
    fill_uniform(x, 22U, -1.F, 1.F);
    round_to_bf16(w);
    round_to_bf16(x);
    const auto acc = contract(w, x, kRows, k, tokens);
    const auto dw = to_device_bf16(w), dx = to_device_bf16(x);
    const Sm100Bf16Operands operands{static_cast<const __nv_bfloat16*>(dx.data()),
                                     static_cast<const __nv_bfloat16*>(dw.data()), kRows, k,
                                     tokens};
    return check_epilogue(std::string("sm100 bf16 ") + kind_name(kind) + " t" +
                              std::to_string(tokens),
                          kind, acc, kRows, tokens, 1.0, {0.05, 0.01},
                          [&](const Sm100Epilogue& e, __nv_bfloat16* y) {
                              launch_sm100_bf16_gemm(operands, e, y, nullptr);
                          });
}

// Router scores: 257 weight rows end inside a 128-row tile and the FP32 results are stored with a
// padded token stride, which the columns past the rows must leave untouched.
int scores_case(int tokens) {
    constexpr int kRows = 257, kStride = 260, kK = 3072;
    std::vector<float> w(static_cast<std::size_t>(kRows) * kK),
        x(static_cast<std::size_t>(tokens) * kK);
    fill_uniform(w, 41U, -1.F, 1.F);
    fill_uniform(x, 42U, -1.F, 1.F);
    round_to_bf16(w);
    round_to_bf16(x);
    const auto acc = contract(w, x, kRows, kK, tokens);
    const auto dw = to_device_bf16(w), dx = to_device_bf16(x);
    DeviceBuffer y(cuda_environment().context.device(),
                   static_cast<std::size_t>(tokens) * kStride * sizeof(float));
    y.fill(0xff);
    Sm100Epilogue epilogue;
    epilogue.kind         = Sm100Epilogue::Scores;
    epilogue.scores       = static_cast<float*>(y.data());
    epilogue.score_stride = kStride;
    const Sm100Bf16Operands operands{static_cast<const __nv_bfloat16*>(dx.data()),
                                     static_cast<const __nv_bfloat16*>(dw.data()), kRows, kK,
                                     tokens};
    launch_sm100_bf16_gemm(operands, epilogue, nullptr, nullptr);
    cuda_check_last_launch("sm100 scores");
    cuda_synchronize();
    const std::vector<float> values =
        from_device<float>(y, static_cast<std::size_t>(tokens) * kStride);
    std::vector<double> got, expected;
    int padding_failures = 0;
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < kRows; ++r) {
            got.push_back(values[static_cast<std::size_t>(t) * kStride + r]);
            expected.push_back(acc[static_cast<std::size_t>(t) * kRows + r]);
        }
        for (int r = kRows; r < kStride; ++r) {
            std::uint32_t bits;
            std::memcpy(&bits, &values[static_cast<std::size_t>(t) * kStride + r], sizeof(bits));
            if (bits != 0xffffffffU) ++padding_failures;
        }
    }
    if (padding_failures != 0)
        std::cerr << "sm100 scores t" << tokens << ": " << padding_failures
                  << " padding columns were written\n";
    return padding_failures + verify_pointwise("sm100 scores t" + std::to_string(tokens), got,
                                               expected, {2e-3, 2e-4});
}

// E4M3 codes drawn over the finite range with one FP32 scale per row; the dequantized values
// stay within [-1, 1] so the contraction is comparable with the BF16 cases.
struct Fp8Matrix {
    std::vector<std::uint8_t> codes; // [rows, k]
    std::vector<float> scales;       // [rows]
    std::vector<float> dequantized;  // [rows, k]
};

Fp8Matrix make_fp8(int rows, int k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.F, 1.F), magnitude(0.25F, 1.F);
    Fp8Matrix m;
    m.codes.resize(static_cast<std::size_t>(rows) * k);
    m.dequantized.resize(m.codes.size());
    m.scales.resize(rows);
    for (int r = 0; r < rows; ++r) {
        m.scales[r] = magnitude(rng) / 448.F;
        for (int c = 0; c < k; ++c) {
            const std::size_t index = static_cast<std::size_t>(r) * k + c;
            m.codes[index]          = float_codec::encode_e4m3fn_rne_satfinite(unit(rng) * 448.F);
            m.dequantized[index]    = float_codec::decode_e4m3fn(m.codes[index]) * m.scales[r];
        }
    }
    return m;
}

Sm100Fp8Operands fp8_operands(const DeviceBuffer& wc, const DeviceBuffer& ws,
                              const DeviceBuffer& xc, const DeviceBuffer& xs, int rows, int k,
                              int tokens) {
    return {static_cast<const std::uint8_t*>(xc.data()), static_cast<const float*>(xs.data()),
            static_cast<const std::uint8_t*>(wc.data()), static_cast<const float*>(ws.data()),
            rows,
            k,
            tokens};
}

int fp8_case(int kind, int tokens, int k = 1024, int rows = 1536, bool split = false) {
    const std::string label = case_label("fp8", kind, tokens, rows, k, split);
    if (split && require_split(label, fp8_sm100_plan(rows, k, tokens, kSm100SplitPartialBytes)))
        return 1;
    const Fp8Matrix w = make_fp8(rows, k, 31U), x = make_fp8(tokens, k, 32U);
    const auto acc    = contract(w.dequantized, x.dequantized, rows, k, tokens);
    const auto wc = to_device(w.codes), ws = to_device(w.scales), xc = to_device(x.codes),
               xs = to_device(x.scales);
    const Sm100Fp8Operands operands = fp8_operands(wc, ws, xc, xs, rows, k, tokens);
    std::optional<SplitScratch> scratch;
    if (split) scratch.emplace();
    return check_epilogue(label, kind, acc, rows, tokens, 1.0, {0.05, 0.01},
                          [&](const Sm100Epilogue& e, __nv_bfloat16* y) {
                              launch_sm100_fp8_gemm(operands, e, y, nullptr,
                                                    scratch ? scratch->poisoned()
                                                            : Sm100Partials{});
                          });
}

// A split call sums the K partials of a tile in a fixed order, so repeated calls store the same
// bits for the whole and the cut tiles alike.
int split_determinism_case() {
    constexpr int kRows = 20480, kK = 3072, kTokens = 40;
    const std::string label = "sm100 fp8 split determinism";
    if (require_split(label, fp8_sm100_plan(kRows, kK, kTokens, kSm100SplitPartialBytes)))
        return 1;
    const Fp8Matrix w = make_fp8(kRows, kK, 41U), x = make_fp8(kTokens, kK, 42U);
    const auto wc = to_device(w.codes), ws = to_device(w.scales), xc = to_device(x.codes),
               xs = to_device(x.scales);
    const Sm100Fp8Operands operands = fp8_operands(wc, ws, xc, xs, kRows, kK, kTokens);
    const std::size_t count         = static_cast<std::size_t>(kTokens) * kRows;
    SplitScratch scratch;
    std::array<std::vector<std::uint16_t>, 2> runs;
    for (auto& run : runs) {
        DeviceBuffer y(cuda_environment().context.device(), count * 2);
        y.fill(0xff);
        launch_sm100_fp8_gemm(operands, {}, static_cast<__nv_bfloat16*>(y.data()), nullptr,
                              scratch.poisoned());
        cuda_check_last_launch("sm100 fp8 split");
        cuda_synchronize();
        run = from_device<std::uint16_t>(y, count);
    }
    if (runs[0] == runs[1]) return 0;
    std::cerr << label << ": two runs stored different bits\n";
    return 1;
}

// Plan invariants: calls above kSm100MaxSplitTokens or without scratch run every tile over the
// whole K loop, and a split cuts at most every tile into at least two non-empty K ranges whose
// partial blocks fit the scratch.
int split_plan_case() {
    int failures      = 0;
    const auto expect = [&](bool holds, const std::string& what) {
        if (holds) return;
        std::cerr << "sm100 split plan: " << what << '\n';
        ++failures;
    };
    struct Shape {
        const char* format;
        int rows, k, k_iters;
    };
    for (const Shape& shape : {Shape{"fp8", 20480, 3072, 24}, Shape{"fp8", 3072, 8192, 64},
                               Shape{"fp8", 1536, 1024, 8}, Shape{"nvfp4", 3072, 8192, 32},
                               Shape{"nvfp4", 1536, 1024, 4}}) {
        const bool fp8   = std::string(shape.format) == "fp8";
        const auto plan  = [&](int tokens, std::size_t bytes) {
            return fp8 ? fp8_sm100_plan(shape.rows, shape.k, tokens, bytes)
                       : nvfp4_sm100_plan(shape.rows, shape.k, tokens, bytes);
        };
        for (int tokens = 1; tokens <= 260; ++tokens) {
            const std::string what = std::string(shape.format) + " r" + std::to_string(shape.rows) +
                                     " k" + std::to_string(shape.k) + " t" +
                                     std::to_string(tokens);
            expect(plan(tokens, 0).split_tiles == 0, what + " splits without scratch");
            const Sm100GemmPlan p = plan(tokens, kSm100SplitPartialBytes);
            if (p.split_tiles == 0) continue;
            const long tiles = static_cast<long>(shape.rows / 128) *
                               ((tokens + p.tile_tokens - 1) / p.tile_tokens);
            const std::size_t bytes = static_cast<std::size_t>(p.split_tiles) * p.k_splits *
                                      p.tile_tokens * 128 * sizeof(float);
            expect(tokens <= kSm100MaxSplitTokens, what + " splits a prefill call");
            expect(p.split_tiles <= tiles, what + " cuts more tiles than exist");
            expect(p.k_splits >= 2 && p.k_splits <= shape.k_iters, what + " k_splits out of range");
            expect(p.split_tiles * p.k_splits <= kSm100MaxSplitUnits &&
                       bytes <= kSm100SplitPartialBytes,
                   what + " partials exceed the scratch");
        }
    }
    return failures;
}

// The registered 122B shapes run on tcgen05 at every token count; nothing else does.
int routing_case() {
    int failures = 0;
    const auto expect = [&](QType type, int n, int k, int tokens, LinearBackend backend) {
        if (linear_backend(type, n, k, tokens) != backend) {
            std::cerr << "linear_backend mismatch n=" << n << " k=" << k << " tokens=" << tokens
                      << '\n';
            ++failures;
        }
    };
    for (int t : {1, 2, 130, 520, 8192}) {
        expect(QType::NVFP4, 3072, 8192, t, LinearBackend::Tcgen05);
        expect(QType::NVFP4, 17408, 3072, t, LinearBackend::Tcgen05);
        expect(QType::BF16, 20480, 3072, t, LinearBackend::Tcgen05);
        expect(QType::BF16, 3072, 8192, t, LinearBackend::Tcgen05);
        expect(QType::FP8_E4M3FN_ROW_FP32, 20480, 3072, t, LinearBackend::Tcgen05);
        expect(QType::FP8_E4M3FN_ROW_FP32, 3072, 8192, t, LinearBackend::Tcgen05);
        expect(QType::FP8_E4M3FN_ROW_BF16, 20480, 3072, t,
               t == 1 ? LinearBackend::Gemv : LinearBackend::MmaSync);
        expect(QType::NVFP4, 5120, 6144, t, t == 1 ? LinearBackend::Gemv : LinearBackend::MmaSync);
        expect(QType::BF16, 248320, 3072, t, LinearBackend::Tcgen05);
        expect(QType::BF16, 248320, 2560, t, t == 1 ? LinearBackend::Gemv : LinearBackend::MmaSync);
        expect(QType::Q8_G32_FP16, 248320, 3072, t,
               t == 1 ? LinearBackend::Gemv : LinearBackend::MmaSync);
    }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("sm100/backend_routing", routing_case);
    for (int tokens : {1, 17, 130, 600})
        suite.add("sm100/scores/t" + std::to_string(tokens), [=] { return scores_case(tokens); });
    for (int kind : {Sm100Epilogue::Identity, Sm100Epilogue::Add, Sm100Epilogue::Split,
                     Sm100Epilogue::SwiGLU})
        for (int tokens : {1, 130, 257}) {
            suite.add(std::string("sm100/nvfp4/") + kind_name(kind) + "/t" + std::to_string(tokens),
                      [=] { return nvfp4_case(kind, tokens); });
            suite.add(std::string("sm100/bf16/") + kind_name(kind) + "/t" + std::to_string(tokens),
                      [=] { return bf16_case(kind, tokens); });
        }
    // Large calls run 256-token BF16 tiles; 3200 ends in a half tile.
    for (int kind : {Sm100Epilogue::Identity, Sm100Epilogue::Add, Sm100Epilogue::Split})
        suite.add(std::string("sm100/bf16/") + kind_name(kind) + "/t3200",
                  [=] { return bf16_case(kind, 3200, 256); });
    suite.add("sm100/bf16/identity/t3072",
              [] { return bf16_case(Sm100Epilogue::Identity, 3072, 256); });
    // Token counts that select the 64-, 128- and 256-token tiles of the 1536-row cases.
    for (int tokens : {600, 1100, 2000}) {
        suite.add("sm100/bf16/identity/t" + std::to_string(tokens),
                  [=] { return bf16_case(Sm100Epilogue::Identity, tokens, 256); });
        suite.add("sm100/bf16/swiglu/t" + std::to_string(tokens),
                  [=] { return bf16_case(Sm100Epilogue::SwiGLU, tokens, 256); });
        suite.add("sm100/fp8/identity/t" + std::to_string(tokens),
                  [=] { return fp8_case(Sm100Epilogue::Identity, tokens, 256); });
    }
    for (int kind : {Sm100Epilogue::Identity, Sm100Epilogue::Add, Sm100Epilogue::Split}) {
        for (int tokens : {1, 17, 130, 257})
            suite.add(std::string("sm100/fp8/") + kind_name(kind) + "/t" + std::to_string(tokens),
                      [=] { return fp8_case(kind, tokens); });
        suite.add(std::string("sm100/fp8/") + kind_name(kind) + "/t3200",
                  [=] { return fp8_case(kind, 3200, 256); });
    }
    // Decode calls given split-K scratch: 1536 rows and K 8192 cut every tile, 20480 rows and
    // K 3072 run whole tiles before the cut ones; 100 tokens end inside a 128-token tile.
    suite.add("sm100/split_plan", split_plan_case);
    suite.add("sm100/fp8/split_determinism", split_determinism_case);
    for (int kind : {Sm100Epilogue::Identity, Sm100Epilogue::Add, Sm100Epilogue::Split}) {
        const std::string name = kind_name(kind);
        for (int tokens : {16, 100}) {
            suite.add("sm100/nvfp4/" + name + "/t" + std::to_string(tokens) + "/split",
                      [=] { return nvfp4_case(kind, tokens, 8192, true); });
            suite.add("sm100/fp8/" + name + "/t" + std::to_string(tokens) + "/split",
                      [=] { return fp8_case(kind, tokens, 8192, 1536, true); });
        }
        const int tokens = kind == Sm100Epilogue::Add ? 40 : 16;
        suite.add("sm100/fp8/" + name + "/t" + std::to_string(tokens) + "/r20480/split",
                  [=] { return fp8_case(kind, tokens, 3072, 20480, true); });
    }
    return suite.main(argc, argv);
}
