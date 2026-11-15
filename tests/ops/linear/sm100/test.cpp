// tcgen05 dense GEMM (B200 / sm_100a): direct operand-level checks of the four epilogues and of
// the linear_backend() routing. The shape-level coverage of the registered 122B projections is in
// the projection / split / add suites, which run the same kernels through the public Ops.
//
// The reference is an independent double-precision contraction of the dequantized operands. The
// NVFP4 operands are built here in the K16M128x4 atom order from the format definition
// (offset = atom * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4), not by the
// production quantizer.

#include "ops/linear/backend.h"
#include "ops/linear/kernels/sm100/gemm.h"
#include "ops/op_tester.h"
#include "ops/suite.h"

#include <array>
#include <cmath>
#include <cstdint>
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

constexpr std::array<int, 3> kSegments = {512, 256, 768};
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
        for (int width : kSegments) {
            segment_buffers.emplace_back(cuda_environment().context.device(),
                                         static_cast<std::size_t>(tokens) * width * 2);
            segment_buffers.back().fill(0xff);
            segment_pointers.push_back(static_cast<__nv_bfloat16*>(segment_buffers.back().data()));
        }
        epilogue.segments = kSegments;
        epilogue.outputs  = segment_pointers;
    }
    launch(epilogue, static_cast<__nv_bfloat16*>(y.data()));
    cuda_check_last_launch("sm100 gemm");
    cuda_synchronize();

    std::vector<std::vector<double>> segment_values;
    for (std::size_t s = 0; s < segment_buffers.size(); ++s)
        segment_values.push_back(from_device_bf16(
            segment_buffers[s], static_cast<std::size_t>(tokens) * kSegments[s]));
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
                if (r == base + kSegments[segment]) base += kSegments[segment++];
                got.push_back(segment_values[segment][static_cast<std::size_t>(t) *
                                                          kSegments[segment] + (r - base)]);
            }
        }
        if (kind != Sm100Epilogue::Split)
            got.insert(got.end(), all_values.begin() + static_cast<std::ptrdiff_t>(t) * out_rows,
                       all_values.begin() + static_cast<std::ptrdiff_t>(t + 1) * out_rows);
    }
    return verify_pointwise(label, got, expected, criterion);
}

int nvfp4_case(int kind, int tokens) {
    constexpr int kRows = 1536, kK = 1024;
    constexpr float kAlpha = 1.F / 64.F;
    const Fp4Matrix w = make_fp4(kRows, kK, 11U), x = make_fp4(tokens, kK, 12U);
    const auto acc    = contract(w.dequantized, x.dequantized, kRows, kK, tokens);
    const auto wc = to_device(w.codes), ws = to_device(w.scales), xc = to_device(x.codes),
               xs = to_device(x.scales);
    const Nvfp4A4Operands operands{static_cast<const std::uint8_t*>(xc.data()),
                                   static_cast<const std::uint8_t*>(xs.data()),
                                   static_cast<const std::uint8_t*>(wc.data()),
                                   static_cast<const std::uint8_t*>(ws.data()),
                                   kRows,
                                   kK,
                                   tokens,
                                   kAlpha,
                                   Nvfp4ScaleLayout::K16M128x4};
    return check_epilogue(std::string("sm100 nvfp4 ") + kind_name(kind) + " t" +
                              std::to_string(tokens),
                          kind, acc, kRows, tokens, kAlpha, {0.05, 0.02},
                          [&](const Sm100Epilogue& e, __nv_bfloat16* y) {
                              launch_sm100_nvfp4_gemm(operands, e, y, nullptr);
                          });
}

int bf16_case(int kind, int tokens) {
    constexpr int kRows = 1536, kK = 1024;
    std::vector<float> w(static_cast<std::size_t>(kRows) * kK),
        x(static_cast<std::size_t>(tokens) * kK);
    fill_uniform(w, 21U, -1.F, 1.F);
    fill_uniform(x, 22U, -1.F, 1.F);
    round_to_bf16(w);
    round_to_bf16(x);
    const auto acc = contract(w, x, kRows, kK, tokens);
    const auto dw = to_device_bf16(w), dx = to_device_bf16(x);
    const Sm100Bf16Operands operands{static_cast<const __nv_bfloat16*>(dx.data()),
                                     static_cast<const __nv_bfloat16*>(dw.data()), kRows, kK,
                                     tokens};
    return check_epilogue(std::string("sm100 bf16 ") + kind_name(kind) + " t" +
                              std::to_string(tokens),
                          kind, acc, kRows, tokens, 1.0, {0.05, 0.01},
                          [&](const Sm100Epilogue& e, __nv_bfloat16* y) {
                              launch_sm100_bf16_gemm(operands, e, y, nullptr);
                          });
}

// The four registered 122B shapes run on tcgen05 at every token count; nothing else does.
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
        expect(QType::NVFP4, 5120, 6144, t, t == 1 ? LinearBackend::Gemv : LinearBackend::MmaSync);
        expect(QType::Q8_G32_FP16, 248320, 3072, t,
               t == 1 ? LinearBackend::Gemv : LinearBackend::MmaSync);
    }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("sm100/backend_routing", routing_case);
    for (int kind : {Sm100Epilogue::Identity, Sm100Epilogue::Add, Sm100Epilogue::Split,
                     Sm100Epilogue::SwiGLU})
        for (int tokens : {1, 130, 257}) {
            suite.add(std::string("sm100/nvfp4/") + kind_name(kind) + "/t" + std::to_string(tokens),
                      [=] { return nvfp4_case(kind, tokens); });
            suite.add(std::string("sm100/bf16/") + kind_name(kind) + "/t" + std::to_string(tokens),
                      [=] { return bf16_case(kind, tokens); });
        }
    return suite.main(argc, argv);
}
