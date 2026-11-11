// T2 / F4: linear_geglu Op test against an independent FP64 reference.
//
// Contract under test (include/ninfer/ops/linear_geglu.h):
//   out[i,t] = GELU_erf(gate[i,t]) * up[i,t],  GELU_erf(z) = 0.5*z*(1+erf(z/sqrt(2)))
// with gate rows [0,M) followed by up rows [M,2M) in a [2M,K] BF16 weight.
//
// Token 0 is dense (every K contributes). Tokens 1.. activate only K-column 0, which, together
// with crafted weights in the first rows, drives the pre-activation over [-4,4). On the negative
// tail the tanh-approximate GELU differs from the erf form by far more than the tolerance below,
// so a tanh substitution must fail this test.

#include "cuda_fixture.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear_geglu.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr std::int32_t kM       = 6656; // output rows; gate_up rows = 13312
constexpr std::int32_t kK       = 2048;
constexpr std::int32_t kTokens  = 8;
constexpr std::int32_t kCrafted = 256; // rows with controlled pre-activations

double gelu_erf_fp64(double z) { return 0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0))); }

float crafted_gate_weight(int row) { return static_cast<float>(row - 128) / 32.0F; } // [-4,4)
float crafted_up_weight(int row) { return 1.0F + static_cast<float>(row % 7) / 8.0F; }
constexpr float kTokenScale[kTokens] = {0.0F, 1.0F, 0.75F, 0.5F, -1.0F, 1.0F, 0.5F, -0.75F};

int run() {
    auto host = direct_bf16_weight::make_patterned(2 * kM, kK, 4242U);
    for (int row = 0; row < kCrafted; ++row) {
        for (int base : {row, kM + row}) {
            for (int c = 0; c < kK; ++c) {
                host.bits[std::size_t(base) * kK + c] = 0;
            }
        }
        host.bits[std::size_t(row) * kK] = f32_to_bf16(crafted_gate_weight(row));
        host.bits[std::size_t(kM + row) * kK] = f32_to_bf16(crafted_up_weight(row));
    }

    // Activations [T][K]: token 0 dense, others only column 0.
    std::vector<std::uint16_t> x_bits(std::size_t(kTokens) * kK, 0);
    for (int c = 0; c < kK; ++c) {
        x_bits[c] = f32_to_bf16(static_cast<float>(((c * 37) % 63) - 31) * 2.0e-2F);
    }
    x_bits[0] = f32_to_bf16(0.5F);
    for (int t = 1; t < kTokens; ++t) { x_bits[std::size_t(t) * kK] = f32_to_bf16(kTokenScale[t]); }

    // FP64 reference from the exact stored BF16 values.
    std::vector<double> ref(std::size_t(kM) * kTokens), gate_ref(ref.size()), up_ref(ref.size());
    for (int t = 0; t < kTokens; ++t) {
        std::vector<float> x(kK);
        for (int c = 0; c < kK; ++c) { x[c] = bf16_to_f32(x_bits[std::size_t(t) * kK + c]); }
        for (int i = 0; i < kM; ++i) {
            const double g = direct_bf16_weight::dot_fp64(host, i, x);
            const double u = direct_bf16_weight::dot_fp64(host, kM + i, x);
            const std::size_t at = std::size_t(t) * kM + i;
            gate_ref[at] = g;
            up_ref[at]   = u;
            ref[at]      = gelu_erf_fp64(g) * u;
        }
    }

    direct_bf16_weight::DeviceWeight weight(host);
    GuardedDeviceBuffer x_dev(x_bits.size() * 2), out_dev(ref.size() * 2);
    x_dev.copy_from_host(x_bits.data(), x_bits.size() * 2);
    out_dev.fill(0xff);

    const auto policy = ops::LinearPolicy::A16Only;
    const auto bytes  = ops::linear_geglu_workspace_capacity_bytes(QType::BF16, 2 * kM, kK, policy,
                                                                   kTokens, kTokens);
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena arena(scratch.span());
    Tensor x(x_dev.span(), DType::BF16, {kK, kTokens});
    Tensor out(out_dev.span(), DType::BF16, {kM, kTokens});
    const auto exec = cuda_environment().default_execution;
    ops::linear_geglu(x, weight.view(), out, policy, arena, exec);
    cuda_check(cudaStreamSynchronize(exec.stream()), "synchronize linear_geglu");

    int failures = 0;
    failures += out_dev.verify_guards("geglu output");
    failures += x_dev.verify_guards("geglu input");
    failures += weight.verify_preserved("geglu weight");

    const auto actual = from_device_bf16(out_dev.data(), ref.size());
    int bad = 0, shown = 0;
    for (int t = 0; t < kTokens; ++t) {
        for (int i = 0; i < kM; ++i) {
            const std::size_t at = std::size_t(t) * kM + i;
            const double floor   = t == 0 ? 1.0e-5 : 2.0e-6;
            const double tol     = 4.0e-3 * std::fabs(ref[at]) +
                               floor * std::max(1.0, std::fabs(gate_ref[at])) *
                                   std::max(1.0, std::fabs(up_ref[at]));
            if (!std::isfinite(actual[at]) || std::fabs(actual[at] - ref[at]) > tol) {
                ++bad;
                if (shown++ < 8) {
                    std::cerr << "t=" << t << " row=" << i << " gate=" << gate_ref[at]
                              << " up=" << up_ref[at] << " ref=" << ref[at]
                              << " got=" << actual[at] << " tol=" << tol << '\n';
                }
            }
        }
    }
    if (bad != 0) { std::cerr << "linear_geglu: " << bad << " elements outside tolerance\n"; }
    return failures + (bad != 0 ? 1 : 0);
}

} // namespace

int main() {
    try {
        const int failures = run();
        std::cout << (failures == 0 ? "PASS" : "FAIL") << ": linear_geglu_t2\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "linear_geglu_t2: unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
