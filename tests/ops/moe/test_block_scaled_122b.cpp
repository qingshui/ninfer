// Qwen3.5-122B MoE geometry (3072/1024/256/8) Op-level tests against independent
// references: FP64 softmax+top-k routing, decoded FP64 NVFP4 experts and a BF16
// shared expert. Mirrors the block-scaled suite's contract checks.
#include "ops/suite.h"
#include "core/buffer.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/backend.h"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/schedule.h"
#include "ops/moe/operands.h"
#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <map>
#include <numeric>
#include <tuple>

namespace ninfer::test::moe_122b {
namespace {
constexpr int H = 3072, I = 1024, E = 256, TopK = 8;
// Two activation quantizations with a nonlinear product between them; the FP64
// oracle keeps the original BF16 input and decoded weights (same budget as the
// block-scaled nvfp4 suite).
constexpr ReductionCriterion kA4{0.25, 1e-7, 0.35};
constexpr ReductionCriterion kRoute{3e-5, 5e-6, 0};
constexpr ReductionCriterion kConsistency{1e-2, 1e-3, 1e-2};

ops::MoeProfile make_profile() {
    return {{{H, I, QType::NVFP4, QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
              QuantLayout::BlockScaleK16M128x4, ops::LinearPolicy::AllowA4,
              ops::LinearPolicy::AllowA4},
             E, TopK},
            {H, I, QType::BF16, QType::BF16, QuantLayout::Contiguous,
             QuantLayout::Contiguous}};
}

Weight dense(const DeviceBuffer& buffer, int rows, int columns) {
    Weight w;
    w.device  = buffer.device();
    w.payload = w.qdata = buffer.data();
    w.payload_bytes     = buffer.size();
    w.qtype             = QType::BF16;
    w.layout            = QuantLayout::Contiguous;
    w.ndim              = 2;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = columns;
    return w;
}

// Expert-major bank: expert e holds pattern slot (e & 1) at e * stride, exactly the affine layout
// the artifact stores (code plane, then block-scale plane, per expert).
struct Matrix {
    int n, k;
    std::size_t code_bytes, scale_bytes;
    std::vector<std::vector<std::uint8_t>> payload;
    std::vector<std::vector<float>> decoded;
    DeviceBuffer data, divisors;
    std::vector<float> dw;
    std::size_t stride, scale_offset;

    Matrix(int rows, int columns, unsigned seed)
        : n(rows), k(columns), divisors(cuda_environment().context.device(), E * 4) {
        for (int slot = 0; slot < 2; ++slot) {
            auto p = quantized_weight::make_patterned_weight(
                QType::NVFP4, n, k, seed + slot, {.weight_scale = 1, .input_scale = 1});
            for (std::size_t b = 0; b < p.code_plane_bytes; ++b) {
                p.payload[b] = quantized_weight::detail::mix64(b + seed + slot * 998123ULL) & 255;
            }
            code_bytes   = p.code_plane_bytes;
            scale_bytes  = p.scale_plane_bytes;
            scale_offset = p.scale_plane_offset;
            std::vector<float> values(std::size_t(n) * k);
            for (int r = 0; r < n; ++r) {
                for (int c = 0; c < k; ++c) {
                    values[std::size_t(r) * k + c] =
                        quantized_weight::logical_weight_fp64(p, r, c);
                }
            }
            payload.push_back(std::move(p.payload));
            decoded.push_back(std::move(values));
        }
        stride = (payload[0].size() + 255) / 256 * 256;
        data   = DeviceBuffer(cuda_environment().context.device(), std::size_t(E) * stride);
        for (int e = 0; e < E; ++e) {
            const int slot = e & 1;
            data.copy_from_host(payload[slot].data(), payload[slot].size(), std::size_t(e) * stride);
        }
        dw.resize(E);
        for (int e = 0; e < E; ++e) { dw[e] = 3.0F + (e % 8) * 0.5F; }
        divisors.copy_from_host(dw.data(), divisors.size());
    }

    ExpertMatrixDeviceView view() const {
        ExpertMatrixDeviceView result;
        result.codes               = static_cast<const std::byte*>(data.data());
        result.scales              = result.codes + scale_offset;
        result.code_stride         = stride;
        result.scale_stride        = stride;
        result.matrix_scales       = static_cast<const float*>(divisors.data());
        result.matrix_scale_stride = 1;
        result.physical_count      = E;
        return result;
    }
};

std::vector<float> random_values(std::size_t count, unsigned seed, float limit) {
    std::vector<float> result(count);
    fill_uniform(result, seed, -limit, limit);
    round_to_bf16(result);
    return result;
}

std::vector<double> ffn(std::span<const float> x, const std::vector<float>& gu,
                        const std::vector<float>& down, float gu_divisor, float down_divisor) {
    std::vector<double> h(I), out(H);
    for (int row = 0; row < I; ++row) {
        double gate = 0, up = 0;
        for (int k = 0; k < H; ++k) {
            gate += double(x[k]) * gu[std::size_t(row) * H + k] / gu_divisor;
            up += double(x[k]) * gu[std::size_t(row + I) * H + k] / gu_divisor;
        }
        h[row] = gate / (1 + std::exp(-gate)) * up;
    }
    for (int row = 0; row < H; ++row) {
        for (int k = 0; k < I; ++k) {
            out[row] += h[k] * down[std::size_t(row) * I + k] / down_divisor;
        }
    }
    return out;
}

struct Fixture {
    const ops::MoeProfile profile;
    Matrix gu, down;
    std::vector<std::vector<float>> x;
    std::vector<float> control, shared_gu, shared_down;
    DeviceBuffer control_d, shared_gu_d, shared_down_d;
    ops::MoeWeights weights;
    std::map<std::tuple<int, int>, std::vector<double>> cache;
    std::vector<std::vector<double>> shared_values;
    std::array<std::vector<double>, 3> logits;

    Fixture()
        : profile(make_profile()), gu(2 * I, H, 101), down(H, I, 211),
          control(random_values(std::size_t(E + 1) * H, 303, .0004F)),
          shared_gu(random_values(std::size_t(2 * I) * H, 404, .012F)),
          shared_down(random_values(std::size_t(H) * I, 505, .025F)) {
        for (int p = 0; p < 3; ++p) {
            x.push_back(random_values(H, 1701 + p, .7F));
            for (int k = 0; k < 32; ++k) { x.back()[k] = 0; }
            // Deliberate zero groups and a last-K coordinate expose tail errors.
            x.back()[H - 1] = p == 2 ? 3.5F : .875F;
            shared_values.push_back(ffn(x.back(), shared_gu, shared_down, 1, 1));
        }
        for (int e = 0; e <= E; ++e) { control[std::size_t(e) * H + e + 32] = 2; }
        // The highest two rows are identical dense projections: lower ID must win ties.
        control[std::size_t(E - 2) * H + E - 2 + 32] = 0;
        control[std::size_t(E - 2) * H + H - 1]      = 2;
        std::copy_n(control.begin() + std::size_t(E - 2) * H, H,
                    control.begin() + std::size_t(E - 1) * H);
        for (int p = 0; p < 3; ++p) {
            logits[p].resize(E + 1);
            for (int e = 0; e <= E; ++e) {
                for (int k = 0; k < H; ++k) {
                    logits[p][e] += double(control[std::size_t(e) * H + k]) * x[p][k];
                }
            }
        }
        control_d     = to_device_bf16(control);
        shared_gu_d   = to_device_bf16(shared_gu);
        shared_down_d = to_device_bf16(shared_down);
        weights       = {
            dense(control_d, E + 1, H),
            {{gu.view(), down.view(), E, control_d.device()},
             {ops::ActivationCalibrationView::broadcast(ops::ActivationScaleOperation::Divide,
                                                       719.7991333F),
              ops::ActivationCalibrationView::broadcast(ops::ActivationScaleOperation::Divide,
                                                       313.927002F)}},
            {{dense(shared_gu_d, 2 * I, H)}, {dense(shared_down_d, H, I)}}};
    }

    const std::vector<double>& expert(int p, int e) {
        auto key   = std::tuple{p, e % 8};
        auto found = cache.find(key);
        if (found != cache.end()) { return found->second; }
        const int slot = e & 1;
        return cache
            .emplace(key, ffn(x[p], gu.decoded[slot], down.decoded[slot], gu.dw[e], down.dw[e]))
            .first->second;
    }
};

int route_case(Fixture& f, CudaExecutionView execution, int tokens) {
    std::vector<float> input(std::size_t(tokens) * H);
    for (int t = 0; t < tokens; ++t) {
        std::copy(f.x[t % 3].begin(), f.x[t % 3].end(), input.begin() + std::size_t(t) * H);
    }
    auto dx = to_device_bf16(input);
    GuardedDeviceBuffer ids(std::size_t(tokens) * TopK * 4), alpha(std::size_t(tokens) * TopK * 4),
        beta(tokens * 4);
    GuardedDeviceBuffer scratch(
        ops::topk_softmax_route_workspace_capacity_bytes({H, E, TopK}, tokens, tokens));
    MemoryArena arena(scratch.span());
    Tensor x(dx.span(), DType::BF16, {H, tokens}), shared(beta.span(), DType::FP32, {tokens});
    ops::ExpertRoutes routes{Tensor(ids.span(), DType::I32, {TopK, tokens}),
                             Tensor(alpha.span(), DType::FP32, {TopK, tokens})};
    ops::topk_softmax_route(x, {H, E, TopK}, f.weights.control, routes, shared, arena, execution);
    cuda_synchronize(execution.stream());
    const auto actual_ids   = from_device<int>(routes.ids.data, tokens * TopK);
    const auto actual_alpha = from_device<float>(routes.coefficients.data, tokens * TopK);
    const auto actual_beta  = from_device<float>(shared.data, tokens);
    int failures            = 0;
    std::vector<double> expected_alpha(tokens * TopK), expected_beta(tokens);
    for (int t = 0; t < tokens; ++t) {
        const auto& scores = f.logits[t % 3];
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return scores[a] > scores[b]; });
        double total = 0;
        for (int j = 0; j < TopK; ++j) { total += std::exp(scores[order[j]] - scores[order[0]]); }
        for (int j = 0; j < TopK; ++j) {
            if (actual_ids[t * TopK + j] != order[j]) {
                std::cerr << "route T=" << tokens << " token=" << t << " rank=" << j
                          << ": ID mismatch " << actual_ids[t * TopK + j] << " != " << order[j]
                          << '\n';
                ++failures;
            }
            expected_alpha[t * TopK + j] = std::exp(scores[order[j]] - scores[order[0]]) / total;
        }
        expected_beta[t] = 1 / (1 + std::exp(-scores[E]));
    }
    const auto label = "route nvfp4 h3072 T=" + std::to_string(tokens);
    failures += verify_reduction(label + " alpha",
                                 std::vector<double>(actual_alpha.begin(), actual_alpha.end()),
                                 expected_alpha, kRoute);
    failures += verify_reduction(label + " beta",
                                 std::vector<double>(actual_beta.begin(), actual_beta.end()),
                                 expected_beta, kRoute);
    failures += ids.verify_guards(label) + alpha.verify_guards(label) +
                beta.verify_guards(label) + scratch.verify_guards(label);
    return failures;
}

int run(Fixture& f, CudaExecutionView execution, int tokens, bool shared, bool complete) {
    const auto& profile = f.profile;
    std::vector<float> x(std::size_t(tokens) * H);
    std::vector<int> ids(tokens * TopK);
    std::vector<float> alpha(ids.size()), beta(tokens);
    std::vector<double> reference(x.size());
    for (int t = 0; t < tokens; ++t) {
        const int p = t % 3;
        std::copy(f.x[p].begin(), f.x[p].end(), x.begin() + std::size_t(t) * H);
        std::array<double, TopK> coefficients;
        double total = 0, shared_coefficient = 0;
        if (complete) {
            const auto& scores = f.logits[p];
            std::vector<int> order(E);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return scores[a] > scores[b]; });
            for (int j = 0; j < TopK; ++j) {
                coefficients[j] = std::exp(scores[order[j]] - scores[order[0]]);
                total += coefficients[j];
                ids[t * TopK + j] = order[j];
            }
            for (int j = 0; j < TopK; ++j) { coefficients[j] /= total; }
            shared_coefficient = 1 / (1 + std::exp(-scores[E]));
        } else {
            for (int j = 0; j < TopK; ++j) {
                ids[t * TopK + j]   = (t * 61 + j * 29) % E;
                alpha[t * TopK + j] = .025F + j * .007F;
                coefficients[j]     = alpha[t * TopK + j];
            }
            if (shared) {
                beta[t]            = .25F + p * .125F;
                shared_coefficient = beta[t];
            }
        }
        for (int j = 0; j < TopK; ++j) {
            const int e        = ids[t * TopK + j];
            const auto& value  = f.expert(p, e);
            for (int h = 0; h < H; ++h) {
                reference[std::size_t(t) * H + h] += coefficients[j] * value[h];
            }
        }
        if (shared) {
            for (int h = 0; h < H; ++h) {
                reference[std::size_t(t) * H + h] += shared_coefficient * f.shared_values[p][h];
            }
        }
    }
    auto dx = to_device_bf16(x), di = to_device(ids), da = to_device(alpha), db = to_device(beta);
    Tensor input(dx.span(), DType::BF16, {H, tokens}),
        shared_tensor(db.span(), DType::FP32, {tokens});
    ops::ExpertRoutes routes{Tensor(di.span(), DType::I32, {TopK, tokens}),
                             Tensor(da.span(), DType::FP32, {TopK, tokens})};
    const std::size_t bytes =
        complete ? ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
                       profile, ops::ExpertSubsetKind::All, ops::MoeEpilogue::Write, tokens, tokens)
        : shared ? ops::routed_swiglu_workspace_capacity_bytes(
                         profile.routed, profile.shared, ops::ExpertSubsetKind::All, DType::BF16,
                         ops::MoeEpilogue::Write, tokens, tokens)
                 : ops::routed_swiglu_workspace_capacity_bytes(
                       profile.routed, ops::ExpertSubsetKind::All, DType::BF16,
                       ops::MoeEpilogue::Write, tokens, tokens);
    GuardedDeviceBuffer scratch(bytes), result(x.size() * 2);
    MemoryArena arena(scratch.span());
    Tensor output(result.span(), DType::BF16, {H, tokens});
    const ops::ExpertSubset subset{ops::ExpertSubsetKind::All};
    const auto invoke = [&] {
        if (complete) {
            ops::topk_softmax_swiglu_moe(input, profile, f.weights, {}, ops::MoeEpilogue::Write,
                                         output, arena, execution);
        } else {
            if (shared) {
                ops::routed_swiglu(input, profile.routed, f.weights.routed, routes, subset,
                                   {f.weights.shared, shared_tensor}, ops::MoeEpilogue::Write,
                                   output, arena, execution);
            } else {
                ops::routed_swiglu(input, profile.routed, f.weights.routed, routes, subset,
                                   ops::MoeEpilogue::Write, output, arena, execution);
            }
        }
    };
    invoke();
    cuda_synchronize(execution.stream());
    const auto actual = from_device_bf16(output.data, x.size());
    int failures      = 0;
    const std::string label = std::string(complete ? "complete" : "routed") + " nvfp4 h3072 T=" +
                              std::to_string(tokens) + " shared=" + std::to_string(shared);
    failures += verify_reduction(label, actual, reference, kA4);
    failures += scratch.verify_guards(label + " scratch") + result.verify_guards(label + " output");
    if (complete && tokens <= 520) {
        // Independent non-fused composition of the registered route, routed and
        // shared Ops must agree with the fused output within the same budget.
        auto fused = actual;
        Tensor route_ids(di.span(), DType::I32, {TopK, tokens}),
            route_alpha(da.span(), DType::FP32, {TopK, tokens});
        ops::ExpertRoutes selection{route_ids, route_alpha};
        Tensor shared_coefficients(db.span(), DType::FP32, {tokens});
        const auto route_bytes = ops::topk_softmax_route_workspace_capacity_bytes(
            {H, E, TopK}, tokens, tokens);
        const auto routed_bytes = ops::routed_swiglu_workspace_capacity_bytes(
            profile.routed, profile.shared, ops::ExpertSubsetKind::All, DType::BF16,
            ops::MoeEpilogue::Write, tokens, tokens);
        GuardedDeviceBuffer route_scratch(route_bytes), routed_scratch(routed_bytes),
            split_result(x.size() * 2);
        MemoryArena route_arena(route_scratch.span()), routed_arena(routed_scratch.span());
        Tensor split_output(split_result.span(), DType::BF16, {H, tokens});
        ops::topk_softmax_route(input, {H, E, TopK}, f.weights.control, selection,
                                shared_coefficients, route_arena, execution);
        ops::routed_swiglu(input, profile.routed, f.weights.routed, selection, subset,
                           {f.weights.shared, shared_coefficients}, ops::MoeEpilogue::Write,
                           split_output, routed_arena, execution);
        cuda_synchronize(execution.stream());
        const auto split = from_device_bf16(split_output.data, x.size());
        failures += verify_reduction(label + " vs split", split, fused, kConsistency);
        failures += route_scratch.verify_guards(label + " route scratch") +
                    routed_scratch.verify_guards(label + " routed scratch") +
                    split_result.verify_guards(label + " split output");
    }
    return failures;
}

int validate_cases(Fixture&) {
    namespace detail = ninfer::ops::detail;
    int failures     = 0;
    const auto accepted = [&](auto call, const char* label) {
        try {
            call();
        } catch (const std::invalid_argument& error) {
            std::cerr << "validate: registered " << label << " rejected: " << error.what() << '\n';
            ++failures;
        }
    };
    const auto rejected = [&](auto call, std::initializer_list<const char*> tokens) {
        try {
            call();
            std::cerr << "validate: expected rejection missing\n";
            ++failures;
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            for (const auto* token : tokens) {
                if (message.find(token) == std::string::npos) {
                    std::cerr << "validate: message lacks geometry token " << token << ": "
                              << message << '\n';
                    ++failures;
                }
            }
        }
    };
    accepted([] { detail::validate_route_profile({3072, 256, 8}); }, "route (3072,256,8)");
    accepted([] { detail::validate_route_profile({2048, 256, 8}); }, "route (2048,256,8)");
    accepted([] { detail::validate_route_profile({2560, 512, 10}); }, "route (2560,512,10)");
    accepted([] { detail::validate_expert_profile(make_profile().routed); }, "expert h3072_i1024");
    accepted([] { detail::validate_shared_profile(make_profile().shared, 3072); },
             "BF16 shared h3072_i1024");
    accepted([] {
        auto nvfp4_shared       = make_profile().shared;
        nvfp4_shared.gate_up_type = QType::NVFP4;
        nvfp4_shared.down_type    = QType::NVFP4;
        nvfp4_shared.gate_up_layout = QuantLayout::BlockScaleK16M128x4;
        nvfp4_shared.down_layout    = QuantLayout::BlockScaleK16M128x4;
        nvfp4_shared.gate_up_policy = ops::LinearPolicy::AllowA4;
        nvfp4_shared.down_policy    = ops::LinearPolicy::AllowA4;
        detail::validate_shared_profile(nvfp4_shared, 3072);
    }, "nvfp4 shared h3072_i1024");
    rejected([] { detail::validate_route_profile({3072, 256, 10}); }, {"3072", "256", "10"});
    rejected([] {
        auto mismatched          = make_profile().routed;
        mismatched.ffn.intermediate = 2048;
        detail::validate_expert_profile(mismatched);
    }, {"3072", "2048"});
    rejected([] {
        auto bf16_shared          = make_profile().shared;
        bf16_shared.intermediate     = 2048;
        detail::validate_shared_profile(bf16_shared, 3072);
    }, {"3072", "2048"});
    return failures;
}

// The 122B profile resolves to a kernel family on rtx5090 and b200 for both entries. On b200 the
// 2560/640/512/10 mma.sync profile has no kernel and the rejection names its geometry.
int dispatch_cases(Fixture&) {
    namespace detail = ninfer::ops::detail;
    int failures     = 0;
    try {
        (void)detail::moe_routed::shape_dispatch(make_profile().routed);
        (void)detail::moe_complete::shape_dispatch(make_profile());
    } catch (const std::invalid_argument& error) {
        std::cerr << "dispatch: registered h3072_i1024_e256_k8 rejected: " << error.what() << '\n';
        ++failures;
    }
#if NINFER_TARGET_B200
    auto other                 = make_profile();
    other.routed.ffn.hidden    = 2560;
    other.routed.ffn.intermediate = 640;
    other.routed.experts       = 512;
    other.routed.top_k         = 10;
    other.shared.hidden        = 2560;
    other.shared.intermediate  = 640;
    const auto expect_geometry = [&](auto call, const char* label) {
        try {
            call();
            std::cerr << "dispatch: " << label << " expected rejection missing\n";
            ++failures;
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            if (message.find("2560") == std::string::npos ||
                message.find("640") == std::string::npos) {
                std::cerr << "dispatch: " << label << " message lacks geometry: " << message
                          << '\n';
                ++failures;
            }
        }
    };
    expect_geometry([&] { (void)detail::moe_routed::shape_dispatch(other.routed); }, "routed");
    expect_geometry([&] { (void)detail::moe_complete::shape_dispatch(other); }, "complete");
#endif
    return failures;
}

// Backend routing and tuning contract of the b200 tcgen05 path: moe_backend selects the engine,
// the tuning resolver rejects tiles and groupings the grouped kernels do not provide, and the
// grouped kernels reject indexed (non-affine) expert banks.
int backend_cases(Fixture& f, CudaExecutionView execution) {
    namespace detail = ninfer::ops::detail;
    namespace bs     = ninfer::ops::detail::moe_cuda::block_scaled;
    int failures     = 0;
    const auto expect = [&](bool condition, const char* label) {
        if (!condition) {
            std::cerr << "backend: " << label << '\n';
            ++failures;
        }
    };
    const auto rejected = [&](auto call, const char* token) {
        try {
            call();
            std::cerr << "backend: expected rejection missing\n";
            ++failures;
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            if (message.find(token) == std::string::npos) {
                std::cerr << "backend: rejection message lacks '" << token << "': " << message
                          << '\n';
                ++failures;
            }
        }
    };
    // One token is a weight-bandwidth GEMV on every target; other geometries stay on mma.sync.
    expect(detail::moe_backend(3072, 256, 8, 1) == detail::MoeBackend::Vector,
           "single token must route to the vector kernels");
    expect(detail::moe_backend(3072, 256, 10, 130) == detail::MoeBackend::MmaSync,
           "unregistered geometry must stay on mma.sync");
#if NINFER_TARGET_B200
    expect(detail::moe_backend(3072, 256, 8, 130) == detail::MoeBackend::Tcgen05,
           "122B geometry must route to tcgen05");
    expect(detail::moe_backend(3072, 256, 8, 2048) == detail::MoeBackend::Tcgen05,
           "122B geometry must route to tcgen05 for large calls");
    // Unset tiles default from the token count, separately for the two projections.
    const auto tuned = [&](int tokens) {
        return bs::resolve_nvfp4_sm100_tuning(bs::Nvfp4Sm100Tuning{}, tokens);
    };
    expect(tuned(1).gate_up_tile_tokens == 32 && tuned(768).gate_up_tile_tokens == 32,
           "small gate/up calls default to 32-token tiles");
    expect(tuned(769).gate_up_tile_tokens == 64 && tuned(4096).gate_up_tile_tokens == 64,
           "large gate/up calls default to 64-token tiles");
    expect(tuned(1).down_tile_tokens == 16 && tuned(256).down_tile_tokens == 16,
           "small down calls default to 16-token tiles");
    expect(tuned(257).down_tile_tokens == 32 && tuned(768).down_tile_tokens == 32,
           "medium down calls default to 32-token tiles");
    expect(tuned(769).down_tile_tokens == 64 && tuned(1536).down_tile_tokens == 64,
           "large down calls default to 64-token tiles");
    expect(tuned(1537).down_tile_tokens == 128, "very large down calls default to 128-token tiles");
    // A requested tile survives the defaults.
    expect(bs::resolve_nvfp4_sm100_tuning({.gate_up_tile_tokens = 16, .down_tile_tokens = 64}, 4096)
                   .gate_up_tile_tokens == 16,
           "an explicit gate/up tile overrides the default");
    rejected([&] { bs::resolve_nvfp4_sm100_tuning({.gate_up_tile_tokens = 48}, 130); }, "48");
    rejected([&] { bs::resolve_nvfp4_sm100_tuning({.down_tile_tokens = 48}, 130); }, "48");
    rejected([&] { bs::resolve_nvfp4_sm100_tuning({.cluster = 2}, 130); }, "cluster");
    rejected([&] { bs::resolve_nvfp4_sm100_tuning({.ctas_per_sm = 2}, 130); }, "ctas_per_sm");
    // The grouped kernels address experts as codes + e * stride over one TMA map, so an
    // indexed (offsets) bank is rejected before any launch.
    {
        const int tokens = 130;
        std::vector<float> input(std::size_t(tokens) * H);
        for (int t = 0; t < tokens; ++t) {
            std::copy(f.x[t % 3].begin(), f.x[t % 3].end(), input.begin() + std::size_t(t) * H);
        }
        auto dx = to_device_bf16(input);
        std::vector<int> ids(tokens * TopK);
        std::vector<float> alpha(ids.size());
        for (int t = 0; t < tokens; ++t) {
            for (int j = 0; j < TopK; ++j) {
                ids[t * TopK + j]   = (t * 61 + j * 29) % E;
                alpha[t * TopK + j] = .025F + j * .007F;
            }
        }
        auto di = to_device(ids), da = to_device(alpha);
        Tensor x(dx.span(), DType::BF16, {H, tokens});
        ops::ExpertRoutes routes{Tensor(di.span(), DType::I32, {TopK, tokens}),
                                 Tensor(da.span(), DType::FP32, {TopK, tokens})};
        DeviceBuffer offsets(cuda_environment().context.device(), std::size_t(E) * 8);
        std::vector<std::uint64_t> indices(E, 0);
        offsets.copy_from_host(indices.data(), indices.size(), 0);
        auto experts     = f.weights.routed;
        auto gate_up     = f.gu.view();
        auto down        = f.down.view();
        gate_up.offsets  = static_cast<const std::uint64_t*>(offsets.data());
        gate_up.indexed_storage = f.gu.data.span();
        down.offsets     = gate_up.offsets;
        down.indexed_storage    = f.down.data.span();
        experts.bank.gate_up    = gate_up;
        experts.bank.down       = down;
        const auto bytes        = ops::routed_swiglu_workspace_capacity_bytes(
            f.profile.routed, ops::ExpertSubsetKind::All, DType::BF16, ops::MoeEpilogue::Write,
            tokens, tokens);
        GuardedDeviceBuffer scratch(bytes), result(input.size() * 2);
        MemoryArena arena(scratch.span());
        Tensor output(result.span(), DType::BF16, {H, tokens});
        rejected([&] {
            ops::routed_swiglu(x, f.profile.routed, experts, routes,
                               ops::ExpertSubset{ops::ExpertSubsetKind::All},
                               ops::MoeEpilogue::Write, output, arena, execution);
        }, "affine");
        failures += scratch.verify_guards("backend indexed scratch") +
                    result.verify_guards("backend indexed output");
        cuda_synchronize(execution.stream());
    }
#else
    (void)f;
    (void)execution;
    (void)bs::Nvfp4Tuning{};
#endif
    return failures;
}

void add_cases(OpSuite& suite) {
    std::vector<OpCase> catalogue;
    std::vector<std::function<int(Fixture&, CudaExecutionView)>> runners;
    const auto add = [&](std::string id, auto run) {
        catalogue.push_back(
            {"moe/122b/" + id, {{"format", "nvfp4"}, {"suite", "h3072_i1024_e256_k8"}}});
        runners.push_back(run);
    };
    add("validate", [](Fixture& f, auto) { return validate_cases(f); });
    add("dispatch", [](Fixture& f, auto) { return dispatch_cases(f); });
    add("backend", [](Fixture& f, auto e) { return backend_cases(f, e); });
    for (int t : {1, 33, 130, 520})
        add("route/t" + std::to_string(t),
            [t](Fixture& f, auto e) { return route_case(f, e, t); });
    for (int t : {1, 130, 520, 2048, 4096, 6000, 8500}) {
        add("routed/t" + std::to_string(t),
            [t](Fixture& f, auto e) { return run(f, e, t, false, false); });
        add("routed_shared/t" + std::to_string(t),
            [t](Fixture& f, auto e) { return run(f, e, t, true, false); });
        add("complete/t" + std::to_string(t),
            [t](Fixture& f, auto e) { return run(f, e, t, true, true); });
    }
    // Decode sizes from 8 to 127 tokens split the control and shared gate/up product over K.
    for (int t : {8, 33, 72, 127})
        add("complete/t" + std::to_string(t),
            [t](Fixture& f, auto e) { return run(f, e, t, true, true); });
    suite.add(std::move(catalogue), [=](auto selected) {
        DeviceContext device;
        Fixture fixture;
        int failures = 0;
        for (auto i : selected) failures += runners[i](fixture, device.execution_view());
        return failures;
    });
}
} // namespace
} // namespace ninfer::test::moe_122b

int main(int argc, char** argv) {
    using namespace ninfer::test;
    OpSuite suite;
    moe_122b::add_cases(suite);
    return suite.main(argc, argv);
}
