// Varlen Gated DeltaNet: one call over a row table with traversal boundaries. Contract under test:
// out, row destinations and checkpoints are bit-identical to calling the distinct-state
// gated_delta_net once per sub-segment, and match an independent double-precision recurrence.
#include "ops/suite.h"
#include "ninfer/ops/gated_delta_net.h"
#include "cuda_fixture.h"
#include "ops/op_tester.h"
#include "gdn_ref.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kState = 128;
constexpr ReductionCriterion kOutputCriterion{6.0e-3, 5.0e-6, 8.0e-3};
constexpr ReductionCriterion kStateCriterion{4.5e-3, 1.0e-5, 9.0e-3};

struct Boundary {
    int row, column;
    bool checkpoint;
};

struct Spec {
    std::string name;
    int hq, hv;
    std::vector<int> lengths;
    std::vector<Boundary> boundaries;
    bool normalize;
    bool in_place;
};

struct SubSegment {
    int row, start, length;
    int checkpoint; // boundary index or -1
};

std::vector<SubSegment> sub_segments(const Spec& spec) {
    std::vector<SubSegment> result;
    int begin = 0;
    std::size_t next = 0;
    for (int row = 0; row < static_cast<int>(spec.lengths.size()); ++row) {
        int previous = 0;
        for (; next < spec.boundaries.size() && spec.boundaries[next].row == row; ++next) {
            const int column = spec.boundaries[next].column;
            result.push_back({row, begin + previous, column - previous,
                              spec.boundaries[next].checkpoint ? static_cast<int>(next) : -1});
            previous = column;
        }
        result.push_back({row, begin + previous, spec.lengths[row] - previous, -1});
        begin += spec.lengths[row];
    }
    return result;
}

std::vector<float> uniform(std::size_t count, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(generator);
    return values;
}

void normalize_rows(std::vector<float>& values) {
    for (std::size_t row = 0; row < values.size() / kState; ++row) {
        double sum = 0.0;
        for (int d = 0; d < kState; ++d) sum += double(values[row * kState + d]) * values[row * kState + d];
        for (int d = 0; d < kState; ++d)
            values[row * kState + d] = static_cast<float>(values[row * kState + d] / std::sqrt(sum));
    }
}

gdn_ref::Inputs slice_inputs(const gdn_ref::Inputs& all, int start, int length) {
    gdn_ref::Inputs in;
    in.head_dim = kState;
    in.qk_heads = all.qk_heads;
    in.value_heads = all.value_heads;
    in.tokens = length;
    const auto copy = [&](const std::vector<float>& source, std::int64_t per_token) {
        return std::vector<float>(source.begin() + start * per_token,
                                  source.begin() + (start + length) * per_token);
    };
    in.q = copy(all.q, kState * all.qk_heads);
    in.k = copy(all.k, kState * all.qk_heads);
    in.v = copy(all.v, kState * all.value_heads);
    in.g = copy(all.g, all.value_heads);
    in.beta = copy(all.beta, all.value_heads);
    return in;
}

int verify_close(const std::string& label, const std::vector<double>& got,
                 const std::vector<double>& expected, const ReductionCriterion& criterion) {
    const auto stats = compute_reduction_stats(got.data(), expected.data(), got.size());
    if (reduction_passes(stats, static_cast<std::int64_t>(got.size()), criterion)) return 0;
    std::cerr << label << ": rel_l2=" << stats.relative_l2 << " limit=" << criterion.relative_l2
              << " max_abs=" << stats.maximum_absolute_error << '\n';
    return 1;
}

std::vector<double> widen(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

struct Layout {
    int rows, boundaries;
    std::size_t record_bytes;
    int source(int row) const { return row; }
    int destination(int row) const { return rows + row; }
    int checkpoint(int boundary) const { return 2 * rows + boundary; }
    int scratch(int boundary) const { return 2 * rows + boundaries + boundary; }
    int records() const { return 2 * rows + 2 * boundaries; }
};

StateFieldLayout record_layout(int value_heads, std::size_t record_bytes) {
    TensorRegion region;
    region.dtype  = DType::FP32;
    region.shape  = {kState, kState, value_heads, 1};
    region.region = {0, record_bytes, 16};
    return StateFieldLayout{region, 1, 0};
}

Tensor record_tensor(GuardedDeviceBuffer& arena, const Layout& layout, int record, int value_heads) {
    return Tensor(arena.span().subspan(static_cast<std::size_t>(record) * layout.record_bytes,
                                       layout.record_bytes),
                  DType::FP32, {kState, kState, value_heads});
}

std::vector<float> read_record(const GuardedDeviceBuffer& arena, const Layout& layout, int record) {
    const auto* base = static_cast<const std::uint8_t*>(arena.data()) +
                       static_cast<std::size_t>(record) * layout.record_bytes;
    return from_device<float>(base, layout.record_bytes / sizeof(float));
}

int run_case(const Spec& spec, std::uint32_t seed) {
    const int R = static_cast<int>(spec.lengths.size()), C = static_cast<int>(spec.boundaries.size());
    int T = 0;
    for (int length : spec.lengths) T += length;
    const int hq = spec.hq, hv = spec.hv;
    const float scale = 1.0F / std::sqrt(static_cast<float>(kState));
    const std::string label = "varlen " + spec.name + " Hq=" + std::to_string(hq) +
                              " Hv=" + std::to_string(hv) + " R=" + std::to_string(R) +
                              " C=" + std::to_string(C) + " T=" + std::to_string(T) +
                              (spec.in_place ? " in-place" : " distinct");
    std::cout << "GDN_VARLEN_CASE " << label << std::endl;

    std::mt19937 generator(seed);
    gdn_ref::Inputs in;
    in.head_dim = kState;
    in.qk_heads = hq;
    in.value_heads = hv;
    in.tokens = T;
    in.q = uniform(std::size_t(T) * kState * hq, generator, -1.0F, 1.0F);
    in.k = uniform(std::size_t(T) * kState * hq, generator, -1.0F, 1.0F);
    in.v = uniform(std::size_t(T) * kState * hv, generator, -0.5F, 0.5F);
    in.g = uniform(std::size_t(T) * hv, generator, -0.10F, -0.005F);
    in.beta = uniform(std::size_t(T) * hv, generator, 0.05F, 0.95F);
    if (!spec.normalize) {
        normalize_rows(in.q);
        normalize_rows(in.k);
    }
    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    const std::size_t state_floats = std::size_t(kState) * kState * hv;
    const auto initial = uniform(state_floats * R, generator, -0.02F, 0.02F);

    const auto subs = sub_segments(spec);
    const Layout layout{R, C, state_floats * sizeof(float)};
    const auto source_record = [&](int row) {
        return spec.in_place ? layout.destination(row) : layout.source(row);
    };
    std::vector<std::int32_t> token_offsets(R + 1, 0), b_rows, b_columns;
    std::vector<std::int64_t> source_offsets, destination_offsets, b_offsets;
    for (int row = 0; row < R; ++row) {
        token_offsets[row + 1] = token_offsets[row] + spec.lengths[row];
        source_offsets.push_back(std::int64_t(source_record(row)) * layout.record_bytes);
        destination_offsets.push_back(std::int64_t(layout.destination(row)) * layout.record_bytes);
    }
    for (int b = 0; b < C; ++b) {
        b_rows.push_back(spec.boundaries[b].row);
        b_columns.push_back(spec.boundaries[b].column);
        b_offsets.push_back(spec.boundaries[b].checkpoint
                                ? std::int64_t(layout.checkpoint(b)) * layout.record_bytes
                                : -1);
    }

    const auto prepare_arena = [&](GuardedDeviceBuffer& arena) {
        arena.fill(0xff);
        for (int row = 0; row < R; ++row)
            arena.copy_from_host(initial.data() + state_floats * row, layout.record_bytes,
                                 std::size_t(source_record(row)) * layout.record_bytes);
    };
    const std::size_t arena_bytes = std::size_t(layout.records()) * layout.record_bytes;
    GuardedDeviceBuffer arena_varlen(arena_bytes), arena_chain(arena_bytes);
    prepare_arena(arena_varlen);
    prepare_arena(arena_chain);

    DeviceBuffer d_q = to_device_bf16(in.q), d_k = to_device_bf16(in.k), d_v = to_device_bf16(in.v);
    DeviceBuffer d_g = to_device_f32(in.g), d_beta = to_device_f32(in.beta);
    GuardedDeviceBuffer out_varlen(in.v.size() * 2), out_chain(in.v.size() * 2);
    out_varlen.fill(0xff);
    out_chain.fill(0xff);
    Tensor q(d_q.span(), DType::BF16, {kState, hq, T}), k(d_k.span(), DType::BF16, {kState, hq, T});
    Tensor v(d_v.span(), DType::BF16, {kState, hv, T});
    Tensor g(d_g.span(), DType::FP32, {hv, T}), beta(d_beta.span(), DType::FP32, {hv, T});
    Tensor o_varlen(out_varlen.span(), DType::BF16, {kState, hv, T});
    Tensor o_chain(out_chain.span(), DType::BF16, {kState, hv, T});
    const auto execution = cuda_environment().default_execution;

    // The reference composition: the existing single-segment Op, once per sub-segment, in order.
    int max_length = 1;
    for (const auto& sub : subs) max_length = std::max(max_length, sub.length);
    GuardedDeviceBuffer chain_scratch(
        std::max<std::size_t>(ops::gated_delta_net_workspace_capacity_bytes(hq, hv, 1, max_length), 256));
    MemoryArena chain_ws(chain_scratch.span());
    int scratch_boundary = 0, previous_output = 0;
    for (std::size_t i = 0; i < subs.size(); ++i) {
        const auto& sub = subs[i];
        const bool first = i == 0 || subs[i - 1].row != sub.row;
        const bool last = i + 1 == subs.size() || subs[i + 1].row != sub.row;
        const int input = first ? source_record(sub.row) : previous_output;
        const int output = last ? layout.destination(sub.row)
                                : (sub.checkpoint >= 0 ? layout.checkpoint(sub.checkpoint)
                                                       : layout.scratch(scratch_boundary++));
        Tensor state_in = record_tensor(arena_chain, layout, input, hv);
        Tensor state_out = record_tensor(arena_chain, layout, output, hv);
        Tensor os = o_chain.slice(2, sub.start, sub.length);
        ops::gated_delta_net(q.slice(2, sub.start, sub.length), k.slice(2, sub.start, sub.length),
                             v.slice(2, sub.start, sub.length), g.slice(1, sub.start, sub.length),
                             beta.slice(1, sub.start, sub.length), scale, spec.normalize,
                             chain_ws, state_in, state_out, os, execution);
        previous_output = output;
    }
    cuda_synchronize();

    // The call under test.
    DeviceBuffer d_tokens = to_device(token_offsets);
    DeviceBuffer d_source = to_device(source_offsets), d_destination = to_device(destination_offsets);
    DeviceBuffer d_b_rows, d_b_columns, d_b_offsets;
    ops::GdnVarlenBoundaries boundaries;
    if (C > 0) {
        d_b_rows = to_device(b_rows);
        d_b_columns = to_device(b_columns);
        d_b_offsets = to_device(b_offsets);
        boundaries = {Tensor(d_b_rows.span(), DType::I32, {C}),
                      Tensor(d_b_columns.span(), DType::I32, {C}),
                      Tensor(d_b_offsets.span(), DType::I64, {C})};
    }
    const auto field = record_layout(hv, layout.record_bytes);
    const Tensor source_table(d_source.span(), DType::I64, {R});
    const Tensor destination_table(d_destination.span(), DType::I64, {R});
    ops::GdnVarlenRows rows{Tensor(d_tokens.span(), DType::I32, {R + 1}),
                       ConstStateFieldView(ConstRecordBatchView(ConstMemorySpan(arena_varlen.span()),
                                                                source_table),
                                           field),
                       StateFieldView{{arena_varlen.span(), destination_table}, field}};
    const auto capacity = ops::gated_delta_net_varlen_workspace_capacity_bytes(hq, hv, R, T, C);
    GuardedDeviceBuffer varlen_scratch(std::max<std::size_t>(capacity, 256));
    varlen_scratch.fill(0xff);
    MemoryArena varlen_ws(varlen_scratch.span());
    ops::gated_delta_net_varlen(q, k, v, g, beta, scale, spec.normalize, rows, boundaries, field, R,
                                C, varlen_ws, o_varlen, execution);
    cuda_synchronize();

    int failures = 0;
    failures += verify_exact((label + " out bits").c_str(),
                             from_device<std::uint16_t>(out_varlen.data(), in.v.size()),
                             from_device<std::uint16_t>(out_chain.data(), in.v.size()));
    const auto expect_record = [&](int record, const std::string& what) {
        failures += verify_exact((label + " " + what + " bits").c_str(),
                                 read_record(arena_varlen, layout, record),
                                 read_record(arena_chain, layout, record));
    };
    for (int row = 0; row < R; ++row) expect_record(layout.destination(row), "destination " + std::to_string(row));
    for (int b = 0; b < C; ++b)
        if (spec.boundaries[b].checkpoint) expect_record(layout.checkpoint(b), "checkpoint " + std::to_string(b));
    if (!spec.in_place) {
        for (int row = 0; row < R; ++row) {
            const auto got = read_record(arena_varlen, layout, layout.source(row));
            failures += verify_exact((label + " source unchanged").c_str(), got,
                                     std::vector<float>(initial.begin() + state_floats * row,
                                                        initial.begin() + state_floats * (row + 1)));
        }
    }
    if (varlen_ws.used() != 0 || varlen_ws.peak_used() > capacity) {
        std::cerr << label << ": workspace not released or above its reported capacity\n";
        ++failures;
    }
    failures += arena_varlen.verify_guards(label + " arena") + out_varlen.verify_guards(label + " out") +
                varlen_scratch.verify_guards(label + " workspace");

    // Independent reference: double-precision recurrence per row, from the row's initial state.
    for (int row = 0; row < R; ++row) {
        auto row_in = slice_inputs(in, token_offsets[row], spec.lengths[row]);
        row_in.state.assign(initial.begin() + state_floats * row, initial.begin() + state_floats * (row + 1));
        const auto ref = gdn_ref::evaluate(row_in, scale, spec.normalize);
        const std::size_t out_count = std::size_t(spec.lengths[row]) * hv * kState;
        const auto* out_base = static_cast<const std::uint16_t*>(out_varlen.data()) +
                               std::size_t(token_offsets[row]) * hv * kState;
        failures += verify_close(label + " row " + std::to_string(row) + " out",
                                 from_device_bf16(out_base, out_count), ref.out, kOutputCriterion);
        failures += verify_close(label + " row " + std::to_string(row) + " state",
                                 widen(read_record(arena_varlen, layout, layout.destination(row))),
                                 ref.final_state, kStateCriterion);
        for (int b = 0; b < C; ++b) {
            if (spec.boundaries[b].row != row || !spec.boundaries[b].checkpoint) continue;
            auto prefix = slice_inputs(in, token_offsets[row], spec.boundaries[b].column);
            prefix.state = row_in.state;
            const auto prefix_ref = gdn_ref::evaluate(prefix, scale, spec.normalize);
            failures += verify_close(label + " checkpoint " + std::to_string(b),
                                     widen(read_record(arena_varlen, layout, layout.checkpoint(b))),
                                     prefix_ref.final_state, kStateCriterion);
        }
    }
    return failures;
}

template <class Call>
int expect_invalid(const std::string& label, const std::vector<std::string>& needles, Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        for (const auto& needle : needles) {
            if (message.find(needle) == std::string::npos) {
                std::cerr << label << ": message lacks '" << needle << "': " << message << '\n';
                return 1;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << label << ": wrong exception type: " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": no exception\n";
    return 1;
}

int rejection_case() {
    GuardedDeviceBuffer scratch(256), dummy(256);
    MemoryArena ws(scratch.span());
    const auto device = cuda_environment().context.device();
    const auto execution = cuda_environment().default_execution;
    const auto call = [&](int hq, int hv, int rows, int boundaries, int tokens) {
        Tensor q(dummy.data(), device, DType::BF16, {kState, hq, tokens});
        Tensor v(dummy.data(), device, DType::BF16, {kState, hv, tokens});
        Tensor g(dummy.data(), device, DType::FP32, {hv, tokens});
        Tensor out = v;
        ops::gated_delta_net_varlen(q, q, v, g, g, 1.0F / std::sqrt(128.0F), true, {}, {}, {}, rows,
                                    boundaries, ws, out, execution);
    };
    const std::vector<std::string> names{"Hq=", "Hv=", "R=", "C=", "T="};
    int failures = 0;
    failures += expect_invalid("unsupported heads", {"Hq=8", "Hv=24", "R=2", "C=0", "T=40"},
                               [&] { call(8, 24, 2, 0, 40); });
    failures += expect_invalid("zero rows", {"Hq=16", "Hv=32", "R=0", "C=0", "T=40"},
                               [&] { call(16, 32, 0, 0, 40); });
    failures += expect_invalid("too many rows", {"R=1025", "T=2048"}, [&] { call(16, 64, 1025, 0, 2048); });
    failures += expect_invalid("too many boundaries", {"C=1025", "T=4096"},
                               [&] { call(16, 48, 2, 1025, 4096); });
    failures += expect_invalid("boundaries exceed tokens", {"R=3", "C=9", "T=10"},
                               [&] { call(16, 32, 3, 9, 10); });
    failures += expect_invalid("capacity query", names,
                               [&] { (void)ops::gated_delta_net_varlen_workspace_capacity_bytes(16, 40, 4, 64, 2); });
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("gdn_varlen/mixed-routes", [] {
        return run_case({"mixed", 16, 32, {37, 5, 16, 70, 1},
                         {{0, 20, true}, {1, 3, true}, {3, 7, true}, {3, 33, false}, {3, 50, true},
                          {3, 60, true}},
                         true, false}, 11);
    });
    suite.add("gdn_varlen/mixed-routes-in-place", [] {
        return run_case({"mixed", 16, 32, {37, 5, 16, 70, 1},
                         {{0, 20, true}, {1, 3, true}, {3, 7, true}, {3, 33, false}, {3, 50, true},
                          {3, 60, true}},
                         true, true}, 12);
    });
    suite.add("gdn_varlen/hv64-raw-qk", [] {
        return run_case({"hv64", 16, 64, {130, 33, 4},
                         {{0, 16, true}, {0, 17, false}, {0, 100, true}, {1, 15, true}},
                         false, false}, 13);
    });
    suite.add("gdn_varlen/hv48-no-boundaries", [] {
        return run_case({"hv48", 16, 48, {40, 3, 19}, {}, true, true}, 14);
    });
    // Many rows and boundaries: row scans cross warp boundaries and rows mix both routes.
    suite.add("gdn_varlen/many-rows", [] {
        std::mt19937 generator(2024);
        Spec spec{"many", 16, 32, {}, {}, true, false};
        for (int row = 0; row < 48; ++row) {
            const int length = 1 + static_cast<int>(generator() % 45);
            spec.lengths.push_back(length);
            std::vector<int> columns;
            for (int column = 1; column < length; ++column)
                if (generator() % 9 == 0 && columns.size() < 3) columns.push_back(column);
            for (int column : columns)
                spec.boundaries.push_back({row, column, generator() % 10 < 7});
        }
        return run_case(spec, 15);
    });
    suite.add("gdn_varlen/rejection", [] { return rejection_case(); });
    return suite.main(argc, argv);
}
