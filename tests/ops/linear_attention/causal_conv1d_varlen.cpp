// Varlen split causal conv: one call over a row table with traversal boundaries. Outputs, row
// destination windows and checkpoint windows are bit-identical to calling
// causal_conv1d_silu_split once per sub-segment, and match an independent double-precision oracle.
#include "ops/suite.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "cuda_fixture.h"
#include "ops/op_tester.h"

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

struct Profile {
    int channels, rows0, rows1, rows2;
};

constexpr Profile kProfiles[] = {
    {5120, 1024, 1024, 3072}, {8192, 2048, 2048, 4096}, {10240, 2048, 2048, 6144},
    {12288, 2048, 2048, 8192}};

struct Boundary {
    int row, column;
    bool checkpoint;
};

struct Spec {
    std::string name;
    Profile profile;
    std::vector<int> lengths;
    std::vector<Boundary> boundaries;
    bool in_place;
};

struct SubSegment {
    int row, start, length, checkpoint;
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

std::vector<float> bf16_values(std::size_t count, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(generator);
    round_to_bf16(values);
    return values;
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

StateFieldLayout record_layout(int channels, std::size_t record_bytes) {
    TensorRegion region;
    region.dtype  = DType::BF16;
    region.shape  = {channels, 3, 1, 1};
    region.region = {0, record_bytes, 16};
    return StateFieldLayout{region, 1, 0};
}

Tensor record_tensor(GuardedDeviceBuffer& arena, const Layout& layout, int record, int channels) {
    return Tensor(arena.span().subspan(static_cast<std::size_t>(record) * layout.record_bytes,
                                       layout.record_bytes),
                  DType::BF16, {channels, 3});
}

std::vector<std::uint16_t> read_record(const GuardedDeviceBuffer& arena, const Layout& layout,
                                       int record) {
    const auto* base = static_cast<const std::uint8_t*>(arena.data()) +
                       static_cast<std::size_t>(record) * layout.record_bytes;
    return from_device<std::uint16_t>(base, layout.record_bytes / 2);
}

int run_case(const Spec& spec, std::uint32_t seed) {
    const int R = static_cast<int>(spec.lengths.size()), C = static_cast<int>(spec.boundaries.size());
    int T = 0;
    for (int length : spec.lengths) T += length;
    const Profile p = spec.profile;
    const int channels = p.channels;
    const std::string label = "conv varlen " + spec.name + " channels=" + std::to_string(channels) +
                              " R=" + std::to_string(R) + " C=" + std::to_string(C) +
                              " T=" + std::to_string(T) + (spec.in_place ? " in-place" : " distinct");
    std::cout << "CONV_VARLEN_CASE " << label << std::endl;

    std::mt19937 generator(seed);
    const auto x_host = bf16_values(std::size_t(T) * channels, generator, -2.0F, 2.0F);
    const auto w_host = bf16_values(std::size_t(channels) * 4, generator, -0.6F, 0.6F);
    const std::size_t state_elements = std::size_t(channels) * 3;
    const auto initial = bf16_values(state_elements * R, generator, -2.0F, 2.0F);

    const auto subs = sub_segments(spec);
    const Layout layout{R, C, state_elements * 2};
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
    const std::size_t arena_bytes = std::size_t(layout.records()) * layout.record_bytes;
    GuardedDeviceBuffer arena_varlen(arena_bytes), arena_chain(arena_bytes);
    for (auto* arena : {&arena_varlen, &arena_chain}) {
        arena->fill(0xff);
        for (int row = 0; row < R; ++row) {
            std::vector<std::uint16_t> bits(state_elements);
            for (std::size_t i = 0; i < state_elements; ++i)
                bits[i] = f32_to_bf16(initial[state_elements * row + i]);
            arena->copy_from_host(bits.data(), layout.record_bytes,
                                  std::size_t(source_record(row)) * layout.record_bytes);
        }
    }

    DeviceBuffer d_x = to_device_bf16(x_host), d_w = to_device_bf16(w_host);
    const std::size_t out_bytes[3] = {std::size_t(p.rows0) * T * 2, std::size_t(p.rows1) * T * 2,
                                      std::size_t(p.rows2) * T * 2};
    const int out_rows[3] = {p.rows0, p.rows1, p.rows2};
    GuardedDeviceBuffer outs_varlen[3] = {GuardedDeviceBuffer(out_bytes[0]),
                                          GuardedDeviceBuffer(out_bytes[1]),
                                          GuardedDeviceBuffer(out_bytes[2])};
    GuardedDeviceBuffer outs_chain[3] = {GuardedDeviceBuffer(out_bytes[0]),
                                         GuardedDeviceBuffer(out_bytes[1]),
                                         GuardedDeviceBuffer(out_bytes[2])};
    for (int i = 0; i < 3; ++i) {
        outs_varlen[i].fill(0xff);
        outs_chain[i].fill(0xff);
    }
    Tensor x(d_x.span(), DType::BF16, {channels, T}), weight(d_w.span(), DType::BF16, {channels, 4});
    Tensor v0(outs_varlen[0].span(), DType::BF16, {out_rows[0], T});
    Tensor v1(outs_varlen[1].span(), DType::BF16, {out_rows[1], T});
    Tensor v2(outs_varlen[2].span(), DType::BF16, {out_rows[2], T});
    Tensor c0(outs_chain[0].span(), DType::BF16, {out_rows[0], T});
    Tensor c1(outs_chain[1].span(), DType::BF16, {out_rows[1], T});
    Tensor c2(outs_chain[2].span(), DType::BF16, {out_rows[2], T});
    const auto execution = cuda_environment().default_execution;

    int scratch_boundary = 0, previous_output = 0;
    for (std::size_t i = 0; i < subs.size(); ++i) {
        const auto& sub = subs[i];
        const bool first = i == 0 || subs[i - 1].row != sub.row;
        const bool last = i + 1 == subs.size() || subs[i + 1].row != sub.row;
        const int input = first ? source_record(sub.row) : previous_output;
        const int output = last ? layout.destination(sub.row)
                                : (sub.checkpoint >= 0 ? layout.checkpoint(sub.checkpoint)
                                                       : layout.scratch(scratch_boundary++));
        Tensor state_in = record_tensor(arena_chain, layout, input, channels);
        Tensor state_out = record_tensor(arena_chain, layout, output, channels);
        Tensor o0 = c0.slice(1, sub.start, sub.length), o1 = c1.slice(1, sub.start, sub.length);
        Tensor o2 = c2.slice(1, sub.start, sub.length);
        ops::causal_conv1d_silu_split(x.slice(1, sub.start, sub.length), weight, state_in,
                                      state_out, o0, o1, o2, execution);
        previous_output = output;
    }
    cuda_synchronize();

    DeviceBuffer d_tokens = to_device(token_offsets);
    DeviceBuffer d_source = to_device(source_offsets), d_destination = to_device(destination_offsets);
    DeviceBuffer d_b_rows, d_b_columns, d_b_offsets;
    Tensor t_b_rows, t_b_columns, t_b_offsets;
    if (C > 0) {
        d_b_rows = to_device(b_rows);
        d_b_columns = to_device(b_columns);
        d_b_offsets = to_device(b_offsets);
        t_b_rows = Tensor(d_b_rows.span(), DType::I32, {C});
        t_b_columns = Tensor(d_b_columns.span(), DType::I32, {C});
        t_b_offsets = Tensor(d_b_offsets.span(), DType::I64, {C});
    }
    const auto field = record_layout(channels, layout.record_bytes);
    const Tensor source_table(d_source.span(), DType::I64, {R});
    const Tensor destination_table(d_destination.span(), DType::I64, {R});
    ops::ConvVarlenRows rows{Tensor(d_tokens.span(), DType::I32, {R + 1}),
                             ConstStateFieldView(ConstRecordBatchView(
                                                     ConstMemorySpan(arena_varlen.span()), source_table),
                                                 field),
                             StateFieldView{{arena_varlen.span(), destination_table}, field}};
    ops::causal_conv1d_silu_split_varlen(x, weight, rows, t_b_rows, t_b_columns, t_b_offsets, field,
                                         R, C, v0, v1, v2, execution);
    cuda_synchronize();

    int failures = 0;
    for (int i = 0; i < 3; ++i) {
        failures += verify_exact((label + " out" + std::to_string(i) + " bits").c_str(),
                                 from_device<std::uint16_t>(outs_varlen[i].data(), out_bytes[i] / 2),
                                 from_device<std::uint16_t>(outs_chain[i].data(), out_bytes[i] / 2));
        failures += outs_varlen[i].verify_guards(label + " out guard");
    }
    const auto expect_record = [&](int record, const std::string& what) {
        failures += verify_exact((label + " " + what + " bits").c_str(),
                                 read_record(arena_varlen, layout, record),
                                 read_record(arena_chain, layout, record));
    };
    for (int row = 0; row < R; ++row) expect_record(layout.destination(row), "destination");
    for (int b = 0; b < C; ++b)
        if (spec.boundaries[b].checkpoint) expect_record(layout.checkpoint(b), "checkpoint");
    failures += arena_varlen.verify_guards(label + " arena");

    // Independent oracle over concat(source window, row columns), in double precision.
    const auto extended = [&](int row, int channel, int position) -> double {
        return position < 3
                   ? initial[state_elements * row + std::size_t(position) * channels + channel]
                   : x_host[std::size_t(token_offsets[row] + position - 3) * channels + channel];
    };
    std::vector<double> got(std::size_t(T) * channels), expected(got.size());
    for (int i = 0; i < 3; ++i) {
        const auto bits = from_device<std::uint16_t>(outs_varlen[i].data(), out_bytes[i] / 2);
        const int base = i == 0 ? 0 : (i == 1 ? p.rows0 : p.rows0 + p.rows1);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < out_rows[i]; ++c)
                got[std::size_t(t) * channels + base + c] =
                    bf16_to_f32(bits[std::size_t(t) * out_rows[i] + c]);
    }
    for (int row = 0; row < R; ++row) {
        for (int t = 0; t < spec.lengths[row]; ++t) {
            for (int c = 0; c < channels; ++c) {
                double sum = 0.0;
                for (int j = 0; j < 4; ++j)
                    sum += double(w_host[std::size_t(j) * channels + c]) * extended(row, c, t + j);
                expected[std::size_t(token_offsets[row] + t) * channels + c] =
                    sum / (1.0 + std::exp(-sum));
            }
        }
    }
    const ReductionCriterion criterion{1.0e-2, 1.0e-2, 1.0e-2};
    const auto stats = compute_reduction_stats(got.data(), expected.data(), got.size());
    if (!reduction_passes(stats, static_cast<std::int64_t>(got.size()), criterion)) {
        std::cerr << label << ": oracle rel_l2=" << stats.relative_l2 << '\n';
        ++failures;
    }
    const auto expect_window = [&](int record, int row, int consumed, const std::string& what) {
        std::vector<std::uint16_t> window(state_elements);
        for (int s = 0; s < 3; ++s)
            for (int c = 0; c < channels; ++c)
                window[std::size_t(s) * channels + c] =
                    f32_to_bf16(static_cast<float>(extended(row, c, consumed + s)));
        failures += verify_exact((label + " " + what + " window").c_str(),
                                 read_record(arena_varlen, layout, record), window);
    };
    for (int row = 0; row < R; ++row)
        expect_window(layout.destination(row), row, spec.lengths[row], "destination");
    for (int b = 0; b < C; ++b)
        if (spec.boundaries[b].checkpoint)
            expect_window(layout.checkpoint(b), spec.boundaries[b].row, spec.boundaries[b].column,
                          "checkpoint");
    return failures;
}

int rejection_case() {
    GuardedDeviceBuffer dummy(1 << 20);
    const auto device = cuda_environment().context.device();
    const auto execution = cuda_environment().default_execution;
    const auto call = [&](int channels, int rows, int boundaries, int tokens) {
        Tensor x(dummy.data(), device, DType::BF16, {channels, tokens});
        Tensor w(dummy.data(), device, DType::BF16, {channels, 4});
        Tensor o0(dummy.data(), device, DType::BF16, {2048, tokens});
        Tensor o1 = o0, o2(dummy.data(), device, DType::BF16, {4096, tokens});
        ops::causal_conv1d_silu_split_varlen(x, w, {}, {}, {}, {}, {}, rows, boundaries, o0, o1, o2,
                                             execution);
    };
    const auto expect = [](const std::string& label, const std::vector<std::string>& needles,
                           const auto& run) {
        try {
            run();
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            for (const auto& needle : needles)
                if (message.find(needle) == std::string::npos) {
                    std::cerr << label << ": message lacks '" << needle << "': " << message << '\n';
                    return 1;
                }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << label << ": wrong exception: " << error.what() << '\n';
            return 1;
        }
        std::cerr << label << ": no exception\n";
        return 1;
    };
    int failures = 0;
    failures += expect("unregistered profile", {"channels=6144", "R=2", "C=0", "T=40"},
                       [&] { call(6144, 2, 0, 40); });
    failures += expect("zero rows", {"R=0", "T=40"}, [&] { call(8192, 0, 0, 40); });
    failures += expect("too many rows", {"R=1025", "T=2048"}, [&] { call(8192, 1025, 0, 2048); });
    failures += expect("too many boundaries", {"C=1025", "T=4096"},
                       [&] { call(8192, 2, 1025, 4096); });
    failures += expect("boundaries exceed tokens", {"R=3", "C=9", "T=10"},
                       [&] { call(8192, 3, 9, 10); });
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    const std::vector<Boundary> boundaries{{0, 20, true}, {0, 21, false}, {2, 1, true},
                                           {2, 33, true}, {2, 35, true}};
    const char* names[] = {"c5120", "c8192", "c10240", "c12288"};
    for (int i = 0; i < 4; ++i) {
        suite.add(std::string("conv_varlen/") + names[i], [=] {
            return run_case({names[i], kProfiles[i], {40, 2, 70, 1}, boundaries, false}, 21 + i);
        });
    }
    suite.add("conv_varlen/c8192-in-place", [=] {
        return run_case({"in-place", kProfiles[1], {40, 2, 70, 1}, boundaries, true}, 31);
    });
    suite.add("conv_varlen/c10240-no-boundaries", [=] {
        return run_case({"plain", kProfiles[2], {5, 64, 1}, {}, false}, 32);
    });
    suite.add("conv_varlen/c12288-packed-flight", [=] {
        return run_case({"flight",
                         kProfiles[3],
                         {1, 7, 223, 536, 33, 64, 800, 3, 100},
                         {{2, 100, true}, {3, 64, true}, {6, 700, true}},
                         false},
                        33);
    });
    suite.add("conv_varlen/rejection", [] { return rejection_case(); });
    return suite.main(argc, argv);
}
