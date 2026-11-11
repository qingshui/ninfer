#include "swa_step.h"

namespace {

// Run one scenario of chunked attend-then-append steps for `batch` rows and check, after every
// step, the whole ring against the logical history (committed window written, nothing else).
int run_scenario(const std::string& label, int batch, const std::vector<Step>& steps) {
    std::vector<int> total(batch, 0);
    for (const auto& step : steps)
        for (int b = 0; b < batch; ++b) total[b] += step.valid[b];
    std::vector<History> history;
    for (int b = 0; b < batch; ++b) history.push_back(make_history(b, total[b]));

    std::vector<int> lanes(batch);
    for (int b = 0; b < batch; ++b) lanes[b] = (b * 5 + 3) % Lanes;
    Ring ring;
    GuardedDeviceBuffer dlane(batch * 8);
    Tensor tlane(dlane.span(), DType::I64, {batch});
    const auto view = ring.view(tlane);
    const auto offsets = ring.packed->record_offsets(lanes);
    dlane.copy_from_host(offsets.data(), batch * 8);

    auto expected_k = ring.k_init;
    auto expected_v = ring.v_init;
    std::vector<int> length(batch, 0);
    int failures = 0;
    for (std::size_t s = 0; s < steps.size(); ++s) {
        const auto before = length;
        failures += run_step(batch, steps[s], length, history, view, label, static_cast<int>(s));
        for (int b = 0; b < batch; ++b)
            for (int p = std::max(before[b], length[b] - Window); p < length[b]; ++p)
                for (int h = 0; h < Hkv; ++h)
                    for (int d = 0; d < D; ++d) {
                        const std::size_t src = (static_cast<std::size_t>(p) * Hkv + h) * D + d;
                        expected_k[Ring::index(lanes[b], h, p, d)] = history[b].k_bits[src];
                        expected_v[Ring::index(lanes[b], h, p, d)] = fp16_bits_of(history[b].v16[src]);
                    }
        ring.sync();
        const std::string name = label + " ring after step " + std::to_string(s);
        failures += verify_exact((name + " K").c_str(),
                                 from_device<std::uint16_t>(ring.device_k.data(), expected_k.size()),
                                 expected_k);
        failures += verify_exact((name + " V").c_str(),
                                 from_device<std::uint16_t>(ring.device_v.data(), expected_v.size()),
                                 expected_v);
        if (failures) break;
    }
    ring.sync();
    if (!ring.packed->guards_intact()) {
        std::cerr << label << ": ring state-record guards overwritten\n";
        ++failures;
    }
    failures += ring.device_k.verify_guards("ring K buffer") +
                ring.device_v.verify_guards("ring V buffer") + dlane.verify_guards("lanes");
    return failures;
}

std::vector<Step> uniform_steps(const std::vector<int>& widths) {
    std::vector<Step> steps;
    for (int w : widths) steps.push_back({w, {w}});
    return steps;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    // B=1: lengths 511/512/513 (window edge for T=1 decode), a >window chunk that reaches
    // L=4096, a decode at 4096, then chunks that wrap the ring, including one larger than it.
    suite.add("t3/swa/b1/edges_511_512_513_4096_and_wrap", [] {
        return run_scenario("b1 edges", 1, uniform_steps({511, 1, 1, 1, 3582, 1, 16, 700, 1}));
    });
    // B=1: chunks of exactly window+1 and window tokens from an empty and a wrapped ring.
    suite.add("t3/swa/b1/chunk_window_plus_one", [] {
        return run_scenario("b1 chunk513", 1, uniform_steps({513, 1, 512, 1}));
    });
    // B=16: ragged valid counts (including empty rows) so that every row's queries cross
    // positions 511/512/513 inside 16-wide chunks and its ring wraps.
    suite.add("t3/swa/b16/ragged_wrap", [] {
        std::vector<Step> steps;
        for (int s = 0; s < 44; ++s) {
            Step step{16, std::vector<int>(16)};
            for (int b = 0; b < 16; ++b)
                step.valid[b] = (b % 5 == 0 && s % 11 == 4) ? 0 : 16 - ((b + s) % 4);
            steps.push_back(step);
        }
        return run_scenario("b16 ragged", 16, steps);
    });
    return suite.main(argc, argv);
}
