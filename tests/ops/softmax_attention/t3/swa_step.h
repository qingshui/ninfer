#pragma once

#include "swa_support.h"

namespace {

// Attend over the ring as committed before this chunk, compare with the naive mask reference,
// then append the chunk. `length` holds each row's committed length and is advanced.
int run_step(int batch, const Step& step, std::vector<int>& length,
             const std::vector<History>& history, const CyclicKVCacheLayerView& view,
             const std::string& label, int step_index) {
    const int T                = step.width;
    const std::size_t q_count  = static_cast<std::size_t>(D) * Hq * T * batch;
    const std::size_t kv_count = static_cast<std::size_t>(D) * Hkv * T * batch;
    std::vector<float> q(q_count);
    fill_uniform(q, 500U + step_index * 17U + T, -0.35f, 0.35f);
    round_to_bf16(q);
    auto q_bits = bf16_bits(q);
    std::vector<std::uint16_t> k_bits(kv_count), v_bits(kv_count);
    std::vector<int> positions(static_cast<std::size_t>(T) * batch);
    const std::vector<int>& counts = step.valid;
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < T; ++t) {
            const std::size_t base = (static_cast<std::size_t>(b) * T + t) * Hkv * D;
            if (t < step.valid[b]) {
                const std::size_t src = static_cast<std::size_t>(length[b] + t) * Hkv * D;
                std::copy_n(history[b].k_bits.begin() + src, Hkv * D, k_bits.begin() + base);
                std::copy_n(history[b].v_bf16_bits.begin() + src, Hkv * D, v_bits.begin() + base);
                positions[b * T + t] = length[b] + t;
            } else {
                // Inert physical tail: arbitrary bits, including NaN.
                std::fill_n(q_bits.begin() + (static_cast<std::size_t>(b) * T + t) * Hq * D,
                            Hq * D, 0x7fc1);
                std::fill_n(k_bits.begin() + base, Hkv * D, 0x7fc1);
                std::fill_n(v_bits.begin() + base, Hkv * D, 0x7fc1);
                positions[b * T + t] = -1234567;
            }
        }

    // Naive reference: key j is visible to query position p iff 0 <= p-j <= 511.
    std::vector<double> expected(q_count, 0.0);
    parallel_rows(batch * T, std::uint64_t(batch) * T * Window * D * Hq, [&](int first, int last) {
        for (int item = first; item < last; ++item) {
            const int b = item / T, t = item % T;
            if (t >= step.valid[b]) continue;
            const int p      = length[b] + t;
            const auto& hist = history[b];
            naive_dense_softmax_attention(
                Geometry, 1, p + 1, static_cast<double>(Scale),
                [&](int d, int h, int) {
                    return static_cast<double>(bf16_to_f32(
                        q_bits[((static_cast<std::size_t>(b) * T + t) * Hq + h) * D + d]));
                },
                [&](int d, int h, int key) {
                    return static_cast<double>(
                        hist.k[(static_cast<std::size_t>(key) * Hkv + h) * D + d]);
                },
                [&](int d, int h, int key) {
                    return static_cast<double>(
                        hist.v16[(static_cast<std::size_t>(key) * Hkv + h) * D + d]);
                },
                [&](int, int key) { return p - key >= 0 && p - key <= Window - 1; },
                [&](int d, int h, int, double value) {
                    expected[((static_cast<std::size_t>(b) * T + t) * Hq + h) * D + d] = value;
                });
        }
    });

    auto exec           = cuda_environment().context.execution_view();
    const auto capacity = ops::causal_sliding_window_attention_workspace_capacity_bytes(
        Geometry, Window, batch, T, T, exec.multiprocessor_count());
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    MemoryArena workspace(scratch.span().subspan(0, std::max<std::size_t>(capacity, 1)));
    GuardedDeviceBuffer dq(q_count * 2), dk(kv_count * 2), dv(kv_count * 2),
        dp(positions.size() * 4), dvalid(batch * 4), output(q_count * 2);
    Tensor tq(dq.span(), DType::BF16, {D, Hq, T, batch}),
        tk(dk.span(), DType::BF16, {D, Hkv, T, batch}),
        tv(dv.span(), DType::BF16, {D, Hkv, T, batch}), tp(dp.span(), DType::I32, {T, batch}),
        tvalid(dvalid.span(), DType::I32, {batch}),
        to(output.span(), DType::BF16, {D, Hq, T, batch});
    dq.copy_from_host(q_bits.data(), q_bits.size() * 2);
    dk.copy_from_host(k_bits.data(), k_bits.size() * 2);
    dv.copy_from_host(v_bits.data(), v_bits.size() * 2);
    dp.copy_from_host(positions.data(), positions.size() * 4);
    dvalid.copy_from_host(counts.data(), batch * 4);
    output.fill(0xff);
    cuda_synchronize();

    ops::causal_sliding_window_attention(tq, tk, tv, tp, tvalid, Geometry, Window, Scale,
                                         ConstCyclicKVCacheLayerView(view), workspace, to, exec);
    cuda_synchronize(exec.stream());

    const std::string name = label + " step" + std::to_string(step_index) +
                             " T=" + std::to_string(T) + " L0=" + std::to_string(length[0]);
    int failures =
        verify_reduction(name, from_device_bf16(output.data(), q_count), expected, Criterion);
    const auto out_bits = from_device<std::uint16_t>(output.data(), q_count);
    bool tail_zero      = true;
    for (int b = 0; b < batch; ++b)
        for (int t = step.valid[b]; t < T; ++t)
            for (int i = 0; i < Hq * D; ++i)
                tail_zero = tail_zero &&
                            out_bits[(static_cast<std::size_t>(b) * T + t) * Hq * D + i] == 0;
    if (!tail_zero) {
        std::cerr << name << ": inert tail is not exact zero\n";
        ++failures;
    }
    failures += output.verify_guards(name) + scratch.verify_guards(name + " scratch");

    GuardedDeviceBuffer dcounts(batch * 4);
    Tensor tcounts(dcounts.span(), DType::I32, {batch});
    dcounts.copy_from_host(counts.data(), batch * 4);
    ops::kv_cache_append_prefix(
        tk, tv, tp, tcounts,
        ops::KVCacheAppendPrefixExecutionEnvelope{0, static_cast<std::uint32_t>(T)}, view, exec);
    cuda_synchronize(exec.stream());
    for (int b = 0; b < batch; ++b) length[b] += step.valid[b];
    failures += dq.verify_guards(name + " q") + dk.verify_guards(name + " k") +
                dv.verify_guards(name + " v") + dp.verify_guards(name + " positions");
    return failures;
}


} // namespace
