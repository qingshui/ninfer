#pragma once

#include "core/dtype.h"
#include "ninfer/types.h"
#include "op_measurement.cuh"

namespace ninfer::bench::gdn_recurrence {
enum class Mode { Running, BatchUpdate, ChunkedRaw, ChunkedNormalized, Recurrent };

struct Options {
    measurement::Options timing;
    Mode mode    = Mode::Running;
    int qk_heads = 16, value_heads = 48, batch = 1;
    std::string state_form = "distinct";
    std::vector<int> tokens{1024};
    bool breakdown      = false;
    std::string qk_norm = "fused";
    DType state_dtype   = DType::FP32;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool explicit_tokens = false, explicit_state = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout
                << "Gated DeltaNet recurrence benchmark\n"
                   "  --mode running|batch-update|chunked-raw|chunked-normalized|recurrent\n"
                   "  --tokens T[,T...] or --sweep START:END[:STEP] --qk-heads H "
                   "--value-heads H --batch B\n"
                   "  --qk-norm fused|composed (composed requires B=1)\n"
                   "  --breakdown (chunked stages)\n"
                   "  --state-form distinct|alias (batch default alias, prefill default distinct)\n"
                   "  --state-dtype fp32|bf16 (batch-update State storage, default fp32)\n"
                   "Running and batch-update call public Ops; other modes and breakdown are "
                   "private stage analysis.\n"
                   "chunked-raw normalizes raw Q/K; chunked-normalized consumes "
                   "pre-normalized BF16 Q/K.\n";
            measurement::help();
            std::exit(0);
        }
        if (flag == "--breakdown") {
            o.breakdown = true;
            continue;
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--mode") {
            if (value == "running")
                o.mode = Mode::Running;
            else if (value == "batch-update")
                o.mode = Mode::BatchUpdate;
            else if (value == "chunked-raw")
                o.mode = Mode::ChunkedRaw;
            else if (value == "chunked-normalized")
                o.mode = Mode::ChunkedNormalized;
            else if (value == "recurrent")
                o.mode = Mode::Recurrent;
            else
                throw std::invalid_argument("unknown recurrence mode");
        } else if (flag == "--tokens") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::integers(value);
            explicit_tokens = true;
        } else if (flag == "--sweep") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::sweep(value);
            explicit_tokens = true;
        } else if (flag == "--qk-heads")
            o.qk_heads = measurement::integer(value);
        else if (flag == "--value-heads")
            o.value_heads = measurement::integer(value);
        else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--state-form") {
            o.state_form   = value;
            explicit_state = true;
        } else if (flag == "--qk-norm")
            o.qk_norm = value;
        else if (flag == "--state-dtype") {
            if (value == "fp32")
                o.state_dtype = DType::FP32;
            else if (value == "bf16")
                o.state_dtype = DType::BF16;
            else
                throw std::invalid_argument("unknown state dtype");
        }
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (o.state_form != "distinct" && o.state_form != "alias")
        throw std::invalid_argument("unknown state form");
    if (!explicit_state && o.mode == Mode::BatchUpdate) o.state_form = "alias";
    if (o.mode == Mode::BatchUpdate && !explicit_tokens) o.tokens = {1};
    // Batch update admits one row per decode lane.
    if (o.value_heads < o.qk_heads || o.value_heads % o.qk_heads || o.batch < 1 ||
        o.batch > static_cast<int>(kMaximumConcurrency) ||
        (o.mode != Mode::BatchUpdate && o.batch != 1))
        throw std::invalid_argument("invalid head/batch geometry");
    if (o.qk_norm != "fused" && o.qk_norm != "composed")
        throw std::invalid_argument("unknown normalization");
    if (o.state_dtype != DType::FP32 && o.mode != Mode::BatchUpdate)
        throw std::invalid_argument("--state-dtype bf16 requires batch-update");
    if (o.qk_norm == "composed" && (o.mode != Mode::BatchUpdate || o.batch != 1))
        throw std::invalid_argument("composed normalization requires batch-update with B=1");
    if (o.breakdown && o.mode == Mode::BatchUpdate)
        throw std::invalid_argument("breakdown requires a prefill mode");
    for (int t : o.tokens) {
        if (o.mode == Mode::BatchUpdate && t != 1)
            throw std::invalid_argument("batch-update requires T=1");
        if ((o.mode == Mode::ChunkedRaw || o.mode == Mode::ChunkedNormalized) && t < 16)
            throw std::invalid_argument("chunked mode requires T>=16");
    }
    return o;
}
} // namespace ninfer::bench::gdn_recurrence
