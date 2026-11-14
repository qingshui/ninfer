// Decode step-time benchmark over a real Qwen3.5 artifact. One measured round is one complete
// decode step of a fixed batch — reserve, submit, commit and retire — with every lane licensed
// exactly one token; graph mode replays the captured bucket, eager mode dispatches directly.
// Phase intervals come from the opt-in execution phase clock: eager rounds record their
// segments on the stream, graph rounds read the captured segments a bucket replay refreshed.

#include "artifact/views.h"
#include "models/qwen3_5/execution/phase_timing.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/graphs.h"
#include "runtime/engine/startup.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace q35    = ninfer::models::qwen3_5;
namespace runtime = ninfer::runtime;
using Instance    = runtime::InferenceInstance<q35::RuntimeTypes>;
using Clock       = std::chrono::steady_clock;
using Json        = nlohmann::json;
using q35::execution::DecodePhase;

struct Options {
    std::filesystem::path artifact = "/raid0/models/ninfer/qwen35-122b.ninfer";
    int device                   = 0;
    std::string kv               = "fp8";
    std::string mode             = "both";
    std::vector<std::uint32_t> batches{1, 8, 32, 130, 520};
    std::uint32_t context_tokens = 64;
    std::uint32_t max_context    = 128;
    int warmup                   = 5;
    int repetitions              = 20;
};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--artifact <model.ninfer>] [--device <id>] [--kv bf16|int8|fp8|nvfp4|k8v4]"
                 " [--mode graphs|eager|both] [--batch <list>]"
                 " [--context <tokens>] [--max-context <tokens>] [--warmup <n>] [--reps <n>]\n"
                 "  --batch accepts a single value or a comma-separated list; default 1,8,32,130,520\n"
                 "  prints one JSON object per (batch, mode) on stdout\n";
}

std::vector<std::uint32_t> parse_batch_list(std::string_view text) {
    std::vector<std::uint32_t> result;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = text.find(',', begin);
        const auto item = text.substr(begin, end == std::string_view::npos ? text.size() : end - begin);
        const unsigned long value = std::stoul(std::string(item));
        if (value == 0 || value > ninfer::kMaximumConcurrency) {
            throw std::invalid_argument("--batch entries must be in [1," +
                                        std::to_string(ninfer::kMaximumConcurrency) + "]");
        }
        result.push_back(static_cast<std::uint32_t>(value));
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    if (result.empty()) throw std::invalid_argument("--batch needs at least one value");
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> std::string_view {
            if (++index >= argc) throw std::invalid_argument(std::string(name) + " needs a value");
            return argv[index];
        };
        if (argument == "--artifact") {
            options.artifact = std::filesystem::path(value("--artifact"));
        } else if (argument == "--device") {
            options.device = std::stoi(std::string(value("--device")));
        } else if (argument == "--kv") {
            options.kv = std::string(value("--kv"));
        } else if (argument == "--mode") {
            options.mode = std::string(value("--mode"));
        } else if (argument == "--batch") {
            options.batches = parse_batch_list(value("--batch"));
        } else if (argument == "--context") {
            options.context_tokens = std::stoul(std::string(value("--context")));
        } else if (argument == "--max-context") {
            options.max_context = std::stoul(std::string(value("--max-context")));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(value("--warmup")));
        } else if (argument == "--reps") {
            options.repetitions = std::stoi(std::string(value("--reps")));
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argc > 0 ? argv[0] : "ninfer_qwen3_5_decode_step_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    const std::array<std::string_view, 5> kv_names{"bf16", "int8", "fp8", "nvfp4", "k8v4"};
    if (options.device < 0) throw std::invalid_argument("--device must be nonnegative");
    if (std::find(kv_names.begin(), kv_names.end(), options.kv) == kv_names.end()) {
        throw std::invalid_argument("--kv must be bf16, int8, fp8, nvfp4 or k8v4");
    }
    if (options.mode != "graphs" && options.mode != "eager" && options.mode != "both") {
        throw std::invalid_argument("--mode must be graphs, eager or both");
    }
    if (options.warmup < 1) throw std::invalid_argument("--warmup must be positive");
    if (options.repetitions < 1) throw std::invalid_argument("--reps must be positive");
    if (options.context_tokens < 16 || options.context_tokens + 2 > options.max_context) {
        throw std::invalid_argument("context must fit [16, max-context) with room for the rounds");
    }
    if (static_cast<std::uint64_t>(options.context_tokens) +
            static_cast<std::uint64_t>(options.warmup + options.repetitions) + 2 >
        options.max_context) {
        throw std::invalid_argument("context and rounds exceed max-context");
    }
    return options;
}

ninfer::KvCacheStorage parse_kv(const std::string& text) {
    if (text == "bf16") return ninfer::KvCacheStorage::BFloat16;
    if (text == "int8") return ninfer::KvCacheStorage::Int8Group64;
    if (text == "fp8") return ninfer::KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    if (text == "nvfp4") return ninfer::KvCacheStorage::Nvfp4Group16;
    return ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
}

std::vector<ninfer::TokenId> prompt_tokens(std::uint32_t count) {
    std::vector<ninfer::TokenId> prompt{
        248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
        1710,   248046, 198, 248045, 74455, 198,   248068, 198,
    };
    prompt.insert(prompt.begin() + 9, count - prompt.size(), 374);
    return prompt;
}

// One decode step of a bound batch, from submit to retire. Host admission (unit reservation)
// stays outside the measured interval; commit and retire are inside it.
struct Round {
    double wall_ms                = 0;
    std::uint64_t licensed_tokens = 0;
    std::optional<double> target_stream_ms;
    bool phases_valid     = false;
    double attention_ms   = 0;
    double gdn_ms        = 0;
    double moe_ms        = 0;
    std::string phase_diagnostic;
};

class DecodeFixture {
public:
    explicit DecodeFixture(Instance& instance) : instance_(instance) {
        instance_.executor->bind_to_current_thread();
        clock_ = q35::execution::DecodePhaseClock::current();
    }
    q35::SequenceHandle bind(std::vector<ninfer::TokenId> tokens, runtime::LaneId lane,
                            std::uint32_t output_tokens) {
        auto& session = *instance_.session;
        auto prompt   = instance_.frontend.prepare_tokens(std::move(tokens), false);
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = output_tokens;
        options.allow_prefix_reuse      = false;
        auto plan     = session.plan_request(std::move(prompt), options);
        const auto source = session.inspect_source(plan, std::nullopt);
        if (!source || !session.start_binding(plan, lane, *source)) {
            throw std::runtime_error("benchmark binding could not reserve its first unit");
        }
        auto binding = await<q35::ContextProgress>();
        if (!binding.published || !binding.sequence ||
            binding.kind != q35::ContextOperationKind::Bind) {
            throw std::runtime_error("benchmark binding did not publish a sequence");
        }
        const auto sequence = *binding.sequence;
        for (;;) {
            const std::array<q35::ExecutionUnit, 1> unit{
                q35::ExecutionUnit{sequence, q35::ExecutionUnitKind::Prefill, 0}};
            if (!session.reserve_units(unit)) {
                throw std::runtime_error("benchmark prefill unit could not be reserved");
            }
            const auto start = session.start_prefill({&sequence, 1}, {});
            if (!start) throw std::runtime_error("benchmark prefill execution slot is occupied");
            auto progress = await<q35::PrefillProgress>(start->ticket);
            if (start->segments != 1 || progress.rows.size() != 1) {
                throw std::runtime_error("benchmark prefill admitted several segments");
            }
            auto& row = progress.rows.front();
            if (!row.complete) continue;
            if (!row.pending || row.pending->row_count() != 1) {
                throw std::runtime_error("benchmark prefill did not license exactly one row");
            }
            const std::array<runtime::CommitDecision, 1> decision{
                runtime::CommitDecision{.accepted_tokens = 1}};
            (void)session.commit(std::move(*row.pending), decision);
            retire(start->ticket, {&sequence, 1});
            return sequence;
        }
    }

    Round measure(std::span<const q35::SequenceHandle> sequences, bool graphs) {
        const auto batch = sequences.size();
        auto& session    = *instance_.session;
        std::vector<runtime::RoundBudget> budgets(batch, runtime::RoundBudget{1});
        std::vector<q35::ExecutionUnit> units(batch);
        for (std::size_t row = 0; row < batch; ++row) {
            units[row] = {sequences[row], q35::ExecutionUnitKind::Decode, 1};
        }
        if (!session.reserve_units(units)) {
            throw std::runtime_error("benchmark decode batch could not be reserved");
        }
        if (clock_) clock_->begin_round();
        const auto started = Clock::now();
        const auto ticket  = session.start_decode(sequences, budgets);
        if (!ticket) throw std::runtime_error("benchmark decode execution slot is occupied");
        auto pending = await<q35::PendingBatch>(*ticket);
        if (pending.row_count() != batch) {
            throw std::runtime_error("benchmark decode round returned invalid membership");
        }
        std::vector<runtime::CommitDecision> decisions(batch);
        Round result;
        for (std::size_t row = 0; row < batch; ++row) {
            const auto count = pending.row_counts().empty()
                                   ? std::uint32_t{1}
                                   : static_cast<std::uint32_t>(pending.row_counts()[row]);
            if (count == 0) throw std::runtime_error("benchmark decode round licensed no token");
            decisions[row] = runtime::CommitDecision{.accepted_tokens = count};
            result.licensed_tokens += count;
        }
        (void)session.commit(std::move(pending), decisions);
        auto complete = await<runtime::ExecutionComplete>(*ticket);
        result.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        if (const auto ns = complete.sample.max_rank_gpu.target_stream_ns) {
            result.target_stream_ms = static_cast<double>(*ns) * 1.0e-6;
        }
        for (const auto& sequence : sequences) {
            if (session.execution_pending(sequence)) {
                throw std::runtime_error("benchmark decode round left pending work");
            }
        }
        collect_phases(graphs, result);
        return result;
    }

    void release(q35::SequenceHandle sequence) {
        auto& session = *instance_.session;
        const auto result = session.abort(sequence);
        if (result.status != runtime::ConsumeStatus::Consumed) {
            throw std::runtime_error("benchmark could not release its active sequence");
        }
        while (session.has_context_operation() || instance_.executor->has_pending_work()) {
            if (session.poll()) (void)0;
            instance_.executor->flush();
            std::this_thread::yield();
        }
    }

private:
    // The eager round records its segments on the ring in layer order: [mixer, MoE] per layer.
    // A graph replay refreshes the segments of every captured body it launches, but the bench
    // cannot know those bodies' capture positions statically (profile and expert-subgraph
    // selection differ per plan), so it discovers them once: delta reads report the refreshed
    // pool positions, and once the union carries one mixer and one MoE segment per layer with
    // a layout the model explains, those positions are read directly every later round.
    void collect_phases(bool graphs, Round& round) {
        if (!clock_) return;
        const auto layers = layer_phases_.size();
        if (!layers) return;
        try {
            if (!graphs) {
                if (clock_->eager_segment_count() != 2 * layers) {
                    throw std::runtime_error("eager round recorded " +
                                             std::to_string(clock_->eager_segment_count()) +
                                             " segments, expected " + std::to_string(2 * layers));
                }
                for (std::size_t layer = 0; layer < layers; ++layer) {
                    if (clock_->eager_phase(2 * layer) != layer_phases_[layer] ||
                        clock_->eager_phase(2 * layer + 1) != DecodePhase::Moe) {
                        throw std::runtime_error("eager round phase pattern does not match the model");
                    }
                }
                round.attention_ms = clock_->eager_milliseconds(DecodePhase::Attention);
                round.gdn_ms       = clock_->eager_milliseconds(DecodePhase::Gdn);
                round.moe_ms       = clock_->eager_milliseconds(DecodePhase::Moe);
                round.phases_valid = true;
                return;
            }
            if (!positions_.empty()) {
                for (const auto index : positions_) {
                    const auto elapsed = clock_->capture_milliseconds_at(index);
                    if (!elapsed) throw std::runtime_error("graph replay segment did not complete");
                    switch (clock_->capture_phase_at(index)) {
                    case DecodePhase::Attention: round.attention_ms += *elapsed; break;
                    case DecodePhase::Gdn: round.gdn_ms += *elapsed; break;
                    case DecodePhase::Moe: round.moe_ms += *elapsed; break;
                    }
                }
                round.phases_valid = true;
                return;
            }
            // Discovery: one complete delta read locks directly; otherwise the union of the
            // refreshed positions accumulates across rounds to absorb value collisions. A union
            // that outgrows one mixer-and-MoE-per-layer restarts from the latest read, so a
            // calibration-to-steady transition of the replayed bodies cannot poison the set.
            const auto fresh = clock_->read_capture();
            ++reads_;
            std::set<std::size_t> indices;
            for (const auto& sample : fresh) indices.insert(sample.index);
            const auto lock = [&](const std::set<std::size_t>& positions) {
                if (positions.size() != 2 * layers || !validate_capture_positions(positions)) {
                    return false;
                }
                positions_.assign(positions.begin(), positions.end());
                return true;
            };
            if (lock(indices)) return collect_phases(graphs, round);
            if (reads_ > 1) {
                candidates_.insert(indices.begin(), indices.end());
                if (candidates_.size() > 2 * layers) {
                    candidates_.clear();
                    candidates_.insert(indices.begin(), indices.end());
                }
                if (lock(candidates_)) return collect_phases(graphs, round);
            }
            throw std::runtime_error("graph replay position discovery covers " +
                                     std::to_string(candidates_.size()) + " of " +
                                     std::to_string(2 * layers) + " segments after " +
                                     std::to_string(reads_) + " reads");
        } catch (const std::exception& error) {
            // An unattributable round keeps its step time; its phase columns stay absent.
            round.phases_valid     = false;
            round.phase_diagnostic = error.what();
        }
    }

    // One accepted layout covers a resident-expert Main graph: every attention mixer was
    // captured as its own child body before the layer-ordered main body, so the refreshed
    // positions read [attention mixers][per layer: (GDN mixer)?, MoE]. An offloaded schedule
    // instead captures mixer and MoE of each layer together, reading [mixer, MoE] per layer.
    [[nodiscard]] bool validate_capture_positions(const std::set<std::size_t>& positions) const {
        const auto layers = layer_phases_.size();
        if (positions.size() != 2 * layers) return false;
        std::vector<DecodePhase> phases;
        phases.reserve(positions.size());
        for (const auto index : positions) phases.push_back(clock_->capture_phase_at(index));
        std::vector<DecodePhase> expected;
        expected.reserve(2 * layers);
        const auto offloaded = [&] {
            for (std::size_t layer = 0; layer < layers; ++layer) {
                expected.push_back(layer_phases_[layer]);
                expected.push_back(DecodePhase::Moe);
            }
            return phases == expected;
        };
        if (offloaded()) return true;
        expected.clear();
        const auto resident = [&] {
            for (std::size_t layer = 0; layer < layers; ++layer) {
                if (layer_phases_[layer] == DecodePhase::Attention) {
                    expected.push_back(DecodePhase::Attention);
                }
            }
            for (std::size_t layer = 0; layer < layers; ++layer) {
                if (layer_phases_[layer] == DecodePhase::Gdn) expected.push_back(DecodePhase::Gdn);
                expected.push_back(DecodePhase::Moe);
            }
            return phases == expected;
        };
        return resident();
    }

    template <class Result>
    Result await(std::optional<runtime::ExecutionTicket> ticket = {}) {
        for (;;) {
            instance_.executor->flush();
            auto event = instance_.session->poll();
            if (event) {
                const bool matches = !ticket || event->ticket == *ticket;
                if (matches && std::holds_alternative<Result>(event->result)) {
                    return std::move(std::get<Result>(event->result));
                }
                // Benign completion noise outside the awaited round: unrelated context
                // publications and already retired executions.
                if (std::holds_alternative<runtime::ExecutionComplete>(event->result) ||
                    (std::holds_alternative<q35::ContextProgress>(event->result) &&
                     !std::is_same_v<Result, q35::ContextProgress>)) {
                    continue;
                }
                throw std::runtime_error("benchmark received an unrelated Session result");
            }
            std::this_thread::yield();
        }
    }

    void retire(runtime::ExecutionTicket ticket,
                std::span<const q35::SequenceHandle> sequences) {
        const auto complete = await<runtime::ExecutionComplete>(ticket);
        if (complete.row_count != sequences.size() ||
            std::any_of(sequences.begin(), sequences.end(), [&](const auto sequence) {
                return instance_.session->execution_pending(sequence);
            })) {
            throw std::runtime_error("benchmark commit did not retire its complete membership");
        }
    }

    Instance& instance_;
    q35::execution::DecodePhaseClock* clock_ = nullptr;
    std::span<const DecodePhase> layer_phases_;
    // Graph-mode position discovery state, reset per engine.
    std::set<std::size_t> candidates_;
    std::vector<std::size_t> positions_;
    std::size_t reads_ = 0;

public:
    void bind_layer_phases(std::span<const DecodePhase> phases) { layer_phases_ = phases; }
};

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto mid = values.size() / 2;
    return values.size() % 2 ? values[mid]
                            : 0.5 * (values[mid - 1] + values[mid]);
}

std::string device_name(int ordinal) {
    cudaDeviceProp properties{};
    const auto status = cudaGetDeviceProperties(&properties, ordinal);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return properties.name;
}

int run(const Options& options) {
    if (!std::filesystem::exists(options.artifact)) {
        std::cout << "SKIP: artifact not present: " << options.artifact.string() << '\n';
        return 0;
    }

    // The layer phase table (attention vs GDN per layer) drives per-round validation of the
    // phase clock's segment pattern; the capture pool must cover every graph-mode engine.
    std::vector<DecodePhase> layer_phases;
    {
        const auto definition = [&] {
            ninfer::artifact::Reader reader(options.artifact);
            return q35::read_definition(reader);
        }();
        const auto& layers = definition->weights().text.layers;
        std::vector<std::uint32_t> capture_buckets;
        for (const auto batch : options.batches) {
            const auto buckets = q35::graph_capture_buckets(batch);
            capture_buckets.insert(capture_buckets.end(), buckets.begin(), buckets.end());
        }
        for (const auto& layer : layers) {
            layer_phases.push_back(std::holds_alternative<q35::AttentionWeights>(layer.mixer)
                                       ? DecodePhase::Attention
                                       : DecodePhase::Gdn);
        }
        if (layer_phases.empty()) throw std::runtime_error("model has no Main layers");
        // Empirical worst case per captured bucket (observed on the 122B geometry): each of the
        // three captured attention profiles records one single-segment child body per full
        // attention layer plus one steady body of two segments per layer — 8 segments per layer
        // covers that with headroom. Every engine captures every bucket of its capacity list,
        // and all engines of the process share one pool.
        const auto capture_segments = capture_buckets.size() * layer_phases.size() * 8 + 256;
        q35::execution::DecodePhaseClock::arm(2 * layer_phases.size() + 8, capture_segments);
    }

    std::vector<std::pair<std::uint32_t, bool>> plan;
    for (const auto batch : options.batches) {
        if (options.mode != "eager") plan.emplace_back(batch, true);
        if (options.mode != "graphs") plan.emplace_back(batch, false);
    }
    const auto clock = q35::execution::DecodePhaseClock::current();

    for (const auto& [batch, graphs] : plan) {
        ninfer::EngineOptions engine;
        engine.artifact_path      = options.artifact;
        engine.device             = options.device;
        engine.max_context        = options.max_context;
        engine.prefill_chunk      = 128;
        engine.kv_cache          = parse_kv(options.kv);
        engine.max_concurrency    = batch;
        engine.use_cuda_graph     = graphs;
        engine.observability.gpu_timing = true;
        engine                    = runtime::normalize_engine_options(std::move(engine));
        auto constructed          = runtime::construct_model(engine);
        auto& instance = *std::get<std::unique_ptr<Instance>>(constructed.instance);
        DecodeFixture fixture(instance);
        fixture.bind_layer_phases(layer_phases);

        const auto output_tokens =
            static_cast<std::uint32_t>(options.warmup + options.repetitions) + 16;
        std::vector<q35::SequenceHandle> sequences;
        sequences.reserve(batch);
        for (std::uint32_t lane = 0; lane < batch; ++lane) {
            sequences.push_back(fixture.bind(prompt_tokens(options.context_tokens),
                                              runtime::LaneId{lane}, output_tokens));
        }
        const auto active = std::span<const q35::SequenceHandle>(sequences);
        if (graphs) {
            std::cerr << "capture pool: batch " << batch << " has "
                      << (clock ? clock->capture_segment_count() : 0) << " segments";
            if (clock) {
                std::cerr << ", intervals";
                for (const auto size : clock->capture_intervals()) std::cerr << ' ' << size;
            }
            std::cerr << '\n';
        }

        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            (void)fixture.measure(active, graphs);
        }
        std::vector<Round> rounds;
        rounds.reserve(static_cast<std::size_t>(options.repetitions));
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            rounds.push_back(fixture.measure(active, graphs));
        }
        for (const auto& sequence : sequences) fixture.release(sequence);

        std::vector<double> step_ms;
        std::vector<double> target_stream_ms;
        std::vector<double> licensed;
        std::vector<double> attention_ms, gdn_ms, moe_ms;
        std::size_t phase_rounds = 0;
        for (const auto& round : rounds) {
            step_ms.push_back(round.wall_ms);
            licensed.push_back(static_cast<double>(round.licensed_tokens));
            if (round.target_stream_ms) target_stream_ms.push_back(*round.target_stream_ms);
            if (round.phases_valid) {
                ++phase_rounds;
                attention_ms.push_back(round.attention_ms);
                gdn_ms.push_back(round.gdn_ms);
                moe_ms.push_back(round.moe_ms);
            } else if (!round.phase_diagnostic.empty() && phase_rounds == 0 &&
                       attention_ms.empty()) {
                std::cerr << "phase diagnostic (batch " << batch << ", "
                          << (graphs ? "graphs" : "eager")
                          << "): " << round.phase_diagnostic << '\n';
            }
        }
        const auto phase_mean = [&](const std::vector<double>& values) -> Json {
            if (values.empty()) return Json(nullptr);
            return Json(mean(values));
        };

        const double step_mean = mean(step_ms);
        const double tokens_per_s =
            1000.0 * mean(licensed) / step_mean;
        Json out;
        out["batch"]             = batch;
        out["mode"]              = graphs ? "graphs" : "eager";
        out["step_ms_mean"]      = step_mean;
        out["step_ms_p50"]       = median(step_ms);
        out["decode_tokens_per_s"] = tokens_per_s;
        out["phase_ms"]["attention"] = phase_mean(attention_ms);
        out["phase_ms"]["gdn"]       = phase_mean(gdn_ms);
        out["phase_ms"]["moe"]       = phase_mean(moe_ms);
        // The logits projection and sampling dispatch live in the Program layer, below the
        // phase clock's hooks: the project column stays null rather than inferred.
        out["phase_ms"]["project"]  = nullptr;
        out["target_stream_ms"]     = target_stream_ms.empty() ? Json(nullptr)
                                                               : Json(mean(target_stream_ms));
        out["measurement"]          = Json{
            {"scope",
             "single decode step of the measured batch: submit, graph replay or eager dispatch, "
             "commit and retire; host admission (unit reservation), host queuing and prefill "
             "are outside the timed interval"},
            {"artifact", options.artifact.string()},
            {"device_name", device_name(options.device)},
            {"kv_dtype", options.kv},
            {"context_tokens", options.context_tokens},
            {"max_context", options.max_context},
            {"graph_bucket", batch},
            {"warmup", options.warmup},
            {"repetitions", options.repetitions},
            {"licensed_tokens_per_round", mean(licensed)},
            {"phase_valid_rounds", phase_rounds},
            {"phase_total_rounds", rounds.size()},
            {"phase_notes",
             "attention/gdn/moe are CUDA event intervals of the Main traversal (eager: host-"
             "recorded per segment, graph mode: captured segments refreshed by the bucket "
             "replay; MoE segments span the offload schedule's host handoffs); project is not "
             "instrumented and stays null; phase means cover rounds with a valid full segment "
             "set only"},
        };
        std::cout << out.dump() << '\n';
        std::cout.flush();
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer_qwen3_5_decode_step_bench: " << error.what() << '\n';
        return 1;
    }
}
