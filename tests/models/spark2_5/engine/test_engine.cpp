// Engine-level Spark2_5 contracts on converted artifacts.
//   graph16    16 concurrent requests decode to the same tokens with CUDA Graph and eagerly.
//   capacity   The judged configuration plans room for 8 x 256K and for 1 x 1M tokens.
//   scoring    CausalScoring starts and completes scores.
// The artifacts come from the fixture builders; exit code 77 when they are absent.
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace fs = std::filesystem;

constexpr int kSkip = 77;

struct Skip {};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

fs::path artifact(const char* directory_variable, const char* name) {
    const char* directory = std::getenv(directory_variable);
    if (directory == nullptr || *directory == '\0') {
        std::cout << "SKIP: " << directory_variable << " is not set\n";
        throw Skip{};
    }
    const auto path = fs::path(directory) / name;
    if (!fs::is_regular_file(path)) {
        std::cout << "SKIP: " << path << " is missing\n";
        throw Skip{};
    }
    return path;
}

// The evaluated serving configuration: Graph on, 256 MiB of planning headroom.
EngineOptions judged_options(const fs::path& path, std::uint32_t concurrency,
                             std::uint32_t context) {
    EngineOptions options;
    options.artifact_path         = path;
    options.max_concurrency       = concurrency;
    options.max_context           = context;
    options.use_cuda_graph        = true;
    options.memory_headroom_bytes = 256ULL << 20;
    return options;
}

std::vector<std::vector<TokenId>> run_concurrent(Engine& engine, std::uint32_t requests,
                                                 std::uint32_t output_tokens) {
    std::vector<GenerationHandle> handles;
    for (std::uint32_t i = 0; i < requests; ++i) {
        std::string text;
        // Histories from a few tokens to ~2K cross the decode envelope bounds (128, 512, 2048).
        for (std::uint32_t word = 0; word < 5 + 40 * i; ++word)
            text += "word" + std::to_string(word * i + i) + " ";
        auto prompt = engine.prepare_tokens(engine.tokenize_text(text));
        RequestOptions request;
        request.execution.requested_output_tokens = output_tokens;
        request.execution.sampling.temperature    = 2.0F;
        request.execution.sampling.top_p          = 1.0F;
        request.execution.sampling.seed           = 1000 + i;
        handles.push_back(engine.submit(std::move(prompt), std::move(request)));
    }
    std::vector<std::vector<TokenId>> out;
    for (auto& handle : handles) out.push_back(handle.wait().generated_token_ids);
    return out;
}

std::vector<std::vector<TokenId>> generate(const fs::path& path, bool graph,
                                           std::uint32_t& widest_batch) {
    auto options             = judged_options(path, 16, 8192);
    options.use_cuda_graph   = graph;
    options.context_capacity = ContextCapacityPolicy::explicit_capacity(2ULL << 30);
    Engine engine(options);
    require(engine.options().max_concurrency == 16, "Engine narrowed max_concurrency below 16");
    require(engine.options().kv_cache == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
            "Spark2_5 did not default to FP8 G128 KV");
    auto tokens         = run_concurrent(engine, 16, 24);
    const auto snapshot = engine.snapshot();
    widest_batch        = 0;
    for (std::size_t batch = 0; batch < snapshot.decode_by_batch.size(); ++batch)
        if (snapshot.decode_by_batch[batch].units)
            widest_batch = static_cast<std::uint32_t>(batch + 1);
    return tokens;
}

void test_graph16() {
    const auto path           = artifact("NINFER_SPARK2_5_FIXTURE", "fp8.ninfer");
    std::uint32_t graph_batch = 0, eager_batch = 0;
    const auto graph = generate(path, true, graph_batch);
    const auto eager = generate(path, false, eager_batch);
    require(graph.size() == 16 && eager.size() == 16, "missing results");
    for (std::size_t i = 0; i < graph.size(); ++i) {
        require(!graph[i].empty(), "request " + std::to_string(i) + " produced no tokens");
        require(graph[i] == eager[i],
                "Graph and eager tokens differ for request " + std::to_string(i));
    }
    require(graph_batch > 1, "no decode round batched more than one request");
    std::size_t distinct = 0;
    for (std::size_t i = 0; i < graph.size(); ++i)
        distinct += std::count(graph.begin(), graph.begin() + i, graph[i]) == 0;
    std::cout << "OK graph16 widest_graph_batch=" << graph_batch
              << " widest_eager_batch=" << eager_batch << " distinct_outputs=" << distinct << "\n";
}

void expect_capacity(const fs::path& path, std::uint32_t concurrency, std::uint32_t context) {
    Engine engine(judged_options(path, concurrency, context));
    const auto& description    = engine.description();
    const std::uint64_t needed = std::uint64_t{concurrency} * context;
    require(engine.options().kv_cache == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
            "Spark2_5 did not default to FP8 G128 KV");
    require(description.max_concurrency == concurrency && description.max_context == context,
            "Engine changed the planned concurrency or context");
    require(description.context_token_capacity_estimate >= needed,
            path.filename().string() + ": planned " +
                std::to_string(description.context_token_capacity_estimate) + " tokens for " +
                std::to_string(concurrency) + " x " + std::to_string(context));
    std::cout << "OK capacity " << path.filename().string() << " " << concurrency << "x"
              << context << " planned=" << description.context_token_capacity_estimate << "\n";
}

void test_capacity() {
    for (const char* name : {"bf16.ninfer", "fp8.ninfer"}) {
        const auto path = artifact("NINFER_SPARK2_5_FULL_FIXTURE", name);
        expect_capacity(path, 8, 256U << 10);
        expect_capacity(path, 1, 1U << 20);
    }
}

std::vector<TokenId> scoring_tokens(Engine& engine, std::size_t count) {
    std::string text;
    std::vector<TokenId> tokens;
    while (tokens.size() < count) {
        text += "Every evaluation window owns fresh state and a fresh KV address space. ";
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(count);
    return tokens;
}

EngineOptions scoring_options(const fs::path& path, std::uint32_t chunk) {
    EngineOptions options;
    options.artifact_path   = path;
    options.purpose         = EnginePurpose::CausalScoring;
    options.max_context     = 2048;
    options.max_concurrency = 4;
    options.prefill_chunk   = chunk;
    return options;
}

void test_scoring() {
    const auto path = artifact("NINFER_SPARK2_5_FIXTURE", "fp8.ninfer");
    std::vector<TokenId> tokens;
    std::vector<float> all;
    {
        Engine engine(scoring_options(path, 0));
        const auto& effective = engine.options();
        require(effective.max_concurrency == 1 && !effective.use_cuda_graph &&
                    !effective.context_cache.enabled,
                "causal scoring options were not normalized");
        tokens            = scoring_tokens(engine, 1500);
        all               = engine.score_tokens(tokens, 1);
        auto suffix       = engine.score_tokens(tokens, 301);
        require(all.size() == 1499 && suffix.size() == 1199,
                "score_tokens returned a wrong shape");
        for (const auto* scores : {&all, &suffix})
            for (const float value : *scores)
                require(std::isfinite(value) && value <= 0.0F, "score is not a log probability");
        const auto snapshot = engine.snapshot();
        require(snapshot.scoring.completed_jobs == 2 && snapshot.scoring.failed_jobs == 0 &&
                    snapshot.scoring.active_jobs == 0,
                "scoring jobs were not retired");
    }
    // The ring and KV carry across prefill windows: 128-token windows score the same tokens as
    // the default window up to bf16 rounding of the logits. A wrong carry changes the context
    // and moves scores by their own magnitude.
    Engine windowed(scoring_options(path, 128));
    require(windowed.options().prefill_chunk == 128, "explicit prefill_chunk was not kept");
    const auto split = windowed.score_tokens(tokens, 1);
    require(split.size() == all.size(), "windowed score has a wrong shape");
    double worst = 0, total = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
        const double difference =
            std::abs(double{all[i]} - double{split[i]}) / (1.0 + std::abs(double{all[i]}));
        worst                   = std::max(worst, difference);
        total += difference;
    }
    require(worst < 0.1 && total / all.size() < 0.01,
            "windowed prefill scores differ (relative): worst " + std::to_string(worst) + ", mean " +
                std::to_string(total / all.size()));
    std::cout << "OK scoring tokens=" << all.size() << " window_worst=" << worst
              << " window_mean=" << total / all.size() << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    try {
        if (mode == "graph16")
            test_graph16();
        else if (mode == "capacity")
            test_capacity();
        else if (mode == "scoring")
            test_scoring();
        else {
            std::cerr << "usage: ninfer_spark2_5_engine_test graph16|capacity|scoring\n";
            return 2;
        }
    } catch (const Skip&) { return kSkip; } catch (const std::exception& error) {
        std::cerr << "FAIL " << mode << ": " << error.what() << "\n";
        return 1;
    }
    return 0;
}
