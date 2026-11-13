// Engine concurrency ceiling and model-owned KV defaults.
// Host checks cover the shared ceiling, per-model defaults and construct_model resolution.
// The Qwen3.5 plan check needs a CUDA device handle and returns 77 when none exists.
#include "artifact/fixture.h"
#include "core/device.h"
#include "models/glm5next/config.h"
#include "models/glm5next/program/planning.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/program/planning.h"
#include "runtime/engine/startup.h"

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace {

using namespace ninfer;
namespace fixture = ninfer::test::artifact_fixture;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Returns the invalid_argument text, or an empty string when the call succeeds.
template <class Function>
std::string rejection(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

bool contains(const std::string& text, std::string_view needle) {
    return text.find(needle) != std::string::npos;
}

EngineOptions engine_options(std::uint32_t concurrency) {
    EngineOptions options;
    options.artifact_path   = "model.ninfer";
    options.max_concurrency = concurrency;
    return options;
}

void test_shared_ceiling() {
    static_assert(kMaximumConcurrency == 16);
    for (const std::uint32_t invalid : {0U, 17U}) {
        const auto message = rejection(
            [&] { (void)runtime::normalize_engine_options(engine_options(invalid)); });
        require(contains(message, "max_concurrency must be in [1,16]"),
                "Engine accepted a concurrency outside [1,16]");
    }
    auto accepted = runtime::normalize_engine_options(engine_options(16));
    require(accepted.max_concurrency == 16, "Engine rejected the maximum concurrency");
    require(!accepted.kv_cache, "normalization resolved a model-owned KV default");
    accepted = runtime::normalize_engine_options(engine_options(1));
    require(accepted.max_concurrency == 1, "Engine rejected a single lane");
}

template <class Resolve>
void check_resolution(Resolve resolve, KvCacheStorage default_storage, const char* model) {
    EngineOptions options;
    require(!options.kv_cache, model);
    require(default_storage == KvCacheStorage::BFloat16, model);
    require(resolve(options) == KvCacheStorage::BFloat16, model);
    options.kv_cache = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    require(resolve(options) == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, model);
}

void test_model_defaults() {
    static_assert(noexcept(models::qwen3_5::resolve_kv_cache(EngineOptions{})));
    static_assert(noexcept(models::qwen4_exp::resolve_kv_cache(EngineOptions{})));
    static_assert(noexcept(models::glm5next::resolve_kv_cache(EngineOptions{})));
    static_assert(models::qwen3_5::kMaximumSupportedConcurrency == 8);
    static_assert(models::qwen4_exp::kMaximumSupportedConcurrency == 8);
    static_assert(models::glm5next::kMaximumSupportedConcurrency == 8);
    check_resolution([](const EngineOptions& o) { return models::qwen3_5::resolve_kv_cache(o); },
                     models::qwen3_5::kDefaultKvCacheStorage, "Qwen3.5 KV resolution");
    check_resolution([](const EngineOptions& o) { return models::qwen4_exp::resolve_kv_cache(o); },
                     models::qwen4_exp::kDefaultKvCacheStorage, "Qwen4Exp KV resolution");
    check_resolution([](const EngineOptions& o) { return models::glm5next::resolve_kv_cache(o); },
                     models::glm5next::kDefaultKvCacheStorage, "GLM5Next KV resolution");
}

// The artifact names an architecture but carries no model, so construction fails while parsing
// its configuration. The architecture case has already resolved the KV storage by then.
void test_construct_model_resolution() {
    struct Route {
        const char* architecture;
        const char* model_type;
    };
    for (const auto& route : {Route{"Qwen3_5ForCausalLM", "qwen3_5_text"},
                              Route{"Qwen3_5MoeForCausalLM", "qwen3_5_moe_text"},
                              Route{"Qwen4ExpForCausalLM", "qwen4_exp_text"},
                              Route{"Glm5NextForCausalLM", "glm5next_text"}}) {
        fixture::Fixture artifact;
        artifact.root["components"]["text"]["config"] = {
            {"architectures", {route.architecture}}, {"model_type", route.model_type}};
        artifact.write();
        for (const auto requested : {std::optional<KvCacheStorage>{},
                                     std::optional{KvCacheStorage::Fp8E4m3fnGroup128Ue8m0}}) {
            auto options        = engine_options(2);
            options.artifact_path = artifact.entry;
            options.kv_cache    = requested;
            bool failed         = false;
            try {
                (void)runtime::construct_model(options);
            } catch (const std::exception&) { failed = true; }
            require(failed, "an artifact without a model was constructed");
            require(options.kv_cache == requested.value_or(KvCacheStorage::BFloat16),
                    "construct_model did not resolve the KV storage before building");
        }
    }
}

template <class Plan>
void check_limit(Plan plan, const char* limit_text, const char* model) {
    for (const std::uint32_t concurrency : {9U, 16U}) {
        require(contains(rejection([&] { plan(concurrency); }), limit_text), model);
    }
    // At the model limit the plan reaches later, unrelated checks of the empty profile.
    require(!contains(rejection([&] { plan(8); }), limit_text), model);
}

void test_model_plan_limits() {
    check_limit(
        [](std::uint32_t concurrency) {
            models::qwen4_exp::execution::ModelProfile profile{};
            (void)models::qwen4_exp::describe_execution_shape(profile,
                                                              engine_options(concurrency));
        },
        "up to 8", "Qwen4Exp accepted more lanes than its limit");
    check_limit(
        [](std::uint32_t concurrency) {
            (void)models::glm5next::plan_execution(
                std::make_shared<models::glm5next::execution::ModelProfile>(),
                engine_options(concurrency), KvCacheVectorFormat::BFloat16, 1);
        },
        "up to 8", "GLM5Next accepted more lanes than its limit");
}

// A profile that passes every Qwen3.5 target check ahead of the concurrency check.
int test_qwen3_5_plan_limit() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    namespace q = models::qwen3_5;
    q::execution::ModelProfile profile{};
    profile.config.text.attention              = q::AttentionConfig{};
    profile.config.text.full_attention_layers  = 1;
    profile.config.text.max_position_embeddings = 1U << 20;
    auto shared = std::make_shared<const q::execution::ModelProfile>(std::move(profile));
    DeviceContext device(0);
    for (const std::uint32_t concurrency : {9U, 16U}) {
        auto options          = engine_options(concurrency);
        options.prefill_chunk = 1024;
        require(contains(rejection([&] {
                             (void)q::detail::make_sequence_planner_impl(shared, device, options);
                         }),
                         "[1,8]"),
                "Qwen3.5 accepted more lanes than its limit");
    }
    return 0;
}

} // namespace

int main() {
    try {
        test_shared_ceiling();
        test_model_defaults();
        test_construct_model_resolution();
        test_model_plan_limits();
        const int qwen3_5 = test_qwen3_5_plan_limit();
        if (qwen3_5 == 77) std::cout << "no CUDA device; Qwen3.5 plan check skipped\n";
        return qwen3_5;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_concurrency_limits_test: " << error.what() << '\n';
        return 1;
    }
}
