#include "models/spark2_5/startup.h"

#include "artifact/placement.h"
#include "models/common/frontend/digest.h"
#include "models/spark2_5/program/program.h"

#include <nlohmann/json.hpp>

namespace ninfer::models::spark2_5 {

Startup::Startup(const artifact::Reader& definition_source, const artifact::Reader& rank_source,
                 const EngineOptions& options, Device device)
    : reader_(rank_source), definition_(read_definition(definition_source, load_options(options))) {
    if (options.deployment && options.deployment->ranks.size() != 1)
        throw std::invalid_argument("Spark2_5 does not support distributed generation");
    profile = describe_model(definition_);
    std::vector<artifact::PlacementRequest> requests;
    for (const auto& parameter : definition_->parameters())
        for (const auto& part : parameter.binding.parts)
            requests.push_back({part.object, device, artifact::HostAccess::CpuOnly});
    const auto plan = artifact::plan_materialization(reader_, requests);
    placements_.push_back(
        {WeightPlacement::Resident, plan.device_capacity_bytes, plan.host_capacity_bytes});
}

std::optional<Startup::ExecutionPlan>
Startup::plan(DeviceContext& device, const EngineOptions& options, WeightPlacement placement,
              std::size_t available, std::size_t available_host_bytes) const {
    if (placement != WeightPlacement::Resident) return {};
    auto planned = plan_execution(profile, options, device, available, available_host_bytes);
    if (!planned) return {};
    const auto capacity = planned->capacity;
    const auto chunk    = planned->shape.prefill_chunk;
    return ExecutionPlan{std::move(*planned), capacity, placement, chunk, {}};
}

std::unique_ptr<ModelShard> Startup::materialize(DeviceContext& device, WeightPlacement,
                                                 const StartupObserver& observer) {
    return materialize_resident(reader_, definition_, device, &observer);
}

Frontend Startup::frontend(const EngineOptions& options) const {
    return create_frontend(*definition_, options);
}

GraphPreparationPlan Startup::graph_plan(const EngineOptions& options,
                                         const ExecutionPreparation&) const {
    return describe_graph_preparation(options);
}

Startup::Calibration Startup::calibrate(const RankParameters& parameters, DeviceContext& device,
                                        const EngineOptions& options,
                                        const ExecutionPreparation&) const {
    return calibrate_graphs(parameters, device, options);
}

std::unique_ptr<RankProgram> create_rank_program(const RankParameters& parameters,
                                                 spark2_5::ExecutionPlan&& plan,
                                                 DeviceContext& device,
                                                 const StartupObserver& observer,
                                                 GraphPreparationMode mode) {
    auto program = std::make_unique<RankProgram>(parameters, std::move(plan), device, observer);
    program->prepare_execution(mode);
    return program;
}

std::string prefill_signature(const ModelDefinition& model, std::uint32_t chunk) {
    using Json    = nlohmann::json;
    const auto& c = model.config();
    Json facts{{"implementation", "spark2-5-eager-1"},
               {"chunk", chunk},
               {"hidden", c.hidden},
               {"vocabulary", c.vocabulary},
               {"attention", {c.heads, c.kv_heads, c.head_dim, c.sliding_window}},
               {"intermediate", c.intermediate}};
    auto& layers = facts["layers"] = Json::array();
    for (const auto& layer : c.layers) layers.push_back(static_cast<int>(layer.kind));
    auto& weights = facts["weights"] = Json::array();
    for (const auto& w : model.weight_data()) {
        Json item{{"role", w.name},
                  {"shape", w.view.shape},
                  {"parts", Json::array()},
                  {"uses", Json::array()}};
        for (const auto& part : w.view.parts)
            item["parts"].push_back(
                {part.parent->geometry.format, part.parent->geometry.layout, part.begin, part.end});
        for (const auto& use : w.uses) item["uses"].push_back({use.input, use.policy});
        weights.push_back(std::move(item));
    }
    return models::frontend::sha256_hex(models::frontend::sha256(facts.dump()));
}

std::string Startup::prefill_signature(const ExecutionPreparation&, std::uint32_t chunk) const {
    return spark2_5::prefill_signature(*definition_, chunk);
}

} // namespace ninfer::models::spark2_5
