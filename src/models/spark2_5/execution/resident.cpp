#include "models/spark2_5/execution/resident.h"

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "models/common/bindings.h"

namespace ninfer::models::spark2_5 {

ModelShard::ModelShard(std::shared_ptr<const ModelDefinition> definition,
                       artifact::MaterializedArtifact backing, std::vector<BoundWeight> bound)
    : definition_(std::move(definition)), backing_(std::move(backing)), bound_(std::move(bound)) {}

ModelShard::~ModelShard() = default;

std::unique_ptr<ModelShard> materialize_resident(const artifact::Reader& reader,
                                                 std::shared_ptr<const ModelDefinition> definition,
                                                 DeviceContext& device,
                                                 const StartupObserver* observer) {
    if (!definition || definition->artifact_id() != reader.artifact_id())
        throw artifact::ArtifactError("model definition belongs to another artifact");
    if (device.device().type() != DeviceType::CUDA)
        throw artifact::ArtifactError("Spark2_5 weights are placed on a GPU");
    std::vector<artifact::PlacementRequest> requests;
    for (const auto& parameter : definition->parameters())
        for (const auto& part : parameter.binding.parts)
            requests.push_back({part.object, device.device(), artifact::HostAccess::CpuOnly});
    auto backing = artifact::materialize(reader, artifact::plan_materialization(reader, requests),
                                         device.transfer_view(), observer);
    std::vector<PendingWeight> pending;
    pending.reserve(definition->weight_data().size());
    for (std::size_t i = 0; i < definition->weight_data().size(); ++i) {
        const auto& description = definition->weight_data()[i];
        pending.push_back(
            {definition->parameter({i}), description.uses, description.source_objects});
    }
    const std::vector<Device> devices(pending.size(), device.device());
    auto bound = resolve_weights(std::move(pending), backing, devices);
    return std::unique_ptr<ModelShard>(
        new ModelShard(std::move(definition), std::move(backing), std::move(bound)));
}

} // namespace ninfer::models::spark2_5
