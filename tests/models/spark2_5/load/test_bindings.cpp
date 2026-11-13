#include "models/spark2_5/load.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace ninfer;
namespace spark = models::spark2_5;

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

artifact::ParameterReference reference(std::string name,
                                       std::initializer_list<artifact::Part> parts) {
    artifact::ParameterReference out;
    out.name             = std::move(name);
    out.shape            = {8, 4};
    out.binding.parts    = parts;
    out.binding.elements = 32;
    return out;
}

bool shares(const artifact::ParameterReference& head,
            const artifact::ParameterReference& embedding) {
    try {
        spark::loading::require_shared(head, embedding);
        return true;
    } catch (const artifact::ArtifactError& error) {
        require(std::string(error.what()).find(head.name) != std::string::npos,
                std::string("error does not name the head: ") + error.what());
        return false;
    }
}

void tied_head_must_share_embedding_storage() {
    const auto embedding = reference("text/token_embedding", {{{3}, 0, 32}});
    require(shares(reference("text/output_head", {{{3}, 0, 32}}), embedding),
            "identical payload ranges were rejected");
    require(!shares(reference("text/output_head", {{{4}, 0, 32}}), embedding),
            "a different object was accepted");
    require(!shares(reference("text/output_head", {{{3}, 0, 16}}), embedding),
            "a shorter range was accepted");
    require(!shares(reference("text/output_head", {{{3}, 4, 32}}), embedding),
            "a shifted range was accepted");
    require(!shares(reference("text/output_head", {{{3}, 0, 16}, {{3}, 16, 32}}), embedding),
            "a split binding was accepted as shared");
    require(!shares(reference("text/output_head", {}), embedding), "an empty binding was accepted");
}
} // namespace

int main() {
    try {
        tied_head_must_share_embedding_storage();
        std::cout << "PASS Spark2_5 tied head sharing\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
