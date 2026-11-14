// T8 / F1: the concurrency range accepted by ninfer-serve is 0..1024, where 0 selects
// automatic lane fitting and is the serving default.
//   0 and 1024 are accepted and reach EngineOptions; 1025 is rejected with an error naming
//   the range; the serve default constant is Auto and explicit values override it.
#include "ninfer/types.h"
#include "product/config/options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::product::ServeOptions;

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return ninfer::product::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_default_is_auto() {
    const auto options = parse({"ninfer-serve", "model.ninfer"});
    require(options.launch.engine.max_concurrency == 0,
            "serving default concurrency is not automatic lane fitting");
    require(ninfer::product::kDefaultServeMaxConcurrency == 0,
            "serving default concurrency constant is not zero");
    require(parse({"ninfer-serve", "model.ninfer", "--max-concurrency", "0"})
                    .launch.engine.max_concurrency == 0,
            "--max-concurrency 0 was not accepted as automatic lane fitting");
}

void test_explicit_values_reach_engine_options() {
    require(parse({"ninfer-serve", "model.ninfer", "--max-concurrency", "16"})
                    .launch.engine.max_concurrency == 16,
            "--max-concurrency 16 was not accepted");
    require(parse({"ninfer-serve", "model.ninfer", "--max-concurrency", "1024"})
                    .launch.engine.max_concurrency == 1024,
            "--max-concurrency 1024 was not accepted");
    require(ninfer::kMaximumConcurrency == 1024, "public concurrency ceiling is not 1024");
}

void test_above_ceiling_rejected_with_new_range() {
    try {
        parse({"ninfer-serve", "model.ninfer", "--max-concurrency", "1025"});
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        require(message.find("[0,1024]") != std::string::npos,
                "the error for 1025 does not state the range [0,1024]");
        return;
    }
    throw std::runtime_error("--max-concurrency 1025 was accepted");
}

} // namespace

int main() {
    try {
        test_default_is_auto();
        test_explicit_values_reach_engine_options();
        test_above_ceiling_rejected_with_new_range();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
