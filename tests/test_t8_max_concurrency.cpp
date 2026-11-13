// T8 / F1: the concurrency range accepted by ninfer-serve is [1,16].
//   16 is accepted and reaches EngineOptions; 17 is rejected with an error naming the range.
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

void test_sixteen_accepted() {
    const auto options =
        parse({"ninfer-serve", "model.ninfer", "--max-concurrency", "16"});
    require(options.launch.engine.max_concurrency == 16, "--max-concurrency 16 was not accepted");
    require(ninfer::kMaximumConcurrency == 16, "public concurrency limit is not 16");
}

void test_seventeen_rejected_with_new_range() {
    try {
        parse({"ninfer-serve", "model.ninfer", "--max-concurrency", "17"});
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        require(message.find("[1,16]") != std::string::npos,
                "the error for 17 does not state the range [1,16]");
        return;
    }
    throw std::runtime_error("--max-concurrency 17 was accepted");
}

} // namespace

int main() {
    try {
        test_sixteen_accepted();
        test_seventeen_rejected_with_new_range();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
