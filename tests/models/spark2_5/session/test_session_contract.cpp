// T05 contract: the spark2_5 Session never forks sample trees. The public declaration is the
// authority the engine reads before offering a sample fork; the definition keeps it at zero.
// No GPU or artifact is needed.
#include "models/spark2_5/session/session.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

namespace {

namespace spark = ninfer::models::spark2_5;

static_assert(std::is_same_v<decltype(&spark::Session::sample_fork_limit),
                             std::uint32_t (spark::Session::*)() const noexcept>);

std::string slurp(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

int failures = 0;

void expect(bool ok, std::string_view message) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

} // namespace

int main() {
    const std::filesystem::path root = std::filesystem::path(__FILE__).parent_path()
                                           .parent_path()
                                           .parent_path()
                                           .parent_path()
                                           .parent_path();
    const auto definition = slurp(root / "src/models/spark2_5/session/execution.cpp");
    expect(definition.find("sample_fork_limit() const noexcept { return 0; }") !=
               std::string::npos,
           "spark2_5 sample_fork_limit keeps its never-forks definition");
    if (failures == 0) std::cout << "ok\n";
    return failures == 0 ? 0 : 1;
}
