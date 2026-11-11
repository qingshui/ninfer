// Compares prepared token ids with HF's ids for the checkpoint tokenizer.
//
// NINFER_SPARK2_5_RESOURCES names a directory holding the checkpoint's tokenizer.json,
// tokenizer_config.json and generation_config.json. The test is skipped (77) when it is unset.
#include "spark_fixture.h"

#include <cstdlib>
#include <iostream>

int main() {
    using namespace spark_fixture;
    try {
        const char* root = std::getenv("NINFER_SPARK2_5_RESOURCES");
        if (!root || !*root) {
            std::cout << "NINFER_SPARK2_5_RESOURCES is not set\n";
            return 77;
        }
        const std::filesystem::path directory(root);
        const auto resource =
            resources(directory / "tokenizer.json", read(directory / "tokenizer_config.json"),
                      read(directory / "generation_config.json"));
        const auto data = Json::parse(read(fixtures / "chat-golden.json"));
        require(data.at("resources").at("tokenizer.json") != nullptr,
                "golden lacks tokenizer hash");
        const auto f = frontend(*resource);
        for (const auto& item : data.at("cases")) {
            const auto name     = item.at("name").get<std::string>();
            const auto expected = item.at("expected").at("ids").get<std::vector<TokenId>>();
            const auto input    = prompt(item.at("input"));
            const auto prepared = f.prepare(input);
            require(spark::PreparedPromptAccess::view(prepared).token_ids == expected,
                    name + ": prepared tokens differ from HF");
            require(f.count_tokens(input) == expected.size(), name + ": counted tokens differ");
            require(prepared.summary().prompt_tokens == expected.size(),
                    name + ": summary count differs");
        }
        // Thinking disabled ends with the closed reasoning token pair HF produces.
        const auto off = f.prepare(prompt(Json{{"messages", {{{"role", "user"}, {"content", "x"}}}},
                                               {"options", {{"enable_thinking", false}}}}));
        require(!off.summary().starts_in_reasoning, "disabled thinking starts in reasoning");
        const auto closing = f.tokenize_text("<|Bot|></think>");
        const auto& ids    = spark::PreparedPromptAccess::view(off).token_ids;
        require(closing.size() == 2 && ids.size() >= 2 &&
                    std::equal(closing.begin(), closing.end(), ids.end() - 2),
                "disabled thinking prompt does not end with <|Bot|></think>");
        std::cout << data.at("cases").size() << " Spark2_5 token golden cases passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
