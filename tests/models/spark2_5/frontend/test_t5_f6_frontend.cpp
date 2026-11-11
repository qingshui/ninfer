// T5 / F6 acceptance for the Spark2_5 frontend, written against the Spec only:
//  1. enable_thinking=false prompts tokenize to the text HF apply_chat_template renders
//     (chat-golden.json "text"; ids too when NINFER_SPARK2_5_RESOURCES names the checkpoint).
//  2. Streamed output never retains template tokens, at any chunking, including tool calls.
//  3. Protocol behaviour is covered by ninfer_spark2_5_protocol_test (registered separately).
#include "spark_fixture.h"

#include <cstdlib>
#include <iostream>

namespace {
using namespace ninfer;
using namespace spark_fixture;

const std::string kEnd = "<｜end▁of▁sentence｜>";

bool thinking_off(const Json& item) {
    const auto& options = item.at("input").at("options");
    return options.contains("enable_thinking") && !options.at("enable_thinking").get<bool>() &&
           !options.value("continuation", false);
}

// Token sequence of the HF-rendered text equals the prepared prompt, for thinking disabled.
void golden_tokens() {
    const auto data = Json::parse(read(fixtures / "chat-golden.json"));
    const auto res  = resources(fixtures / "tokenizer.json");
    const auto f    = frontend(*res);
    int checked     = 0;
    for (const auto& item : data.at("cases")) {
        if (!thinking_off(item)) continue;
        const auto name     = item.at("name").get<std::string>();
        const auto input    = prompt(item.at("input"));
        const auto prepared = f.prepare(input);
        const auto expected = f.tokenize_text(item.at("expected").at("text").get<std::string>());
        require(spark::PreparedPromptAccess::view(prepared).token_ids == expected,
                name + ": prepared tokens differ from the HF-rendered text");
        require(f.count_tokens(input) == expected.size(), name + ": counted tokens differ");
        require(!prepared.summary().starts_in_reasoning, name + ": thinking-off starts in reasoning");
        ++checked;
    }
    require(checked >= 5, "too few thinking-off golden cases");

    // Against the real checkpoint tokenizer: ids must equal HF's ids exactly.
    const char* root = std::getenv("NINFER_SPARK2_5_RESOURCES");
    if (!root || !*root) return;
    const std::filesystem::path dir(root);
    const auto real =
        resources(dir / "tokenizer.json", read(dir / "tokenizer_config.json"),
                  read(dir / "generation_config.json"));
    const auto rf = frontend(*real);
    for (const auto& item : data.at("cases")) {
        if (!thinking_off(item)) continue;
        const auto prepared = rf.prepare(prompt(item.at("input")));
        require(spark::PreparedPromptAccess::view(prepared).token_ids ==
                    item.at("expected").at("ids").get<std::vector<TokenId>>(),
                item.at("name").get<std::string>() + ": ids differ from HF");
    }
}

void require_clean(const Output& out, const std::string& label) {
    for (const auto* token : {"<｜end▁of▁sentence｜>", "<｜start▁of▁sentence｜>", "<|Bot|>",
                              "<|User|>", "<|System|>", "<|Tool|>", "<think>", "</think>"})
        require(out.reasoning.find(token) == std::string::npos &&
                    out.content.find(token) == std::string::npos,
                label + ": streamed output retained template token " + token);
}

// Streams thinking-off and thinking-on generations token by token and in one piece.
void streaming_clean() {
    const auto res = resources(fixtures / "tokenizer.json");
    const auto f   = frontend(*res);
    const auto& tk = *res->parsed.text.tokenizer;
    const auto tool = declaration(tool_schema(Json{{"city", {{"type", "string"}}}}), false, "weather");

    for (const bool thinking : {false, true}) {
        auto in                   = input({tool});
        in.options.enable_thinking = thinking;
        const auto prepared       = f.prepare(in);
        const std::string wire =
            std::string(thinking ? "plan</think>" : "") +
            "hello<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>" +
            kEnd;
        const auto ids = tk.encode(wire);
        for (auto width : {ids.size(), std::size_t(1)}) {
            auto session = f.make_output_session(
                prepared, {}, {}, {}, {}, {.constraints = ToolConstraintMode::Automatic});
            Output out;
            for (std::size_t i = 0; i < ids.size(); i += width)
                commit(session, std::span(ids).subspan(i, std::min(width, ids.size() - i)), out);
            const auto label = std::string(thinking ? "thinking" : "no-thinking") + " width " +
                               std::to_string(width);
            require_clean(out, label);
            require(out.content == "hello" && out.reasoning == (thinking ? "plan" : ""),
                    label + ": channels differ");
            require(out.calls.size() == 1 && out.calls[0].name == "weather" &&
                        out.calls[0].status == ToolCallStatus::Complete &&
                        Json::parse(out.calls[0].arguments_json) == Json{{"city", "Paris"}},
                    label + ": tool call differs");
        }
    }
}
} // namespace

int main() {
    try {
        golden_tokens();
        streaming_clean();
        std::cout << "Spark2_5 T5 F6 tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
