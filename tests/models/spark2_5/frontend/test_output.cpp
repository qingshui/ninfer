// Output side of the Spark2_5 frontend: reasoning and content channels, tool calls, thinking
// controls and the absence of template tokens in published text.
#include "spark_fixture.h"

#include <iostream>

namespace {
using namespace ninfer;
using namespace spark_fixture;
using Session = models::frontend::OutputSession;

const std::string kEnd = "<｜end▁of▁sentence｜>";

std::vector<std::string> declarations() {
    const auto properties = Json{{"s", {{"type", "string"}}},  {"n", {{"type", "integer"}}},
                                 {"b", {{"type", "boolean"}}}, {"z", {{"type", "null"}}},
                                 {"o", {{"type", "object"}}},  {"a", {{"type", "array"}}}};
    return {declaration(tool_schema(properties)),
            declaration(tool_schema(Json::object()), false, "noop")};
}

struct Fixture {
    std::unique_ptr<Resources> resource = resources(fixtures / "tokenizer.json");
    spark::Frontend frontend            = spark_fixture::frontend(*resource);
    const text::Tokenizer& tokenizer    = *resource->parsed.text.tokenizer;

    std::vector<TokenId> encode(std::string_view text) const { return tokenizer.encode(text); }
};

// Template tokens are structure, never output text.
void require_clean(const Output& out, const std::string& label) {
    for (const auto* token : {"<｜end▁of▁sentence｜>", "<｜start▁of▁sentence｜>", "<|Bot|>",
                              "<|User|>", "<|System|>", "<|Tool|>", "<think>", "</think>"}) {
        require(out.reasoning.find(token) == std::string::npos &&
                    out.content.find(token) == std::string::npos,
                label + ": output retained template token " + token);
    }
}

void channels(const Fixture& t) {
    const auto prompt = t.frontend.prepare(input());
    require(prompt.summary().starts_in_reasoning, "default prompt must start in reasoning");
    const auto ids = t.encode("some reasoning</think>the answer" + kEnd);
    for (auto width : {ids.size(), std::size_t(1), std::size_t(2), std::size_t(5)}) {
        auto session = t.frontend.make_output_session(prompt, {}, {}, {}, {}, {});
        Output out;
        for (std::size_t i = 0; i < ids.size(); i += width)
            commit(session, std::span(ids).subspan(i, std::min(width, ids.size() - i)), out);
        require(out.reasoning == "some reasoning" && out.content == "the answer" &&
                    out.calls.empty(),
                "reasoning and content were not separated at width " + std::to_string(width));
        require_clean(out, "channels");
        require(session.reasoning_tokens() > 0, "reasoning tokens were not counted");
    }

    // A prompt whose thinking is disabled ends in a closed block: generation is all content.
    auto off                    = input();
    off.options.enable_thinking = false;
    const auto closed           = t.frontend.prepare(off);
    require(!closed.summary().starts_in_reasoning, "disabled thinking starts in reasoning");
    auto session = t.frontend.make_output_session(closed, {}, {}, {}, {}, {});
    Output out;
    commit(session, t.encode("direct answer" + kEnd), out);
    require(out.reasoning.empty() && out.content == "direct answer",
            "disabled thinking leaked reasoning");
    require_clean(out, "closed");

    // Tool framing stays literal text when no tool is declared.
    auto plain = t.frontend.make_output_session(prompt, {}, {}, {}, {}, {});
    Output literal;
    commit(plain, t.encode("r</think><tool_call>x</tool_call>" + kEnd), literal);
    require(literal.calls.empty() && literal.content == "<tool_call>x</tool_call>",
            "undeclared tool framing was interpreted");
}

void tools(const Fixture& t) {
    const auto prompt = t.frontend.prepare(input(declarations()));
    const std::string wire =
        "think</think>before<tool_call>call<arg_key>s</arg_key><arg_value>你\\\" \n</arg_value>"
        "<arg_key>n</arg_key><arg_value>42</arg_value><arg_key>b</arg_key><arg_value>true"
        "</arg_value><arg_key>z</arg_key><arg_value>null</arg_value><arg_key>o</arg_key>"
        "<arg_value>{\"tag\":\"</arg_value>\"}</arg_value><arg_key>a</arg_key>"
        "<arg_value>[1, \"x\"]</arg_value></tool_call><tool_call>noop</tool_call>" +
        kEnd;
    const auto ids = t.encode(wire);
    const Json expected{{"s", "你\\\" \n"},
                        {"n", 42},
                        {"b", true},
                        {"z", nullptr},
                        {"o", {{"tag", "</arg_value>"}}},
                        {"a", Json::array({1, "x"})}};
    for (auto width : {ids.size(), std::size_t(1), std::size_t(3)}) {
        auto session = t.frontend.make_output_session(
            prompt, {}, {}, {}, {}, {.constraints = ToolConstraintMode::Automatic});
        Output out;
        for (std::size_t i = 0; i < ids.size(); i += width)
            commit(session, std::span(ids).subspan(i, std::min(width, ids.size() - i)), out);
        require(out.reasoning == "think" && out.content == "before" && out.calls.size() == 2,
                "tool turn channels differ at width " + std::to_string(width));
        require_clean(out, "tools");
        require(out.calls[0].name == "call" &&
                    Json::parse(out.calls[0].arguments_json) == expected &&
                    out.calls[1].name == "noop" && Json::parse(out.calls[1].arguments_json).empty(),
                "tool arguments did not follow the declared types");
        for (const auto& call : out.calls)
            require(call.status == ToolCallStatus::Complete, "completed call was not complete");
        require(out.calls == session.take_tool_calls(),
                "tool aggregate differs from published events");
    }

    // The calls come back through the template as the compact framing the model emitted.
    auto history                = input(declarations());
    auto assistant              = message(ChatRole::Assistant, "before");
    assistant.reasoning_content = "think";
    assistant.tool_calls        = {{"0", "call", R"({"s":"x","n":42,"o":{"k":[1,2]}})"}};
    history.messages.push_back(assistant);
    auto response         = message(ChatRole::Tool, "ok");
    response.tool_call_id = "0";
    history.messages.push_back(response);
    const auto text = t.tokenizer.decode(
        spark::PreparedPromptAccess::view(t.frontend.prepare(history)).token_ids);
    require(text.find("<｜start▁of▁sentence｜><|Bot|><think>think</think>before<tool_call>call"
                      "<arg_key>s</arg_key><arg_value>x</arg_value><arg_key>n</arg_key>"
                      "<arg_value>42</arg_value><arg_key>o</arg_key><arg_value>{\"k\": [1, 2]}"
                      "</arg_value></tool_call><｜end▁of▁sentence｜>"
                      "<｜start▁of▁sentence｜><|Tool|><tool_response>ok</tool_response>"
                      "<｜end▁of▁sentence｜>") != std::string::npos,
            "tool history did not render as the checkpoint's tool framing");
    require(text.ends_with("<｜start▁of▁sentence｜><|Bot|><think>"),
            "a tool response must start a new thinking turn");
}

void controls(const Fixture& t) {
    const auto prompt = t.frontend.prepare(input());
    const auto word   = t.encode("x");
    require(word.size() == 1, "fixture must contain x");

    // Thinking accounting does not depend on presentation or on the content after the close.
    for (const bool raw : {false, true}) {
        auto unlimited = t.frontend.make_output_session(prompt, {}, {.raw = raw}, {}, {}, {});
        (void)unlimited.preview_model(word, 100, FinishReason::OutputLimit);
        unlimited.discard_preview();
        require(unlimited.thinking_stats().model_thinking_tokens == 0,
                "discard advanced unbudgeted thinking");
        Output visible;
        commit(unlimited, t.encode("x</think>answer"), visible);
        const auto stats = unlimited.thinking_stats();
        require(stats.model_thinking_tokens == t.encode("x</think>").size() &&
                    !stats.configured_budget && !stats.applied && stats.injected_tokens == 0,
                "unbudgeted thinking depends on presentation or includes post-close content");
    }

    // An exhausted budget asks the target to close reasoning with the checkpoint's own token.
    auto session = t.frontend.make_output_session(prompt, {}, {}, {.budget = 2}, {}, {});
    Output output;
    commit(session, word, output);
    const auto decision = session.preview_model(word, 100, FinishReason::OutputLimit);
    require(decision.continuation == runtime::ContinuationAction::ApplyTargetControl,
            "budget did not request control");
    session.discard_preview();
    require(session.pending_control_tokens().empty() &&
                session.thinking_stats().model_thinking_tokens == 1,
            "discard advanced thinking budget");
    commit(session, word, output);
    const auto control = std::vector<TokenId>(session.pending_control_tokens().begin(),
                                              session.pending_control_tokens().end());
    require(t.tokenizer.decode(control) == "</think>", "control contains foreign framing");
    (void)session.preview_control(control, 100);
    session.discard_preview();
    require(!session.pending_control_tokens().empty(), "discard consumed pending control");
    const auto closed = session.preview_control(control, 100);
    require(closed.prefix_execution_split_after == control.size(), "control lost execution split");
    output.append(session.commit_preview());
    commit(session, t.encode("  answer" + kEnd), output);
    require(output.reasoning == "xx" && output.content == "  answer" &&
                session.thinking_stats().applied &&
                session.thinking_stats().injected_tokens == control.size(),
            "forced close changed content bytes or reasoning");

    // Default stop tokens end generation and nothing after them is accepted.
    require(!t.frontend.default_stop_policy().token_ids.empty(), "checkpoint has no stop token");
    for (const auto eos : t.frontend.default_stop_policy().token_ids) {
        auto terminal = t.frontend.make_output_session(prompt, {}, {}, {}, {}, {});
        auto ids      = t.encode("r</think>answer");
        ids.push_back(eos);
        ids.push_back(word.front());
        const auto stopped = terminal.preview_model(ids, 100, FinishReason::OutputLimit);
        require(stopped.finish_reason == FinishReason::StopToken &&
                    stopped.accepted_tokens == ids.size() - 1,
                "default EOS did not truncate the accepted prefix");
        (void)terminal.commit_preview();
    }

    // Raw output is the model's text with template tokens left in place.
    auto raw = t.frontend.make_output_session(prompt, {}, {.raw = true}, {}, {}, {});
    Output raw_out;
    commit(raw, t.encode("r</think>answer" + kEnd), raw_out);
    require(raw_out.reasoning.empty() && raw_out.content == "r</think>answer" + kEnd,
            "raw output was interpreted");

    // A stop string split across tokens is withheld, not leaked.
    auto stopped = t.frontend.make_output_session(
        prompt, {.strings = {{.text = "END", .channel = OutputChannel::Content}}}, {}, {}, {}, {});
    Output visible;
    commit(stopped, t.encode("r</think>aE"), visible);
    const auto stop = stopped.preview_model(t.encode("NDtail"), 100, FinishReason::OutputLimit);
    visible.append(stopped.commit_preview());
    require(stop.finish_reason == FinishReason::StopString && visible.content == "a" &&
                stopped.matched_stop_string() == "END",
            "split stop string leaked");

    // A tool opener while reasoning ends the reasoning without an explicit close.
    const auto tool_prompt = t.frontend.prepare(input(declarations()));
    auto implicit = t.frontend.make_output_session(tool_prompt, {}, {}, {.budget = 100}, {},
                                                   {.constraints = ToolConstraintMode::Automatic});
    Output tool_out;
    const auto implicit_ids = t.encode("reason<tool_call>noop</tool_call>" + kEnd);
    for (auto id : implicit_ids) commit(implicit, std::span(&id, 1), tool_out);
    require(tool_out.reasoning == "reason" && tool_out.calls.size() == 1 &&
                !implicit.thinking_stats().applied,
            "implicit tool opener did not end reasoning");
    require_clean(tool_out, "implicit");

    // Continuation resumes the final assistant text and never reopens reasoning.
    auto resumed_input = input();
    resumed_input.messages.push_back(message(ChatRole::Assistant, "prefix"));
    resumed_input.options.continuation = PromptContinuationMode::ContinueFinalAssistant;
    const auto resumed                 = t.frontend.prepare(resumed_input);
    auto continuation = t.frontend.make_output_session(resumed, {}, {}, {.budget = 1}, {}, {});
    Output continued;
    commit(continuation, t.encode(" suffix" + kEnd), continued);
    require(continued.content == " suffix" && !continuation.thinking_stats().configured_budget &&
                continuation.thinking_stats().model_thinking_tokens == 0,
            "continuation restarted reasoning or repeated its prefix");

    // The close marker split across ordinary tokens is still recognized.
    auto split = t.frontend.make_output_session(prompt, {}, {}, {}, {}, {});
    Output split_out;
    for (char c : std::string("reason</think> content"))
        commit(split, t.encode(std::string(1, c)), split_out);
    commit(split, t.encode(kEnd), split_out);
    require(split_out.reasoning == "reason" && split_out.content == " content",
            "reasoning delimiter split across ordinary tokens changed the output");
}
} // namespace

int main() {
    try {
        const Fixture t;
        channels(t);
        tools(t);
        controls(t);
        std::cout << "Spark2_5 output channels, tool calls and controls passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
