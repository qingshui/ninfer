"""Regenerate chat-golden.json with HF: --resources DIR --output GOLDEN.json.

DIR holds the Spark2_5 checkpoint's tokenizer.json and tokenizer_config.json. The chat template is
the checked-in fixture, which must match the checkpoint's chat_template.jinja byte for byte.
Each case lists the engine input (`input`) and the HF call that is equivalent to it
(`reference`). `expected` holds HF's rendered text and token ids.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}
CLOCK = {
    "type": "function",
    "function": {"name": "clock", "parameters": {"type": "object", "properties": {}}},
}


def user(text):
    return {"role": "user", "content": text}


def assistant(text, reasoning=None, calls=None):
    message = {"role": "assistant", "content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if calls is not None:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
            for i, (n, a) in enumerate(calls)
        ]
    return message


def tool(text, call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


CALLS = [
    ("get_weather", {"city": "Zürich", "days": 3, "units": ["c", "k"], "extra": {"a": 1.5}}),
    ("clock", {}),
]

# name, messages, options (enable_thinking, reasoning_effort, tools) and the engine's equivalents.
CASES = [
    ("thinking-default", [user("hello")], {}),
    ("thinking-off", [user("hello")], {"enable_thinking": False}),
    ("thinking-on-explicit", [user("hello")], {"enable_thinking": True}),
    ("effort-low", [user("hello")], {"reasoning_effort": "low"}),
    ("effort-high", [user("hello")], {"reasoning_effort": "high"}),
    (
        "system-thinking-off",
        [{"role": "system", "content": "Be brief."}, user("hello")],
        {"enable_thinking": False},
    ),
    (
        "system-thinking-on",
        [{"role": "system", "content": "Be brief."}, user("hello")],
        {},
    ),
    (
        "history-with-reasoning",
        [user("a"), assistant("b", "because"), user("c")],
        {},
    ),
    (
        "history-without-reasoning-off",
        [user("a"), assistant("b"), user("c")],
        {"enable_thinking": False},
    ),
    (
        "later-system-message",
        [
            {"role": "system", "content": "first"},
            user("a"),
            {"role": "system", "content": "second"},
            user("b"),
        ],
        {"enable_thinking": False},
    ),
    (
        "tools-off",
        [{"role": "system", "content": "Use tools."}, user("weather in Zürich?")],
        {"enable_thinking": False, "tools": [WEATHER, CLOCK]},
    ),
    ("tools-no-system", [user("time?")], {"tools": [CLOCK]}),
    (
        "tool-round-off",
        [
            user("weather and time"),
            assistant("Checking.", None, CALLS),
            tool("sunny", "call_0"),
            tool("noon", "call_1"),
            user("thanks"),
        ],
        {"enable_thinking": False, "tools": [WEATHER, CLOCK]},
    ),
    (
        "tool-round-reasoning",
        [
            user("weather"),
            assistant("", "need the tool", [CALLS[0]]),
            tool("sunny", "call_0"),
        ],
        {"tools": [WEATHER]},
    ),
    (
        "tool-argument-string-values",
        [
            user("x"),
            assistant("", None, [("clock", {"s": "plain", "n": None, "b": True, "f": 2.0})]),
            tool("ok", "call_0"),
        ],
        {"enable_thinking": False},
    ),
    (
        "text-blocks",
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "cd"}],
            }
        ],
        {"enable_thinking": False},
    ),
    ("unicode", [user("你好 \U0001F600 é")], {"enable_thinking": False}),
    (
        "empty-assistant",
        [user("a"), assistant(""), user("b")],
        {"enable_thinking": False},
    ),
    (
        "developer-is-system",
        [{"role": "system", "content": "dev rules"}, user("hello")],
        {"enable_thinking": False},
        {"developer": True},
    ),
    (
        "continuation",
        [user("a"), assistant("The answer is", "think first")],
        {},
        {"continuation": True},
    ),
    (
        "continuation-no-reasoning",
        [user("a"), assistant("The answer is")],
        {"enable_thinking": False},
        {"continuation": True},
    ),
]


def engine_input(messages, options, extra):
    """The engine's equivalent of an HF call: role names and argument encoding differ."""
    result = []
    for index, message in enumerate(messages):
        item = copy.deepcopy(message)
        if extra.get("developer") and index == 0:
            item["role"] = "developer"
        result.append(item)
    keep = {k: v for k, v in options.items() if k != "tools"}
    keep["tools"] = options.get("tools", [])
    if extra.get("continuation"):
        keep["continuation"] = True
    return dict(messages=result, options=keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = Path(__file__).parent / "fixtures"
    source = (fixtures / "chat_template.jinja").read_text()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.resources, local_files_only=True)
    tokenizer.chat_template = source
    cases = []
    for name, messages, options, *rest in CASES:
        extra = rest[0] if rest else {}
        continuation = extra.get("continuation", False)
        kwargs = {k: v for k, v in options.items() if k != "tools"}
        text = tokenizer.apply_chat_template(
            messages,
            tools=options.get("tools"),
            tokenize=False,
            add_generation_prompt=not continuation,
            **kwargs,
        )
        if continuation:
            # The final assistant turn stays open: HF has no continue_final_message here.
            end = "<｜end▁of▁sentence｜>"
            assert text.endswith(end)
            text = text[: -len(end)]
        cases.append(
            dict(
                name=name,
                input=engine_input(messages, options, extra),
                reference=dict(messages=messages, options=options),
                expected=dict(text=text, ids=tokenizer.encode(text, add_special_tokens=False)),
            )
        )
    data = dict(
        source=dict(sha256=hashlib.sha256(source.encode()).hexdigest()),
        resources={
            name: hashlib.sha256((args.resources / name).read_bytes()).hexdigest()
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        reference={
            name: importlib.metadata.version(name)
            for name in ("transformers", "jinja2", "tokenizers")
        },
    )
    header = json.dumps(data, indent=2)
    rows = [json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cases]
    args.output.write_text(
        header[:-2] + ',\n  "cases": [\n    ' + ",\n    ".join(rows) + "\n  ]\n}\n"
    )
    print(f"Wrote {len(cases)} HF cases to {args.output}")


if __name__ == "__main__":
    main()
