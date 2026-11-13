"""Write the small byte-level tokenizer used by the Spark2_5 frontend tests.

The vocabulary reuses the GLM test tokenizer's byte-level base and declares the added tokens that
the Spark2_5 chat template and tool framing need. Ids differ from the checkpoint: golden token ids
are only compared against the real checkpoint tokenizer.
"""

import argparse
import json
from pathlib import Path

ADDED = [
    ("<｜start▁of▁sentence｜>", True),
    ("<｜end▁of▁sentence｜>", True),
    ("<｜▁pad▁｜>", True),
    ("<think>", False),
    ("</think>", False),
    ("<|System|>", False),
    ("<|User|>", False),
    ("<|Tool|>", False),
    ("<|Bot|>", False),
    ("<tool_call>", False),
    ("</tool_call>", False),
    ("<tool_response>", False),
    ("</tool_response>", False),
    ("<arg_key>", False),
    ("</arg_key>", False),
    ("<arg_value>", False),
    ("</arg_value>", False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="GLM prompt-tokenizer.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.base.read_text())
    first = len(data["model"]["vocab"])
    data["added_tokens"] = [
        dict(
            id=first + i,
            content=content,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=special,
        )
        for i, (content, special) in enumerate(ADDED)
    ]
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
