#!/usr/bin/env python3
"""Generation sanity for the b200 Qwen3.5-122B NVFP4 adaptation.

Samples prompts from the serving dataset with a fixed seed, requests them from a
running ninfer-serve with the sampling used by the locked serving benchmark, and
applies rule checks against the sglang outputs recorded in the t1 benchmark.json.
The sglang side is a recorded reference, not a live server, and sampling is
stochastic, so the checks are rule based (no token-level comparison). Entries that
fail any rule are marked ``needs_review`` for manual inspection.

The tool only reads the dataset and the recorded results.
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SAMPLING: dict[str, Any] = {
    "temperature": 0.9,
    "top_p": 0.9,
    "max_tokens": 15,
    "n": 4,
    "enable_thinking": False,
}
REQUEST_TIMEOUT_S = 600.0
CHECK_NAMES = (
    "nonempty",
    "no_replacement_char",
    "no_repetition_collapse",
    "script_consistent",
)

# Special markers the model may print literally, e.g. "[EOS]".
_MARKER = re.compile(r"\[[A-Z_]{2,}\]")
_CHAR_RUN = re.compile(r"(.)\1{5,}")
_UNIT_REPEAT = re.compile(r"(.{2,8}?)\1{3,}")
# Scripts that are routine in customer-service Chinese/English text.
_COMMON_SCRIPTS = frozenset({"HAN", "LATIN"})
_FOREIGN_SCRIPT_SHARE = 0.10


class SanityError(RuntimeError):
    """A request could not be completed against ninfer-serve."""


def load_examples(path: Path) -> list[list[dict[str, Any]]]:
    """Return message lists in the order the benchmark client numbers them (qid = index + 1)."""
    examples: list[list[dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                messages = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(messages, list):
                examples.append(messages)
    return examples


def load_sglang_texts(path: Path) -> dict[int, list[str]]:
    """Map qid to the recorded sglang completions of successful requests."""
    document = json.loads(path.read_text(encoding="utf-8"))
    texts: dict[int, list[str]] = {}
    for result in document["results"]:
        generated = result.get("generated_text")
        if not result.get("success") or generated is None:
            continue
        texts[int(result["qid"])] = [generated] if isinstance(generated, str) else list(generated)
    return texts


def select_qids(candidates: set[int], count: int, seed: int) -> list[int]:
    """Sample count qids from candidates with the seed and return them sorted."""
    if count > len(candidates):
        raise SanityError(f"requested {count} prompts but only {len(candidates)} have a recorded sglang output")
    return sorted(random.Random(seed).sample(sorted(candidates), count))


def request_completions(base_url: str, model: str, messages: list[dict[str, Any]]) -> list[str]:
    """Stream one chat completion request and return the n choice texts in index order."""
    parts = urlsplit(base_url)
    if parts.scheme != "http" or not parts.hostname:
        raise SanityError(f"--ninfer-url must be an http URL, got {base_url!r}")
    path = parts.path.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": SAMPLING["max_tokens"],
        "temperature": SAMPLING["temperature"],
        "top_p": SAMPLING["top_p"],
        "n": SAMPLING["n"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": SAMPLING["enable_thinking"]},
    }
    connection = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=REQUEST_TIMEOUT_S)
    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise SanityError(f"HTTP {response.status} {response.reason}: {response.read(500)!r}")
        texts = [""] * SAMPLING["n"]
        done = False
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue  # blank separators, ": keep-alive" comments
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise SanityError(f"error event in stream: {event['error']}")
            for choice in event.get("choices") or []:
                index = choice.get("index", 0)
                if not 0 <= index < len(texts):
                    raise SanityError(f"choice index {index} outside n={len(texts)}")
                texts[index] += (choice.get("delta") or {}).get("content") or ""
        if not done:
            raise SanityError("stream ended without [DONE]")
        return texts
    except (OSError, http.client.HTTPException, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SanityError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        connection.close()


def _visible(text: str) -> str:
    """Text without literal markers and whitespace, for repetition and script analysis."""
    return "".join(_MARKER.sub("", text).split())


def check_nonempty(texts: list[str]) -> bool:
    """Require all n sampled texts to have nonempty visible content."""
    return len(texts) == SAMPLING["n"] and all(_visible(text) for text in texts)


def check_no_replacement_char(texts: list[str]) -> bool:
    """Report whether no text contains the Unicode replacement character."""
    return not any("\ufffd" in text for text in texts)


def _collapsed(text: str) -> bool:
    """Detect repetition collapse: long character runs, repeated units, or tiny alphabets."""
    visible = _visible(text)
    for match in _CHAR_RUN.finditer(visible):
        if unicodedata.category(match.group(1))[0] in "LN":
            return True  # a letter/digit repeated 6+ times in a row
    if any(len(match.group(0)) >= 8 for match in _UNIT_REPEAT.finditer(visible)):
        return True  # a 2-8 character unit repeated 4+ times
    return len(visible) >= 12 and len(set(visible)) <= 2


def check_no_repetition_collapse(texts: list[str]) -> bool:
    """Report whether no text shows repetition collapse."""
    return not any(_collapsed(text) for text in texts)


def script_counts(texts: list[str]) -> Counter[str]:
    """Letter counts per Unicode script (CJK ideographs reported as HAN)."""
    counts: Counter[str] = Counter()
    for text in texts:
        for char in unicodedata.normalize("NFKC", _MARKER.sub("", text)):
            if char.isalpha():
                script = unicodedata.name(char, "UNKNOWN").split()[0]
                counts["HAN" if script == "CJK" else script] += 1
    return counts


def check_script_consistent(ninfer_texts: list[str], sglang_texts: list[str]) -> bool:
    """Check the dominant script matches the reference and no foreign script dominates."""
    ours, reference = script_counts(ninfer_texts), script_counts(sglang_texts)
    if not ours and not reference:
        return True
    if not ours or not reference:
        return False
    if ours.most_common(1)[0][0] != reference.most_common(1)[0][0]:
        return False
    total = sum(ours.values())
    return not any(
        count / total >= _FOREIGN_SCRIPT_SHARE and script not in _COMMON_SCRIPTS and script not in reference
        for script, count in ours.items()
    )


def evaluate(qid: int, ninfer_texts: list[str], sglang_texts: list[str]) -> dict[str, Any]:
    """Run all rule checks for one qid and return its verdict entry."""
    checks = {
        "nonempty": check_nonempty(ninfer_texts),
        "no_replacement_char": check_no_replacement_char(ninfer_texts),
        "no_repetition_collapse": check_no_repetition_collapse(ninfer_texts),
        "script_consistent": check_script_consistent(ninfer_texts, sglang_texts),
    }
    return {
        "qid": qid,
        "ninfer_text": ninfer_texts,
        "sglang_text": sglang_texts,
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "needs_review",
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ninfer-url", required=True, help="OpenAI base URL, e.g. http://127.0.0.1:30100/v1")
    parser.add_argument("--model", required=True, help="model id reported by /v1/models")
    parser.add_argument("--sglang-results", type=Path, required=True, help="recorded sglang benchmark.json")
    parser.add_argument("--data", type=Path, required=True, help="serving dataset jsonl (one message list per line)")
    parser.add_argument("--count", type=int, default=20, help="prompts to sample (default 20)")
    parser.add_argument("--seed", type=int, default=20261116, help="prompt sampling seed")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    return parser


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """Run the sanity sweep, returning the result document and request errors."""
    examples = load_examples(args.data)
    sglang = load_sglang_texts(args.sglang_results)
    qids = select_qids({qid for qid in sglang if 1 <= qid <= len(examples)}, args.count, args.seed)
    prompts: list[dict[str, Any]] = []
    errors: list[str] = []
    for qid in qids:
        try:
            ninfer_texts = request_completions(args.ninfer_url, args.model, examples[qid - 1])
        except SanityError as exc:
            errors.append(f"qid {qid}: {exc}")
            ninfer_texts = []
        entry = evaluate(qid, ninfer_texts, sglang[qid])
        prompts.append(entry)
        print(f"[b200_sanity] qid={qid} {entry['verdict']}", file=sys.stderr)
    passed = sum(1 for entry in prompts if entry["verdict"] == "pass")
    document = {
        "sampling": dict(SAMPLING),
        "sglang_reference": "recorded_t1_benchmark",
        "prompts": prompts,
        "summary": {"count": len(prompts), "pass": passed, "needs_review": len(prompts) - passed},
    }
    return document, errors


def main(argv: list[str] | None = None) -> int:
    """Run the sweep, write the JSON report, and print the summary."""
    args = build_parser().parse_args(argv)
    try:
        document, errors = run(args)
    except (SanityError, OSError, KeyError, ValueError) as exc:
        print(f"[b200_sanity] error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = document["summary"]
    print(
        f"[b200_sanity] count={summary['count']} pass={summary['pass']} "
        f"needs_review={summary['needs_review']} -> {args.out}"
    )
    for error in errors:
        print(f"[b200_sanity] request failed: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
