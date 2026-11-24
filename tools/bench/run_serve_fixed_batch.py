#!/usr/bin/env python3
"""Fixed-batch decode measurement against an already-running ninfer-serve.

N streaming requests are prepared on their own HTTP connections and released
together; each generates `--max-tokens` output tokens, so the Engine decodes N
resident rows for the whole run. The per-stream TPOT is
(last non-empty output arrival - first) / (completion tokens - 1), and its
median across streams is the decode step at N rows. Usage counters provide the
token count; without them event arrivals are used and the row is marked as an
estimate. Only public HTTP is used; start the server separately.

Example:
    python3 -I tools/bench/run_serve_fixed_batch.py \
      --base-url http://127.0.0.1:8080 --rows 32,48,56,64,72,96 \
      --max-tokens 1200 --output .agent-tmp/fixed_batch/run.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ninfer_serve.client import NInferServeClient, ProtocolRequest

# Long-form prompts keep every stream generating for the full budget; they are
# assigned round-robin so concurrent rows do not share one prefix.
PROMPTS = (
    "请写一篇关于人工智能发展历史的长文，至少8000字，分"
    "十个章节详细论述每个时期的代表人物、关键技术和社会影响。",
    "请写一篇关于中国古代科技成就的长文，至少8000字，"
    "分十个章节详细介绍天文、数学、医学、农业、建筑等方面。",
    "请写一篇关于全球气候变化的长篇报告，至少8000字"
    "，分十个章节详细分析成因、影响、各国政策与未来展望。",
    "请写一部长篇科幻小说的前五章，每章至少1500"
    "字，讲述人类第一次星际移民的故事，细节要丰富。",
)


def parse_rows(value: str) -> list[int]:
    """Parse a comma-separated row-count list into positive integers."""
    rows: list[int] = []
    for part in value.split(","):
        if not part:
            continue
        row = int(part)
        if row <= 0:
            raise argparse.ArgumentTypeError(f"row count must be positive: {row}")
        rows.append(row)
    if not rows:
        raise argparse.ArgumentTypeError("no rows requested")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, sorting and deduplicating the requested row sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True,
                        help="serve root URL without /v1, e.g. http://127.0.0.1:8080")
    parser.add_argument("--rows", nargs="+", required=True, type=parse_rows,
                        metavar="N[,N...]",
                        help="decode row counts; repeat or comma-separate for a sweep")
    parser.add_argument("--max-tokens", type=int, default=1200,
                        help="per-stream output token budget (default: 1200)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=0.0,
                        help="explicit presence penalty keeps cross-engine loads comparable")
    parser.add_argument("--api-key", default=None, help="bearer token when serve requires one")
    parser.add_argument("--timeout", type=float, default=3600.0, help="per-request timeout seconds")
    parser.add_argument("--output", type=Path, default=None,
                        help="write per-row summaries and raw streams to this JSON file")
    args = parser.parse_args(argv)
    args.rows = sorted({row for value in args.rows for row in value})
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return args


def run_row(
    client: NInferServeClient,
    model: str,
    rows: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Release one fixed batch of decoding streams together and summarize their TPOTs."""
    barrier = threading.Barrier(rows)
    streams: list[dict[str, Any]] = [{} for _ in range(rows)]

    def worker(index: int) -> None:
        """Prepare one stream, wait for the barrier, execute it, and record its timing."""
        record: dict[str, Any] = streams[index]
        request = ProtocolRequest(
            protocol="openai_chat",
            path="/v1/chat/completions",
            payload={
                "model": model,
                "messages": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_completion_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "presence_penalty": args.presence_penalty,
            },
        )
        exchange = client.prepare(request)  # body encoded and connection established before timing
        barrier.wait()
        result = exchange.execute()
        record["http_status"] = result.http.status
        record["error"] = result.protocol_error or result.error_code or result.http.error
        record["cancelled"] = result.http.cancelled
        record["body_bytes"] = result.body_bytes
        first_ns = last_ns = None
        outputs = 0
        completion_tokens = None
        for event in result.events:
            if event.kind == "model_output" and event.output:
                outputs += 1
                first_ns = event.received_ns if first_ns is None else first_ns
                last_ns = event.received_ns
            usage = (event.payload or {}).get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                completion_tokens = usage["completion_tokens"]
        record["first_output_ns"] = first_ns
        record["last_output_ns"] = last_ns
        record["output_events"] = outputs
        record["completion_tokens"] = completion_tokens

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(rows)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ok = [record for record in streams if record.get("first_output_ns") is not None]
    tpots = []
    tokens_source = "usage"
    for record in ok:
        tokens = record["completion_tokens"]
        if tokens is None:
            tokens = record["output_events"]
            tokens_source = "events"
        if tokens and tokens > 1 and record["last_output_ns"] is not None:
            tpots.append((record["last_output_ns"] - record["first_output_ns"]) / (tokens - 1) / 1e6)
    tpots.sort()
    summary: dict[str, Any] = {
        "rows": rows,
        "streams": streams,
        "completed_streams": len(ok),
        "failed_streams": rows - len(ok),
        "tokens_source": tokens_source if ok else None,
        "tpot_ms": tpots,
    }
    if tpots:
        summary["tpot_ms_median"] = statistics.median(tpots)
        summary["tpot_ms_min"] = tpots[0]
        summary["tpot_ms_max"] = tpots[-1]
    if ok:
        firsts = [record["first_output_ns"] for record in ok]
        lasts = [record["last_output_ns"] for record in ok if record["last_output_ns"] is not None]
        if lasts:
            summary["all_active_window_s"] = (min(lasts) - max(firsts)) / 1e9
        counts = sorted(
            record["completion_tokens"] or record["output_events"] or 0 for record in ok
        )
        summary["completion_tokens"] = {"min": counts[0], "median": counts[len(counts) // 2], "max": counts[-1]}
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the row sweep against Serve and write the optional JSON report."""
    args = parse_args(argv)
    client = NInferServeClient(args.base_url, args.timeout, args.api_key)
    model = client.discover_model()
    report: dict[str, Any] = {
        "base_url": args.base_url,
        "model": model,
        "max_tokens": args.max_tokens,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
        },
        "rows": [],
    }
    for rows in args.rows:
        summary = run_row(client, model, rows, args)
        report["rows"].append(summary)
        if "tpot_ms_median" not in summary:
            print(f"rows {rows}: no completed stream measured")
            continue
        tokens = summary["completion_tokens"]
        window = summary.get("all_active_window_s")
        window_text = f"{window:.2f} s" if window is not None else "unavailable"
        print(
            f"rows {rows}: streams {summary['completed_streams']}/{rows} ok"
            f"; tokens min {tokens['min']} median {tokens['median']} max {tokens['max']}"
            f" ({summary['tokens_source']}); all-active window {window_text}"
            f"; per-stream TPOT median {summary['tpot_ms_median']:.2f} ms"
            f" (min {summary['tpot_ms_min']:.2f}, max {summary['tpot_ms_max']:.2f})"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
