#!/usr/bin/env python3
"""Performance and numerical report for the b200 Qwen3.5-122B NVFP4 adaptation.

Reads the locked serving benchmark's metrics.json and the ninfer-perplexity
report.json, validates the measurements, and writes a JSON and a Markdown
report. Inputs are read-only. The only file this tool may create is the PPL
reference (--ppl-ref), initialised from the first run whose PPL is in (1, 20).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_REQ_THROUGHPUT = 61.0
TUNING_TARGET_RATIO = 1.5
DATASET_REQUESTS = 2000  # the locked dataset; request_throughput cannot exceed 2000 / duration_s
PPL_OPEN_RANGE = (1.0, 20.0)
PPL_TOLERANCE_PCT = 1.0
# ninfer_bench.sh rounds request_throughput to 4 and duration_s to 2 decimals.
THROUGHPUT_ROUNDING = 0.5e-4
DURATION_ROUNDING = 0.5e-2

METRIC_KEYS = (
    "request_throughput",
    "completed",
    "failed",
    "duration_s",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "mean_e2el_ms",
    "sglang_baseline_req_throughput",
)

LOCKED_PPL_GATE_NOTE = (
    "The locked ppl_check.sh takes the last line of the ninfer-perplexity log matching "
    "'perplexity<non-digits><number>'. That line is the 'report: .../profiles/perplexity/qwen35-122b/...' "
    "path, so it parses 35 (from the model name) and compares it with ppl_reference.txt, which also "
    "contains 35. Passing that gate is not numerical evidence. The PPL in this report comes from "
    "report.json overall.perplexity and the ppl_ref file; the ninfer-perplexity stdout format is unchanged."
)


class ReportError(RuntimeError):
    """Inputs are missing, malformed, or inconsistent."""


def read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ReportError(f"{what} {path} is not a JSON object")
    return document


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReportError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def load_metrics(path: Path) -> dict[str, float]:
    document = read_json(path, "metrics")
    missing = [key for key in METRIC_KEYS if key not in document]
    if missing:
        raise ReportError(f"metrics {path} lacks {', '.join(missing)}")
    return {key: finite_number(document[key], key) for key in METRIC_KEYS}


def throughput_bound_ok(throughput: float, duration_s: float) -> bool:
    """True when request_throughput is positive and no larger than 2000 / duration_s.

    The bound is widened by the rounding ninfer_bench.sh applies to the two inputs.
    """
    if throughput <= 0.0 or duration_s <= DURATION_ROUNDING:
        return False
    upper = DATASET_REQUESTS / (duration_s - DURATION_ROUNDING) + THROUGHPUT_ROUNDING
    return throughput <= upper


def load_perplexity(path: Path, artifact: Path) -> tuple[float, str]:
    """Return (overall perplexity, artifact id) after checking the report belongs to the artifact."""
    document = read_json(path, "perplexity report")
    try:
        ppl = finite_number(document["overall"]["perplexity"], "overall.perplexity")
        described = document["artifact"]
        artifact_id = str(described["prefill_signature"])
        reported_path = str(described["path"])
    except (KeyError, TypeError) as exc:
        raise ReportError(f"perplexity report {path} lacks field {exc}") from exc
    if os.path.realpath(reported_path) != os.path.realpath(artifact):
        raise ReportError(f"perplexity report scored {reported_path}, not --artifact {artifact}")
    if not PPL_OPEN_RANGE[0] < ppl < PPL_OPEN_RANGE[1]:
        raise ReportError(f"perplexity {ppl} is outside the sanity range {PPL_OPEN_RANGE}")
    return ppl, artifact_id


def compare_ppl_ref(ref_path: Path, ppl: float, artifact_id: str) -> dict[str, Any]:
    """Compare against the recorded reference, or initialise it from this run."""
    if not ref_path.exists():
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(
            json.dumps({"perplexity": ppl, "artifact_id": artifact_id}, indent=2) + "\n", encoding="utf-8"
        )
        return {"ppl_ref": None, "ppl_ref_status": "initialized_from_this_run", "ppl_delta_pct": None, "ppl_within_1pct": None}
    reference = read_json(ref_path, "ppl reference")
    ref = finite_number(reference.get("perplexity"), "ppl reference perplexity")
    if ref <= 0.0:
        raise ReportError(f"ppl reference {ref_path} must be positive")
    if reference.get("artifact_id") != artifact_id:
        raise ReportError(
            f"ppl reference {ref_path} belongs to artifact {reference.get('artifact_id')!r}, not {artifact_id!r}; "
            "remove it deliberately to re-initialise"
        )
    delta_pct = (ppl - ref) / ref * 100.0
    return {
        "ppl_ref": ref,
        "ppl_ref_status": "recorded",
        "ppl_delta_pct": delta_pct,
        "ppl_within_1pct": abs(delta_pct) <= PPL_TOLERANCE_PCT,
    }


def git_state() -> tuple[str, bool]:
    def run(*args: str) -> str:
        try:
            done = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ReportError(f"git {' '.join(args)} failed: {exc}") from exc
        return done.stdout.strip()

    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))


def gpu_name(index: int) -> str:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(index)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReportError(f"cannot query GPU {index} name: {exc}") from exc
    name = done.stdout.strip()
    if not name:
        raise ReportError(f"nvidia-smi returned no name for GPU {index}")
    return name


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    metrics = load_metrics(args.metrics)
    ppl, artifact_id = load_perplexity(args.perplexity_report, args.artifact)
    comparison = compare_ppl_ref(args.ppl_ref, ppl, artifact_id)
    baseline = metrics["sglang_baseline_req_throughput"]
    if baseline != BASELINE_REQ_THROUGHPUT:
        raise ReportError(f"metrics baseline {baseline} differs from the locked anchor {BASELINE_REQ_THROUGHPUT}")
    commit, dirty = git_state()
    throughput = metrics["request_throughput"]
    return {
        "request_throughput": throughput,
        "sglang_baseline_req_throughput": baseline,
        "ratio_vs_baseline": throughput / baseline,
        "completed": int(metrics["completed"]),
        "failed": int(metrics["failed"]),
        "duration_s": metrics["duration_s"],
        "throughput_bound_ok": throughput_bound_ok(throughput, metrics["duration_s"]),
        "mean_ttft_ms": metrics["mean_ttft_ms"],
        "p99_ttft_ms": metrics["p99_ttft_ms"],
        "mean_tpot_ms": metrics["mean_tpot_ms"],
        "mean_e2el_ms": metrics["mean_e2el_ms"],
        "ppl": ppl,
        **comparison,
        "gpu": {"index": args.gpu, "name": gpu_name(args.gpu)},
        "build": {"preset": args.build_dir.name, "git_commit": commit, "dirty": dirty},
        "artifact": {"path": str(args.artifact), "artifact_id": artifact_id},
        "serve_args": args.serve_args,
        "tuning_target": {
            "request_throughput_min": TUNING_TARGET_RATIO * baseline,
            "status": "owned_by_tuning",
        },
        "locked_ppl_gate_note": LOCKED_PPL_GATE_NOTE,
    }


def render_markdown(report: dict[str, Any]) -> str:
    build, gpu, artifact = report["build"], report["gpu"], report["artifact"]
    target = report["tuning_target"]
    ref_status = report["ppl_ref_status"]
    if ref_status == "initialized_from_this_run":
        ppl_ref_line = f"initialised from this run ({report['ppl']:.6f}); nothing to compare yet"
    else:
        ppl_ref_line = (
            f"{report['ppl_ref']:.6f}, delta {report['ppl_delta_pct']:+.3f}%, "
            f"within 1%: {report['ppl_within_1pct']}"
        )
    rows = [
        ("request_throughput (req/s)", f"{report['request_throughput']:.4f}"),
        ("sglang baseline (req/s)", f"{report['sglang_baseline_req_throughput']:.1f}"),
        ("ratio vs baseline", f"{report['ratio_vs_baseline']:.4f}"),
        ("completed / failed", f"{report['completed']} / {report['failed']}"),
        ("duration_s", f"{report['duration_s']:.2f}"),
        ("throughput <= 2000/duration_s", str(report["throughput_bound_ok"])),
        ("mean TTFT (ms)", f"{report['mean_ttft_ms']:.2f}"),
        ("p99 TTFT (ms)", f"{report['p99_ttft_ms']:.2f}"),
        ("mean TPOT (ms)", f"{report['mean_tpot_ms']:.2f}"),
        ("mean E2EL (ms)", f"{report['mean_e2el_ms']:.2f}"),
    ]
    lines = [
        "# b200 Qwen3.5-122B NVFP4 report",
        "",
        "## Environment",
        "",
        f"- GPU: index {gpu['index']}, {gpu['name']}",
        f"- Build: preset `{build['preset']}`, commit `{build['git_commit']}`, uncommitted changes: {build['dirty']}",
        f"- Artifact: `{artifact['path']}`, id `{artifact['artifact_id']}`",
        f"- ninfer-serve args: `{report['serve_args']}`",
        "",
        "## Serving benchmark",
        "",
        "Scope: end-to-end HTTP serving through the locked ninfer_bench.sh protocol (unmodified client,",
        "2000 dataset requests, concurrency 130, n=4, max_tokens 15). TTFT and TPOT are client-side",
        "means over requests. This does not isolate any single Op or Engine stage.",
        "",
        "| metric | value |",
        "| --- | --- |",
        *(f"| {name} | {value} |" for name, value in rows),
        "",
        f"Throughput target {target['request_throughput_min']:.1f} req/s ({TUNING_TARGET_RATIO}x baseline) is "
        f"`{target['status']}`: it is not a development-stage gate.",
        "",
        "## Numerical check",
        "",
        f"- Perplexity (report.json overall.perplexity, ninfer-perplexity --quick): {report['ppl']:.6f}",
        f"- Reference: {ppl_ref_line}",
        f"- {report['locked_ppl_gate_note']}",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, required=True, help="metrics.json from ninfer_bench.sh")
    parser.add_argument("--perplexity-report", type=Path, required=True, help="ninfer-perplexity report.json")
    parser.add_argument("--ppl-ref", type=Path, required=True, help="PPL reference JSON (created if absent)")
    parser.add_argument("--artifact", type=Path, required=True, help="the .ninfer artifact that was measured")
    parser.add_argument("--build-dir", type=Path, required=True, help="CMake build directory (named after the preset)")
    parser.add_argument("--gpu", type=int, required=True, help="CUDA device index used for the measurement")
    parser.add_argument("--serve-args", required=True, help="ninfer-serve arguments used for the benchmark")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(args)
    except ReportError as exc:
        print(f"[b200_report] error: {exc}", file=sys.stderr)
        return 1
    for path, text in (
        (args.out_json, json.dumps(report, indent=2, allow_nan=False) + "\n"),
        (args.out_md, render_markdown(report)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(
        f"[b200_report] req/s={report['request_throughput']:.4f} ratio={report['ratio_vs_baseline']:.4f} "
        f"ppl={report['ppl']:.6f} bound_ok={report['throughput_bound_ok']} -> {args.out_json}"
    )
    failed = not report["throughput_bound_ok"] or report["ppl_within_1pct"] is False
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
