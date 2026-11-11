"""T7 acceptance: Spark2.5 gate metrics (F8), serving (F5) and numeric-test registration (F7).

Written from the Spec only. Throughput targets owned by tuning (overall_ratio_min >= 2) are
asserted only as finite and inside (0, 50). Environment knobs:

  NINFER_T7_METRICS   comma-separated metrics.json files of full-matrix gate runs (one per
                      variant, or one covering both). Default: .agent-tmp/spark_gate/full_{bf16,fp8}.
  NINFER_SPARK_ROOT   directory with the Spark-X2.6-1.7B checkpoints.
  GATE_GPU            idle GPU index for the serving cases.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "tools" / "bench"
GATE = BENCH / "spark_gate.py"
BASELINES = BENCH / "baselines_spark17b.json"
GATE_COMMIT = "84f157ef"  # "bench: add locked spark-x2.6-1.7b gate and frozen sglang baselines"
WORK = REPO / ".agent-tmp" / "t7_tests"
ROOT = Path(os.environ.get("NINFER_SPARK_ROOT", "/home/users/humingqing/qwen"))
CHECKPOINTS = {"bf16": ROOT / "Spark-X2.6-1.7B", "fp8": ROOT / "Spark-X2.6-1.7B-FP8"}
RECIPES = {"bf16": "spark_x2_6_17b_bf16", "fp8": "spark_x2_6_17b_fp8"}
NVIDIA_LIBS = "/home/opt/nvidia_lib:/root/local-deps/usr/lib"

FULL_POINTS = {
    "ctx1024_c1", "ctx1024_c8", "ctx1024_c16", "ctx8192_c16",
    "ctx65536_c8", "ctx262144_c8", "ctx1048576_c1",
}
PREFILL_GATE = {"ctx8192_c16", "ctx65536_c8", "ctx262144_c8", "ctx1048576_c1"}
DECODE_GATE = {"ctx1024_c1", "ctx1024_c8", "ctx1024_c16", "ctx65536_c8", "ctx262144_c8"}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# --------------------------------------------------------------------------- F8: judge + metrics


def test_f8_judge_files_match_locked_version():
    """spark_gate.py and baselines_spark17b.json are byte-identical to the locked commit."""
    for path in (GATE, BASELINES):
        rel = path.relative_to(REPO).as_posix()
        diff = subprocess.run(
            ["git", "diff", "--quiet", GATE_COMMIT, "--", rel], cwd=REPO
        )
        assert diff.returncode == 0, f"{rel} differs from locked commit {GATE_COMMIT}"


def _load_metrics():
    env = os.environ.get("NINFER_T7_METRICS")
    paths = (
        [Path(p) for p in env.split(",") if p]
        if env
        else [REPO / ".agent-tmp/spark_gate" / f"full_{v}" / "metrics.json" for v in ("bf16", "fp8")]
    )
    missing = [str(p) for p in paths if not p.is_file()]
    assert not missing, f"full-matrix gate metrics not found: {missing}"
    merged = {"variants": {}, "functional_ok": True}
    for path in paths:
        data = json.loads(path.read_text())
        merged["variants"].update(data["variants"])
        merged["functional_ok"] = merged["functional_ok"] and data["functional_ok"] is True
        merged.setdefault("sources", []).append(data)
    return merged


def test_f8_full_matrix_metrics_contract():
    """Full matrix: functional_ok, 0 failed, finite in-range metrics, correct measurement scope,
    prefill/decode ratio minima >= 2.0. overall_ratio_min is only checked as finite in (0, 50)."""
    metrics = _load_metrics()
    baselines = json.loads(BASELINES.read_text())
    assert metrics["functional_ok"] is True
    assert set(metrics["variants"]) == {"bf16", "fp8"}

    prefill_mins, decode_mins = [], []
    for variant, result in metrics["variants"].items():
        points = result["points"]
        assert set(points) == FULL_POINTS, f"{variant}: not a full matrix"
        for key, point in points.items():
            assert point["failed"] == 0, f"{variant} {key}: failed requests"
            assert _finite(point["prefill_tps"]) and point["prefill_tps"] > 0, (variant, key)
            assert _finite(point["decode_tps_mean"]) and point["decode_tps_mean"] > 0, (variant, key)
            base = baselines[variant]["points"][key]
            if key in PREFILL_GATE:
                ratio = point["prefill_tps"] / base["sglang_prefill_tps"]
                assert point["prefill_ratio"] == pytest.approx(ratio, rel=1e-9)
            if key in DECODE_GATE:
                ratio = point["decode_tps_mean"] / base["sglang_decode_tps"]
                assert point["decode_ratio"] == pytest.approx(ratio, rel=1e-9)
        pre = [points[k]["prefill_ratio"] for k in PREFILL_GATE]
        dec = [points[k]["decode_ratio"] for k in DECODE_GATE]
        assert result["prefill_ratio_min"] == pytest.approx(min(pre))
        assert result["decode_ratio_min"] == pytest.approx(min(dec))
        prefill_mins.append(result["prefill_ratio_min"])
        decode_mins.append(result["decode_ratio_min"])

    top = {"prefill_ratio_min": min(prefill_mins), "decode_ratio_min": min(decode_mins)}
    top["overall_ratio_min"] = min(top["prefill_ratio_min"], top["decode_ratio_min"])
    for name, value in top.items():
        assert _finite(value) and 0 < value < 50, f"{name}={value}"
    # Development-phase gates (overall_ratio_min is intentionally not compared with 2.0).
    assert top["prefill_ratio_min"] >= 2.0, f"prefill_ratio_min={top['prefill_ratio_min']:.3f}"
    assert top["decode_ratio_min"] >= 2.0, f"decode_ratio_min={top['decode_ratio_min']:.3f}"
