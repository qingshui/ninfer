"""Spark2.5 gate judge lock (F8): spark_gate.py and baselines_spark17b.json stay unchanged."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "tools" / "bench"
GATE = BENCH / "spark_gate.py"
BASELINES = BENCH / "baselines_spark17b.json"
GATE_COMMIT = "84f157ef"  # "bench: add locked spark-x2.6-1.7b gate and frozen sglang baselines"


def test_f8_judge_files_match_locked_version():
    """spark_gate.py and baselines_spark17b.json are byte-identical to the locked commit."""
    for path in (GATE, BASELINES):
        rel = path.relative_to(REPO).as_posix()
        diff = subprocess.run(
            ["git", "diff", "--quiet", GATE_COMMIT, "--", rel], cwd=REPO
        )
        assert diff.returncode == 0, f"{rel} differs from locked commit {GATE_COMMIT}"
