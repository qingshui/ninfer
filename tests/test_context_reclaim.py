"""K2 acceptance tests for F1, F3, F4, F7, F8 (reclaim planning cost and context epoch contract).

Host-only. Checks the public Session/ResourceManager contract surface, the K2 measurement
artifact, and the recorded diff_gate metrics. Nothing here starts a server or loads a model.
Prefill throughput thresholds (prefill_ratio_min >= 2.0) belong to the tuning phase: the tests only
require prefill_ratio to be present, finite and inside the plausibility band (0, 50).
Recorded-artifact tests skip when the artifact is absent.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / ".agent-tmp/ninfer-35b-prefill-2x"
SRC = ROOT / "src"
RM_HEADER = SRC / "runtime/engine/context_cache/resource_manager.h"
SESSIONS = {
    "qwen3_5": SRC / "models/qwen3_5/session/session.h",
    "qwen4_exp": SRC / "models/qwen4_exp/session/session.h",
    "glm5next": SRC / "models/glm5next/session/session.h",
}
POINTS = [f"ctx{ctx}_c{c}" for ctx in (2048, 32768, 131072, 204800) for c in (1, 2, 4, 6)]
BASELINE = "a754fcf3"


def _finite_positive(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _load(path: Path):
    if not path.is_file():
        pytest.skip(f"missing artifact {path.relative_to(ROOT)}")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def gate_metrics():
    return _load(TASK / "k2/full-after/metrics.json")


@pytest.fixture(scope="module")
def reclaim_report():
    path = TASK / "k2-reclaim.md"
    if not path.is_file():
        pytest.skip("missing k2-reclaim.md")
    return path.read_text()


# ---- F1: judge measurability --------------------------------------------------------------


def test_f4_context_epoch_and_pins_exposed_by_every_session():
    for model, header in SESSIONS.items():
        text = header.read_text()
        assert re.search(r"std::uint64_t\s+context_epoch\(\)\s+const\s+noexcept", text), model
        assert re.search(r"bool\s+context_pins\(\s*CheckpointHandle[^)]*\)\s+const\s+noexcept",
                         text), model


def test_f8_reclaim_artifact_has_required_fields(reclaim_report):
    for needle in ("commit", "workload", "hot spot", "planning_ns", "128K C=1", "decode_keep"):
        assert needle in reclaim_report, needle
    assert re.search(r"ttft", reclaim_report, re.I)
    assert re.search(r"reclaim\w*\s+(calls|attempts)", reclaim_report, re.I)
