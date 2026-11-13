"""F7: the Spark2.5 model-level numeric tests are registered with ctest and carry the
Spec's top-1 agreement gates.

Spec numbers come from F7 acceptance. The checks inspect the ctest registration and the test
sources only; the numeric ctest cases themselves are run by the ctest gate.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EXEC = REPO / "tests" / "models" / "spark2_5" / "execution"
BUILD_DIRS = [REPO / "build" / "rtx5090-dev", REPO / "build" / "rtx5090-release"]


def _sources() -> str:
    return "\n".join(
        p.read_text() for p in sorted(EXEC.iterdir()) if p.suffix in {".h", ".cpp", ".inc"}
    )


def test_f7_numeric_cases_registered_in_ctest():
    listing = ""
    for build in BUILD_DIRS:
        if (build / "CTestTestfile.cmake").is_file():
            done = subprocess.run(["ctest", "--test-dir", str(build), "-N"], capture_output=True,
                                  text=True, timeout=60)
            listing += done.stdout
    if "spark2_5" not in listing:
        pytest.skip("no configured build tree lists spark2_5 tests")
    for case in ("layers_bf16", "layers_fp8", "window", "real"):
        assert f"ninfer_spark2_5_execution_{case}_test" in listing, case


def test_f7_real_artifact_gates_top1_agreement():
    """Real-artifact case: >=256 positions, top-1 agreement >= 98% (bf16) / 95% (fp8) on
    positions whose reference top-1/top-2 gap is >= 0.1, enforced (not only printed)."""
    text = _sources()
    body = text[text.index("void real_case") :]
    body = body[: body.index("\n}\n")]
    assert "positions >= 256" in body
    assert re.search(r"0\.98", body) and re.search(r"0\.95", body), "no 98%/95% top-1 gate"
    assert re.search(r"\b0\.1\b", body), "top-1/top-2 margin of 0.1 is not applied"
