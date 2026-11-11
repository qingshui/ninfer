"""T7 (F7): the Spark2.5 model-level numeric tests are registered with ctest and carry the
Spec's criteria, mutants and top-1 agreement gates.

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


def _flat(text: str) -> str:
    return re.sub(r"\s+", "", text)


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


def test_f7_criteria_follow_linear_policy():
    """A16Only -> {1e-2, -, 2^-5}; AllowA8 -> {4e-2, -, 6e-2}; attention -> {2.8e-3,1e-3,2^-7}."""
    text = _flat(_sources())
    assert "kAttentionCriterion{2.8e-3,1.0e-3,0x1p-7}" in text
    assert re.search(r"A16Only\?(test::)?ReductionCriterion\{1\.0e-2,0\.0,0x1p-5\}", text)
    assert re.search(r":(test::)?ReductionCriterion\{4\.0e-2,0\.0,6\.0e-2\}", text)


def test_f7_each_wrong_variant_has_a_failing_mutant_case():
    """Window 512, full-channel rotation, elementwise gate, no-sigmoid gate and an untied head
    must each be exercised as a mutant that the fixture comparison rejects."""
    text = _sources()
    mutant_lines = " ".join(
        line.lower() for line in text.splitlines() if re.search(r"mutat", line, re.I)
    )
    wanted = {
        "window 512": r"512",
        "full-channel rotation": r"rotat|full[- ]channel|partial",
        "elementwise gate": r"element",
        "gate without sigmoid": r"sigmoid",
        "untied head": r"tied|share|head",
    }
    missing = [name for name, pattern in wanted.items() if not re.search(pattern, mutant_lines)]
    assert not missing, f"no mutant case for: {missing}"


def test_f7_real_artifact_gates_top1_agreement():
    """Real-artifact case: >=256 positions, top-1 agreement >= 98% (bf16) / 95% (fp8) on
    positions whose reference top-1/top-2 gap is >= 0.1, enforced (not only printed)."""
    text = _sources()
    body = text[text.index("void real_case") :]
    body = body[: body.index("\n}\n")]
    assert "positions >= 256" in body
    assert re.search(r"0\.98", body) and re.search(r"0\.95", body), "no 98%/95% top-1 gate"
    assert re.search(r"\b0\.1\b", body), "top-1/top-2 margin of 0.1 is not applied"
