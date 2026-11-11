"""Builds the fixture of the Spark2_5 HF parity cases.

For both variants it converts the checkpoint with the official recipe (the converter contract:
python3 -m tools.convert --model CKPT --recipe spark_x2_6_17b_<variant> --out PATH) and writes the
fp32 reference of make_hf_reference.py:

    OUT_DIR/artifacts/{bf16,fp8}.ninfer
    OUT_DIR/reference/hf-{bf16,fp8}.json and hf-{bf16,fp8}.logits.f32

Usage: build_hf_fixture.py OUT_DIR   (checkpoint root: $NINFER_SPARK_ROOT, shared with tests/convert)
Exit code 77 when a checkpoint or the reference environment is absent. OUT_DIR/HF_FIXTURE_OK marks
a complete build.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
REFERENCE = Path(__file__).resolve().parent / "make_hf_reference.py"
CHECKPOINTS = Path(os.environ.get("NINFER_SPARK_ROOT", "/home/users/humingqing/qwen"))
VARIANTS = {"bf16": "Spark-X2.6-1.7B", "fp8": "Spark-X2.6-1.7B-FP8"}
SKIP = 77


def main() -> int:
    out = Path(sys.argv[1])
    if not all((CHECKPOINTS / name / "config.json").is_file() for name in VARIANTS.values()):
        print("Spark2_5 checkpoints are not available", file=sys.stderr)
        return SKIP
    if (out / "HF_FIXTURE_OK").is_file():
        return 0
    shutil.rmtree(out, ignore_errors=True)
    (out / "artifacts").mkdir(parents=True)
    for variant, name in VARIANTS.items():
        # The reference first: it is the step that can report an absent environment.
        done = subprocess.run(
            [sys.executable, "-I", str(REFERENCE), "--model", str(CHECKPOINTS / name),
             "--variant", variant, "--out", str(out / "reference")],
            check=False,
        )
        if done.returncode != 0:
            return done.returncode
        subprocess.run(
            [sys.executable, "-B", "-m", "tools.convert", "--model", str(CHECKPOINTS / name),
             "--recipe", f"spark_x2_6_17b_{variant}", "--out", str(out / "artifacts" / f"{variant}.ninfer"),
             "--device", "cpu", "--cpu-threads", "8"],
            cwd=ROOT,
            check=True,
        )
    (out / "HF_FIXTURE_OK").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
