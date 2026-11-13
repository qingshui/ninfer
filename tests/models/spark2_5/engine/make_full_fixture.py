"""Builds the full-depth Spark2_5 fixture used by the Engine capacity case.

It reuses the 2-layer fixture builder with every layer of the checkpoints kept, so the planned
weights, State rings and KV geometry equal the real model's. Usage: make_full_fixture.py OUT_DIR
(checkpoint root: $NINFER_SPARK_ROOT). Exit code 77 when a checkpoint is absent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "execution" / "make_fixture.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("spark_execution_fixture", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.CHECKPOINTS / module.VARIANTS["bf16"] / "config.json"
    if not config.is_file():
        print("Spark2_5 checkpoints are not available", file=sys.stderr)
        return 77
    module.SOURCE_LAYERS = tuple(range(len(json.loads(config.read_text())["layer_types"])))
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
