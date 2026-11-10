"""T2 / F2: real-checkpoint conversion through the public CLI (size, time, reproducibility, source immutability)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("NINFER_SPARK_ROOT", "/home/users/humingqing/qwen"))
CHECKPOINTS = {
    "spark_x2_6_17b_bf16": ROOT / "Spark-X2.6-1.7B",
    "spark_x2_6_17b_fp8": ROOT / "Spark-X2.6-1.7B-FP8",
}
GIB = 1 << 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(directory: Path) -> dict:
    return {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns, sha256(p))
        for p in sorted(directory.iterdir())
        if p.is_file()
    } | {"__entries__": tuple(sorted(p.name for p in directory.iterdir()))}


def run_convert(model: Path, recipe: str, out: Path) -> tuple[int, float, str]:
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "5")
    env.setdefault("LD_LIBRARY_PATH", "/home/opt/nvidia_lib:/root/local-deps/usr/lib")
    start = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-m", "tools.convert", "--model", str(model), "--recipe", recipe,
         "--out", str(out), "--cpu-threads", "16"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=1800,
    )
    return done.returncode, time.monotonic() - start, done.stderr[-2000:]


def inspect_json(path: Path) -> dict:
    done = subprocess.run(
        [sys.executable, "-m", "tools.artifact.inspect", "--json", "--bindings", str(path)],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def conversions():
    for model in CHECKPOINTS.values():
        if not model.is_dir():
            pytest.skip(f"checkpoint missing: {model}")
    work = REPO / ".agent-tmp" / "t2_f2_tests"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    if shutil.disk_usage(work).free < 12 * GIB:
        shutil.rmtree(work, ignore_errors=True)
        pytest.skip("less than 12 GiB free disk")
    results = {}
    try:
        for recipe, model in CHECKPOINTS.items():
            before = snapshot(model)
            out = work / f"{recipe}_a.ninfer"
            code, seconds, err = run_convert(model, recipe, out)
            first = sha256(out) if out.exists() else None
            again = work / f"{recipe}_b.ninfer"
            code2, _, err2 = run_convert(model, recipe, again)
            second = sha256(again) if again.exists() else None
            again.unlink(missing_ok=True)
            results[recipe] = {
                "code": code, "code2": code2, "err": err + err2, "seconds": seconds,
                "size": out.stat().st_size if out.exists() else None,
                "shas": (first, second), "unchanged": before == snapshot(model),
                "info": inspect_json(out) if code == 0 else None,
            }
            out.unlink(missing_ok=True)
        yield results
    finally:
        shutil.rmtree(work, ignore_errors=True)


@pytest.mark.parametrize("recipe", list(CHECKPOINTS))
def test_real_conversion_succeeds_within_size_and_time(conversions, recipe):
    r = conversions[recipe]
    assert r["code"] == 0, r["err"]
    assert r["size"] <= 4 * GIB
    assert r["seconds"] < 1800


@pytest.mark.parametrize("recipe", list(CHECKPOINTS))
def test_conversion_is_reproducible(conversions, recipe):
    r = conversions[recipe]
    assert r["code"] == 0 and r["code2"] == 0, r["err"]
    assert r["shas"][0] is not None and r["shas"][0] == r["shas"][1]


@pytest.mark.parametrize("recipe", list(CHECKPOINTS))
def test_artifact_text_component_is_spark2_5_with_tied_head(conversions, recipe):
    info = conversions[recipe]["info"]
    assert info is not None
    config = info["components"]["text"]["config"]
    assert config["architectures"] == ["Spark2_5ForCausalLM"]
    assert config["model_type"] == "spark2_5_text"
    assert config["sliding_window"] == 512
    types = config["layer_types"]
    assert [i for i, t in enumerate(types) if t == "full_attention"] == list(range(3, 28, 4))
    assert config["rope"]["full_attention"]["rotary_dim"] == 64
    assert config["rope"]["sliding_attention"]["rotary_dim"] == 256
    records = info["binding_records"]
    assert records["text/output_head"] == records["text/token_embedding"]


@pytest.mark.parametrize("recipe", list(CHECKPOINTS))
def test_source_directory_is_not_modified(conversions, recipe):
    assert conversions[recipe]["unchanged"]
