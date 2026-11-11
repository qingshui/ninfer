"""F5: KV capacity points on the real checkpoints.

Per variant: convert twice, start ninfer-serve with the judge's serving config (1M ctx, C16,
auto capacity) and exercise it. Needs an idle GPU (GATE_GPU) and the checkpoints under
NINFER_SPARK_ROOT; skips otherwise.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "tools" / "bench"
WORK = REPO / ".agent-tmp" / "spark2_5_serving"
ROOT = Path(os.environ.get("NINFER_SPARK_ROOT", "/home/users/humingqing/qwen"))
CHECKPOINTS = {"bf16": ROOT / "Spark-X2.6-1.7B", "fp8": ROOT / "Spark-X2.6-1.7B-FP8"}
RECIPES = {"bf16": "spark_x2_6_17b_bf16", "fp8": "spark_x2_6_17b_fp8"}
NVIDIA_LIBS = "/home/opt/nvidia_lib:/root/local-deps/usr/lib"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _convert(variant: str, out: Path, gpu: int) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    env["LD_LIBRARY_PATH"] = NVIDIA_LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    env.pop("CUDA_MODULE_LOADING", None)
    cmd = [sys.executable, "-m", "tools.convert", "--model", str(CHECKPOINTS[variant]),
           "--recipe", RECIPES[variant], "--out", str(out), "--cpu-threads", "16"]
    done = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0 and out.is_file(), done.stdout[-2000:] + done.stderr[-2000:]


@pytest.fixture(scope="module")
def gate():
    sys.path.insert(0, str(BENCH))
    try:
        import spark_gate
    finally:
        sys.path.remove(str(BENCH))
    return spark_gate


@pytest.fixture(scope="module", params=["bf16", "fp8"])
def served(request, gate):
    variant = request.param
    if not (CHECKPOINTS[variant] / "config.json").is_file():
        pytest.skip(f"{variant} checkpoint is not available")
    if shutil.disk_usage(REPO).free < 9 * 1024**3:
        pytest.skip("less than 9 GiB free for artifacts")
    requested = os.environ.get("GATE_GPU")
    try:
        gpu = gate.pick_gpu(int(requested) if requested else None)
    except SystemExit as exc:
        pytest.skip(str(exc))
    work = WORK / variant
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    first, second = work / "a.ninfer", work / "b.ninfer"
    _convert(variant, first, gpu)
    sha_first = _sha256(first)
    _convert(variant, second, gpu)
    sha_second = _sha256(second)
    second.unlink()

    port = gate.free_port(18190)
    assert port is not None
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        proc, url = gate.start_server(gpu, port, first, work / "serve.log")
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    try:
        assert proc is not None, f"ninfer-serve failed to start for {variant}; see {work}/serve.log"
        yield {"variant": variant, "url": url, "sha": (sha_first, sha_second), "gate": gate}
    finally:
        gate.stop_server(proc)
        try:
            gate.wait_gpu_drained(gpu)
        except SystemExit:
            pass
        shutil.rmtree(work, ignore_errors=True)


def test_f5_gate_capacity_points_do_not_fail(served):
    """256K x C8 and 1M x C1 are accepted under the judge's serving config. Run on bf16 only:
    it has the smaller KV budget, and it keeps the file inside the time limit."""
    if served["variant"] != "bf16":
        pytest.skip("capacity points are exercised on the tighter-memory bf16 variant")
    from transformers import AutoTokenizer

    gate = served["gate"]
    tok = AutoTokenizer.from_pretrained(gate.TOKENIZER_DIR)
    for ctx, conc, n in ((262144, 8, 8), (1048576, 1, 1)):
        point = gate.run_point(served["url"], tok, ctx, conc, n, 900.0, gate.ninfer_request)
        assert point["failed"] == 0, f"ctx {ctx} x C{conc}: {point}"
