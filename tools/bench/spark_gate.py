#!/usr/bin/env python3
"""Spark-X2.6-1.7B dual gate: ninfer-serve vs the locked sglang baselines.

One self-contained judge for the spark-x2.6-1.7b-2x adaptation task. It covers
BOTH checkpoints (BF16 and compressed-tensors FP8) in one invocation and gates
TWO metrics against per-variant frozen sglang baselines:

  prefill throughput = sum(prompt_tokens) / max(TTFT of the wave)
  decode throughput  = mean per-stream (completion-1)/(e2e - ttft)

  decode points  (decode_ratio gated):  1K x C1/C8/C16, 64K x C8, 256K x C8
  prefill points (prefill_ratio gated):  8K x C16, 64K x C8, 256K x C8, 1M x C1

Methodology (identical to the baseline measurement this judge froze):
- streaming greedy (temperature 0), thinking disabled, 512 output tokens
- unique-nonce prose prompts defeat prefix/radix caching on every request
- ninfer side: Anthropic /v1/messages; baseline side was sglang's OpenAI
  /v1/chat/completions via --measure-sglang with the same corpus and wave
  shapes (tuned run_spark_x26_17b_{bf16,fp8}.sh configs, sglang 0.5.23).

The judge rebuilds the .ninfer artifact for every variant from the current
tree before serving (converter contract: `python -m tools.convert --model
<src> --recipe <official recipe> --out <file>`; recipes spark_x2_6_17b_bf16 /
spark_x2_6_17b_fp8 must be registered in tools/convert/official_recipes.py).
It starts its own ninfer-serve on the gate GPU and stops it at exit. It never
launches sglang: the sglang baseline is frozen in tools/bench/baselines_spark17b.json.

Matrices (n = requests per point, mirrors the frozen baseline waves):
- smoke: 1K x C1, 64K x C2        (functional verification)
- quick: 1K x C8, 64K x C8, 1M x C1 (tuning iterations)
- full:  all 7 points               (final acceptance, both variants)

Metrics written to $AUTODEVX_ANALYZE_OUTPUT_DIR/metrics.json:
  overall_ratio_min  min over both variants of min(prefill_ratio_min,
                      decode_ratio_min) -- the acceptance number (>= 2.0)
  prefill_ratio_min / decode_ratio_min  min over variants, per metric
  functional_ok      zero failed requests across every point

Exit codes: 0 pass / 1 perf or functional failure / 2 environment error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINES = Path(__file__).resolve().parent / "baselines_spark17b.json"
SERVE = REPO / "build/rtx5090-release/apps/ninfer-serve"
ARTIFACT_DIR = Path(
    "/home/users/humingqing/work/ninfer/.agent-tmp/diff-bench/"
    "autodevx-spark17b/artifacts"
)
# Tokenizer shared by both checkpoints; prompts are built with it on every side.
TOKENIZER_DIR = "/home/users/humingqing/qwen/Spark-X2.6-1.7B"
NINFER_MODEL_ID = "spark-x2.6-1.7b"
SGLANG_MODEL_ID = "spark2.6"
OUT_TOKENS = 512
RATIO_TARGET = 2.0
NVIDIA_LIBS = "/home/opt/nvidia_lib:/root/local-deps/usr/lib"

VARIANTS = ("bf16", "fp8")
VARIANT_SOURCES = {
    "bf16": (
        Path("/home/users/humingqing/qwen/Spark-X2.6-1.7B"),
        "spark_x2_6_17b_bf16",
    ),
    "fp8": (
        Path("/home/users/humingqing/qwen/Spark-X2.6-1.7B-FP8"),
        "spark_x2_6_17b_fp8",
    ),
}

# (ctx, concurrency, requests) — request counts mirror the baseline waves.
SMOKE_MATRIX = [
    (1024, 1, 2), (65536, 2, 4),
]
QUICK_MATRIX = [
    (1024, 8, 8), (65536, 8, 8), (1048576, 1, 2),
]
FULL_MATRIX = [
    (1024, 1, 3), (1024, 8, 8), (1024, 16, 16),
    (8192, 16, 16),
    (65536, 8, 8),
    (262144, 8, 8),
    (1048576, 1, 2),
]

# Which metric is gated on which (ctx, concurrency) point.
DECODE_GATE = {(1024, 1), (1024, 8), (1024, 16), (65536, 8), (262144, 8)}
PREFILL_GATE = {(8192, 16), (65536, 8), (262144, 8), (1048576, 1)}

_FILLER = (
    "The transformer architecture processes sequences through stacked attention layers. "
    "Each layer projects its input into query, key and value vectors, then mixes information "
    "across positions using softmax-normalized dot products. Residual connections and "
    "normalization keep gradients stable across depth. Feed-forward sublayers expand and "
    "re-project features channel-wise, giving the network per-position nonlinearity. "
    "Training minimizes next-token cross-entropy over large corpora; inference reuses the "
    "resulting representations to predict continuations one token at a time. "
)

import requests  # noqa: E402  (kept after docstring for fast --help)


def build_prompt(tokenizer, target_tokens: int, nonce: str) -> str:
    head = f"Document {nonce}: "
    body_units = max(1, (target_tokens - 64) // 100 + 1)
    text = head + (_FILLER * (body_units // 1 + 1))
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target_tokens - 32:
        text = tokenizer.decode(ids[: target_tokens - 32])
    tail = (
        f"\n\nAbove is document {nonce}. Write a detailed, thorough summary of at "
        f"least 450 words covering every topic it mentions. Do not stop early, "
        f"do not think; answer directly."
    )
    return text + tail


def ninfer_request(url: str, prompt: str, timeout: float) -> dict:
    """Anthropic /v1/messages streaming request against ninfer-serve."""
    payload = {
        "model": NINFER_MODEL_ID,
        "max_tokens": OUT_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": prompt}],
    }
    rec = {"ok": False}
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{url}/v1/messages", json=payload,
                             timeout=timeout, stream=True)
        rec["http_status"] = resp.status_code
        if resp.status_code != 200:
            rec["error"] = resp.text[:300]
            return rec
        ttft = None
        usage_in = None
        usage_out = None
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            try:
                chunk = json.loads(raw[5:].strip())
            except json.JSONDecodeError:
                continue
            kind = chunk.get("type")
            if kind == "message_start":
                usage_in = chunk["message"].get("usage", {}).get("input_tokens")
            elif kind == "content_block_delta":
                delta = chunk.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
            elif kind == "message_delta":
                usage_out = (chunk.get("usage") or {}).get("output_tokens")
        rec.update(ok=True, ttft_s=ttft, e2e_s=time.perf_counter() - t0,
                   prompt_tokens=usage_in, completion_tokens=usage_out)
        return rec
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


def sglang_request(url: str, prompt: str, timeout: float) -> dict:
    """OpenAI /v1/chat/completions streaming request against a sglang server."""
    payload = {
        "model": SGLANG_MODEL_ID,
        "max_tokens": OUT_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }
    rec = {"ok": False}
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload,
                             timeout=timeout, stream=True)
        rec["http_status"] = resp.status_code
        if resp.status_code != 200:
            rec["error"] = resp.text[:300]
            return rec
        ttft = None
        usage_in = None
        usage_out = None
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage:
                if usage.get("prompt_tokens") is not None:
                    usage_in = usage["prompt_tokens"]
                if usage.get("completion_tokens") is not None:
                    usage_out = usage["completion_tokens"]
            choices = chunk.get("choices") or []
            if choices and ttft is None:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    ttft = time.perf_counter() - t0
        rec.update(ok=True, ttft_s=ttft, e2e_s=time.perf_counter() - t0,
                   prompt_tokens=usage_in, completion_tokens=usage_out)
        return rec
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


def run_point(url, tokenizer, ctx, conc, n, timeout, request_fn) -> dict:
    run_id = uuid.uuid4().hex[:8]
    prompts = [build_prompt(tokenizer, ctx, f"{run_id}-g-{i:04d}") for i in range(n)]
    t0 = time.perf_counter()
    if conc == 1:
        recs = [request_fn(url, p, timeout) for p in prompts]
    else:
        with ThreadPoolExecutor(max_workers=conc) as pool:
            recs = list(pool.map(lambda p: request_fn(url, p, timeout), prompts))
    wall = time.perf_counter() - t0

    ok = [r for r in recs if r.get("ok") and r.get("ttft_s")]
    failed = len(recs) - len(ok)
    if not ok:
        return {"ctx": ctx, "c": conc, "n": n, "failed": failed,
                "error": recs[0].get("error", "no successful request")}
    prompt_total = sum(r["prompt_tokens"] or 0 for r in ok)
    max_ttft = max(r["ttft_s"] for r in ok)
    decodes = []
    for r in ok:
        out = r.get("completion_tokens")
        if out and r["e2e_s"] > r["ttft_s"]:
            decodes.append((out - 1) / (r["e2e_s"] - r["ttft_s"]))
    return {
        "ctx": ctx, "c": conc, "n": n, "failed": failed,
        "prefill_tps": prompt_total / max_ttft if max_ttft else None,
        "decode_tps_mean": sum(decodes) / len(decodes) if decodes else None,
        "prompt_tokens_total": prompt_total,
        "max_ttft_s": max_ttft,
        "wall_s": wall,
    }


def gpu_free_mib(gpu: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu)],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    return int(out)


def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def free_port(base: int) -> int | None:
    for offset in range(9):
        port = base + offset
        if port_free(port):
            return port
    return None


def pick_gpu(requested: int | None) -> int:
    """Requested GPU, else the first idle card among the identical 5090Ds."""
    candidates = [requested] if requested is not None else [5, 1]
    for gpu in candidates:
        if gpu is None:
            continue
        if gpu_free_mib(gpu) <= 2000:
            return gpu
    sys.exit("environment error: no free GPU among "
             f"{[g for g in candidates if g is not None]}")


def wait_gpu_drained(gpu: int, limit_mib: int = 2000, timeout: float = 180.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if gpu_free_mib(gpu) <= limit_mib:
            return
        time.sleep(3)
    sys.exit(f"environment error: GPU {gpu} did not drain after server stop; "
             f"check for stray processes")


def build_artifact(variant: str, gpu: int, log_path: Path) -> Path | None:
    """Convert the HF checkpoint into a .ninfer artifact with the current tree.

    The converter contract is part of the task: tools/convert must accept
    `--model <spark dir> --recipe <official recipe> --out <file>` and register
    the spark_x2_6_17b_bf16 / spark_x2_6_17b_fp8 recipes. GATE_RECIPE_<VARIANT>
    overrides the recipe name for exploratory runs.
    """
    src, recipe = VARIANT_SOURCES[variant]
    recipe = os.environ.get(f"GATE_RECIPE_{variant.upper()}", recipe)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(ARTIFACT_DIR).free
    if free_bytes < 12 * 1024 * 1024 * 1024:
        sys.exit(f"environment error: only {free_bytes >> 30} GiB free for the "
                 f"artifact under {ARTIFACT_DIR}; free some disk first")
    artifact = ARTIFACT_DIR / f"spark_x2_6_17b_{variant}.ninfer"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LD_LIBRARY_PATH"] = NVIDIA_LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    env.pop("CUDA_MODULE_LOADING", None)
    cmd = [sys.executable, "-m", "tools.convert",
           "--model", str(src), "--recipe", recipe,
           "--out", str(artifact), "--cpu-threads", "16"]
    print(f"[gate] converting {variant} (recipe {recipe}, gpu {gpu})", flush=True)
    with open(log_path, "a") as log:
        try:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"[gate] artifact conversion for {variant} timed out; "
                  f"see {log_path}", flush=True)
            return None
    if proc.returncode != 0 or not artifact.exists():
        print(f"[gate] artifact conversion for {variant} failed; see {log_path}",
              flush=True)
        return None
    return artifact


def start_server(gpu: int, port: int, artifact: Path, log_path: Path):
    if not SERVE.exists():
        sys.exit(f"environment error: {SERVE} not built")
    used = gpu_free_mib(gpu)
    if used > 2000:
        sys.exit(f"environment error: GPU {gpu} has {used} MiB in use; "
                 f"free the card or point GATE_GPU at a free one")
    if not port_free(port):
        sys.exit(f"environment error: port {port} is busy")
    toml = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    toml.write(f"""# spark_gate auto-generated serving config.
# kv_dtype and prefill_chunk are deliberately NOT set: they resolve from the
# engine/model defaults, making them legitimate tuning surfaces (change the
# default in the source or artifact config; this judge must not pin them).
# Fixed here are only requirement-scope items: the 1M context, concurrency 16
# (both parity with the frozen sglang baseline configs) and CUDA Graph.
[model]
path = "{artifact}"

[engine]
max_context = 1048576
max_concurrency = 16
context_capacity = "auto"
memory_headroom = "256MiB"
cuda_graph = true

[server]
host = "127.0.0.1"
port = {port}
model_id = "{NINFER_MODEL_ID}"
""")
    toml.close()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LD_LIBRARY_PATH"] = (
        NVIDIA_LIBS + ":" + env.get("LD_LIBRARY_PATH", ""))
    env.pop("CUDA_MODULE_LOADING", None)
    proc = subprocess.Popen(
        [str(SERVE), "--config", toml.name],
        stdout=open(log_path, "a"), stderr=subprocess.STDOUT, env=env)
    deadline = time.time() + 600
    url = f"http://127.0.0.1:{port}"
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[gate] ninfer-serve exited during startup; see {log_path}",
                  flush=True)
            return None, None
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return proc, url
        except Exception:
            pass
        time.sleep(3)
    proc.terminate()
    print(f"[gate] ninfer-serve not healthy in 600s; see {log_path}", flush=True)
    return None, None


def stop_server(proc) -> None:
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        proc.kill()


def measure_sglang(args, tokenizer, matrix) -> int:
    """Baseline mode: measure a live sglang server and write a JSON fragment."""
    url = args.sglang_url
    try:
        if requests.get(f"{url}/health", timeout=5).status_code != 200:
            sys.exit(f"environment error: sglang at {url} is not healthy")
    except Exception as exc:
        sys.exit(f"environment error: cannot reach sglang at {url}: {exc}")
    points = {}
    failed_total = 0
    for ctx, conc, n in matrix:
        point = run_point(url, tokenizer, ctx, conc, n,
                          args.request_timeout, sglang_request)
        key = f"ctx{ctx}_c{conc}"
        points[key] = {
            "sglang_prefill_tps": point.get("prefill_tps"),
            "sglang_decode_tps": point.get("decode_tps_mean"),
            "n": n, "failed": point["failed"],
        }
        failed_total += point["failed"]
        print(f"[measure] {key}: prefill "
              f"{point.get('prefill_tps') and round(point['prefill_tps'])} tok/s "
              f"decode {point.get('decode_tps_mean') and round(point['decode_tps_mean'], 1)} "
              f"tok/s failed={point['failed']}", flush=True)
    fragment = {
        "sglang_url": url,
        "measured_unix": time.time(),
        "points": points,
    }
    out = Path(args.out_file) if args.out_file else \
        Path(tempfile.gettempdir()) / f"sglang_baseline_{args.variants}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fragment, indent=2))
    print(f"[measure] wrote {out}", flush=True)
    return 1 if failed_total else 0


def gate_ninfer(args, tokenizer, matrix, baselines) -> int:
    out_dir = os.environ.get("AUTODEVX_ANALYZE_OUTPUT_DIR")
    base_log = Path(out_dir or tempfile.gettempdir())
    gpu = pick_gpu(args.gpu if args.gpu is not None
                   else int(os.environ["GATE_GPU"]) if os.environ.get("GATE_GPU")
                   else None)
    print(f"[gate] matrix={args.matrix} gpu={gpu} "
          f"variants={args.variants} enforce={args.enforce}", flush=True)
    variants_result = {}
    functional_ok = True
    for variant in args.variants.split(","):
        src, _recipe = VARIANT_SOURCES[variant]
        log_path = base_log / f"gate_server_{variant}.log"
        log_path.write_text("")
        artifact = build_artifact(variant, gpu, log_path)
        if artifact is None:
            return 1
        port = free_port(args.port)
        if port is None:
            sys.exit("environment error: no free port in the gate range")
        proc, url = start_server(gpu, port, artifact, log_path)
        if proc is None:
            return 1
        points = {}
        try:
            for ctx, conc, n in matrix:
                point = run_point(url, tokenizer, ctx, conc, n,
                                  args.request_timeout, ninfer_request)
                key = f"ctx{ctx}_c{conc}"
                base = baselines.get(variant, {}).get("points", {}).get(key)
                if base is None and ((ctx, conc) in DECODE_GATE
                                     or (ctx, conc) in PREFILL_GATE):
                    sys.exit(f"environment error: no frozen baseline for "
                             f"{variant} {key}")
                prefill_ratio = (point.get("prefill_tps")
                                 / base["sglang_prefill_tps"]
                                 if point.get("prefill_tps")
                                 and base and base.get("sglang_prefill_tps")
                                 else None)
                decode_ratio = (point.get("decode_tps_mean")
                                / base["sglang_decode_tps"]
                                if point.get("decode_tps_mean")
                                and base and base.get("sglang_decode_tps")
                                else None)
                points[key] = {**point, "prefill_ratio": prefill_ratio,
                                "decode_ratio": decode_ratio,
                                "gates_decode": (ctx, conc) in DECODE_GATE,
                                "gates_prefill": (ctx, conc) in PREFILL_GATE}
                print(f"[gate] {variant} {key}: prefill "
                      f"{point.get('prefill_tps') and round(point['prefill_tps'])} tok/s "
                      f"ratio={prefill_ratio and round(prefill_ratio, 3)} "
                      f"decode {point.get('decode_tps_mean') and round(point['decode_tps_mean'], 1)} tok/s "
                      f"ratio={decode_ratio and round(decode_ratio, 3)} "
                      f"failed={point['failed']}", flush=True)
                functional_ok = functional_ok and point["failed"] == 0
        finally:
            stop_server(proc)
            wait_gpu_drained(gpu)
        prefill_ratios = [p["prefill_ratio"] for p in points.values()
                          if p["gates_prefill"] and p["prefill_ratio"]]
        decode_ratios = [p["decode_ratio"] for p in points.values()
                         if p["gates_decode"] and p["decode_ratio"]]
        variants_result[variant] = {
            "prefill_ratio_min": min(prefill_ratios) if prefill_ratios else None,
            "decode_ratio_min": min(decode_ratios) if decode_ratios else None,
            "points": points,
        }
        print(f"[gate] {variant}: prefill_ratio_min="
              f"{variants_result[variant]['prefill_ratio_min'] and round(variants_result[variant]['prefill_ratio_min'], 3)} "
              f"decode_ratio_min="
              f"{variants_result[variant]['decode_ratio_min'] and round(variants_result[variant]['decode_ratio_min'], 3)}",
              flush=True)

    mins = [min(v["prefill_ratio_min"] or float("inf"),
                v["decode_ratio_min"] or float("inf"))
            for v in variants_result.values()]
    overall = min(mins) if mins and all(m != float("inf") for m in mins) else None
    prefill_min = min((v["prefill_ratio_min"] for v in variants_result.values()
                       if v["prefill_ratio_min"] is not None), default=None)
    decode_min = min((v["decode_ratio_min"] for v in variants_result.values()
                      if v["decode_ratio_min"] is not None), default=None)
    metrics = {
        "overall_ratio_min": overall,
        "prefill_ratio_min": prefill_min,
        "decode_ratio_min": decode_min,
        "functional_ok": functional_ok,
        "variants": variants_result,
    }
    summary = {k: v for k, v in metrics.items() if k != "variants"}
    print(json.dumps(summary, indent=2), flush=True)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (Path(__file__).resolve().parent / "gate_last_result_spark.json").write_text(
        json.dumps(metrics, indent=2))

    if not functional_ok:
        return 1
    if args.enforce:
        for name, value in (("overall_ratio_min", overall),
                            ("prefill_ratio_min", prefill_min),
                            ("decode_ratio_min", decode_min)):
            if value is None or value < RATIO_TARGET:
                print(f"[gate] enforce: {name} below {RATIO_TARGET}", flush=True)
                return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=("smoke", "quick", "full"),
                        default=os.environ.get("GATE_MATRIX", "full"))
    parser.add_argument("--variants", default=os.environ.get("GATE_VARIANT", "bf16,fp8"),
                        help="comma-separated subset of bf16,fp8")
    parser.add_argument("--gpu", type=int, default=None,
                        help="explicit GPU; default GATE_GPU or first free card")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("GATE_PORT", "18082")))
    parser.add_argument("--enforce", action="store_true",
                        help="fail (exit 1) when the 2.0x targets are not met; "
                             "without it only request failures fail")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--measure-sglang", action="store_true",
                        help="baseline mode: measure the live sglang server at "
                             "--sglang-url instead of gating ninfer")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30012")
    parser.add_argument("--out-file", default=None,
                        help="fragment path for --measure-sglang output")
    args = parser.parse_args()

    bad = [v for v in args.variants.split(",") if v not in VARIANTS]
    if bad:
        sys.exit(f"environment error: unknown variants {bad}")
    matrices = {"smoke": SMOKE_MATRIX, "quick": QUICK_MATRIX,
                "full": FULL_MATRIX}
    matrix = matrices[args.matrix]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR,
                                              trust_remote_code=True)

    if args.measure_sglang:
        return measure_sglang(args, tokenizer, matrix)

    if not BASELINES.exists():
        sys.exit(f"environment error: {BASELINES} not found")
    baselines = json.loads(BASELINES.read_text())
    return gate_ninfer(args, tokenizer, matrix, baselines)


if __name__ == "__main__":
    sys.exit(main())
