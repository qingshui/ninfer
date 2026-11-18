"""Collect measured data for the report into data.json."""
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import telemetry as T

W = "/home/users/humingqing/work/ninfer/.agent-tmp/qwen122b-report"
B = "/raid0/models/ninfer/bench_results/"
KEYS = ["request_throughput", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "mean_e2el_ms",
        "duration_s", "completed", "failed"]


def summ(path):
    s = json.load(open(path))["summary"]
    return {k: s[k] for k in KEYS}


def group(paths, logs=()):
    runs = [summ(p) for p in paths]
    out = {"runs": runs, "n": len(runs)}
    if runs:
        for k in KEYS:
            out["mean_" + k if not k.startswith("mean_") else k] = st.mean(r[k] for r in runs)
    recs = []
    for lg in logs:
        recs += T.steady(T.parse(lg))
    if recs:
        out["telemetry"] = T.summarize(recs)
    return out


def first(globpat):
    return sorted(glob.glob(globpat))


def main():
    d = {}
    d["sglang"] = summ("/sgl-workspace/load_test/results_v2_ts/t1/benchmark.json")
    d["target"] = 91.5
    ph = [B + f"benchmark_20261118_{t}.json" for t in ("001541", "001716", "002120")]
    pl = [B + f"serve_20261118_{t}.log" for t in ("001541", "001716", "002120")]
    d["phase_a"] = group(ph, pl)
    d["pipeline_head_recorded"] = summ(B + "benchmark_20261117_000202.json")
    official = B + "benchmark_20261118_123503.json"
    d["final_official"] = summ(official)
    d["final_official"]["telemetry"] = T.summarize(T.steady(T.parse(B + "serve_20261118_123503.log")))
    for tag, pat in (("head", "head_*"), ("final_c", "final_c*"), ("final_o", "final_o*")):
        js, lg = [], []
        for dd in first(f"{W}/ab/runs/{pat}"):
            js += first(dd + "/benchmark_*.json")
            lg += first(dd + "/serve_*.log")
        d[tag] = group(js, lg)
    json.dump(d, open(W + "/data.json", "w"), indent=1)
    for k in ("sglang", "phase_a", "head", "final_c", "final_o", "final_official"):
        v = d[k]
        if "runs" in v:
            print(k, [round(r["request_throughput"], 2) for r in v["runs"]])
        else:
            print(k, round(v["request_throughput"], 2))
        tl = v.get("telemetry")
        if tl:
            print("   ", {a: round(b, 1) for a, b in tl.items() if a in
                          ("flights", "prefill_per_flight", "decode_rows_per_flight", "host_pct", "gpu_busy_pct", "prefill_ktok_s")})


if __name__ == "__main__":
    main()
