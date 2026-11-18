import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import json
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, legend, hbars, save, md_table

W0 = ROOT + "/data/"
PH = {"head": ("HEAD", "muted"), "A": ("Phase A, engine and control plane", "b250"), "B": ("Phase B, kernels and artifact", "s1")}

STAGES = [
    ("HEAD 3e31d5a4", 2.21, "head", "2.21", "re-measured, 1 run (pipeline recorded 2.14)", "Pipeline baseline"),
    ("A1 Engine rework, no helper lane", 17.32, "A", "~17.3", "run3 on 2000 requests, from notes", "fork_batch, in-flight capture, no quiescent iterations; pool-full admission stalls remain"),
    ("A2 Retained-point cap 176", 36.13, "A", "~36.1", "cap sweep, from notes", "Cap on retained checkpoints removes the admission stalls"),
    ("A3 MoE down epilogue, sampler", 48.07, "A", "48.07", "run13, 1 run", "red.global.add.v4 epilogue, sampler rewrite, slice cap 8192, dense control gate/up"),
    ("A5 Attention split heuristic", 49.95, "A", "49.95", "run17, 1 run", "kMinSplitTarget=3"),
    ("A6 Conv varlen rewrite", 50.98, "A", "50.98", "run18, 1 run", "causal conv varlen kernel"),
    ("A3/A7 MoE gather, batched completions", 52.01, "A", "52.01", "run21, 1 run", "single-thread gather, 8-warp epilogue, completion batching, batched reserve"),
    ("A5 fmha K/V loader, envelope", 55.99, "A", "55.99", "run22, 1 run", "batched K/V loads, planner contract fix, per-segment envelope"),
    ("A2/A7 Headroom controller, host path", 59.45, "A", "59.45", "3 runs: 59.41, 59.66, 59.28", "pool-headroom retention, lineage-first eviction, cheap exposure accounting"),
    ("B1 MoE gate/up gather_input", 65.59, "B", "65.59", "1 run", "contiguous TMA loads, silu_approx, shared-reciprocal quantizer"),
    ("B2 MoE ordered combine", 67.40, "B", "67.40", "4 runs: 67.19 to 67.60", "fixed-order fp32 contributions, deterministic"),
    ("B3 BF16 TileN=256, Add epilogue", 70.60, "B", "70.60", "2 runs: 70.79, 70.40", "256-token tile, vectorized residual epilogue"),
    ("B4 Conv, prepare, rmsnorm tuning", 72.31, "B", "72.31", "3 runs: 71.94 to 72.54", "launch bounds, ftz silu, router columns"),
    ("B5 Grouped ragged attention", 75.75, "B", "75.75", "3 runs: 75.49 to 76.17", "one launch for all prefill segments"),
    ("B5 Grouped FP8 KV write", 76.09, "B", "76.09", "3 runs: 76.03 to 76.12", "one launch writes all KV segments"),
    ("B6 FP8 GDN projections", 83.95, "B", "83.95", "3 runs: 83.66 to 84.24", "tcgen05 E4M3 W8A8 for GDN in/out projections"),
    ("B7 BF16 output head", 85.91, "B", "85.91", "2 runs: 85.69, 86.13", "lm_head Q8 to BF16 tcgen05"),
    ("B8 In-flight checkpoint capture", 87.17, "B", "~87.2", "runs 87.11 to 87.28", "checkpoints ride the final prefill window, one unit per request"),
    ("B9 Register-state GDN step", 90.71, "B", "90.71", "1 run", "state held in MMA accumulators, 1 sync per chunk"),
    ("B10 Recurrent route, grouped prepare", 94.21, "B", "94.21", "3 runs: 93.77 to 94.49", "warp-shared loads in recurrent rows, 4 value heads per prepare block"),
    ("Official run, final sources", 95.26, "B", "95.26", "ninfer_e2e.sh, 1 run", "canonical artifact, diagnostics removed"),
]


def build(d, outdir):
    rows = []
    for lab, v, ph, txt, basis, what in STAGES:
        rows.append(dict(label=lab, value=v, fill=PH[ph][1], text=txt, bold=(lab.startswith("Official")),
                         tip=f"{lab}: {txt} req/s. {basis}. {what}"))
    W = 880
    sc = Scene(W, 800, "Throughput after each stage of the optimization",
               "Request throughput of the locked harness after each development stage, from HEAD to the official run")
    y = header(sc, "Throughput after each development stage (locked harness, c=130, n=4)",
               "req/s, higher is better. Stages are cumulative, in the order the changes were made.")
    y = legend(sc, 24, y + 8, [PH[k] for k in ("head", "A", "B")], 830)
    y = hbars(sc, 24, y + 22, W - 48, rows, vmax=100,
              refs=[(d["sglang"]["request_throughput"], "sglang 61.0"), (d["target"], f"target {d['target']}")],
              bar_h=16, pitch=24, label_w=260, dec=2, xlabel="req/s")
    y = footer(sc, y + 8, [
        "Values are the mean of the stage's harness runs; ~ marks values recovered from development notes, not from a retained benchmark file.",
        "Several stages bundle more than one change, so a step is not the isolated gain of its label. Phase A runs use the original artifact (BF16 GDN, Q8 head);",
        "B6 switches GDN projections to FP8, B7 the output head to BF16. The 12 diagnostic NVTX ranges were removed only before the official run."])
    sc.h = y + 14
    svg, png = save(sc, outdir, "fig3_stages")
    table = md_table(["stage", "req/s", "runs", "change"], [[s[0], s[3], s[4], s[5]] for s in STAGES])
    return svg, png, table


if __name__ == "__main__":
    d = json.load(open(W0 + "data.json"))
    r = build(d, "/tmp/figtest")
    print(r[2])
