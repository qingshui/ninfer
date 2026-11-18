import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import json
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, legend, stacked_hbars, save, fmt, md_table

W0 = ROOT + "/data/"
ORDER = ["MoE experts", "Dense linear", "GDN decode (state update)", "GDN prefill (varlen/chunked)",
         "Full attention", "MoE routing and shared expert", "other"]
NAMES = ["MoE routed experts", "Dense linear (GEMM)", "GDN decode state update", "GDN prefill (varlen/chunked)",
         "Full attention", "MoE routing and shared expert", "Norm, sampling, other"]
TOK = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
SERIES = list(zip(NAMES, TOK))


def per_request(p, head_rate=None):
    cats = p["categories"]
    total = sum(v["ms"] for v in cats.values())
    merged = {k: 0.0 for k in ORDER}
    for k, v in cats.items():
        merged[k if k in merged else "other"] += v["ms"]
    if head_rate is None:
        n = p["requests"]["n"]
        return {k: v / n for k, v in merged.items()}, p["requests"]["n"]
    scale = p["util"] * 1000.0 / head_rate / total
    return {k: v * scale for k, v in merged.items()}, None


def build(d, outdir):
    P = {k: json.load(open(W0 + f"prof/{k}.json")) for k in ("head", "final_o", "final_c")}
    head_rate = d["head"]["mean_request_throughput"]
    pr = {"head": per_request(P["head"], head_rate)[0], "final_o": per_request(P["final_o"])[0],
          "final_c": per_request(P["final_c"])[0]}
    tot = {k: sum(v.values()) for k, v in pr.items()}

    def row(label, key, bold=False):
        segs = [(i, pr[key][c], f"{NAMES[i]}: {pr[key][c]:.2f} ms per request") for i, c in enumerate(ORDER)]
        labs = {i: (f"{pr[key][c]:.0f}" if tot[key] > 100 else f"{pr[key][c]:.1f}") for i, c in enumerate(ORDER)}
        return dict(label=label, segs=segs, labels=labs)

    W = 880
    sc = Scene(W, 600, "GPU kernel time per request by category",
               "Sum of GPU kernel time per completed request for HEAD and the final build, from nsys profiles of the locked workload")
    y = header(sc, "GPU kernel time per request (ms), by category",
               "Same workload (2000 requests, c=130, n=4). Kernel durations from nsys, node-level CUDA graph trace.")
    y = legend(sc, 24, y + 8, SERIES, 830)
    sc.text(24, y + 14, "A. HEAD against the final build (full scale)", 12, "ink1", bold=True)
    y = stacked_hbars(sc, 24, y + 24, W - 48, [row("HEAD", "head"), row("Final", "final_c")],
                      SERIES, vmax=400, unit=" ms", pitch=44, label_w=70,
                      total_fmt=lambda v: f"{v:,.1f}" if v < 100 else f"{v:,.0f}")
    sc.text(24, y + 16, "B. Zoom on the final build, original artifact against canonical artifact", 12, "ink1", bold=True)
    y = stacked_hbars(sc, 24, y + 26, W - 48, [row("Final, original artifact", "final_o"), row("Final, canonical artifact", "final_c")],
                      SERIES, vmax=12, unit=" ms", pitch=44, label_w=170,
                      total_fmt=lambda v: f"{v:,.1f}")
    nh = P["head"]["requests"]["n"]
    y = footer(sc, y + 10, [
        f"HEAD: only {nh} requests finished inside its 10 s window, so its per-request values use the run's measured {head_rate:.2f} req/s and the window's GPU-busy",
        f"fraction ({P['head']['util']:.3f}); the final builds divide by requests finished in their 5 s windows ({P['final_c']['requests']['n']} canonical, {P['final_o']['requests']['n']} original).",
        "Profiled runs are slower than unprofiled ones. Segment labels are ms per request and appear only where they fit."])
    sc.h = y + 14
    svg, png = save(sc, outdir, "fig5_gpu_time")
    rows = []
    for i, c in enumerate(ORDER):
        rows.append([NAMES[i]] + [f"{pr[k][c]:.2f}" for k in ("head", "final_o", "final_c")] +
                    [f"{pr['head'][c] / pr['final_c'][c]:.0f}x" if pr['final_c'][c] > 0 else "-"])
    rows.append(["Total GPU kernel time per request"] + [f"{tot[k]:.2f}" for k in ("head", "final_o", "final_c")] +
                [f"{tot['head'] / tot['final_c']:.0f}x"])
    rows.append(["GPU idle share of the window"] + [f"{100 * (1 - P[k]['util']):.1f}%" for k in ("head", "final_o", "final_c")] + ["-"])
    table = md_table(["category (ms per request)", "HEAD", "Final, original artifact", "Final, canonical artifact", "HEAD / final"], rows)
    return svg, png, table, pr, tot


if __name__ == "__main__":
    d = json.load(open(W0 + "data.json"))
    r = build(d, "/tmp/figtest")
    print(r[2])
