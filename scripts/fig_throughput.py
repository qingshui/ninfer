import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import json
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, hbars, save, md_table

W0 = ROOT + "/data/"


def build(d, outdir):
    head, pa, fc, fo = d["head"], d["phase_a"], d["final_c"], d["final_official"]
    rows = [
        dict(label="HEAD 3e31d5a4", value=head["mean_request_throughput"], fill="muted", text="2.21",
             tip=f"HEAD, re-measured, 1 run: {head['mean_request_throughput']:.2f} req/s"),
        dict(label="After phase A", value=pa["mean_request_throughput"], fill="b250", text=f"{pa['mean_request_throughput']:.2f}",
             tip=f"After phase A, mean of {pa['n']} runs: {pa['mean_request_throughput']:.2f} req/s"),
        dict(label="sglang reference", value=d["sglang"]["request_throughput"], fill="s2",
             text=f"{d['sglang']['request_throughput']:.2f}", tip="sglang baseline, recorded run"),
        dict(label="Final, mean of 3 runs", value=fc["mean_request_throughput"], fill="s1",
             text=f"{fc['mean_request_throughput']:.2f}", tip=f"Final, canonical artifact, mean of {fc['n']} runs"),
        dict(label="Final, official run", value=fo["request_throughput"], fill="s1", bold=True,
             text=f"{fo['request_throughput']:.2f}", tip="Final, official ninfer_e2e.sh run"),
    ]
    W = 880
    sc = Scene(W, 330, "Serving throughput before and after",
               "Request throughput of the locked harness for HEAD, the build after phase A, the sglang reference and the final build")
    y = header(sc, "Request throughput of the locked harness (2000 requests, c=130, n=4)",
               "req/s, higher is better. Same client, dataset and serving flags for every bar. The vertical line is the 91.5 req/s target.")
    y = hbars(sc, 24, y + 22, W - 48, rows, vmax=100, refs=[(d["target"], f"target {d['target']}")],
              bar_h=20, pitch=34, label_w=170, dec=2, xlabel="req/s")
    y = footer(sc, y + 8, [
        "HEAD was re-measured for this report with the same locked client (905 s for 2000 requests); the pipeline's own recorded run of HEAD was 2.14 req/s.",
        "After phase A and Final are means over 3 runs of the same harness; the official run is one execution of the locked ninfer_e2e.sh."])
    sc.h = y + 14
    svg, png = save(sc, outdir, "fig1_throughput")
    table = md_table(["build", "req/s", "mean E2EL (s)", "mean TTFT (ms)", "p99 TTFT (ms)", "runs"], [
        ["HEAD 3e31d5a4 (re-measured)", f"{head['mean_request_throughput']:.2f}", f"{head['mean_e2el_ms'] / 1000:.1f}",
         f"{head['mean_ttft_ms']:.0f}", f"{head['mean_p99_ttft_ms']:.0f}", head["n"]],
        ["After phase A", f"{pa['mean_request_throughput']:.2f}", f"{pa['mean_e2el_ms'] / 1000:.2f}",
         f"{pa['mean_ttft_ms']:.0f}", f"{pa['mean_p99_ttft_ms']:.0f}", pa["n"]],
        ["sglang reference", f"{d['sglang']['request_throughput']:.2f}", f"{d['sglang']['mean_e2el_ms'] / 1000:.2f}",
         f"{d['sglang']['mean_ttft_ms']:.0f}", f"{d['sglang']['p99_ttft_ms']:.0f}", 1],
        ["Final, canonical artifact", f"{fc['mean_request_throughput']:.2f}", f"{fc['mean_e2el_ms'] / 1000:.2f}",
         f"{fc['mean_ttft_ms']:.0f}", f"{fc['mean_p99_ttft_ms']:.0f}", fc["n"]],
        ["Final, official run", f"{fo['request_throughput']:.2f}", f"{fo['mean_e2el_ms'] / 1000:.2f}",
         f"{fo['mean_ttft_ms']:.0f}", f"{fo['p99_ttft_ms']:.0f}", 1],
    ])
    return svg, png, table


if __name__ == "__main__":
    d = json.load(open(W0 + "data.json"))
    r = build(d, "/tmp/figtest")
    print(r[2])
