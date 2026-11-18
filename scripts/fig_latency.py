import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, hbars, save, fmt, md_table, legend

SYS = [("sglang", "s2"), ("After phase A", "b250"), ("Final", "s1")]


def build(d, outdir):
    sg, pa = d["sglang"], d["phase_a"]
    fn = d["final_c"] if d["final_c"].get("runs") else None
    def final(k):
        return fn["mean_" + k] if fn and k.startswith("p99") else (fn[k] if fn else d["final_official"][k])
    def get(system, k):
        if system == 0:
            return sg[k]
        if system == 1:
            return pa["mean_" + k] if k.startswith("p99") or not k.startswith("mean") else pa[k]
        return final(k)
    panels = [("Mean end-to-end latency (s)", "mean_e2el_ms", 1000.0, 2),
              ("Mean time to first token (ms)", "mean_ttft_ms", 1.0, 0),
              ("p99 time to first token (ms)", "p99_ttft_ms", 1.0, 0),
              ("Mean time per output token (ms)", "mean_tpot_ms", 1.0, 1)]
    W, PW, GAP = 880, 410, 30
    sc = Scene(W, 420, "Latency of the locked harness",
               "Mean and p99 latency of sglang, the build after phase A and the final build")
    y0 = header(sc, "Latency at concurrency 130 (n=4, max_tokens 15)",
                "Lower is better. With a fixed client, throughput is about 130 divided by the mean end-to-end latency.")
    y0 = legend(sc, 24, y0 + 8, SYS, 800)
    rows_md, bottoms = [], []
    for i, (title, key, div, dec) in enumerate(panels):
        px = 24 + (i % 2) * (PW + GAP)
        py = y0 + 14 + (i // 2) * 150
        vals = [get(j, key) / div for j in range(3)]
        sc.text(px, py, title, 12, "ink1", bold=True)
        rows = [dict(label=SYS[j][0], value=vals[j], fill=SYS[j][1], bold=(j == 2),
                     tip=f"{SYS[j][0]}: {fmt(vals[j], dec)}", text=fmt(vals[j], dec)) for j in range(3)]
        bottoms.append(hbars(sc, px, py + 12, PW, rows, dec=dec, bar_h=16, pitch=28, label_w=96, tick_n=4))
        rows_md.append([title] + [fmt(v, dec) for v in vals])
    yb = max(bottoms)
    h = d["head"]["runs"][0]
    yb = footer(sc, yb + 14, [
        f"HEAD, re-measured, is off scale: mean end-to-end {h['mean_e2el_ms'] / 1000:.1f} s, mean TTFT {h['mean_ttft_ms'] / 1000:.2f} s, "
        f"p99 TTFT {h['p99_ttft_ms'] / 1000:.1f} s, mean TPOT {h['mean_tpot_ms'] / 1000:.2f} s.",
        "Final and after-phase-A bars are the mean of their runs; the sglang bars are the recorded baseline run."])
    sc.h = yb + 14
    svg, png = save(sc, outdir, "fig2_latency")
    return svg, png, md_table(["metric", "sglang", "After phase A", "Final"], rows_md)


if __name__ == "__main__":
    import json
    d = json.load(open(ROOT + "/data/data.json"))
    print(build(d, "/tmp/figtest")[2])
