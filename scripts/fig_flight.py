import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, hbars, save, fmt, md_table, legend

SYS = [("HEAD", "muted"), ("After phase A", "b250"), ("Final", "s1")]


def build(d, outdir):
    tl = [d["head"]["telemetry"], d["phase_a"]["telemetry"],
          (d["final_c"].get("telemetry") or d["final_official"]["telemetry"])]
    panels = [
        ("Prefill tokens per flight", [t["prefill_per_flight"] for t in tl], "", 0),
        ("Decode rows per flight", [t["decode_rows_per_flight"] for t in tl], "", 1),
        ("Mean flight time (ms)", [5000.0 / t["flights"] for t in tl], "", 0),
        ("Prefill throughput (k tok/s)", [t["prefill_ktok_s"] for t in tl], "", 1),
    ]
    W, PW, GAP = 880, 410, 30
    sc = Scene(W, 420, "Flight shape of the engine before and after",
               "Median engine telemetry over the steady state of the locked harness for HEAD, after phase A and the final build")
    y0 = header(sc, "What one flight carries (engine telemetry, steady state)",
                "Median of the 5 s throughput log lines while 100 or more requests are running.")
    y0 = legend(sc, 24, y0 + 8, [(n, c) for n, c in SYS], 800)
    rows_md = []
    bottoms = []
    for i, (title, vals, unit, dec) in enumerate(panels):
        px = 24 + (i % 2) * (PW + GAP)
        py = y0 + 14 + (i // 2) * 150
        sc.text(px, py, title, 12, "ink1", bold=True)
        rows = [dict(label=SYS[j][0], value=vals[j], fill=SYS[j][1], bold=(j == 2),
                     tip=f"{SYS[j][0]}: {fmt(vals[j], dec)}{unit}", text=fmt(vals[j], dec) + unit)
                for j in range(3)]
        b = hbars(sc, px, py + 12, PW, rows, dec=dec, bar_h=16, pitch=28, label_w=96, tick_n=4)
        bottoms.append(b)
        rows_md.append([title] + [fmt(v, dec) for v in vals])
    yb = max(bottoms)
    yb = footer(sc, yb + 14, ["Flight time is 5 s divided by the flights logged in the interval. HEAD and final were re-measured for this report;",
                              "after phase A comes from the serve logs of its three recorded runs (59.41, 59.66 and 59.28 req/s)."])
    sc.h = yb + 14
    svg, png = save(sc, outdir, "fig4_flight_shape")
    table = md_table(["metric", "HEAD", "After phase A", "Final"], rows_md)
    return svg, png, table


if __name__ == "__main__":
    import json
    d = json.load(open(ROOT + "/data/data.json"))
    print(build(d, "/tmp/figtest")[2])
