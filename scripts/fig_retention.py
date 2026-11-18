import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import json
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, legend, hbars, save, md_table

STATIC = [
    ("cap 290", 57.78, 434, 2228), ("cap 330", 58.99, 417, 2184),
    ("cap 380", 46.15, 595, 2793), ("cap 420", 38.82, 911, 3323),
    ("cap 480", 33.65, 1274, 3813), ("cap 700", 32.86, 1261, 3918),
]
CTRL = [
    ("free pool >= 12.5% (1/8)", 57.65, None, None), ("free pool >= 10% (1/10)", 57.62, None, None),
    ("free pool >= 8.3% (1/12)", 57.69, None, None), ("free pool >= 6.25% (1/16)", 58.10, None, None),
    ("free pool >= 5% (1/20, default)", 58.49, 427, 2200), ("free pool >= 4.2% (1/24)", 59.18, 421, 2175),
]
SER = [("Static retained-point cap (superseded)", "muted"), ("Pool-headroom controller", "b250")]


def panel(sc, y, title, rows, fill):
    sc.text(24, y + 14, title, 12, "ink1", bold=True)
    items = [dict(label=r[0], value=r[1], fill=fill, text=f"{r[1]:.2f}", bold=r[0].endswith("default)"),
                tip=f"{r[0]}: {r[1]:.2f} req/s") for r in rows]
    return hbars(sc, 24, y + 24, 832, items, vmax=70, bar_h=16, pitch=24, label_w=250, dec=2)


def build(outdir):
    W = 880
    sc = Scene(W, 560, "Static retention cap against the pool-headroom controller",
               "Request throughput of the locked harness for static retained-point caps and for the pool-headroom controller on the phase A tree")
    y = header(sc, "Retention control: static cap against pool-headroom controller (req/s)",
               "Single runs of the locked harness (c=130, n=4) on the phase A tree, original artifact. Higher is better.")
    y = legend(sc, 24, y + 8, SER, 830)
    y = panel(sc, y + 6, "A. Static cap on retained checkpoints: a cliff once the pool fills", STATIC, "muted")
    y = panel(sc, y + 4, "B. Controller: trim the least valuable checkpoints when free pool bytes fall below a share of capacity", CTRL, "b250")
    y = footer(sc, y + 6, [
        "Run-to-run spread in phase A was about 0.5 req/s, so differences among the controller rows are not resolved by single runs.",
        "Lineage-first eviction was on for caps 290 and 330 (cap 330 without it: 57.33); the notes do not record it for caps 380 to 700.",
        "The 1/24 row had persistent skipped checkpoints and was not chosen; 1/20 had none in steady state. Rows are runs 23 to 35."])
    sc.h = y + 14
    svg, png = save(sc, outdir, "fig7_retention")
    rows = [[r[0], f"{r[1]:.2f}", r[2] if r[2] else "-", r[3] if r[3] else "-"] for r in STATIC + CTRL]
    table = md_table(["setting", "req/s", "mean TTFT (ms)", "mean E2EL (ms)"], rows)
    return svg, png, table


if __name__ == "__main__":
    print(build("/tmp/figtest")[2])
