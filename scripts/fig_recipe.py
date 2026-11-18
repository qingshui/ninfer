import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, hbars, grouped_hbars, save, fmt, md_table, legend

# Verified against the two artifact conversion reports and the shapes in the recipe.
GB = 1e9
BYTES = {
    "GDN input projection, 36 layers (20480 x 3072)": (36 * 125_829_120, 36 * 62_996_480),
    "GDN output projection, 36 layers (3072 x 8192)": (36 * 50_331_648, 36 * 25_178_112),
    "Output head (248320 x 3072)": (810_516_480, 1_525_678_080),
}
PAYLOAD = (74_757_843_968, 72_305_503_232)


def build(d, outdir, prof=None):
    W = 880
    sc = Scene(W, 560, "FP8 GDN projections and BF16 head: artifact and throughput",
               "Weight bytes before and after the recipe change, and throughput of the same final build on the two artifacts")
    y = header(sc, "Recipe change: GDN projections in FP8, output head in BF16",
               f"Artifact payload {PAYLOAD[0] / GB:.2f} GB to {PAYLOAD[1] / GB:.2f} GB ({(PAYLOAD[1] - PAYLOAD[0]) / GB:+.2f} GB). Weight bytes by tensor group:")
    y = legend(sc, 24, y + 8, [("Before (HEAD recipe)", "muted"), ("After (qwen3_5_122b_nvfp4 now)", "s1")], 800)
    cats = []
    for k, (b, a) in BYTES.items():
        cats.append(dict(label=k, values=[b / GB, a / GB],
                         tips=[f"{k}: before {b:,} B", f"{k}: after {a:,} B"],
                         texts=[f"{b / GB:.2f} GB", f"{a / GB:.2f} GB ({(a - b) / GB:+.2f})"]))
    y = grouped_hbars(sc, 24, y + 8, W - 48, cats, [("before", "muted"), ("after", "s1")], vmax=5.5,
                      dec=2, pitch=50, label_w=290)
    sc.text(24, y + 14, "Same final build, same client: throughput on the two artifacts", 12, "ink1", bold=True)
    fo, fc = d["final_o"], d["final_c"]
    rows = [dict(label="Original artifact (BF16 GDN, Q8 head)", value=fo["mean_request_throughput"], fill="muted",
                 text=f"{fo['mean_request_throughput']:.1f} req/s", tip="mean of " + ", ".join(f"{r['request_throughput']:.2f}" for r in fo["runs"])),
            dict(label="Canonical artifact (FP8 GDN, BF16 head)", value=fc["mean_request_throughput"], fill="s1", bold=True,
                 text=f"{fc['mean_request_throughput']:.1f} req/s", tip="mean of " + ", ".join(f"{r['request_throughput']:.2f}" for r in fc["runs"]))]
    y = hbars(sc, 24, y + 24, W - 48, rows, vmax=100, label_w=290, bar_h=18, pitch=32)
    gain = fc["mean_request_throughput"] / fo["mean_request_throughput"] - 1
    y = footer(sc, y + 12, [f"The canonical artifact is {100 * gain:.1f}% faster on this workload. Runs: original " +
                            ", ".join(f"{r['request_throughput']:.2f}" for r in fo["runs"]) + "; canonical " +
                            ", ".join(f"{r['request_throughput']:.2f}" for r in fc["runs"]) + " req/s."])
    sc.h = y + 14
    svg, png = save(sc, outdir, "fig9_recipe")
    tb = md_table(["tensor group", "before (B)", "after (B)", "delta (B)"],
                  [[k, f"{b:,}", f"{a:,}", f"{a - b:+,}"] for k, (b, a) in BYTES.items()] +
                  [["artifact payload (conversion report)", f"{PAYLOAD[0]:,}", f"{PAYLOAD[1]:,}", f"{PAYLOAD[1] - PAYLOAD[0]:+,}"]])
    return svg, png, tb


if __name__ == "__main__":
    import json
    d = json.load(open(ROOT + "/data/data.json"))
    print(build(d, "/tmp/figtest")[2])
