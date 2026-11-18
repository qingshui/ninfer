import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import fig_control
import fig_flight
import fig_gdn
import fig_gpu
import fig_latency
import fig_moe
import fig_recipe
import fig_retention
import fig_stages
import fig_throughput

W0 = os.environ.get("REPORT_DATA", ROOT + "/data") + "/"
OUT = os.environ.get("REPORT_OUT", ROOT)


def main():
    d = json.load(open(W0 + "data.json"))
    fig = os.path.join(OUT, "figures")
    tab = os.path.join(OUT, "tables")
    os.makedirs(fig, exist_ok=True)
    os.makedirs(tab, exist_ok=True)
    jobs = [
        ("fig1", lambda: fig_throughput.build(d, fig)),
        ("fig3", lambda: fig_stages.build(d, fig)),
        ("fig4", lambda: fig_flight.build(d, fig)),
        ("fig2", lambda: fig_latency.build(d, fig)),
        ("fig5", lambda: fig_gpu.build(d, fig)),
        ("fig9", lambda: fig_recipe.build(d, fig)),
        ("fig7", lambda: fig_retention.build(fig)),
        ("fig6", lambda: fig_control.build(d, fig)),
        ("fig10", lambda: fig_gdn.build(fig)),
        ("fig8", lambda: fig_moe.build(fig)),
    ]
    for name, fn in jobs:
        r = fn()
        open(os.path.join(tab, name + ".md"), "w", encoding="utf-8").write(r[2] + "\n")
        print(name, os.path.basename(r[0]), os.path.basename(r[1]))


if __name__ == "__main__":
    main()
