import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import json
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, save, md_table
from viz_flow import row, bracket, strip

W0 = ROOT + "/data/"

BEFORE = [
    (["Helper lane", "prefill", "hidden extra lane:", "n+1 = 5 lanes, n=4"], None,
     "HEAD prefills the prompt on a hidden helper lane, so a group of n samples occupies n+1 lanes"),
    (["Capture", "context operation", "copies the State", "(about 150 MiB)"], "s2",
     "Needs the Session's single context-operation slot and an iteration with no flight running"),
    (["Fork", "context operation", "one State copy", "per sample"], "s2",
     "Each sample receives its own copy of the parent State"),
    (["Bind lanes", "context operation", "at a quiescent", "iteration"], "s2",
     "Binding also waits for the context-operation slot"),
    (["Samples decode", "n lanes join the", "running batch"], None, "Decode starts after the chain"),
]
AFTER = [
    (["Parent prefill", "one of the n sample", "lanes, no helper", "lane"], None,
     "The parent prefill runs on the first sample lane"),
    (["Checkpoint", "written in flight", "the final prefill", "window writes it"], "s1",
     "Checkpoints ride the final prefill window, at most 2 per request"),
    (["Publish", "host registers it", "when the flight", "retires"], None,
     "take_flight_checkpoints, then publish_flight_checkpoint at retirement"),
    (["fork_batch", "n lanes share one", "frozen StateAnchor,", "no State copy"], "s1",
     "One Session call forks all sample lanes from a frozen shared anchor"),
    (["Samples decode", "n lanes join the", "running batch"], None, "Decode starts right after the fork"),
]


def build(d, outdir):
    th, tf = d["head"]["telemetry"], d["final_c"]["telemetry"]
    W = 880
    sc = Scene(W, 520, "Control plane before and after",
               "Schematic of how a request with n samples is prefilled, checkpointed and forked in HEAD and in the final engine")
    y = header(sc, "Control plane: how n samples of one request are prefilled, checkpointed and forked",
               "Schematic, not to scale. Orange boxes are the serialized steps, blue boxes are the steps that replace them.")
    sc.text(24, y + 16, "Before: HEAD 3e31d5a4", 13, "ink1", "start", True)
    xs, bw = row(sc, 24, y + 28, 832, 76, BEFORE)
    bracket(sc, xs[1], xs[3] + bw, y + 118, "one context-operation slot, a quiescent iteration for each: flights are held")
    strip(sc, 24, y + 142, 832, "Measured:", f"{th['prefill_per_flight']:.0f} prefill tokens and {th['decode_rows_per_flight']:.1f} decode rows per flight (median), "
          f"{d['head']['mean_request_throughput']:.2f} req/s")
    y2 = y + 196
    sc.text(24, y2 + 16, "After: final engine", 13, "ink1", "start", True)
    xs, bw = row(sc, 24, y2 + 28, 832, 76, AFTER)
    bracket(sc, xs[1], xs[3] + bw, y2 + 118, "bindings are synchronous: no context operation, no quiescent iteration, flights keep running")
    strip(sc, 24, y2 + 142, 832, "Measured:", f"{tf['prefill_per_flight']:,.0f} prefill tokens and {tf['decode_rows_per_flight']:.0f} decode rows per flight (median), "
          f"{d['final_c']['mean_request_throughput']:.2f} req/s", "b100")
    y = footer(sc, y2 + 192, [
        "Flight shape is the median of the engine's 5 s throughput log lines while 100 or more requests run (same data as the flight-shape figure).",
        "Only qwen3_5 uses the new path (Session::captures_in_flight); the other model Sessions keep the old capture and context-operation path."])
    sc.h = y + 14
    svg, png = save(sc, outdir, "fig6_control_plane")
    rows = [["Lanes per group of n samples", "n + 1 (hidden helper)", "n"],
            ["Checkpoint capture", "context operation, State copy, quiescent iteration", "written in the final prefill window, published at retirement"],
            ["Fork", "one State copy per sample, one context operation", "one fork_batch call, shared frozen StateAnchor"],
            ["Prefill tokens per flight (median)", f"{th['prefill_per_flight']:.0f}", f"{tf['prefill_per_flight']:,.0f}"],
            ["Decode rows per flight (median)", f"{th['decode_rows_per_flight']:.1f}", f"{tf['decode_rows_per_flight']:.0f}"]]
    return svg, png, md_table(["", "HEAD", "Final"], rows)


if __name__ == "__main__":
    d = json.load(open(W0 + "data.json"))
    print(build(d, "/tmp/figtest")[2])
