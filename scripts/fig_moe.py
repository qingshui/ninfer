import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, save, md_table
from viz_flow import row, strip

GU_BEFORE = [
    (["Producer warp", "16 TMA gather4 per", "stage, about 94", "cycles each"], "s2",
     "The producer spent about 83% of the kernel in issue, about 1870 cycles per stage"),
    (["MMA warp starved", "tensor cores wait", "for the next stage"], "s2", "Gate/up ran at about 453 us for 3700 tokens"),
    (["Epilogue", "IEEE expf and", "division chains"], None, "SwiGLU and NVFP4 quantization"),
    (["Result", "gate/up 453 us"], None, "3700 tokens, skew 0.4, isolated op bench"),
]
GU_AFTER = [
    (["gather_input kernel", "pre-gathers token", "rows in one pass"], "s1", "count, scan, index, then gather"),
    (["Producer warp", "contiguous 2D TMA", "loads"], "s1", "No per-row gather4 instructions"),
    (["Epilogue", "silu_approx and a", "shared-reciprocal", "quantizer"], None, "silu_approx worst relative error 3.2e-7"),
    (["Result", "gate/up 212 us"], None, "3700 tokens, skew 0.4, isolated op bench"),
]
CB_BEFORE = [
    (["Down GEMM", "per-slot outputs", "fp32, per token"], None, "Each token has 8 routed slots"),
    (["Epilogue", "red.global.add", "into the output"], "s2", "fp32 atomic adds, v4 vectors"),
    (["L2 atomic bound", "order of additions", "varies between runs"], "s2", "Repeated scores of one prompt differed at about 1530 of 1537 positions"),
    (["Result", "routed op 480.5 us", "nondeterministic"], None, "3700 tokens, isolated op bench"),
]
CB_AFTER = [
    (["Down GEMM", "per-slot outputs", "fp32, per token"], None, "Same GEMM"),
    (["Epilogue", "writes contributions", "[slot][hidden], fp32"], "s1", "No atomics"),
    (["Gather kernel", "sums the slots in", "fixed order"], "s1", "reduce for the routed op, stage_routed for the complete op"),
    (["Result", "routed op 437.7 us", "deterministic"], None, "3700 tokens, isolated op bench"),
]


def chain(sc, y, tag, boxes):
    sc.text(24, y + 10, tag, 11, "muted", "start", True)
    row(sc, 24, y + 16, 832, 70, boxes)
    return y + 94


def build(outdir):
    W = 880
    sc = Scene(W, 700, "MoE gate/up and combine before and after",
               "Schematic of the routed-expert gate/up GEMM and of the combine step before and after phase B")
    y = header(sc, "MoE: gate/up producer and combine, before and after",
               "Schematic, not to scale. Orange boxes are replaced by the blue ones.")
    sc.text(24, y + 16, "1. Gate/up GEMM: remove the gather4 producer bottleneck", 13, "ink1", "start", True)
    y = chain(sc, y + 22, "Before", GU_BEFORE)
    y = chain(sc, y, "After", GU_AFTER)
    strip(sc, 24, y, 832, "Measured:", "gate/up 453 to 212 us, routed op 765 to 481.5 us (op bench); end to end 59.45 to 65.59 req/s", "b100")
    sc.text(24, y + 52, "2. Combine: fixed-order sum instead of fp32 atomics", 13, "ink1", "start", True)
    y = chain(sc, y + 58, "Before", CB_BEFORE)
    y = chain(sc, y, "After", CB_AFTER)
    strip(sc, 24, y, 832, "Measured:", "routed op 480.5 to 437.7 us (op bench); end to end 65.59 to about 67.4 req/s, bundled with an ftz silu in conv", "b100")
    y = footer(sc, y + 46, [
        "Op times come from isolated benches on GPU 1 during development and are recorded in notes; the end-to-end figures are harness runs of the stage builds.",
        "The contributions buffer costs 768 MiB (805 MB) at the 8192-token slice limit (computed, not measured); other geometries still use the atomic form."])
    sc.h = y + 14
    svg, png = save(sc, outdir, "fig8_moe")
    table = md_table(["item", "before", "after"], [
        ["Gate/up GEMM, 3700 tokens, skew 0.4 (us)", "453", "212"],
        ["Routed op, 3700 tokens, skew 0.4 (us)", "765", "481.5"],
        ["Routed op, atomic combine against ordered combine (us)", "480.5", "437.7"],
        ["Complete op with shared experts, ordered combine (us)", "-", "650.7"],
        ["End to end, locked harness (req/s)", "59.45 to 65.59 to about 67.4", ""]])
    return svg, png, table


if __name__ == "__main__":
    print(build("/tmp/figtest")[2])
