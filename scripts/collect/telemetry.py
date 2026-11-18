"""Parse `throughput |` telemetry lines from ninfer-serve logs."""
import re
import statistics as st

FIELDS = [
    ("prefill_ktok_s", r"prefill ([\d.]+)k tok/s"),
    ("prefill_tok", r"prefill [\d.]+k tok/s \(([\d,]+) tok\)"),
    ("decode_tok_s", r"decode ([\d.]+) tok/s"),
    ("running", r"running (\d+)"),
    ("waiting", r"waiting (\d+)"),
    ("capture_pending", r"capture-pending (\d+)"),
    ("batch", r"batch ([\d.]+)"),
    ("chunk_ms", r"chunk ([\d.]+) ms"),
    ("step_ms", r"step ([\d.]+) ms"),
    ("host_pct", r"host ([\d.]+)%"),
    ("flights", r"flights (\d+)"),
    ("prefill_per_flight", r"prefill/flight ([\d.]+) tok"),
    ("decode_rows_per_flight", r"decode rows/flight ([\d.]+)"),
    ("gpu_busy_pct", r"gpu busy ([\d.]+)%"),
]


def parse(path):
    recs = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if "throughput |" not in line:
            continue
        r = {"t": line[:23]}
        for name, pat in FIELDS:
            m = re.search(pat, line)
            if m:
                r[name] = float(m.group(1).replace(",", ""))
        recs.append(r)
    return recs


def steady(recs, key="running", lo=0.15, hi=0.15, min_running=100):
    n = len(recs)
    sel = recs[int(n * lo): n - int(n * hi)] if n > 4 else recs
    return [r for r in sel if r.get(key, 0) >= min_running] or sel


def summarize(recs):
    out = {}
    for name, _ in FIELDS:
        vals = [r[name] for r in recs if name in r]
        if vals:
            out[name] = st.median(vals)
    counted = [r for r in recs if {"prefill_tok", "flights", "prefill_per_flight"} <= r.keys()]
    if counted:
        tok = sum(r["prefill_tok"] for r in counted)
        out["counter_prefill_per_flight"] = tok / sum(r["flights"] for r in counted)
        out["flight_counter_share"] = sum(r["prefill_per_flight"] * r["flights"] for r in counted) / tok
    out["n"] = len(recs)
    return out
