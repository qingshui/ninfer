"""Analyze an nsys sqlite export of a ninfer-serve window: categories, busy/idle, flights, requests."""
import collections
import json
import re
import sqlite3
import sys
import time

CATS = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else []


def classify(name):
    for cat, pat in CATS:
        if re.search(pat, name):
            return cat
    return "other"


def merge(intervals, gap=0):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1] + gap:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def requests_done(log, t_lo, t_hi):
    n = prompt = output = cached = 0
    pat = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\s+INFO\s+req#\d+ done .*?prompt (\d+) \| output (\d+) \| cache (\d+)")
    for line in open(log, encoding="utf-8", errors="replace"):
        m = pat.match(line)
        if not m:
            continue
        ts = time.mktime(time.strptime(m.group(1)[:19], "%Y-%m-%d %H:%M:%S")) + float("0." + m.group(1)[20:])
        if t_lo <= ts <= t_hi:
            n += 1
            prompt += int(m.group(2))
            output += int(m.group(3))
            cached += int(m.group(4))
    return dict(n=n, prompt=prompt, output=output, cached=cached)


def main(db, times_file, serve_log, out_json):
    con = sqlite3.connect(db)
    names = dict(con.execute("select id, value from StringIds"))
    rows = con.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    t0, t1 = rows[0][0], max(r[1] for r in rows)
    wall = (t1 - t0) / 1e9
    cat_ns = collections.Counter()
    cat_n = collections.Counter()
    kern = collections.defaultdict(lambda: [0, 0])
    for s, e, sn in rows:
        nm = names.get(sn, str(sn))
        c = classify(nm)
        cat_ns[c] += e - s
        cat_n[c] += 1
        kern[nm][0] += 1
        kern[nm][1] += e - s
    busy = merge([(s, e) for s, e, _ in rows], 0)
    busy_ns = sum(e - s for s, e in busy)
    gaps = [(busy[i + 1][0] - busy[i][1]) for i in range(len(busy) - 1)]
    big = [g for g in gaps if g >= 1_000_000]
    flights = merge([(s, e) for s, e, _ in rows], 1_500_000)
    fl = [dict(start_ms=(f[0] - t0) / 1e6, span_ms=(f[1] - f[0]) / 1e6) for f in flights]
    fbusy = []
    for f in flights:
        fb = sum(min(e, f[1]) - max(s, f[0]) for s, e in busy if e > f[0] and s < f[1])
        fbusy.append(fb / 1e6)
    sess = con.execute("select utcEpochNs, systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()
    ep_lo = sess[0] / 1e9 + t0 / 1e9
    ep_hi = sess[0] / 1e9 + t1 / 1e9
    req = requests_done(serve_log, ep_lo, ep_hi)
    out = dict(
        wall_s=wall, kernels=len(rows), busy_s=busy_ns / 1e9, util=busy_ns / (t1 - t0),
        gaps_ge_1ms=len(big), gap_ge_1ms_total_ms=sum(big) / 1e6,
        categories={k: dict(ms=v / 1e6, n=cat_n[k]) for k, v in cat_ns.items()},
        top=[dict(name=k, n=v[0], ms=v[1] / 1e6) for k, v in sorted(kern.items(), key=lambda kv: -kv[1][1])[:60]],
        flights=dict(n=len(flights), spans_ms=[round(f["span_ms"], 2) for f in fl][:200],
                     busy_ms=[round(b, 2) for b in fbusy][:200]),
        requests=req, window_epoch=[ep_lo, ep_hi],
        busy_intervals_ms=[[round((s - t0) / 1e6, 3), round((e - t0) / 1e6, 3)] for s, e in merge([(s, e) for s, e, _ in rows], 20_000)
                           if (s - t0) < 3_000_000_000],
    )
    json.dump(out, open(out_json, "w"))
    print(f"wall {wall:.2f}s kernels {len(rows)} util {out['util']:.3f} flights {len(flights)} requests {req}")
    for k, v in sorted(out["categories"].items(), key=lambda kv: -kv[1]["ms"]):
        print(f"  {k:28s} {v['ms']:10.1f} ms  n={v['n']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5] if len(sys.argv) > 5 else "/tmp/prof.json")
