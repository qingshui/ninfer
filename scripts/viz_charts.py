"""Chart helpers on top of viz_core / viz_render."""
import math
import os
from viz_core import Scene, text_width
from viz_render import to_svg, to_png

PADL = 24
FILL_TEXT_DARK = {"s4"}


def nice_ticks(vmax, n=5):
    raw = max(vmax, 1e-9) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step * n >= vmax:
            break
    top = math.ceil(vmax / step - 1e-9) * step
    return top, [i * step for i in range(int(round(top / step)) + 1)]


def fmt(v, dec=None):
    if dec is None:
        dec = 0 if abs(v) >= 100 else (1 if abs(v) >= 10 else 2)
    return f"{v:,.{dec}f}"


def header(sc, title, sub=None):
    sc.text(PADL, 30, title, 16, "ink1", bold=True)
    if sub:
        sc.text(PADL, 50, sub, 12, "ink2")
    return 72 if sub else 54


def footer(sc, y, lines):
    for i, s in enumerate(lines):
        sc.text(PADL, y + 14 * i, s, 10.5, "muted")
    return y + 14 * len(lines)


def legend(sc, x, y, items, maxw):
    cx, cy = x, y
    for label, tok in items:
        w = 15 + text_width(label, 12) + 20
        if cx + w > x + maxw and cx > x:
            cx, cy = x, cy + 20
        sc.rect(cx, cy - 9, 10, 10, tok, r=2)
        sc.text(cx + 15, cy, label, 12, "ink2")
        cx += w
    return cy + 10


def snap(v):
    return math.floor(v) + 0.5


def hbars(sc, x, y, w, rows, vmax=None, unit="", refs=(), bar_h=18, pitch=32,
          label_w=None, tick_n=5, dec=None, xlabel=None, tip_pad=8):
    lw = label_w if label_w is not None else max(text_width(r["label"], 12) for r in rows) + 14
    tip_w = max(text_width(r.get("text") or fmt(r["value"], dec) + unit, 12, True) for r in rows) + tip_pad + 6
    px0, px1 = x + lw, x + w - tip_w
    top, tk = nice_ticks(vmax or max(r["value"] for r in rows) * 1.0, tick_n)
    sx = lambda v: px0 + (px1 - px0) * v / top
    ph = pitch * len(rows) + 6
    for t in tk:
        gx = snap(sx(t))
        sc.line(gx, y, gx, y + ph, "grid", 1)
        sc.text(gx, y + ph + 15, f"{t:,g}", 11, "muted", "middle")
    sc.line(snap(px0), y, snap(px0), y + ph, "axis", 1)
    for i, r in enumerate(rows):
        cy = y + 3 + pitch * i + pitch / 2
        sc.text(x + lw - 10, cy + 4, r["label"], 12, "ink1" if r.get("bold") else "ink2", "end", r.get("bold", False))
        bw = max(0.0, sx(r["value"]) - px0)
        if bw > 0:
            sc.rect(px0, cy - bar_h / 2, bw, bar_h, r["fill"], r=4 if bw > 6 else 0,
                    side="right" if bw > 6 else "none", tip=r.get("tip"))
        if not refs:
            sc.text(px0 + bw + tip_pad, cy + 4, r.get("text") or fmt(r["value"], dec) + unit, 12,
                    "ink1", "start", r.get("bold", False))
    for rv, rl in refs:
        gx = snap(sx(rv))
        sc.line(gx, y - 6, gx, y + ph, "ink2", 1.5)
        sc.text(gx, y - 10, rl, 11, "ink2", "middle")
    for i, r in enumerate(rows if refs else []):
        cy = y + 3 + pitch * i + pitch / 2
        bw = max(0.0, sx(r["value"]) - px0)
        lab = r.get("text") or fmt(r["value"], dec) + unit
        bold = r.get("bold", False)
        sc.rect(px0 + bw + tip_pad - 3, cy - 8, text_width(lab, 12, bold) + 6, 16, "surface")
        sc.text(px0 + bw + tip_pad, cy + 4, lab, 12, "ink1", "start", bold)
    if xlabel:
        sc.text((px0 + px1) / 2, y + ph + 32, xlabel, 11, "muted", "middle")
    return y + ph + (40 if xlabel else 22)


def stacked_hbars(sc, x, y, w, rows, series, vmax=None, unit="", bar_h=24, pitch=46,
                  label_w=None, tick_n=5, total_fmt=None):
    lw = label_w if label_w is not None else max(text_width(r["label"], 12, True) for r in rows) + 14
    tot = [sum(s[1] for s in r["segs"]) for r in rows]
    tw = max(text_width((total_fmt or fmt)(t) + unit, 12, True) for t in tot) + 16
    px0, px1 = x + lw, x + w - tw
    top, tk = nice_ticks(vmax or max(tot), tick_n)
    sx = lambda v: px0 + (px1 - px0) * v / top
    ph = pitch * len(rows) + 4
    for t in tk:
        gx = snap(sx(t))
        sc.line(gx, y, gx, y + ph, "grid", 1)
        sc.text(gx, y + ph + 15, f"{t:,g}", 11, "muted", "middle")
    sc.line(snap(px0), y, snap(px0), y + ph, "axis", 1)
    for i, r in enumerate(rows):
        cy = y + 2 + pitch * i + pitch / 2
        sc.text(x + lw - 10, cy + 4, r["label"], 12, "ink1", "end", True)
        acc = 0.0
        for j, (si, val, tip) in enumerate(r["segs"]):
            x0, x1 = sx(acc), sx(acc + val)
            acc += val
            gap = 2 if j > 0 else 0
            sw = x1 - x0 - gap
            if sw <= 0.5:
                continue
            last = j == len(r["segs"]) - 1
            tok = series[si][1]
            sc.rect(x0 + gap, cy - bar_h / 2, sw, bar_h, tok, r=4 if last and sw > 6 else 0,
                    side="right" if last and sw > 6 else "none", tip=tip)
            lab = r.get("labels", {}).get(si)
            if lab and text_width(lab, 11) + 12 < sw:
                sc.text(x0 + gap + sw / 2, cy + 4, lab, 11, "ink1" if tok in FILL_TEXT_DARK else "white", "middle")
        sc.text(sx(acc) + 8, cy + 4, (total_fmt or fmt)(acc) + unit, 12, "ink1", "start", True)
    return y + ph + 22


def grouped_hbars(sc, x, y, w, cats, series, vmax=None, bar_h=14, gap=3, pitch=56,
                  label_w=None, tick_n=5, dec=None, texts=None):
    lw = label_w if label_w is not None else max(text_width(c["label"], 12) for c in cats) + 14
    allv = [v for c in cats for v in c["values"]]
    tw = max(text_width(fmt(v, dec), 12, True) for v in allv) + 20
    px0, px1 = x + lw, x + w - tw
    top, tk = nice_ticks(vmax or max(allv), tick_n)
    sx = lambda v: px0 + (px1 - px0) * v / top
    ph = pitch * len(cats) + 4
    for t in tk:
        gx = snap(sx(t))
        sc.line(gx, y, gx, y + ph, "grid", 1)
        sc.text(gx, y + ph + 15, f"{t:,g}", 11, "muted", "middle")
    sc.line(snap(px0), y, snap(px0), y + ph, "axis", 1)
    ns = len(series)
    for i, c in enumerate(cats):
        cy = y + 2 + pitch * i + pitch / 2
        sc.text(x + lw - 10, cy + 4, c["label"], 12, "ink2", "end")
        block = ns * bar_h + (ns - 1) * gap
        for j, v in enumerate(c["values"]):
            by = cy - block / 2 + j * (bar_h + gap)
            bw = max(0.0, sx(v) - px0)
            if bw > 0:
                sc.rect(px0, by, bw, bar_h, series[j][1], r=4 if bw > 6 else 0,
                        side="right" if bw > 6 else "none",
                        tip=(c.get("tips") or [None] * ns)[j])
            lab = (c.get("texts") or [None] * ns)[j] or fmt(v, dec)
            sc.text(px0 + bw + 8, by + bar_h - 3, lab, 11.5, "ink1", "start", j == ns - 1)
    return y + ph + 22


def save(sc, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    svg = os.path.join(outdir, name + ".svg")
    png = os.path.join(outdir, name + ".png")
    open(svg, "w", encoding="utf-8").write(to_svg(sc))
    to_png(sc, png)
    return svg, png


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
