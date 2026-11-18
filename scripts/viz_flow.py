"""Box and arrow helpers for schematic figures."""
import math
from viz_core import text_width


def box(sc, x, y, w, h, lines, kind=None, tip=None, size=11):
    for s in lines:
        if text_width(s, size, True) > w - 10:
            raise ValueError(f"text too wide for box ({text_width(s, size, True):.0f} > {w - 10}): {s}")
    if kind:
        sc.rect(x, y, w, h, kind, r=6, side="all", op=0.16, stroke=kind, sw=1.25, tip=tip)
    else:
        sc.rect(x, y, w, h, "surface", r=6, side="all", stroke="axis", sw=1, tip=tip)
    lh = size + 3.5
    y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + size * 0.35
    for i, s in enumerate(lines):
        sc.text(x + w / 2, y0 + i * lh, s, size, "ink1", "middle", i == 0 and len(lines) > 1)


def arrow(sc, x1, y1, x2, y2, stroke="ink2", w=1.5):
    L = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    sc.line(x1, y1, x2 - ux * 5, y2 - uy * 5, stroke, w)
    hl, hw = 8, 4
    bx, by = x2 - ux * hl, y2 - uy * hl
    sc.poly([(x2, y2), (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)], fill=stroke, closed=True)


def row(sc, x, y, w, h, boxes, gap=24):
    n = len(boxes)
    bw = (w - gap * (n - 1)) / n
    xs = []
    for i, (lines, kind, tip) in enumerate(boxes):
        bx = x + i * (bw + gap)
        box(sc, bx, y, bw, h, lines, kind, tip)
        xs.append(bx)
        if i:
            arrow(sc, bx - gap + 3, y + h / 2, bx - 2, y + h / 2)
    return xs, bw


def bracket(sc, x1, x2, y, label, stroke="ink2"):
    sc.line(x1, y, x2, y, stroke, 1.25)
    sc.line(x1, y - 5, x1, y, stroke, 1.25)
    sc.line(x2, y - 5, x2, y, stroke, 1.25)
    sc.text((x1 + x2) / 2, y + 15, label, 11, "ink2", "middle")


def strip(sc, x, y, w, label, text, tone="grid"):
    sc.rect(x, y, w, 26, tone, r=4, op=0.7)
    sc.text(x + 10, y + 17, label, 12, "ink1", "start", True)
    sc.text(x + 10 + text_width(label, 12, True) + 8, y + 17, text, 12, "ink1")


def column(sc, x, y, w, h, boxes, gap=18):
    for i, (lines, kind, tip) in enumerate(boxes):
        by = y + i * (h + gap)
        box(sc, x, by, w, h, lines, kind, tip)
        if i:
            arrow(sc, x + w / 2, by - gap + 2, x + w / 2, by - 1)
    return y + len(boxes) * h + (len(boxes) - 1) * gap


def kv_table(sc, x, y, w, head, rows, c1=0.46, c2=0.73):
    sc.text(x, y, head[0], 11, "muted", "start", True)
    sc.text(x + w * c1, y, head[1], 11, "muted", "start", True)
    sc.text(x + w * c2, y, head[2], 11, "muted", "start", True)
    sc.line(x, y + 6, x + w, y + 6, "axis", 1)
    for i, (a, b, c) in enumerate(rows):
        yy = y + 22 + i * 18
        sc.text(x, yy, a, 11, "ink2")
        sc.text(x + w * c1, yy, b, 11, "ink1")
        sc.text(x + w * c2, yy, c, 11, "ink1", "start", True)
    return y + 22 + len(rows) * 18
