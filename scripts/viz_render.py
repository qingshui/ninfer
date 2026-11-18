"""SVG and PNG renderers for viz_core.Scene."""
import html
from PIL import Image, ImageDraw
from viz_core import (LIGHT, FAMILY, _css, _n, _rect_path, _wrap, _font)


def to_svg(sc):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_n(sc.w)} {_n(sc.h)}" '
         f'width="{_n(sc.w)}" height="{_n(sc.h)}" role="img" aria-labelledby="t d" '
         f'font-family="{FAMILY}">',
         f'<title id="t">{html.escape(sc.title)}</title><desc id="d">{html.escape(sc.desc)}</desc>',
         _css(), f'<rect class="f-surface" width="{_n(sc.w)}" height="{_n(sc.h)}"/>']
    for it in sc.items:
        k = it["k"]
        if k == "rect":
            cls = f'f-{it["fill"]}' + (f' s-{it["stroke"]}' if it["stroke"] else "")
            extra = f' stroke-width="{_n(it["sw"])}"' if it["stroke"] else ""
            if it["op"] != 1.0:
                extra += f' fill-opacity="{_n(it["op"])}"'
            if it["r"] and it["side"] != "none":
                a = f'd="{_rect_path(it["x"], it["y"], it["w"], it["h"], it["r"], it["side"])}" class="{cls}"{extra}'
                o.append(_wrap("path", a, it["tip"]))
            else:
                a = (f'x="{_n(it["x"])}" y="{_n(it["y"])}" width="{_n(it["w"])}" '
                     f'height="{_n(it["h"])}" class="{cls}"{extra}')
                o.append(_wrap("rect", a, it["tip"]))
        elif k == "line":
            a = (f'x1="{_n(it["x1"])}" y1="{_n(it["y1"])}" x2="{_n(it["x2"])}" y2="{_n(it["y2"])}" '
                 f'class="s-{it["stroke"]}" stroke-width="{_n(it["w"])}" stroke-linecap="{it["cap"]}"')
            if it["op"] != 1.0:
                a += f' stroke-opacity="{_n(it["op"])}"'
            o.append(f"<line {a}/>")
        elif k == "poly":
            pts = " ".join(f"{_n(px)},{_n(py)}" for px, py in it["pts"])
            tag = "polygon" if it["closed"] else "polyline"
            cls = (f'f-{it["fill"]}' if it["fill"] else "")
            if it["stroke"]:
                cls += f' s-{it["stroke"]}'
            a = f'points="{pts}" class="{cls.strip()}"'
            if not it["fill"]:
                a += ' fill="none"'
            if it["stroke"]:
                a += f' stroke-width="{_n(it["w"])}" stroke-linejoin="round" stroke-linecap="round"'
            if it["op"] != 1.0:
                a += f' opacity="{_n(it["op"])}"'
            o.append(_wrap(tag, a, it["tip"]))
        elif k == "circle":
            if it["ring"]:
                o.append(f'<circle cx="{_n(it["cx"])}" cy="{_n(it["cy"])}" r="{_n(it["r"] + 2)}" class="f-surface"/>')
            a = f'cx="{_n(it["cx"])}" cy="{_n(it["cy"])}" r="{_n(it["r"])}" class="f-{it["fill"]}"'
            o.append(_wrap("circle", a, it["tip"]))
        elif k == "text":
            anc = {"start": "start", "middle": "middle", "end": "end"}[it["anchor"]]
            wgt = ' font-weight="700"' if it["bold"] else ""
            o.append(f'<text x="{_n(it["x"])}" y="{_n(it["y"])}" font-size="{_n(it["size"])}" '
                     f'text-anchor="{anc}" class="f-{it["fill"]}"{wgt}>{html.escape(it["s"])}</text>')
    o.append("</svg>")
    return "\n".join(o)


def to_png(sc, path, tokens=LIGHT, out_scale=2, draw_scale=4):
    S = draw_scale
    img = Image.new("RGB", (int(sc.w * S), int(sc.h * S)), tokens["surface"])
    d = ImageDraw.Draw(img, "RGBA")

    def col(tok, op=1.0):
        h = tokens[tok].lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(round(255 * op)))

    for it in sc.items:
        k = it["k"]
        if k == "rect":
            x0, y0 = it["x"] * S, it["y"] * S
            x1, y1 = (it["x"] + it["w"]) * S - 1, (it["y"] + it["h"]) * S - 1
            side = it["side"]
            r = min(it["r"], it["w"] / 2, it["h"] / 2) * S if it["r"] and side != "none" else 0
            corners = {"all": (1, 1, 1, 1), "right": (0, 1, 1, 0), "left": (1, 0, 0, 1),
                       "top": (1, 1, 0, 0), "bottom": (0, 0, 1, 1), "none": (0, 0, 0, 0)}[side]
            kw = {}
            if it["stroke"]:
                kw = dict(outline=col(it["stroke"]), width=int(round(it["sw"] * S)))
            if r:
                d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=col(it["fill"], it["op"]),
                                    corners=tuple(bool(c) for c in corners), **kw)
            else:
                d.rectangle([x0, y0, x1, y1], fill=col(it["fill"], it["op"]), **kw)
        elif k == "line":
            w = max(1, int(round(it["w"] * S)))
            p1, p2 = (it["x1"] * S, it["y1"] * S), (it["x2"] * S, it["y2"] * S)
            d.line([p1, p2], fill=col(it["stroke"], it["op"]), width=w)
            if it["cap"] == "round":
                for p in (p1, p2):
                    d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2],
                              fill=col(it["stroke"], it["op"]))
        elif k == "poly":
            pts = [(px * S, py * S) for px, py in it["pts"]]
            if it["fill"]:
                d.polygon(pts, fill=col(it["fill"], it["op"]))
            if it["stroke"]:
                w = max(1, int(round(it["w"] * S)))
                seq = pts + ([pts[0]] if it["closed"] else [])
                d.line(seq, fill=col(it["stroke"], it["op"]), width=w, joint="curve")
                for p in pts:
                    d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2],
                              fill=col(it["stroke"], it["op"]))
        elif k == "circle":
            cx, cy, r = it["cx"] * S, it["cy"] * S, it["r"] * S
            if it["ring"]:
                rr = r + 2 * S
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col("surface"))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col(it["fill"]))
        elif k == "text":
            f = _font(it["size"] * S, it["bold"])
            anc = {"start": "ls", "middle": "ms", "end": "rs"}[it["anchor"]]
            d.text((it["x"] * S, it["y"] * S), it["s"], font=f, fill=col(it["fill"]), anchor=anc)
    img = img.resize((int(sc.w * out_scale), int(sc.h * out_scale)), Image.LANCZOS)
    img.save(path, optimize=True)
