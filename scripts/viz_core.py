"""Tiny scene graph with SVG and PNG backends (stdlib + Pillow)."""
import html
from PIL import Image, ImageDraw, ImageFont

FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FAMILY = "'DejaVu Sans', Verdana, system-ui, -apple-system, 'Segoe UI', sans-serif"

LIGHT = dict(surface="#fcfcfb", ink1="#0b0b0b", ink2="#52514e", muted="#898781",
             grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
             s4="#eda100", s5="#e87ba4", s6="#008300", s7="#6250d6", s8="#e34948",
             b250="#86b6ef", b100="#cde2fb", white="#ffffff", good="#006300")
DARK = dict(surface="#1a1a19", ink1="#f0efec", ink2="#c3c2b7", muted="#898781",
            grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926", s3="#199e70",
            s4="#c98500", s5="#d55181", s6="#008300", s7="#9085e9", s8="#e66767",
            b250="#3987e5", b100="#184f95", white="#ffffff", good="#0ca30c")

_fonts = {}


def _font(px, bold=False):
    key = (int(round(px)), bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_B if bold else FONT_R, size=max(1, key[0]))
    return _fonts[key]


def text_width(s, size, bold=False):
    return _font(size * 10, bold).getlength(s) / 10.0


class Scene:
    def __init__(self, w, h, title, desc):
        self.w, self.h, self.title, self.desc = w, h, title, desc
        self.items = []

    def rect(self, x, y, w, h, fill, r=0, side="all", tip=None, op=1.0, stroke=None, sw=1):
        self.items.append(dict(k="rect", x=x, y=y, w=w, h=h, fill=fill, r=r, side=side,
                               tip=tip, op=op, stroke=stroke, sw=sw))

    def line(self, x1, y1, x2, y2, stroke="grid", w=1, cap="butt", op=1.0):
        self.items.append(dict(k="line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=stroke, w=w,
                               cap=cap, op=op))

    def poly(self, pts, stroke=None, fill=None, w=2, closed=False, op=1.0, tip=None):
        self.items.append(dict(k="poly", pts=pts, stroke=stroke, fill=fill, w=w,
                               closed=closed, op=op, tip=tip))

    def circle(self, cx, cy, r, fill, ring=True, tip=None):
        self.items.append(dict(k="circle", cx=cx, cy=cy, r=r, fill=fill, ring=ring, tip=tip))

    def text(self, x, y, s, size=12, fill="ink1", anchor="start", bold=False):
        self.items.append(dict(k="text", x=x, y=y, s=s, size=size, fill=fill,
                               anchor=anchor, bold=bold))


def _n(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _css():
    def block(tok):
        return ";".join(f"--{k}:{v}" for k, v in tok.items())
    cls = []
    for k, v in LIGHT.items():
        cls.append(f".f-{k}{{fill:{v};fill:var(--{k})}}.s-{k}{{stroke:{v};stroke:var(--{k})}}")
    return (f"<style>svg{{{block(LIGHT)}}}"
            f"@media (prefers-color-scheme:dark){{svg{{{block(DARK)}}}}}"
            + "".join(cls) + "</style>")


def _rect_path(x, y, w, h, r, side):
    r = min(r, w / 2, h / 2)
    tl, tr, br, bl = {"all": (1, 1, 1, 1), "right": (0, 1, 1, 0), "left": (1, 0, 0, 1),
                      "top": (1, 1, 0, 0), "bottom": (0, 0, 1, 1), "none": (0, 0, 0, 0)}[side]
    x2, y2 = x + w, y + h
    p = [f"M{_n(x + r * tl)},{_n(y)}", f"H{_n(x2 - r * tr)}"]
    if tr:
        p.append(f"A{_n(r)},{_n(r)} 0 0 1 {_n(x2)},{_n(y + r)}")
    p.append(f"V{_n(y2 - r * br)}")
    if br:
        p.append(f"A{_n(r)},{_n(r)} 0 0 1 {_n(x2 - r)},{_n(y2)}")
    p.append(f"H{_n(x + r * bl)}")
    if bl:
        p.append(f"A{_n(r)},{_n(r)} 0 0 1 {_n(x)},{_n(y2 - r)}")
    p.append(f"V{_n(y + r * tl)}")
    if tl:
        p.append(f"A{_n(r)},{_n(r)} 0 0 1 {_n(x + r)},{_n(y)}")
    return "".join(p) + "Z"


def _wrap(tag, attrs, tip):
    if tip:
        return f"<{tag} {attrs}><title>{html.escape(tip)}</title></{tag}>"
    return f"<{tag} {attrs}/>"
