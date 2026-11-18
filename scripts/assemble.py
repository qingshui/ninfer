import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import glob
import os
import re
import sys

PARTS = os.environ.get("REPORT_PARTS", HERE + "/report_parts")
OUT = os.environ.get("REPORT_OUT", os.path.join(HERE, ".."))


def fig_file(fid):
    m = glob.glob(f"{OUT}/figures/{fid}_*.svg")
    assert len(m) == 1, (fid, m)
    return os.path.basename(m[0])[:-4]


def table(fid):
    return open(f"{OUT}/tables/{fid}.md", encoding="utf-8").read().strip()


def expand(text, ku):
    ext = "png" if ku else "svg"
    keep, drop = ("KU", "LOCAL") if ku else ("LOCAL", "KU")
    text = re.sub(r"\{\{%s\}\}.*?\{\{/%s\}\}\n?" % (drop, drop), "", text, flags=re.S)
    text = re.sub(r"\{\{/?%s\}\}\n?" % keep, "", text)

    def fig(m):
        kind, fid, cap = m.group(1), m.group(2), m.group(3)
        s = f"![{cap}](figures/{fig_file(fid)}.{ext})\n\n*{cap}*\n"
        if kind == "FIG":
            s += f"\n数据表（与{cap.split('　')[0]} 对应）：\n\n{table(fid)}\n"
        return s

    text = re.sub(r"\{\{(FIG|FIGNT):(fig\d+)\|([^}]*)\}\}", fig, text)
    text = re.sub(r"\{\{TABLE:([a-z0-9_]+)\}\}", lambda m: table(m.group(1)), text)
    return text


def main():
    files = sorted(glob.glob(PARTS + "/*.md"))
    body = "\n".join(open(f, encoding="utf-8").read().rstrip() + "\n" for f in files)
    left = re.findall(r"@@[A-Z_]+@@", body)
    if left:
        print("unresolved markers:", sorted(set(left)))
    open(OUT + "/report.md", "w", encoding="utf-8").write(expand(body, False))
    open(OUT + "/report_ku.md", "w", encoding="utf-8").write(expand(body, True))
    print("parts:", len(files), "bytes:", os.path.getsize(OUT + "/report.md"))


if __name__ == "__main__":
    main()
