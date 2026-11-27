"""Assemble report/TECHNICAL_REPORT.md from report/src/TECHNICAL_REPORT.src.md and render report/build/TECHNICAL_REPORT.html for PDF printing.

  {{table:<name>}}  is replaced by report/tables/<name>.md (generated from frozen artifacts by make_tables.py)
  $$ ... $$         display equations: kept as-is in the Markdown (GitHub renders them); rendered to SVG with matplotlib mathtext for the HTML/PDF

PYTHONPATH=~/a5_plotlib python report/scripts/build_report.py
then print the HTML to PDF with a headless Chromium browser (see report/REPRODUCIBILITY.md).
"""
import html, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC, TAB, BUILD = os.path.join(R, "src"), os.path.join(R, "tables"), os.path.join(R, "build")
os.makedirs(os.path.join(BUILD, "eq"), exist_ok=True)


def assemble(name):
    s = open(os.path.join(SRC, name + ".src.md"), encoding="utf-8").read()

    def tab(m):
        return open(os.path.join(TAB, m[1] + ".md"), encoding="utf-8").read().strip()
    s = re.sub(r"\{\{table:([a-z0-9_]+)\}\}", tab, s)
    left = re.findall(r"\{\{[^}]*\}\}", s)
    assert not left, left
    out = os.path.join(R, name + ".md")
    open(out, "w", encoding="utf-8", newline="\n").write(s)
    return s


def eq_svg(tex, i):
    p = os.path.join(BUILD, "eq", f"eq{i:02d}.svg")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=13, color="#1f2328")
    fig.savefig(p, format="svg", bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    return os.path.relpath(p, BUILD).replace("\\", "/")


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; @bottom-center { content: counter(page); } }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #1f2328; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 6pt; }
h2 { font-size: 14pt; border-bottom: 1px solid #d0d7de; padding-bottom: 3pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 12pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 8pt; width: 100%; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 2.5pt 4pt; vertical-align: top; }
th { background: #f6f8fa; text-align: left; }
code { font-family: Consolas, "SFMono-Regular", monospace; font-size: 8.6pt; background: #f6f8fa; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f6f8fa; padding: 6pt 8pt; font-size: 8pt; overflow-x: hidden; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
.eq { text-align: center; margin: 8pt 0; page-break-inside: avoid; }
.eq img { display: inline; margin: 0; }
blockquote { border-left: 3px solid #d0d7de; margin: 8pt 0; padding: 2pt 10pt; color: #59636e; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 14pt 0; }
a { color: #0969da; text-decoration: none; }
"""


def to_html(md_text, title, depth_to_root):
    eqs = []

    def keep(m):
        eqs.append(m[1].strip())
        return f"\n<div class=\"eq\"><img src=\"{eq_svg(m[1].strip(), len(eqs))}\" alt=\"{html.escape(m[1].strip())}\"></div>\n"
    body_md = re.sub(r"\$\$(.+?)\$\$", keep, md_text, flags=re.S)
    md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table")
    body = md.render(body_md)
    body = re.sub(r'src="(?!eq/|data:|http)([^"]+)"', lambda m: f'src="{depth_to_root}{m[1]}"', body)
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>", len(eqs)


for name in ("TECHNICAL_REPORT",):
    text = assemble(name)
    title = text.splitlines()[0].lstrip("# ").strip()
    page, n = to_html(text, title, "../")
    open(os.path.join(BUILD, name + ".html"), "w", encoding="utf-8").write(page)
    print(f"{name}: {len(text):,} chars, {n} equations -> report/{name}.md, report/build/{name}.html")
