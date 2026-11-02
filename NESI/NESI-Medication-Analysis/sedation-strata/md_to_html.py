# md_to_html.py
"""
Converts every report_*.md in the model folders to .html, for copying tables into Outlook.
Open the .html in a browser, select a table, copy, paste into Outlook: it stays a table.
No models are run. Run with Play. Paths are printed.
"""

import html
import re
from pathlib import Path

# Repo layout: med_config.py lives in ../medication-exposure (no effect if already importable)
import sys as _sys
from pathlib import Path as _Path
_MC = _Path(__file__).resolve().parent.parent / "medication-exposure"
if _MC.is_dir() and str(_MC) not in _sys.path:
    _sys.path.insert(0, str(_MC))
from med_config import I0001_ASM_ANTIPSYCH_DIR

ROOT = Path(I0001_ASM_ANTIPSYCH_DIR) / "stratified_feasibility" / "model"

CSS = """
body { font-family: Calibri, Arial, sans-serif; font-size: 11pt; margin: 24px; max-width: 1100px; }
h1 { font-size: 15pt; } h2 { font-size: 12.5pt; margin-top: 22px; }
table { border-collapse: collapse; margin: 8px 0 14px; }
th, td { border: 1px solid #999; padding: 3px 8px; font-size: 10pt; }
th { background: #e8e8e8; text-align: left; }
td:not(:first-child) { white-space: nowrap; }
p { margin: 6px 0; }
"""


def inline(t):
    t = html.escape(t)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", t)


def convert(md):
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            body = [c for c in cells[1:] if not all(re.fullmatch(r":?-+:?", x or "-") for x in c)]
            out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells[0]) + "</tr>")
            for r in body:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</table>")
            continue
        if ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln:
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out)


print("SEARCHING")
print(f"  {ROOT.resolve()} (and subfolders)\n")
n = 0
for md in sorted(ROOT.rglob("report_*.md")):
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{md.stem}</title>"
            f"<style>{CSS}</style></head><body>{convert(md.read_text(encoding='utf-8'))}</body></html>")
    out = md.with_suffix(".html")
    out.write_text(page, encoding="utf-8")
    print(f"  wrote {out.resolve()}")
    n += 1
print(f"\n{n} report(s) converted." if n else "No report_*.md files found.")
