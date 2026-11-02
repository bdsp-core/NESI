#!/usr/bin/env python3
"""Supp Figs 18-23: replace calendar timestamps on the time axes with elapsed hours. Edits pixels only."""
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ["NESI_DATA_ROOT"])
F = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "figures" / "final"
# Recording length in hours, from each figure's first and last axis labels
HOURS = {18: 15.64, 19: 20.74, 20: 21.48, 21: 23.82, 22: 3.97, 23: 14.93}
X0, X1, TICKS = 81, 1985, 10
# (top of tick labels, top of axis title, bottom of axis title), for the two time axes
AXES = [(671, 704, 722), (1115, 1149, 1167)]
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
tick, title = ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT if len(sys.argv) < 2 else sys.argv[1], 16)

for k, hours in HOURS.items():
    p = F / f"SuppFig{k}.png"
    im = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(im)
    for top, mid, bottom in AXES:
        d.rectangle((0, top, im.size[0], bottom), fill="white")
        for i in range(TICKS):
            x = X0 + i * (X1 - X0) / (TICKS - 1)
            d.text((x, top + 3), f"{i * hours / (TICKS - 1):.1f}", fill="black", font=tick, anchor="ma")
        d.text(((X0 + X1) / 2, top + 25), "Elapsed time (hours)", fill="black", font=title, anchor="ma")
    im.save(p)
