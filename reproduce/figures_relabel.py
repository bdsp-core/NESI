#!/usr/bin/env python3
"""Corrected labels on figures with no plotting code in the repo: CAMS to CAM-S, Prediction to Estimate.

Only text is touched: each word is erased, redrawn at the original size and
position, and any text following it on the line is moved over to fit. Sources are
the published images, kept in NESI_DATA_ROOT/intermediates/figure_sources.
"""
import os
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ["NESI_DATA_ROOT"])
SRC = ROOT / "intermediates" / "figure_sources"
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "figures" / "final"
TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
DEJAVU, DEJAVU_B = TTF / "DejaVuSans.ttf", TTF / "DejaVuSans-Bold.ttf"
ARIAL, ARIAL_B = "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
SS = 4  # text is drawn at this multiple and reduced, to match the antialiasing around it


def font_fit(path, text, width):
    """Font at the size that sets `text` to `width` pixels at SS times the image scale."""
    size = 10
    while ImageFont.truetype(str(path), size + 1).getlength(text) <= width * SS:
        size += 1
    return ImageFont.truetype(str(path), size)


def relabel(a, box, old, new, font, keep_right=True):
    """Replace one word in an RGB array; returns the array with the rest of its line moved to fit."""
    l, t, w, h = box
    pad = max(2, h // 5)
    ring = np.concatenate([a[t - pad - 1, l - pad : l + w + pad], a[t + h + pad, l - pad : l + w + pad]])
    bg = np.median(ring, axis=0)
    patch = a[t : t + h, l : l + w].astype(float)
    dist = np.abs(patch - bg).sum(2)
    ink = patch[dist >= np.percentile(dist, 97)].mean(0)
    f = font_fit(font, old, w)
    delta = int(round((f.getlength(new) - f.getlength(old)) / SS))
    delta = max(delta, 0)  # a shorter word keeps its left edge
    left = delta // 2  # half goes into the gap before the word, half moves what follows
    right = delta - left
    top, bot = t - pad, t + h + pad + 2
    x = l + w + 1
    blank = np.abs(a[top:bot].astype(float) - bg).sum(2).max(0) < 40
    run, end = 0, x
    for c in range(x, a.shape[1]):
        run = run + 1 if blank[c] else 0
        if not blank[c]:
            end = c + 1
        if run > 1.6 * h:
            break
    if keep_right and end > x and right > 0:
        a[top:bot, x + right : end + right] = a[top:bot, x:end].copy()
    a[top:bot, l - pad : l + w + right + 1] = bg
    bx = (l - left - pad, top, w + delta + 2 * pad + 4, bot - top)
    im = Image.new("RGB", (bx[2] * SS, bx[3] * SS), tuple(int(v) for v in bg))
    d = ImageDraw.Draw(im)
    asc = f.getbbox(old)
    d.text(((pad) * SS - f.getbbox(new)[0], (t - top) * SS - asc[1] + (h * SS - (asc[3] - asc[1])) / 2), new, font=f, fill=tuple(int(v) for v in ink))
    a[bx[1] : bx[1] + bx[3], bx[0] : bx[0] + bx[2]] = np.asarray(im.resize((bx[2], bx[3]), Image.LANCZOS))
    return a


# name: replacements as (box, old, new, font); boxes are left, top, width, height in pixels
LEGEND = [(x, y, 33, 9) for y in (34, 52) for x in (722, 883, 1055, 1228)]
JOBS = {
    "SuppFig9": ("supp_image24.png", [(b, "CAMS", "CAM-S", DEJAVU) for b in LEGEND]),
    "SuppFig12": ("supp_image5.png", [((832, 1054, 80, 21), "CAMS", "CAM-S", DEJAVU)]),
    "SuppFig17": ("supp_image44.png", [((217, 1302, 43, 11), "CAMS", "CAM-S", DEJAVU)]),
    "SuppFig27": ("supp_image31.png", [((1092, 55, 57, 16), "CAMS", "CAM-S", DEJAVU)]),
    "SuppFig31": ("supp_image7.png", [((1267, 1839, 54, 15), "CAMS", "CAM-S", DEJAVU)]),
    "SuppFig37": (
        "supp_image51.png",
        [((1313, 11, 56, 14), "CAMS", "CAM-S", DEJAVU_B), ((1313, 593, 56, 14), "CAMS", "CAM-S", DEJAVU_B), ((1289, 1061, 51, 13), "CAMS", "CAM-S", DEJAVU)],
    ),
    "SuppFig18": ("supp_image30.png", [((1082, 787, 96, 13), "Prediction", "Estimate", DEJAVU_B)]),
    "SuppFig19": ("supp_image28.png", [((1082, 787, 96, 13), "Prediction", "Estimate", DEJAVU_B)]),
    "SuppFig20": ("supp_image38.png", [((1082, 787, 96, 13), "Prediction", "Estimate", DEJAVU_B)]),
    "SuppFig21": ("supp_image42.png", [((1076, 787, 96, 13), "Prediction", "Estimate", DEJAVU_B)]),
    "SuppFig22": ("supp_image36.png", [((1076, 787, 96, 13), "Prediction", "Estimate", DEJAVU_B)]),
    "SuppFig23": ("supp_image15.png", [((1076, 787, 96, 13), "Prediction", "Estimate", DEJAVU_B)]),
}


def fig1():
    """Fig 1, from the exported PNG. Its embedded draw.io source is not carried over, as it holds the old labels."""
    a = np.array(Image.open(SRC / "Figure1.drawio.png").convert("RGB"))
    a = relabel(a, (1793, 723, 149, 27), "Prediction", "Estimation", ARIAL, keep_right=False)
    a = relabel(a, (1706, 1414, 152, 23), "Prediction", "Estimation", ARIAL_B)
    a = relabel(a, (1252, 1713, 77, 20), "CAMS", "CAM-S", ARIAL)
    # Direction of the index, below the output box
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(ARIAL, 22)
    d.multiline_text((2082, 800), "Higher NESI =\ngreater impairment", font=f, fill=(0, 0, 0), anchor="ma", align="center", spacing=4)
    im.save(OUT / "Fig1.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (src, jobs) in JOBS.items():
        a = np.array(Image.open(SRC / src).convert("RGB"))
        for box, old, new, font in jobs:
            a = relabel(a, box, old, new, font)
        Image.fromarray(a).save(OUT / f"{name}.png")
    fig1()


if __name__ == "__main__":
    main()
