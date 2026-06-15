"""
Shared figure style for the NESI main-paper figures.

Locks in one visual identity across all regenerable figures (Figs 2-6) so the
set is harmonized for journal submission (Scientific Data / Nature Portfolio):
  - single sans-serif family (Arial) at consistent sizes
  - one NESI colormap (red-blue diverging, red = more impaired)
  - one 4-dataset categorical palette (Okabe-Ito, colorblind-safe)
  - bold uppercase top-left panel labels
  - export to high-res PNG + PDF at >= 300 DPI

Import and call apply_style() at the top of each figure script, then use
save_fig(fig, "Figure5") to write both Figure5.png and Figure5.pdf into
MainPaperFigures/.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- locations -------------------------------------------------------------
CODES_DIR = Path(__file__).resolve().parent
FIG_DIR = CODES_DIR.parent          # MainPaperFigures/

# ---- canonical colors ------------------------------------------------------
# NESI continuous: diverging, red = more impaired (high NESI), blue = less.
NESI_CMAP = "RdBu_r"
NESI_VMIN, NESI_VMAX = -3, 3

# Okabe-Ito colorblind-safe palette for the four clinical scales, used wherever
# RASS/GCS/CAMS/ICANS appear as distinct series/panels.
DATASET_COLORS = {
    "RASS":  "#0072B2",   # blue
    "GCS":   "#E69F00",   # orange
    "CAMS":  "#009E73",   # green
    "CAM-S": "#009E73",
    "ICANS": "#CC79A7",   # reddish purple
}

# Two-series line palette (NESI vs GCS in Fig 5). NESI = black (hero, distinct
# from the dataset palette); GCS uses its dataset color for cross-figure
# consistency with Fig 2/Fig 4.
SERIES_COLORS = {"NESI": "#000000", "GCS": DATASET_COLORS["GCS"]}

# Identical NESI colorbar label text wherever the diverging NESI scale appears
# (Fig 3, Fig 6) so the same value is described the same way.
NESI_CBAR_LABEL = "NESI (z-scored; red = more impaired)"

# ---- typography / rc -------------------------------------------------------
FS_TITLE = 11      # panel labels / bold headers
FS_AXLABEL = 10    # axis titles
FS_TICK = 8        # tick labels
FS_LEGEND = 8

def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.labelsize": FS_AXLABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType (editable text in PDF)
        "ps.fonttype": 42,
    })

def panel_label(ax, letter, x=-0.02, y=1.04, fontsize=FS_TITLE + 1):
    """Bold uppercase panel label at top-left, journal convention."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="right")

def save_fig(fig, basename, dpi=400):
    """Write <basename>.png and <basename>.pdf into MainPaperFigures/."""
    png = FIG_DIR / f"{basename}.png"
    pdf = FIG_DIR / f"{basename}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {png}\nsaved {pdf}")
