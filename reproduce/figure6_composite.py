#!/usr/bin/env python3
"""Fig. 6: six-panel PaCMAP composite, from saved coordinates and Qwen3 indication labels.

Follows Arka Roy's NESI/PacMAP/STEP3B_NESI_composite_pacmap.py. Panels A-C use the saved coordinates
(NESI/PacMAP/NESI_pacmap_coords.csv). Panels D-F flag windows whose EEG report the Qwen3-14B labeler
assigned Post-Cardiac Arrest or Stroke / Cerebrovascular Disease (intermediates/indication_llm_labels.csv,
matched by cohort and filename). Panel letters and selection rules follow the published figure,
including its D, F, E order.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output"
(BUILD / "figures").mkdir(parents=True, exist_ok=True)
(BUILD / "tables").mkdir(parents=True, exist_ok=True)

IIIC_PALETTE = {"Seizure": "#E13238", "LPD": "#F08C2A", "GPD": "#F2D549", "LRDA": "#AABF45",
                "GRDA": "#7AC8E3", "Other": "#5C3A87", "Burst": "#000000"}
IIIC_LEGEND_ORDER = ["Burst", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
CAT_PROB_COL = {"Seizure": "IIIC_Seizure", "LPD": "IIIC_LPD", "GPD": "IIIC_GPD", "LRDA": "IIIC_LRDA",
                "GRDA": "IIIC_GRDA", "Other": "IIIC_Other", "Burst": "Burst_vs_NoBurst"}
NAMES = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}
# Severe states (panel C), as in the published call
EXTREME = {"RASS": ([-5], 1.5, "#E13238"), "GCS": ([3], 1.5, "#000000"),
           "CAMS": ([7, 6], 0.0, "#1F77B4"), "ICANS": ([4, 3], -0.05, "green")}
# Indication panels, in published grid order: (title, indication flag, label, pattern, threshold, NESI min)
PANELS = [("D. Post-Cardiac Arrest with Burst", "Arrest", "Post-Cardiac Arrest", "Burst", 0.7, 0.0),
          ("F. Post-Cardiac Arrest with GPD", "Arrest", "Post-Cardiac Arrest", "GPD", 0.7, 0.0),
          ("E. Stroke with LPD", "Stroke", "Stroke", "LPD", 0.7, 0.25)]


def indications(df):
    """Per-window indication flags from the Qwen3 label of the window's own EEG report, matched by
    cohort and filename as in the published script."""
    lab = pd.read_csv(ROOT / "intermediates" / "indication_llm_labels.csv", usecols=["Dataset", "Filename", "IndicationNeuroreport_LLM"])
    norm = lambda x: x.astype("string").str.strip().str.replace(r"(?i)\.csv$", "", regex=True)
    lab = lab.assign(ds=lab.Dataset.astype("string").str.strip().str.upper(), fn=norm(lab.Filename)).dropna(subset=["fn"])
    lab = lab[lab.fn != ""].drop_duplicates(["ds", "fn"])
    key = pd.DataFrame({"ds": df.Dataset.astype("string").str.strip().str.upper(), "fn": norm(df.MorgothOutputFilename)})
    m = key.merge(lab[["ds", "fn", "IndicationNeuroreport_LLM"]], on=["ds", "fn"], how="left")
    m.index = df.index
    txt = m.IndicationNeuroreport_LLM
    out = pd.DataFrame({f: txt.str.contains(pat, case=False, regex=False).fillna(False).astype(bool)
                        for f, pat in [("Arrest", "Post-Cardiac Arrest"), ("Stroke", "Stroke / Cerebrovascular Disease")]})
    return out, txt.notna()


def category_cmap(cat):
    base = IIIC_PALETTE[cat]
    pale = tuple(0.8 + 0.2 * np.array(to_rgb(base)))
    if cat == "Other":
        return LinearSegmentedColormap.from_list(cat, [base, tuple(0.85 + 0.15 * np.array(to_rgb(base)))])
    return LinearSegmentedColormap.from_list(cat, [pale, base])


def iiic_labels(df, burst=0.5):
    cols = ["IIIC_Seizure", "IIIC_LPD", "IIIC_GPD", "IIIC_LRDA", "IIIC_GRDA", "IIIC_Other"]
    lab = np.array(["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"])[np.argmax(df[cols].to_numpy(), axis=1)]
    lab[df.Burst_vs_NoBurst.to_numpy() >= burst] = "Burst"
    rgb = np.zeros((len(df), 3))
    for cat in IIIC_PALETTE:
        m = lab == cat
        if m.any():
            pr = df.loc[m, CAT_PROB_COL[cat]].to_numpy()
            lo, hi = np.quantile(pr, [0.05, 0.95]) if len(pr) > 20 else (pr.min(), pr.max())
            rgb[m] = category_cmap(cat)(np.clip((pr - lo) / max(hi - lo, 1e-6), 0, 1))[:, :3]
    return lab, rgb


df = pd.read_csv(ROOT / "github_moved/NESI/PacMAP/NESI_pacmap_coords.csv", low_memory=False)
ind, linked = indications(df)

# Same orientation, jitter and styling as the published script
xy = df[["pacmap_x_nesi", "pacmap_y_nesi"]].to_numpy(float)
th = np.deg2rad(90)
xy = (xy - xy.mean(0)) @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).T
xy[:, 1] *= -1
J, rng = 4, np.random.default_rng(0)
sx, sy = xy.std(0) * 0.006
xa = np.vstack([xy] + [xy + np.column_stack([rng.normal(0, sx, len(df)), rng.normal(0, sy, len(df))]) for _ in range(J)])
tile = lambda a: np.tile(a, J + 1) if a.ndim == 1 else np.tile(a, (J + 1, 1))
nesi = pd.to_numeric(df.NESI, errors="coerce").to_numpy()
lab, rgb = iiic_labels(df)
ds = df.Dataset.astype(str).str.upper().to_numpy()
score = pd.to_numeric(df.TrueRawScores, errors="coerce").to_numpy()
ext = np.full(len(df), "Other", dtype=object)
for d, (vals, nmin, _) in EXTREME.items():
    ext[(ds == d) & np.isin(score, vals) & (nesi >= nmin)] = d

S, LF, TF = 4.0, 10, 14
fig = plt.figure(figsize=(20, 13))
gs = fig.add_gridspec(2, 3, wspace=0.06, hspace=0.08, left=0.03, right=0.97, top=0.96, bottom=0.03)
ax = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

na = tile(nesi)
o = np.argsort(np.abs(na))
sc = ax[0].scatter(xa[o, 0], xa[o, 1], c=na[o], cmap="RdBu_r", vmin=-3, vmax=3, s=S, edgecolors="none", alpha=0.55)
cb = fig.colorbar(sc, ax=ax[0], orientation="horizontal", fraction=0.035, pad=0.02, shrink=0.65)
cb.set_label("NESI", fontsize=LF); cb.ax.tick_params(labelsize=LF); cb.set_ticks(range(-3, 4))

la, ra = tile(lab), tile(rgb)
for cat in ["Other", "GRDA", "LRDA", "GPD", "LPD", "Seizure", "Burst"]:
    m = la == cat
    ax[1].scatter(xa[m, 0], xa[m, 1], c=ra[m], s=S, edgecolors="none", alpha=0.55)
cnt = pd.Series(lab).value_counts()
ax[1].legend(handles=[mpatches.Patch(color=IIIC_PALETTE[c], label=f"{c} (n={int(cnt.get(c, 0))})") for c in IIIC_LEGEND_ORDER],
             loc="upper right", fontsize=LF, frameon=False, title="IIIC category", title_fontsize=LF)

ea = tile(ext)
m = ea == "Other"
ax[2].scatter(xa[m, 0], xa[m, 1], c="#D3D3D3", s=S, edgecolors="none", alpha=0.30)
h = []
for d, (vals, _, col) in EXTREME.items():
    m = ea == d
    ax[2].scatter(xa[m, 0], xa[m, 1], c=col, s=S, edgecolors="none", alpha=0.75)
    h.append(mpatches.Patch(color=col, label=f"{NAMES[d]} {', '.join(map(str, vals))}, (n={int((ext == d).sum())})"))
h.append(mpatches.Patch(color="#D3D3D3", label=f"Other (n={int((ext == 'Other').sum())})"))
ax[2].legend(handles=h, loc="upper right", fontsize=LF, frameon=False, title="Clinical state", title_fontsize=LF)

rows = []
for a, (title, flag, name, pat, thr, nmin) in zip(ax[3:], PANELS):
    sel = ind[flag].to_numpy() & (pd.to_numeric(df[CAT_PROB_COL[pat]], errors="coerce").to_numpy() >= thr) & (nesi >= nmin)
    s = tile(sel)
    a.scatter(xa[~s, 0], xa[~s, 1], c="#D3D3D3", s=S, edgecolors="none", alpha=0.30, zorder=1)
    o = np.argsort(np.abs(na[s]))
    a.scatter(xa[s][o, 0], xa[s][o, 1], c=na[s][o], cmap="RdBu_r", vmin=-3, vmax=3, s=S, edgecolors="none", alpha=0.75, zorder=2)
    a.legend(handles=[mpatches.Patch(color="#D62728", label=f"{name} (n={int(sel.sum())})"),
                      mpatches.Patch(color="#D3D3D3", label=f"Other (n={int(len(df) - sel.sum())})")],
             loc="upper right", fontsize=LF, frameon=False, title=f"{pat} Probability ≥ {thr:g}", title_fontsize=LF)
    a.set_title(title, fontsize=TF, fontweight="bold")
    rows.append(dict(Panel=title, Indication=flag, Pattern=pat, Selected=int(sel.sum()), IndicationRows=int(ind[flag].sum())))
for a, t in zip(ax[:3], ["A. NESI (continuous)", "B. Dominant IIIC category", "C. Severe clinical states"]):
    a.set_title(t, fontsize=TF, fontweight="bold")
for a in ax:
    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal", adjustable="box")
    for sp in a.spines.values():
        sp.set_visible(False)
fig.savefig(BUILD / "figures" / "Figure6_composite.png", dpi=200, bbox_inches="tight")

counts = pd.DataFrame(rows)
for k, v in cnt.items():
    counts.loc[len(counts), ["Panel", "Selected"]] = [f"B. {k}", int(v)]
for d in list(EXTREME) + ["Other"]:
    counts.loc[len(counts), ["Panel", "Selected"]] = [f"C. {d}", int((ext == d).sum())]
counts.to_csv(BUILD / "tables" / "figure6_counts.csv", index=False)
print(f"{len(df):,} windows; {int(linked.sum()):,} with an indication label")
print(counts.to_string(index=False))
