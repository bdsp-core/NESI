# obs_distribution.py
"""
Observations per patient, by stratum, to choose the cap K for the cap sensitivity analysis.
Counts only: NESI is not read, so choosing K here does not depend on any result.

Uses the same rows as agent_stratified_model.py (same exclusions, RASS -5 to 0).
Outputs (in stratified_feasibility/model/obs_distribution/):
  - obs_per_patient_percentiles.csv : percentiles of obs per patient, by stratum
  - cap_candidates.csv              : for candidate K, % of rows dropped and % of patients
                                      affected in each stratum, and the top-5-patient share after capping
  - obs_per_patient_distribution.png/.svg : cumulative distribution, one line per stratum

Run with Play. Paths are printed at start.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo layout: med_config.py lives in ../medication-exposure (no effect if already importable)
import sys as _sys
from pathlib import Path as _Path
_MC = _Path(__file__).resolve().parent.parent / "medication-exposure"
if _MC.is_dir() and str(_MC) not in _sys.path:
    _sys.path.insert(0, str(_MC))
from med_config import I0001_ASM_ANTIPSYCH_DIR

IN_DIR  = Path(I0001_ASM_ANTIPSYCH_DIR) / "stratified_feasibility"
IN_ROWS = IN_DIR / "strata_analysis_rows.csv"
OUT_DIR = IN_DIR / "model" / "obs_distribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATUM_COL = "stratum"
REFERENCE   = "no sedative or opioid"
EXCLUDE     = {"intermittent dose involved", "unknown medication status", "other combinations"}
CANDIDATE_K = [5, 10, 15, 20, 30, 50]
PCTS        = [50, 75, 90, 95, 99]

print("INPUT")
print(f"  {IN_ROWS.resolve()}")
print("OUTPUT")
print(f"  {OUT_DIR.resolve()}\n")
if not IN_ROWS.exists():
    sys.exit(f"File not found: {IN_ROWS.resolve()}")

d = pd.read_csv(IN_ROWS, usecols=["_pid", "rass", STRATUM_COL], low_memory=False)
d = d.rename(columns={STRATUM_COL: "grp"})
d = d[~d["grp"].isin(EXCLUDE)].dropna(subset=["rass", "grp"])
d = d[d["rass"].between(-5, 0)]
n = d.groupby(["grp", "_pid"]).size().rename("n").reset_index()   # obs per patient per stratum
order = [REFERENCE] + [g for g in n.groupby("grp")["n"].sum().sort_values(ascending=False).index
                       if g != REFERENCE]

# 1. Percentiles
rows = []
for g in order:
    x = n.loc[n["grp"] == g, "n"].sort_values(ascending=False)
    r = {"stratum": g, "patients": len(x), "obs": int(x.sum()), "mean": round(x.mean(), 1)}
    r.update({f"p{p}": int(np.percentile(x, p)) for p in PCTS})
    r["max"] = int(x.max())
    r["% obs from top 5 patients"] = round(100 * x.head(5).sum() / x.sum(), 1)
    rows.append(r)
pct = pd.DataFrame(rows)
pct.to_csv(OUT_DIR / "obs_per_patient_percentiles.csv", index=False)
print("Observations per patient, by stratum:")
print(pct.to_string(index=False))

# 2. What each candidate K would do
rows = []
for g in order:
    x = n.loc[n["grp"] == g, "n"]
    for k in CANDIDATE_K:
        capped = np.minimum(x, k).sort_values(ascending=False)
        rows.append({"stratum": g, "K": k,
                     "% rows dropped": round(100 * (1 - capped.sum() / x.sum()), 1),
                     "% patients capped": round(100 * (x > k).mean(), 1),
                     "top-5 share after cap (%)": round(100 * capped.head(5).sum() / capped.sum(), 1)})
cand = pd.DataFrame(rows)
cand.to_csv(OUT_DIR / "cap_candidates.csv", index=False)
print("\n% of rows dropped by candidate K:")
print(cand.pivot(index="stratum", columns="K", values="% rows dropped").loc[order].to_string())
print("\nTop-5-patient share of observations after capping (%):")
print(cand.pivot(index="stratum", columns="K", values="top-5 share after cap (%)").loc[order].to_string())

# 3. Cumulative distribution plot
try:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(order):
        x = np.sort(n.loc[n["grp"] == g, "n"].values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post",
                color="black" if g == REFERENCE else cmap(i % 10),
                lw=2.2 if g == REFERENCE else 1.2,
                label=("Reference (no sedative or opioid)" if g == REFERENCE else g) + f" (n={len(x)})")
    for k in CANDIDATE_K:
        ax.axvline(k, color="#C9CCD1", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Observations per patient within stratum (log scale; dotted = candidate K)")
    ax.set_ylabel("Share of patients with at most this many")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"obs_per_patient_distribution.{ext}", dpi=200)
    plt.close(fig)
except Exception as e:
    print(f"Plot skipped: {e}")

print(f"\nSaved to {OUT_DIR.resolve()}")
