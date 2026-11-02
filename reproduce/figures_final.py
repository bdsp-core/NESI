#!/usr/bin/env python3
"""Figs 2-4 with their final labels, from the repo's plotting scripts.

The scripts themselves stay as they are: each is copied into the working copy
built by run_all.py with the label changes below, and run there. Fig 2's
patient-level CIs are cached in NESI_DATA_ROOT/intermediates/fig2_patient_ci.csv,
so later runs only replot.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build"))
WORK, OUT, INTER = BUILD / "NESI", BUILD / "output" / "figures" / "final", ROOT / "intermediates"
SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
FOOT2 = (
    "Boxes: median & IQR; whiskers: 1.5×IQR; points: outliers. Values above each subplot indicate Spearman's ρ with 95% patient-level bootstrap confidence intervals.\\n"
    "n denotes the number of EEG observations at each clinical score level. Higher NESI values indicate greater neurological impairment.\\n"
    "Panel A: Scale-specific bespoke models: Trained independently on data from one of the four clinical scales (RASS, GCS, CAM-S, or ICANS) in SPECTRA.\\n"
    "Panel B: Universal model: Trained jointly on data pooled across all four clinical scales in SPECTRA."
)


def patient_ci(boot=1000):
    """95% CIs resampling patients, for the universal and bespoke models on the test set."""
    path = INTER / "fig2_patient_ci.csv"
    if path.exists():
        return pd.read_csv(path)
    rng = np.random.default_rng(0)
    f = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
    rows = []
    for ds in SCALES:
        b = pd.read_csv(ROOT / f"github_moved/NESI/Bespoke_models/Results/{ds}Bespokescore_results.csv")
        t = f[(f.Dataset == ds) & (f.Split == "Test")].merge(b[["MorgothOutputFilename", "BespokeBadnessScore"]], on="MorgothOutputFilename").reset_index(drop=True)
        groups = list(t.groupby("BDSPPatientID").indices.values())
        for model, col in [("universal", "NESI"), ("bespoke", "BespokeBadnessScore")]:
            a, x, v = t.RawScore.values, t[col].values, []
            for _ in range(boot):
                i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                v.append(abs(stats.spearmanr(a[i], x[i])[0]))
            rows.append(dict(Scale=ds, Model=model, lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    r = pd.DataFrame(rows)
    INTER.mkdir(exist_ok=True)
    r.to_csv(path, index=False)
    return r


# CIs as printed in the manuscript's Results text. Fresh bootstraps agree to within Monte Carlo error (checked in main).
REPORTED_CI = {
    ("RASS", "universal"): (0.66, 0.76), ("GCS", "universal"): (0.68, 0.76), ("CAMS", "universal"): (0.79, 0.88), ("ICANS", "universal"): (0.72, 0.90),
    ("RASS", "bespoke"): (0.60, 0.71), ("GCS", "bespoke"): (0.63, 0.72), ("CAMS", "bespoke"): (0.70, 0.84), ("ICANS", "bespoke"): (0.62, 0.84),
}


def run(rel, patches, outputs):
    """Run a patched copy of a script in the working copy, then collect its figures."""
    src = WORK / rel
    text = src.read_text()
    for old, new in patches:
        if old not in text:
            sys.exit(f"Expected text not found in {rel}: {old}")
        text = text.replace(old, new)
    tmp = src.with_name(f"_final_{src.name}")
    tmp.write_text(text)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore", PYTHONPATH=str(BUILD / "stubs"), OMP_NUM_THREADS="4")
    logdir = BUILD / "output" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(HERE / "_run_script.py"), str(tmp), str(OUT / "_extra")], cwd=tmp.parent, env=env, capture_output=True, text=True)
    (logdir / f"final_{src.stem}.log").write_text(r.stdout[-100000:] + "\n--STDERR--\n" + r.stderr[-20000:])
    tmp.unlink()
    if r.returncode:
        sys.exit(f"{rel} failed: {r.stderr.strip().splitlines()[-1]}")
    for a, b in outputs:
        shutil.copy2(WORK / a, OUT / b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ci = patient_ci()
    for r in ci.itertuples():
        lo, hi = REPORTED_CI[(r.Scale, r.Model)]
        if abs(r.lo - lo) > 0.015 or abs(r.hi - hi) > 0.015:
            sys.exit(f"Fig 2 CI for {r.Scale} {r.Model} is {r.lo:.3f}-{r.hi:.3f}; the manuscript says {lo}-{hi}")
    table = REPORTED_CI
    run(
        "figures/main/Codes/FAST_NESIvsBeSpoke_plot.py",
        [
            ("    def _rho_compact(s):\n", f"    PT_CI = {table!r}\n\n    def _rho_ci(s, name, model):\n        m = float(re.findall(r'-?\\d+\\.\\d+', str(s))[0])\n        lo, hi = PT_CI[(name, model)]\n        return f\"ρ = {{m:.2f}} [{{lo:.2f}}, {{hi:.2f}}]\"\n\n    def _rho_compact(s):\n"),
            ('ax_top.set_title(f"{name}\\n{_rho_compact(rho_ind)}"', 'ax_top.set_title(f"{shown}\\n{_rho_ci(rho_ind, name, \'bespoke\')}"'),
            ('ax_bot.set_title(f"{name}\\n{_rho_compact(rho_global)}"', 'ax_bot.set_title(f"{shown}\\n{_rho_ci(rho_global, name, \'universal\')}"'),
            ('ax_bot.set_xlabel(f"{name} score")', 'ax_bot.set_xlabel(f"{shown} score")'),
            ('        face = DATASET_COLORS.get(name, "#9ecae1")', '        shown = name.replace("CAMS", "CAM-S")\n        face = DATASET_COLORS.get(name, "#9ecae1")'),
            ("rotation=0, fontsize=FS_TICK - 2.5", 'rotation=45 if name == "GCS" else 0, fontsize=FS_TICK - 2.5'),
            ('"Bespoke model\'s NESI"', '"Scale-specific bespoke\\nmodel\'s NESI"'),
            ('"Boxes: median & IQR; whiskers: 1.5×IQR; points: outliers."', f'"{FOOT2}"'),
            ("ha='center', fontsize=FS_LEGEND, style='italic')", "ha='center', va='bottom', fontsize=FS_LEGEND, style='italic')"),
            ("plt.tight_layout(rect=[0, 0.03, 1, 1])", "plt.tight_layout(rect=[0, 0.12, 1, 1])"),
        ],
        [("figures/main/Figure2.png", "Fig2.png")],
    )
    run(
        "NESI/ScaleVsNESI/fit_nesi_curves.py",
        [
            ('title_text = f"{sy} vs NESI"', 'title_text = f"{sy} vs NESI".replace("CAMS", "CAM-S")'),
            ('title_text = f"{sy} vs {sx}"', 'title_text = f"{sy} vs {sx}".replace("CAMS", "CAM-S")'),
            ('ax.set_ylabel(sy, fontsize=10)', 'ax.set_ylabel(sy.replace("CAMS", "CAM-S"), fontsize=10)'),
            ('ax.set_xlabel("NESI" if i == j else sx, fontsize=10)', 'ax.set_xlabel("NESI" if i == j else sx.replace("CAMS", "CAM-S"), fontsize=10)'),
        ],
        [("figures/main/Figure3.png", "Fig3.png")],
    )
    run(
        "figures/main/Codes/FAST_NESI_group_morgothactivation.py",
        [("f'{scale}\\n(n={end - start})'", "f'{scale}\\n(n={end - start})'.replace('CAMS', 'CAM-S')")],
        [("figures/main/Figure4.png", "Fig4.png")],
    )


if __name__ == "__main__":
    main()
