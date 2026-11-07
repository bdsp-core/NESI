#!/usr/bin/env python3
"""Figs 2-4 with their final labels, from the repo's plotting scripts.

The scripts themselves stay as they are: each is copied into the working copy
built by run_all.py with the label changes below, and run there. Fig 2's
patient-level CIs come from analyses.correlations, the same values the text reports.
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


def text_ci():
    """The patient-level CIs reported in the text, computed by analyses.correlations (claims.csv checks them), so the
    panel labels and the text cannot disagree."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyses

    ctx = analyses.Context(Path(__file__).resolve().parents[1], ROOT, BUILD / "output", 1000)
    analyses.correlations(ctx)
    n = ctx.num
    return {(ds, m): (round(float(n[f"{ds}.{k}_lo"]), 2), round(float(n[f"{ds}.{k}_hi"]), 2))
            for ds in SCALES for m, k in [("universal", "rho_pt"), ("bespoke", "bespoke_pt")]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = text_ci()
    print("Fig 2 patient-level CIs:", table)
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
