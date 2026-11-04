#!/usr/bin/env python3
"""Supplementary Figs 18-23: NESI over full-length recordings, with spectrograms and RASS or GCS.

    python reproduce/long_eeg_figures.py     # minutes; needs MATLAB with the Signal Processing Toolbox

Runs Arka Roy's MATLAB plotting functions (RevisionRelatedMaterials/LongEEG_NESI_Tracking/Code/
MatlabMDC_RASS and MatlabMDC_GCS, with the spectrogram helpers in cohort_models/ExternalValidation/
EEG_viewing_codes/Callbacks) on the saved inputs in intermediates/long_eeg, a copy of
s3://bdsp-opendata-credentialed/yama/RevisionRelatedMaterials/LongEEG_NESI_Tracking/:
  Products_<scale>_MDC/<scale>SegmentEEG/<recording>/<recording>_Segment.mat   EEG, RASS or GCS with times
  Products_<scale>_MDC/NESIPredictions/<recording>_Segment_predictions.csv      NESI at 1-s steps
intermediates/long_eeg/figures.csv says which recording is in which figure. The NESI series were made
from the raw EDFs (ContiniousNESI_*_EEG/) by RevisionRelatedMaterials/LongEEG_NESI_Tracking/Code/
*_continiousNESI_Prediction.py, which runs MORGOTH and the published NESI model.
Set NESI_MATLAB if matlab is not at the default path.
"""
import csv
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build"))
REPO = Path(__file__).resolve().parent.parent
DATA = ROOT / "intermediates" / "long_eeg"
CODE = REPO / "RevisionRelatedMaterials/LongEEG_NESI_Tracking/Code"
HELPERS = REPO / "cohort_models/ExternalValidation/EEG_viewing_codes/Callbacks"
MATLAB = os.environ.get("NESI_MATLAB", shutil.which("matlab") or "/Applications/MATLAB_R2025b.app/bin/matlab")
FOLDER = {"RASS": "RASSegmentEEG", "GCS": "GCSegmentEEG"}


def main():
    work = BUILD / "long_eeg"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    out = BUILD / "output" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(DATA / "figures.csv")))
    calls = []
    for r in rows:
        prod = DATA / f"Products_{r['scale']}_MDC"
        rec = r["recording"]
        eeg = prod / FOLDER[r["scale"]] / rec / f"{rec}_Segment.mat"
        nesi = prod / "NESIPredictions" / f"{rec}_Segment_predictions.csv"
        d = work / r["figure"]
        d.mkdir()
        calls.append(f"{r['scale']}_MDC_NESI_analysis4('{eeg}','{nesi}','{d}',\"yes\",[],[]);close all;")
    paths = [HELPERS, CODE / "MatlabMDC_RASS", CODE / "MatlabMDC_GCS"]
    script = "".join(f"addpath('{p}');" for p in paths) + "set(0,'DefaultFigureVisible','off');" + "".join(calls)
    subprocess.run([MATLAB, "-batch", script], check=True)
    for r in rows:
        png = sorted((work / r["figure"]).glob("*.png"))
        if len(png) != 1:
            raise SystemExit(f"Supp Fig {r['figure']}: expected one image, found {len(png)}")
        shutil.copy2(png[0], out / f"SuppFig{r['figure']}_long_eeg.png")
    print(f"{len(rows)} figures written to {out}")


if __name__ == "__main__":
    main()
