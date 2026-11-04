#!/usr/bin/env python3
"""Supplementary Fig. 30: NESI heads trained on conventional qEEG features (Arka Roy's analysis).

    python reproduce/qeeg_heads.py     # about a minute, from intermediates/qeeg_heads

Runs RevisionRelatedMaterials/TraditionalFeaturevsNESI/Result_All_model_qEEG_Fea.py unchanged on the
saved per-segment estimates of the four qEEG heads (intermediates/qeeg_heads/Results_NESI_from_qEEGFea,
from s3://bdsp-opendata-credentialed/yama/RevisionRelatedMaterials/TraditionalFeaturevsNESI/). The
per-cohort qEEG features the heads were trained on are beside them in CohortFeatureData/; the MATLAB
feature extraction and the head training scripts are in the same repo folder.
"""
import os
import runpy
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build"))
REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "RevisionRelatedMaterials/TraditionalFeaturevsNESI/Result_All_model_qEEG_Fea.py"
SAVED = ROOT / "intermediates/qeeg_heads/Results_NESI_from_qEEGFea"
OUT = BUILD / "output"


def main():
    # the script finds its data relative to a parent folder named NESI, so run a copy in the build area
    work = BUILD / "qeeg_heads" / "NESI" / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copy2(SRC, work / SRC.name)
    (work / "Results_NESI_from_qEEGFea").symlink_to(SAVED)
    stubs = BUILD / "stubs"  # the script imports torchsummary but never uses it
    stubs.mkdir(parents=True, exist_ok=True)
    (stubs / "torchsummary.py").write_text("def summary(*a, **k):\n    pass\n")
    sys.path.insert(0, str(stubs))
    g = runpy.run_path(str(work / SRC.name), run_name="__main__")
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    g["df_corr_results"].round(4).to_csv(OUT / "tables/qeeg_heads.csv", index=False)
    shutil.copy2(work / "qEEG_based_NESI.png", OUT / "figures/SuppFig30_qeeg_heads.png")


if __name__ == "__main__":
    main()
