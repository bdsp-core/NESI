# Figures

`final/` holds every figure exactly as it appears in the manuscript and supplement, one image per figure
(`Fig1`–`Fig8`, `GraphicalAbstract`, `SuppFig01`–`SuppFig37`; Supplementary Fig. 5 has three panels, a–c).

The scripts that draw them are listed by `python reproduce/run_all.py --list`. The original figure code in
`main/Codes/` and `supplementary_score_prediction/` is run unchanged by `run_all.py`, which writes its output
to `$NESI_BUILD_ROOT/output/figures/`, not to this folder.
