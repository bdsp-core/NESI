# Revision analyses: source code

Code from the 2026 revision that starts from raw EEG or MORGOTH activations, which are not public. Each
analysis here has a fast route in `reproduce/` that regenerates the reported numbers and figures from saved
intermediates in minutes (see `reproduce/README.md`).

| Folder | What it does | Fast route in `reproduce/` |
|---|---|---|
| `ICANS_Awake_Segment_Ablation/` | Selects random-awake and all-day ICANS segments and scores them with the published model (Supp Fig 32) | `icans_segment_ablation.py` (saved segment scores) |
| `TraditionalFeaturevsNESI/` | MATLAB qEEG features and NESI heads trained on them (Supp Fig 30) | none yet: needs per-channel qEEG features for the test set |
| `LongEEG_NESI_Tracking/` | Continuous NESI on full-length recordings and the trajectory plots (Supp Figs 18–23) | none yet: needs per-recording NESI series |

Analyses from this folder that were redone on the corrected split (leave-one-cohort-out, the 24 SPECTRA
mappings), or that duplicated existing scripts, were removed on 2026-11-02; they remain in git history. The
indication analysis moved to `reproduce/indication_stratified.py`. Patient-level files are in credentialed
storage (`DATA_MOVED.md`).
