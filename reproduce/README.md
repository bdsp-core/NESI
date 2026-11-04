# Reproducing the manuscript

`run_all.py` is the single entry point. It builds a working copy of the code with the credentialed data linked in, recomputes manuscript numbers, runs the existing figure and table scripts, and compares each registered number with its manuscript value in `claims.csv`.

```bash
export NESI_DATA_ROOT=/path/to/data        # holds github_moved/, s3/ and drive/
export NESI_BUILD_ROOT=/path/to/build      # optional; working copy and output/
export NESI_TORCH_PYTHON=/path/to/python   # optional; a Python with requirements.txt installed
python reproduce/run_all.py                # everything available
python reproduce/run_all.py --list         # every figure and table, with status
```

Results are written to `$NESI_BUILD_ROOT/output/`: `REPORT.md`, `claims_checked.csv`, `numbers.json`, `tables/` and `logs/`.

## Three tiers

Checking a number or adjusting a figure never needs a retrain or raw EEG. `run_all.py` works from saved intermediates, and each intermediate has its own documented route back to source data.

| Tier | What | Time |
|---|---|---|
| 1 | Source data to intermediates, table below. Run once. | Hours |
| 2 | Saved intermediates in `$NESI_DATA_ROOT/intermediates/` and `retrain/`, with checksums in `intermediates/MANIFEST.csv` (`run_all.py --freeze`). | |
| 3 | `run_all.py`: every number, figure and table from the intermediates. | Minutes |

| Intermediate | Made by | Time |
|---|---|---|
| Encoder and head weights, MORGOTH features, per-segment NESI | `NESI/model/Training/`, then `NESI/model/Testing/` | About 2.5 h on one GPU, after feature extraction |
| `retrain/*/test_scores.csv`: sensitivity retrains | `retrain_split_fix.py` and its options | About 2.5 h per run |
| `retrain/loco_*`, `retrain/mapping_*`: leave-one-cohort-out and SPECTRA mappings | `retrain_split_fix.py --leave-out`, `--mapping` | About 2.5 h per run |
| `intermediates/mortality/`: NESI on the mortality cohort | `mortality.py features` | About 5 h |
| `intermediates/sedation/`: medication models and bootstrap draws (1,000 resamples), restored into `sedation_repo/` by the fast route | `sedation.py --from-source` | About 7 h on 16 cores |
| `intermediates/artifact_per_segment.csv` | `artifact_tolerance.py --refresh` | About 1 h, depending on the network |
| `intermediates/rass_positive_features.npz` | Delete it, then run `rass_positive.py` | Minutes, depending on the network |
| `intermediates/indication_llm_labels.csv`: Qwen3-14B indication label per EEG report (labels and IDs only) | Arka Roy's labeling run; full files with report-derived text at `s3://bdsp-opendata-credentialed/yama/RevisionRelatedMaterials/EEGIndication_Analysis/` | Hours on one GPU |
| `intermediates/icans_segment_ablation/`: NESI for random-awake and all-day ICANS segments | `RevisionRelatedMaterials/ICANS_Awake_Segment_Ablation/` (needs MORGOTH activations) | Hours |

Medication steps use `NESI_SEDATION_PYTHON`, a Python with `pandas<3` and the packages in `NESI/NESI-Medication-Analysis`.

## Data

`github_moved/` mirrors the paths in [`docs/DATA_FILES.md`](../docs/DATA_FILES.md). `s3/` holds files from `s3://bdsp-opendata-credentialed/yama/`. `drive/` holds `i0001_RASS_Meds_NESI.csv`, which is also in the S3 release at `yama/NESI/NESI-Medication-Analysis/sedation-strata/data/`.

## Coverage

`python reproduce/run_all.py --list` shows how every figure and table is produced. Three items are waiting for a saved input, and their code is in the repo: Supplementary Figs 18–23 (per-recording NESI series for six full-length recordings; `RevisionRelatedMaterials/LongEEG_NESI_Tracking/`) and Supplementary Fig 30 (per-channel qEEG features for the test set; `RevisionRelatedMaterials/TraditionalFeaturevsNESI/`), both requested from Arka Roy, and the demographic, death, diagnosis and indication rows of Supplementary Table 20, which need Northwestern's per-patient file (the score and EEG-length rows are computed from `recordings.csv`). Supplementary Table 19 (the Eleveld implementation check) runs in `sedation.py`; `NESI_SEDATION_PYTHON` needs `pip install --no-deps python-anesthesia-simulator "control>=0.10.1" casadi`. `claims.csv` registers the manuscript's numbers; a `DIFF` in the report means a recomputed value differs from the manuscript. The final figure images are in `figures/final/`.
