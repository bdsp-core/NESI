# Reproducing the NESI figures and tables

This file maps each figure and table in the manuscript to the code that produces it.
All paths are relative to the repository root. See `NESI/requirements.txt` for the
Python environment (Python 3.9).

## Data access

The EEG recordings and derived data come from the **Harvard EEG Database (HEEDB)** and are
published under credentialed access on the BDSP platform:

- Dataset: `s3://bdsp-opendata-credentialed/yama/`  (see `yama/README.md`)
- Project page / terms: https://bdsp.io/projects/1yizvr4l41wmiljc6lou/overview/

The data are de-identified (surrogate subject IDs; shifted dates) and available to credentialed
users under a Data Use Agreement; they are **not** redistributed in this repository. A unified
per-segment index with source-EEG provenance is at `yama/segment_index.csv`; the list of unique
continuous source recordings is at `yama/source_eeg_files.csv`. ICANS source EEGs live in the
companion dataset `s3://bdsp-opendata-credentialed/icans/`.

## End-to-end pipeline (run order)

1. **EEG download + preprocessing + SQA**
   - `cohort_models/RASS/EEGPreprocessingDownloadSQA/`, `cohort_models/GCS/EEGPreprocessingDownloadSQA/`
   - 10-minute segment creation: `create_individual_eeg10min_{RASS,GCS}.py`
   - signal-quality assessment / bad-segment discard: `cohort_models/RASS/EEGPreprocessingDownloadSQA/EEG_SQA_based_segment_discard.py`
   - ICANS awake-segment selection: `cohort_models/ICANS/SLEEPHeadbasedSelectEEGSegments/ICANS_group_Best10minEEGSelection.ipynb`
2. **MORGOTH feature extraction** → 591×17 activation matrices (foundation-model inference)
3. **Model training**
   - contrastive triplet encoder: `NESI/model/Training/Train_TripletEncoder.py`
   - NESI ranking head: `NESI/model/Training/Train_NESI.py`
   - bespoke per-scale models: `NESI/Bespoke_models/Training/{RASS,GCS,CAMS,ICANS}/`
4. **Evaluation / analysis** → figures and tables below.

## Figures, tables and numbers

The fast route from saved intermediate results is `reproduce/run_all.py`; `python reproduce/run_all.py --list`
maps every figure and table in the revised manuscript to the script that produces it, and
`reproduce/README.md` documents the route from source data to each intermediate. The final figure images
are in `figures/final/`.

The published model was trained with learning rate 0.005 (`Train_TripletEncoder.py`, `Train_NESI.py`);
the sensitivity retrains in `reproduce/retrain_split_fix.py` used 0.0005 (see its `--lr` option).

## Comparator models

External baselines are kept under `comparators/` and are **not** part of the manuscript's
figures, tables or numbers; they are run separately.

| Comparator | Code | Docs |
|---|---|---|
| AIrhythm (PhysioNet/CinC 2023 coma prognostication; EEG+ECG features → boosted-tree ensemble) | `comparators/AIrhythm/team_code/` | `comparators/AIrhythm/README.md` |

AIrhythm needs its own Python 3.8 environment (`comparators/AIrhythm/team_code/requirements.txt`,
or the provided `Dockerfile`) — `esig` and the pinned CatBoost/XGBoost versions conflict with the
NESI environment. It also expects continuous PhysioNet-layout recordings rather than our 10-minute
HEEDB segments; the required adapter and the label realignment to in-hospital death are described
in its README.

## Trained weights

Trained model weights are committed under `NESI/model/ModelCheckpoints/`
(`ResNetGAP_BestModel.pth` encoder, `NESI_best_model.pth` ranking head) and
`NESI/Bespoke_models/ModelCheckpoints/`. Per-segment scores/result CSVs are in the credentialed
bucket under `yama/NESI/Bespoke_models/Results/` and `yama/NESI/Data/`.
