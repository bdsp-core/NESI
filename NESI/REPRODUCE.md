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
   - `RASS/EEGPreprocessingDownloadSQA/`, `GCS/EEGPreprocessingDownloadSQA/`
   - 10-minute segment creation: `create_individual_eeg10min_{RASS,GCS}.py`
   - signal-quality assessment / bad-segment discard: `RASS/EEGPreprocessingDownloadSQA/EEG_SQA_based_segment_discard.py`
   - ICANS awake-segment selection: `ICANS/SLEEPHeadbasedSelectEEGSegments/ICANS_group_Best10minEEGSelection.ipynb`
2. **MORGOTH feature extraction** → 591×17 activation matrices (foundation-model inference)
3. **Model training**
   - contrastive triplet encoder: `NESI/model/Training/Train_TripletEncoder.py`
   - NESI ranking head: `NESI/model/Training/Train_NESI.py`
   - bespoke per-scale models: `NESI/Bespoke_models/Training/{RASS,GCS,CAMS,ICANS}/`
4. **Evaluation / analysis** → figures and tables below.

## Figures

| Figure | What it shows | Script(s) |
|--------|---------------|-----------|
| Fig. 1 | Pipeline schematic | Conceptual diagram (method image in `NESI/README.md`) |
| Fig. 2 | Universal vs. bespoke Spearman performance (boxplots + ρ, Wilcoxon) | `NESI/model/Testing/FAST_NESIvsBeSpoke_plot.py`; `NESI/Bespoke_models/Testing/TESTING_NESIvsBeSpokeBadness_Compare.py`; `NESI/FigureNESI/NESI_vs_Bespoke.py` |
| Fig. 3 | Worm-model latent fits (NESI vs each scale) | `NESI/ScaleVsNESI/fit_nesi_curves.py` |
| Fig. 4 | MORGOTH feature activations stratified by NESI | `NESI/MORGOTHActivationViz_GroupedbyNESI/NESI_grouped_morgoth_activation_visualization.py` (fast variant: `FAST_NESI_group_morgothactivation.py`) |
| Fig. 5 | Longitudinal mortality AUROC, NESI vs GCS | train: `NESI/DeathPrediction_NESIvsGCS/model_SeqLR/Train_Death_Prediction_with_NESI_vsGCS.py`; plot: `NESI/DeathPrediction_NESIvsGCS/model_SeqLR/Test_results_plot_GCSvsNESI.py` |
| Supp. Fig. 18 | Ablation: mean/median features vs contrastive encoder | `NESI/model/AblationStudy/` (Training/, Testing/, FiguresAblation/) |

## Tables

| Table | What it shows | Script(s) |
|-------|---------------|-----------|
| Table 1 | Patient characteristics per cohort | `RASS/Cohort/Table1_RASS.py`, `GCS/Cohort/Table1_GCS.py`, `CAMS/Cohort/Table1_CAMS.py`, `ICANS/Cohort/Table1_ICANS.py` |
| Table 2 + Supp. Table 1 | Medication exposure (patient- and EEG-level) | `NESI/NESI-Medication-Analysis/medication-exposure/` (drivers `run_*_pipeline_meds.py`; assembled by `build_final_exposure_tables.py`) |
| Table 3 | Propofol Cₑ variance decomposition (NESI vs RASS) | `NESI/NESI-Medication-Analysis/eleveld-pk-analysis/variance_decomposition_eleveld_ce_vs_p.py` |

## Key in-text results

| Result | Script(s) |
|--------|-----------|
| Propofol effect-site concentration (Cₑ) via Eleveld 2018 PK/PD | `NESI/NESI-Medication-Analysis/eleveld-pk-analysis/eleveld_propofol.py`, `eleveld_run_cohort.py` |
| NESI-vs-RASS Cₑ correlation (paired cluster bootstrap) | `NESI/NESI-Medication-Analysis/eleveld-pk-analysis/nesi_vs_rass_correlation_comparison.py` |

## Trained weights

Trained model weights are committed under `NESI/model/ModelCheckpoints/`
(`ResNetGAP_BestModel.pth` encoder, `NESI_best_model.pth` ranking head) and
`NESI/Bespoke_models/ModelCheckpoints/`. Per-segment scores/result CSVs are in the credentialed
bucket under `yama/NESI/Bespoke_models/Results/` and `yama/NESI/Data/`.
