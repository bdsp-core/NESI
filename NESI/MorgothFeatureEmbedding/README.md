# MORGOTH Feature Embedding (NESI atlas)

PaCMAP embedding of MORGOTH event-level activations, with each EEG window colored by NESI score and by clinically-grouped feature loadings. Built to give a single interpretable atlas of "where in the MORGOTH feature space does each pattern live, and how does NESI organize it."

## Pipeline overview

```
UniversalBadnessModelResult_Full.csv               <-- 209,766 snippets, NESI per snippet
            │
            │  build_window_features_csv.py
            │  (10 snippets/patient, 10 windows/patient, drop spike & IIIC_Other)
            ▼
NESI_window_features.csv                           <-- 84,090 rows, 13 features + meta
            │
            │  nesi_pacmap_main.py
            │  (NESI-stratify 400 patients/bin -> ~44k windows;
            │   logit + per-column z-score;
            │   PaCMAP with NESI weighted 2x)
            ▼
NESI_pacmap_fig1_overview.png   ┐
NESI_pacmap_fig2_atlas.png      ├─  canonical figures
NESI_pacmap_iiic_with_nesi.png  │
NESI_pacmap_iiic_color.png      │   (without NESI weighting, for comparison)
NESI_pacmap_weight_sweep.png    │
NESI_pacmap_iiic_smooth.png     ┘
NESI_pacmap_iiic_data.npz       <-- XY + RGB + labels for HTML explorer

            │  build_smoothness_explorer.py
            ▼
NESI_pacmap_explorer.html                          <-- self-contained interactive
```

The full snippet-level CSV and per-patient median variants are also supported, see [Alternative aggregations](#alternative-aggregations).

## Inputs (live elsewhere in YAMA)

- **`../MORGOTHActivationViz_GroupedbyNESI/UniversalBadnessModelResult_Full.csv`** — one row per 10-min snippet with `MorgothOutputFilename, Dataset, TrueRawScores, TransformedRawScores, NESI, WhichSet`.
- **`../../{GCS,RASS,CAMS,ICANS}/MorgothActivations/{SLEEP,NM,BS,FOCGEN,SLOWING,IIIC}/<snippet>.csv`** — per-snippet head activations (one row per ~1-second window). For GCS and RASS, sync from `s3://bdsp-opendata-credentialed/yama/` before running the pipeline.

## Quick start

```bash
cd YAMA/NESI/MorgothFeatureEmbedding/

# 1. Build the per-window feature CSV (~40 s on cpu_count-2 workers)
python3 build_window_features_csv.py

# 2. Run PaCMAP + generate all canonical figures (~3-5 min total)
python3 nesi_pacmap_main.py

# 3. (Optional) Regenerate the interactive smoothing explorer
python3 build_smoothness_explorer.py
```

All outputs land in this directory. Large CSVs (`NESI_window_features.csv`, `NESI_full_features.csv`, etc.) and the `.npz` are gitignored — they're regenerable.

## Scripts

| File | Purpose |
|---|---|
| `build_window_features_csv.py` | Main feature builder. For each patient, randomly pick up to 10 snippets and up to 10 window-vectors total. Output is `NESI_window_features.csv` (~84k rows × 19 cols). |
| `nesi_pacmap_main.py` | Canonical: NESI-stratified subsample, logit+z normalization, PaCMAP with NESI weight 2×, produces all final figures + npz. |
| `build_smoothness_explorer.py` | Reads `NESI_pacmap_iiic_data.npz`, renders ~13 Gaussian-splat smoothed versions at different σ, packages them with a slider into a single self-contained HTML file. |
| `build_full_features_csv.py` | Alternative: one feature vector per snippet (median across windows). All 209,766 snippets. ~2 min on cpu_count-2 workers. |
| `aggregate_per_patient.py` | Alternative: collapses the full snippet CSV to per-patient median (14,131 patients). |
| `build_balanced_features_csv.py` | Alternative: 4,367-row balanced subset matching Arka's `MORGOTHActivationViz_GroupedbyNESI` pipeline. |
| `embedding_preview_cli.py` | Scratchpad: CLI tool to run PCA, t-SNE, UMAP, PaCMAP side-by-side on any features CSV. Useful for method comparison. |

## Pipeline knobs (in `nesi_pacmap_main.py`)

- `TARGET_PATIENTS_PER_NESI_BIN = 400` — number of patients to sample per NESI bin (12 bins of width 0.5 covering [−3, +3]).
- `NESI_FEATURE_WEIGHT = 2.0` — how much extra weight NESI gets in the input matrix (2.0 → NESI contributes ~24% of total variance; 3.0 → ~41%).
- `NESI_WEIGHT_SWEEP = [0.0, 1.0, 2.0, 3.0]` — values used in the weight-sweep figure.
- `JITTER_COPIES = 4`, `JITTER_SIGMA_FRAC = 0.006`, `DOT_SIZE = 4.0` — visual fill for scatter plots.
- `INPUT_FEATURES` — the 13 features fed to PaCMAP (3 sleep + 1 Normal + 1 Burst + 3 Slowing + 5 IIIC).
- `PROB_CMAP_HEX` — shared light-gray→dark-blue colormap for feature-loading panels.
- `IIIC_PALETTE` — standard Westover-lab IIIC colors (Seizure/LPD/GPD/LRDA/GRDA/Other) + black for burst.

## Figure conventions

- **NESI**: diverging `RdBu_r`, symmetric ±3.
- **All feature-loading panels**: shared light-gray → dark-blue sequential scale, fixed `vmin=0, vmax=1`, single shared colorbar (so cross-feature intensities are honestly comparable).
- **IIIC categorical**: per-point RGB built from the dominant pattern × within-category probability gradient; black for burst when Burst_vs_NoBurst ≥ 0.5.
- **Point order**: low-to-high so high-loading points draw on top.

## Outputs

| File | What it shows |
|---|---|
| `NESI_pacmap_fig1_overview.png` | 1×3 orientation: NESI, dominant IIIC pattern, P(Normal). |
| `NESI_pacmap_fig2_atlas.png` | Clinically grouped feature-loading atlas (State, Global, Focal, Generalized, Ictal). |
| `NESI_pacmap_iiic_with_nesi.png` | 2-panel (IIIC categorical + NESI), PaCMAP with NESI weight 2×. |
| `NESI_pacmap_iiic_color.png` | Same 2-panel layout but PaCMAP from features only (NESI not in feature set). |
| `NESI_pacmap_iiic_smooth.png` | Gaussian-splat smoothed version of the categorical IIIC map. |
| `NESI_pacmap_weight_sweep.png` | 4 rows × 2 cols; rows are NESI weights 0, 1, 2, 3 with both IIIC and NESI colorings. |
| `NESI_pacmap_explorer.html` | Self-contained HTML with a slider that varies the Gaussian-splat smoothing σ. |

## Abbreviations

- **NESI** — Neurophysiological EEG-based Severity Index.
- **MORGOTH** — bdsp-core foundation model providing the EEG feature embeddings used here.
- **IIIC** — Ictal-Interictal Continuum (Seizure, LPD, GPD, LRDA, GRDA, Other).
- **LPD / GPD** — Lateralized / Generalized Periodic Discharges.
- **LRDA / GRDA** — Lateralized / Generalized Rhythmic Delta Activity.

## Alternative aggregations

If you want a different feature CSV (e.g. per-patient median, or one row per snippet), run one of the alternative builders before `nesi_pacmap_main.py` and point its `IN_CSV` to the resulting file.

| Builder | n rows | What each row is |
|---|---|---|
| `build_window_features_csv.py` | 84,090 | one ~1-s window-vector, up to 10 per patient |
| `aggregate_per_patient.py` | 14,131 | one patient (median across all their snippets) |
| `build_full_features_csv.py` | 209,766 | one snippet (median across its windows) |
| `build_balanced_features_csv.py` | 4,367 | Arka's NESI-binned + per-class stratified subset |
