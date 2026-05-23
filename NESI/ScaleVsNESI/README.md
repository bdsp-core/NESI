# Scale ↔ NESI relationship (ordinal-logistic curves)

Fits an ordinal-logistic (proportional-odds cumulative-link) model relating each clinical rating scale (GCS, RASS, CAMS, ICANS) to the continuous NESI score. Direction-normalized so that all scales share the convention *higher = worse clinical state* (GCS and RASS are reversed). Produces the per-scale curves used to map any NESI value to expected scale categories, along with a supplementary figure that visualizes inter-rater noise under the fitted noise model.

## Inputs

- **`../MORGOTHActivationViz_GroupedbyNESI/UniversalBadnessModelResult_Full.csv`** — one row per 10-min snippet with `MorgothOutputFilename, Dataset, TrueRawScores, TransformedRawScores, NESI, WhichSet`. The script fits the model on rows with `WhichSet == "Train"`.

## Outputs (committed)

| File | What it is |
|---|---|
| `figure_main.png` | Per-scale ordinal-logistic curves (cumulative-link). |
| `figure_supp_rater_noise.png` | Joint density of two independent draws from the fitted noise model — visualizes expected inter-rater agreement vs NESI. |
| `predicted_scores_vs_NESI.csv` | Lookup: each NESI value → expected raw/transformed score per scale. |
| `nesi_to_scale_lookup.csv` | Compact lookup table for mapping NESI to scale categories. |
| `methods_and_captions.md` | Full methods writeup and figure captions for the paper. |

## How to regenerate

```bash
cd YAMA/NESI/ScaleVsNESI/
python3 fit_nesi_curves.py
```

Reads `UniversalBadnessModelResult_Full.csv` from the sibling folder, writes both lookup CSVs and both figures into this directory.

## Related

- See `../MorgothFeatureEmbedding/` for the PaCMAP feature-loading atlas (uses the same metadata CSV but a different downstream analysis path).
