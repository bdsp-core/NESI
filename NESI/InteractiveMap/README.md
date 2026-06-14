# Interactive NESI embedding explorer

An interactive 2-D map of the MORGOTH/NESI EEG embedding in which **clicking a point displays the
corresponding 10-minute EEG segment**. Points can be colored by **NESI**, by **dominant
ictal–interictal-continuum (IIC) pattern**, or by any single MORGOTH feature. The page is a
single self-contained `explorer.html` (per-point data embedded inline; it works from `file://`
without a server) plus a folder of pre-rendered EEG PNGs that are shown on click.

This folder contains the **code** to build the explorer. The rendered artifacts (`explorer.html`,
`eeg_pngs/`, `deploy/`, `coverage.png`) are large and are git-ignored; build them locally, or use
the hosted demo (see below).

## Scripts

| Script | What it does |
|---|---|
| `build_explorer.py` | Generates `explorer.html` — the interactive PaCMAP scatter with click-to-EEG and the NESI / IIC / feature color selector. Reads PaCMAP coordinates from `../MorgothFeatureEmbedding/NESI_pacmap_coords.csv` and EEG images from `--png-dir`. |
| `batch_render.py` | Renders 10-minute EEG segments to PNGs (`eeg_pngs/<cohort>/...`) used by the explorer. |
| `poc_render.py` | Proof-of-concept renderer for a single EEG segment (filter/montage settings). |
| `subsample_for_deploy.py` | Builds a lightweight, deployable `deploy/` bundle (subsamples points/PNGs to keep the site small enough for static hosting). |
| `serve_deploy.py` | Serves the `deploy/` bundle locally over HTTP for testing. |
| `visualize_coverage.py` | Plots which embedding points have a rendered EEG PNG (coverage map). |

## Build it

```bash
# 1. Render EEG segment images (needs the EEG segments from the credentialed dataset)
python batch_render.py            # -> eeg_pngs/<cohort>/*.png

# 2. Build the interactive page
python build_explorer.py          # -> explorer.html  (open directly in a browser)

# 3. (optional) Make a lightweight deployable bundle and preview it
python subsample_for_deploy.py    # -> deploy/explorer.html + deploy/eeg_pngs/
python serve_deploy.py            # serves deploy/ at http://localhost:8000
```

## Inputs

- **PaCMAP coordinates + per-point metadata:** `../MorgothFeatureEmbedding/NESI_pacmap_coords.csv`
  (produced by the MORGOTH feature-embedding pipeline).
- **EEG segments:** the 10-minute `.mat` segments from the credentialed dataset
  (`s3://bdsp-opendata-credentialed/yama/`; see the repository's top-level README for access).

## Hosted demo

A live, clickable version is hosted at: **[URL to be added]**.
