# NESI PaCMAP Visualization

This folder contains scripts and data for visualizing the Neurological Evaluation and Severity Index (NESI) using PaCMAP dimensionality reduction.

## Files

- `build_window_features_csv_NEW.py`: Extracts window-level MORGOTH features from EEG recordings.
- `nesi_pacmap_Coordinate_estimation.py`: Computes two-dimensional PaCMAP embeddings using MORGOTH features and NESI scores.
- `NESI_PACMAP_main.py`: Generates the PaCMAP visualization colored by continuous NESI scores and dominant EEG patterns.
- `NESI_pacmap_coords.csv`: Precomputed PaCMAP coordinates, NESI scores, and EEG features.
- `UniversalBadnessModelResult_Full.csv`: NESI predictions and associated clinical metadata.
- `NESI_pacmap.png`: Generated visualization.

## Visualization

The figure contains two panels:

- **Panel A:** EEG windows colored by continuous NESI scores.
- **Panel B:** EEG windows colored by dominant MORGOTH EEG patterns, including burst suppression, seizures, LPD, GPD, LRDA, GRDA, and other activity.

## Usage

To reproduce the visualization using the provided coordinates:

```bash
python NESI_PACMAP_main.py

[NESI_PACMAP_main.py](https://github.com/user-attachments/files/33686027/NESI_PACMAP_main.py)
<img width="6680" height="3279" alt="NESI_pacmap" src="https://github.com/user-attachments/assets/b2e1a51e-d46d-4981-8405-8d41268460da" />

