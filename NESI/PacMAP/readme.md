# NESI PaCMAP Visualization

This folder contains scripts and data for visualizing the Neurological Evaluation and Severity Index (NESI) using PaCMAP dimensionality reduction.

## Files

- `STEOP1_build_window_features_csv_NEW.py`: Extracts window-level MORGOTH features from EEG recordings.
- `STEP2_nesi_pacmap_Coordinate_estimation.py`: Computes two-dimensional PaCMAP embeddings using MORGOTH features and NESI scores.
- `STEP3A_NESI_PACMAP_Twoanel.py`: Generates the PaCMAP visualization colored by continuous NESI scores and dominant EEG patterns.
- `STEP3B_NESI_composite_pacmap.py`: Generates the PaCMAP visualization colored by continuous NESI scores and dominant EEG patterns and Disease Association and Severe Ench Category.
- `NESI_pacmap_coords.csv`: Precomputed PaCMAP coordinates, NESI scores, and EEG features.
- `UniversalBadnessModelResult_Full.csv`: NESI predictions and associated clinical metadata.
- `NESI_pacmap.png`: Two Panel PacMAP visualization.
- `NESI_composite_pacmap.png`: Multi Panel PacMAP visualization.

## Visualization

The figure contains two panels:

- **Panel A:** EEG windows colored by continuous NESI scores.
- **Panel B:** EEG windows colored by dominant MORGOTH EEG patterns, including burst suppression, seizures, LPD, GPD, LRDA, GRDA, and other activity.

## Usage

To reproduce the visualization using the provided coordinates:

Directly Run
```bash
python STEP3A_NESI_PACMAP_Twoanel.py
python STEP3B_NESI_composite_pacmap.py
```

<img width="6680" height="3279" alt="NESI_pacmap" src="https://github.com/user-attachments/assets/2faa1134-29b4-4f9e-8b63-ed15646c9c12" />


