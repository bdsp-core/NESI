# 🧠 NESI External Validation Pipeline

This directory contains the complete pipeline for **external validation of NESI-derived EEG models** on independent EEG recordings.

The pipeline processes raw EDF EEG recordings, extracts continuous EEG representations using **MORGOTH**, and generates continuous predictions for:

- **RASS** — Richmond Agitation-Sedation Scale
- **GCS** — Glasgow Coma Scale severity
- **CAMS** — Confusion Assessment Method severity
- **NESI** — Neurophysiologic Encephalopathy Severity Index

For EEG segments longer than 10 minutes, predictions are generated continuously using overlapping 10-minute windows. Segment-level predictions can subsequently be summarized using the **mode / majority vote** to obtain one final prediction for each EEG segment.

---

## 🔄 Pipeline Overview

```text
Raw EDF EEG
     │
     ▼
EEG Preprocessing
0.5–70 Hz filtering
50/60 Hz notch filtering
200 Hz resampling
Common average reference
     │
     ▼
MORGOTH
     │
     ├── Sleep
     ├── Normal / Abnormal
     ├── Burst Suppression
     ├── Spikes
     ├── Slowing
     └── IIIC
     │
     ▼
Continuous MORGOTH Feature Matrix
17 EEG features
10-s windows / 1-s stride
     │
     ▼
┌──────────────────────────────────────┐
│      Continuous 10-min Models        │
├──────────────────────────────────────┤
│ RASS ordinal model                   │
│ GCS ordinal model                    │
│ CAMS ordinal model                   │
│ NESI continuous severity model       │
└──────────────────────────────────────┘
     │
     ▼
Continuous Predictions
     │
     ▼
Mode / Majority Vote
     │
     ▼
Final Segment-Level Prediction
     │
     ▼
External Validation
Confusion matrices / Accuracy / MAE
```

---

# 📂 Directory Structure

```text
ExternalValidation/
│
├── Continious_RASS_GCS_CAMS_NESI_Prediction.py
│
├── RASS_GCS_CAMS_Prediction_Results_MajorityVote_All.py
│
├── KIMCHI_FINAL_METADATA.xlsx
│
├── environment.yml
├── requirements.txt
├── Readme.md
│
├── EDF_EEGS/
│   └── readme.md
│
├── DeliriumScoreModel/
│   ├── RESNETGAP_Best_RASS.pth
│   ├── RESNETGAP_Best_GCS.pth
│   ├── RESNETGAP_Best_CAMS.pth
│   └── readme.md
│
├── NESImodel/
│   ├── ResNetGAP_BestModel.pth
│   └── NESI_best_model.pth
│
├── morgoth/
│   ├── finetune_classification.py
│   ├── task_model.py
│   ├── tokenizer.py
│   ├── utils.py
│   ├── EEG_level_head.py
│   ├── segment_long_eeg.py
│   ├── requirements.txt
│   ├── ...
│   └── checkpoints/
│
├── MATLABPlotCodesFor_ContiniousPrediction/
│   ├── RASS_GCS_CAMS_NESI_EEG_Prediction_viz.m
│   ├── EEG_Spectrum_WithAll_NESI_Results.m
│   └── readme.md
│
└── EEG_viewing_codes/
    ├── main_visuals.m
    ├── fcn_bipolar.m
    ├── sample_EEG_10min.mat
    └── Callbacks/
```

---

# 📁 Main Components

## 🧠 `Continious_RASS_GCS_CAMS_NESI_Prediction.py`

This is the main inference pipeline.

For each EDF recording, the script:

1. Loads the raw EDF EEG.
2. Standardizes EEG channel names.
3. Selects the required 19-channel EEG montage.
4. Resamples EEG to **200 Hz**.
5. Applies **0.5–70 Hz band-pass filtering**.
6. Applies **50 Hz and 60 Hz notch filtering**.
7. Applies common average referencing.
8. Runs the six MORGOTH EEG heads.
9. Combines MORGOTH outputs into a **17-dimensional feature representation**.
10. Applies continuous 10-minute prediction windows.
11. Generates:
   - RASS predictions
   - GCS predictions
   - CAMS predictions
   - NESI scores
12. Saves continuous predictions as CSV files.

The MORGOTH feature sequence is generated using approximately:

```text
10-second EEG window
1-second sliding step
17 MORGOTH features
```

A 10-minute EEG interval therefore produces a feature matrix of approximately:

```text
591 × 17
```

---

# 🧠 Clinical Prediction Models

## RASS

RASS is modeled as an **ordinal classification problem**.

Predicted model classes are mapped to:

```text
0 → -5
1 → -4
2 → -3
3 → -2
4 → -1
5 →  0
```

The model weights are stored in:

```text
DeliriumScoreModel/RESNETGAP_Best_RASS.pth
```

---

## GCS

GCS severity is modeled as an **ordinal classification problem** with three categories:

```text
Severe
Moderate
Mild
```

Clinical GCS values are grouped as:

```text
3–8   → Severe
9–12  → Moderate
13–15 → Mild
```

The model weights are stored in:

```text
DeliriumScoreModel/RESNETGAP_Best_GCS.pth
```

---

## CAMS

CAMS severity is modeled as an **ordinal classification problem** with three categories:

```text
Mild
Moderate
Severe
```

The model weights are stored in:

```text
DeliriumScoreModel/RESNETGAP_Best_CAMS.pth
```

---

## NESI

NESI provides a continuous EEG-derived neurological severity score.

The NESI pipeline uses:

```text
NESImodel/ResNetGAP_BestModel.pth
```

for EEG representation extraction and:

```text
NESImodel/NESI_best_model.pth
```

for NESI score generation.

---

# 🧬 MORGOTH Feature Extraction

The `morgoth/` directory contains the MORGOTH inference framework used to generate EEG representations.

The external-validation pipeline runs the following EEG heads:

| Head | Representation |
|---|---|
| Sleep | Awake / N1 / N2 |
| Normality | Normal / Abnormal |
| Burst Suppression | Burst / No Burst |
| Spikes | No / Focal / Generalized |
| Slowing | No / Focal / Generalized |
| IIIC | Other / Seizure / LPD / GPD / LRDA / GRDA |

These outputs form the **17-dimensional MORGOTH representation** used by the downstream RASS, GCS, CAMS, and NESI models.

---

# 📈 Continuous Prediction

For recordings longer than 10 minutes, predictions are generated continuously.

The downstream models operate on:

```text
591 × 17
```

MORGOTH feature matrices representing approximately **10 minutes of EEG**.

The 10-minute prediction window is moved through the recording to produce a sequence of predictions:

```text
EEG
│
├── 10-min window 1 → Prediction 1
├── 10-min window 2 → Prediction 2
├── 10-min window 3 → Prediction 3
├── ...
└── 10-min window N → Prediction N
```

This allows RASS, GCS, CAMS, and NESI to be followed continuously across longer EEG segments.

---

# 🗳️ Final Segment-Level Prediction

`RASS_GCS_CAMS_Prediction_Results_MajorityVote_All.py` converts continuous predictions into a single prediction for each EEG segment.

For RASS, GCS, and CAMS, the **mode of the continuous predictions** is used:

```text
Continuous predictions
        │
        ▼
      Mode
        │
        ▼
Final segment-level prediction
```

For example:

```text
Continuous GCS predictions:

Mild
Mild
Moderate
Mild
Mild
Moderate
Mild

                 ↓

Final GCS prediction = Mild
```

The script also matches predictions to the clinical metadata contained in:

```text
KIMCHI_FINAL_METADATA.xlsx
```

using both:

```text
SubjectID
seg_start_rel_s
```

---

# 📊 External Validation Metrics

The cohort-level evaluation script currently calculates:

### RASS

- Exact accuracy
- ±1-level accuracy
- Mean Absolute Error (MAE)
- Confusion matrix

### GCS

- Exact classification accuracy
- Confusion matrix

The final combined results are saved as:

```text
Final_Results_Prediction_RASS_GCS_CAMS_Cohort_Model.csv
```

---

# 📦 Generated Output Structure

Running the continuous prediction pipeline generates a `Products/` directory similar to:

```text
Products/
│
├── MorgothActivations/
│   └── <EDF_ID>/
│       ├── IIIC/
│       ├── FOCGEN/
│       ├── BS/
│       ├── NM/
│       ├── SLEEP/
│       └── SLOWING/
│
├── RASSPredictions/
│   └── *_predictions.csv
│
├── GCSPredictions/
│   └── *_predictions.csv
│
├── CAMSPredictions/
│   └── *_predictions.csv
│
└── NESIPredictions/
    └── *_predictions.csv
```

---

# 📊 MATLAB Visualization

The folder:

```text
MATLABPlotCodesFor_ContiniousPrediction/
```

contains MATLAB scripts for visualization of continuous predictions.

### `RASS_GCS_CAMS_NESI_EEG_Prediction_viz.m`

Visualizes continuous:

- RASS
- GCS
- CAMS
- NESI

predictions together with the EEG.

### `EEG_Spectrum_WithAll_NESI_Results.m`

Generates EEG spectral / spectrogram visualization together with model outputs.

These plots are useful for visually comparing changes in EEG background activity with changes in the predicted neurological state.

---

# 👁️ EEG Visualization Utilities

The folder:

```text
EEG_viewing_codes/
```

contains MATLAB utilities for inspecting EEG recordings.

Important files include:

```text
main_visuals.m
fcn_bipolar.m
sample_EEG_10min.mat
Callbacks/
```

A sample 10-minute EEG segment is included for testing the visualization workflow.

---

# 📥 Input EEG

Place external-validation EDF files inside:

```text
EDF_EEGS/
```

The preprocessing code expects a standard 19-channel EEG montage.

Two common channel naming conventions are supported, including:

```text
FP1 F3 C3 P3 F7 T3 T5 O1
FZ CZ PZ
FP2 F4 C4 P4 F8 T4 T6 O2
```

and the equivalent modern temporal-channel nomenclature:

```text
T7 / P7
T8 / P8
```

Channel names are standardized automatically where possible.

---

# 🐍 Environment Setup

Two environments are used because MORGOTH and the downstream NESI models have different dependency requirements.

## 1. MORGOTH Environment (morgoth)

The `morgoth` environment is responsible for EEG representation extraction.

Follow the MORGOTH installation instructions and activate the environment using:

```bash
conda activate morgoth
```

The external-validation pipeline calls MORGOTH from this environment automatically.

---

## 2. NESI / Prediction Environment (torchenv)

Create the prediction environment using:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate torchenv
```

Alternatively:

```bash
pip install -r requirements.txt
```

For a CUDA-enabled installation, install the appropriate PyTorch version for your system.

Verify the environment:

```bash
python -c "import torch, numpy, pandas, mne; print('Environment OK')"
```

Check GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

---

# 🚀 Running External Validation

From the `ExternalValidation` directory:

## Step 1 — Place EDF files

```text
ExternalValidation/EDF_EEGS/
```

## Step 2 — Confirm metadata

Ensure that:

```text
KIMCHI_FINAL_METADATA.xlsx
```

contains the required subject and EEG-segment information.

## Step 3 — Run continuous inference

```bash
conda activate torchenv
```

```bash
python Continious_RASS_GCS_CAMS_NESI_Prediction.py
```

This performs:

```text
EDF
 ↓
Preprocessing
 ↓
MORGOTH
 ↓
17-D EEG representation
 ↓
RASS / GCS / CAMS / NESI
 ↓
Continuous prediction CSVs
```

## Step 4 — Generate segment-level predictions

```bash
python RASS_GCS_CAMS_Prediction_Results_MajorityVote_All.py
```

This performs:

```text
Continuous predictions
        ↓
Mode / majority vote
        ↓
Single prediction per EEG segment
        ↓
Match clinical annotation
        ↓
External-validation metrics
```

## Step 5 — Visualize

Use the scripts under:

```text
MATLABPlotCodesFor_ContiniousPrediction/
```

to visualize EEG, spectrograms, and continuous neurological predictions.

---

# ⚠️ Notes

- External EEG recordings must contain the required EEG channels.
- EEG is internally resampled to **200 Hz**.
- Model weights should remain in their existing directories because the inference code resolves model paths relative to `ExternalValidation`.
- MORGOTH and the prediction environment should both be tested before processing a large external cohort.
- Continuous predictions are intentionally retained before majority voting so that temporal changes in neurological state can also be analyzed.
- Missing or incompatible EEG channels may cause a recording to be skipped.

---

# 🧠 End-to-End Summary

```text
EDF EEG
  ↓
Preprocessing
  ↓
MORGOTH
  ↓
17-D continuous EEG representation
  ↓
┌────────┬────────┬────────┬────────┐
│  RASS  │  GCS   │  CAMS  │  NESI  │
└────────┴────────┴────────┴────────┘
  ↓
Continuous neurological predictions
  ↓
Mode / Majority Vote
  ↓
Segment-level prediction
  ↓
Clinical annotation matching
  ↓
External validation
```

This directory therefore provides an end-to-end workflow for evaluating **EEG-derived neurological state and severity models on independent external EEG datasets**.
