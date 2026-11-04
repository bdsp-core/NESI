# Long-Term NESI Tracking and MDC Analysis

This folder contains the code and data required to reproduce the longitudinal NESI tracking and Minimum Detectable Change (MDC) analyses for GCS and RASS.

## Option 1: Complete End-to-End Analysis

### Step 1: Download EEG Data from AWS

Download the following folders containing EDF EEG recordings:

- `ContiniousNESI_GCS_EEG`
- `ContiniousNESI_RASS_EEG`

Place both folders inside `LongEEG_NESI_Tracking/`.

### Step 2: Generate Continuous NESI Predictions

Run the following Python scripts:

**For GCS EEGs:**
`code/GCS_MDC_continiousNESI_Prediction.py`

**For RASS EEGs:**
`code/RASS_MDC_continiousNESI_Prediction.py`

These scripts generate the following output folders:

**GCS Outputs:**
```text
Products_GCS_MDC/
├── GCSSegmentEEG/
├── MorgothActivations/
├── GCSPredictions/
├── NESIPredictions/
├── RASSPredictions/
└── CAMSPrediction/
```

**RASS Outputs:**
```text
Products_RASS_MDC/
├── GCSSegmentEEG/
├── MorgothActivations/
├── RASSPredictions/
├── NESIPredictions/
├── GCSPredictions/
└── CAMSPrediction/
```

### Step 3: Generate Supplementary Figures

Run the following MATLAB scripts:

- **GCS:** `code/MatlabMDC_GCS/Implement_GCS_MDC.m` (Supplementary Figs. 21–23)
- **RASS:** `code/MatlabMDC_RASS/Implement_RASS_MDC.m` (Supplementary Figs. 18–20)

---

## Option 2: Direct Reproduction Using Precomputed Results

To reproduce the supplementary figures without rerunning EEG preprocessing and model predictions, follow these steps.

### Step 1: Download Precomputed Data from AWS

Download the following four folders:

- `ContiniousNESI_GCS_EEG`
- `ContiniousNESI_RASS_EEG`
- `Products_GCS_MDC`
- `Products_RASS_MDC`

### Step 2: Organize the Directory Structure

Ensure the `LongEEG_NESI_Tracking` folder has the following structure:

```text
LongEEG_NESI_Tracking/
├── code/
├── ContiniousNESI_GCS_EEG/
├── ContiniousNESI_RASS_EEG/
├── EEG_ViewingCodes/
├── Metadata/
├── Products_GCS_MDC/
└── Products_RASS_MDC/
```

### Step 3: Reproduce Supplementary Figures

Directly run the following MATLAB scripts:

- **GCS:** `code/MatlabMDC_GCS/Implement_GCS_MDC.m` (Supplementary Figs. 21–23)
- **RASS:** `code/MatlabMDC_RASS/Implement_RASS_MDC.m` (Supplementary Figs. 18–20)

These scripts use the precomputed results to reproduce the supplementary figures without rerunning the Python prediction pipeline.

**Note:** Option 2 is recommended for direct reproduction of the supplementary figures. Option 1 is intended for complete end-to-end EEG processing and analysis.
