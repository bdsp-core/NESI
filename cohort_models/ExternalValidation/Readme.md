
# 🧠 NESI External Validation Pipeline

This guide provides step-by-step instructions for reproducing the external validation pipeline of the **Neurophysiologic Encephalopathy Severity Index (NESI)** using EEG recordings.

The pipeline consists of three main stages:

**📊 Raw EEG (EDF) → Preprocessed EEG (MAT) → MORGOTH Features (10 min window 591 × 17) → NESI Predictions**

---

## 📦 Prerequisites: Environment Setup

Two separate Conda environments are required to execute the complete pipeline:

| Environment | Purpose |
|-------------|---------|
| 🧠 `morgoth` | EEG-based feature extraction using the MORGOTH foundation model |
| 🔬 `torchenv` | EEG preprocessing and NESI prediction from MORGOTH features |

### 1️⃣ Set Up the MORGOTH Environment

NESI uses **MORGOTH** as its EEG feature-extraction foundation model.

MORGOTH converts preprocessed EEG recordings into feature representations consisting of **591 × 17 matrices**, which are subsequently used for NESI prediction.

📌 **Installation guide:**

🔗 **[MORGOTH GitHub Repository](https://github.com/bdsp-core/morgoth)**

Follow the installation instructions provided in the MORGOTH repository and ensure that the `morgoth` environment is correctly configured before proceeding.

> ⚠️ **Important:** MORGOTH is a required dependency for extracting EEG features from raw recordings. Ensure that the MORGOTH environment is functional before running Step 2.

### 2️⃣ Set Up the NESI Environment (`torchenv`)

The `torchenv` environment is used for EEG preprocessing and generating NESI predictions from MORGOTH-extracted features.

**📥 Create and activate the Conda environment:**

```bash
conda env create -f environment.yml
conda activate torchenv
```

**🚀 Install GPU-enabled PyTorch (CUDA 12.4):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> 💡 **Note:** The CUDA 12.4 command is an example. Install the PyTorch build compatible with your GPU, CUDA drivers, and operating system.

**📚 Install the remaining dependencies:**

```bash
pip install -r requirements.txt
```

Alternatively, if running from the parent directory containing `NESI`:

```bash
pip install -r NESI/cohort_models/ExternalValidation/requirements.txt
```

---

## ⚙️ External Validation: Step-by-Step Execution

### 🔹 STEP 1: Preprocess EDF Recordings into MAT Files

**🎯 Objective:** Convert raw EEG recordings stored in European Data Format (`.edf`) into MATLAB-compatible (`.mat`) files for MORGOTH feature extraction.

**Activate the NESI environment:**

```bash
conda activate torchenv
```

**Run the preprocessing script:**

```bash
python NESI/cohort_models/ExternalValidation/CodesForExternalVal/STEP1_Preprocessing_EDF_EEGs.py
```

**📤 Output:** Preprocessed EEG recordings saved as `.mat` files in the `NESI/cohort_models/ExternalValidation/ProductsV2/ErikaSegmentEEG/` directory.

> 📢 **Important Note for Reproducibility**
>
> The original EDF recordings are not included in this repository because of their large file sizes.
>
> However, the corresponding preprocessed **MAT files are provided through our AWS S3 data repository**.
>
> ✅ **Researchers reproducing our reported external validation results can skip Step 1 and directly use the provided MAT files to begin Step 2.**
>
> Researchers applying NESI to their own raw EDF recordings should complete Step 1.

---

### 🔹 STEP 2: Extract MORGOTH Features from EEG MAT Files

**🎯 Objective:** Use the pretrained MORGOTH foundation model to extract EEG representations from the preprocessed MAT recordings.

**Activate the MORGOTH environment:**

```bash
conda activate morgoth
```

**Navigate to the external validation MORGOTH directory:**

```bash
cd NESI/cohort_models/ExternalValidation/morgoth
```

**🚀 Execute the GPU-based feature extraction script:**

```bash
bash STEP2_MORGOTH_Continous_Feature_Extraction_GPU.sh
```


**📤 Output:** MORGOTH-derived EEG feature matrices for downstream continuous NESI prediction. The extracted features are organized into six subdirectories under:

`NESI/cohort_models/ExternalValidation/ProductsV2/MorgothAcivtaions_Segments/`

**📂 Output directory structure:**

```text
MorgothAcivtaions_Segments/
├── BS/
├── NM/
├── IIIC/
├── SLOWING/
├── SLEEP/
└── FOCGEN/
```

> ⚠️ **CAUTION: Configure Paths Before Execution**
>
> Before running the shell script, open and review:
>
> `STEP2_MORGOTH_Continous_Feature_Extraction_GPU.sh`
>
> Ensure that the following paths are configured correctly:
>
> - 📂 **Input directory (`eval_sub_dir`):** Location of the preprocessed EEG MAT files.
> - 📁 **Output directory (`eval_results_dir`):** Destination where the extracted MORGOTH feature matrices will be stored.
>
> Incorrect input or output paths may cause feature extraction to fail.
>
> 💡 **GPU Recommendation:** A compatible NVIDIA GPU is recommended for efficient MORGOTH feature extraction.

---

### 🔹 STEP 3: Generate NESI and Clinical Scale Predictions

**🎯 Objective:** Use the extracted MORGOTH features to generate NESI scores and corresponding neurological clinical scale predictions.

**Activate the NESI environment:**

```bash
conda activate torchenv
```

**Run the NESI prediction script:**

```bash
python NESI/cohort_models/ExternalValidation/CodesForExternalVal/STEP3__Extract_NESI_and_RASS_GCS_CAMS_prediction.py
```


**📤 Output:** Model-generated predictions for NESI and three neurological clinical assessment scales, including:

- 🧠 **NESI:** Continuous Neurophysiologic Encephalopathy Severity Index.
- 📊 **RASS:** Richmond Agitation-Sedation Scale.
- 🏥 **GCS:** Glasgow Coma Scale.
- 🔬 **CAM-S:** Confusion Assessment Method Severity.

The predictions are stored in separate subdirectories under:

`NESI/cohort_models/ExternalValidation/ProductsV2/`

**📂 Output directory structure:**

```text
ProductsV2/
├── NESIPredictions/
├── RASSPredictions/
├── CAMSPredictions/
└── GCSPrediction/
```

Each subdirectory contains the corresponding model-generated predictions for downstream analysis and external validation.


> ✅ **Interpretation:** Higher NESI scores indicate greater neurological dysfunction or encephalopathy severity.

---

## 🔄 Complete Pipeline Overview

```text
                 🧠 RAW EEG RECORDINGS
                         (.edf)
                           |
                           v
             ┌─────────────────────────┐
             │ STEP 1: EEG Preprocessing│
             │ Environment: torchenv    │
             └─────────────────────────┘
                           |
                           v
                📂 Preprocessed EEG
                       (.mat)
                           |
                           v
             ┌─────────────────────────┐
             │ STEP 2: MORGOTH Features │
             │ Environment: morgoth     │
             │ GPU-based extraction     │
             └─────────────────────────┘
                           |
                           v
                 📊 EEG Features
                10 min window (591 × 17)
                           |
                           v
             ┌─────────────────────────┐
             │ STEP 3: NESI Prediction |
             │(with 1 sample stride on |
             |       MORGOTH features) |
             │ Environment: torchenv   │
             └─────────────────────────┘
                           |
                           v
             🧠 NESI + RASS + GCS + CAM-S
                Continuous Predictions
```

---

## ✅ Reproducing Our External Validation Results

For researchers interested in reproducing the external validation results reported in our study:

1. ✅ Install and configure both Conda environments (`morgoth` and `torchenv`).
2. ✅ Download the preprocessed EEG MAT files from the accompanying AWS S3 data repository.
3. ✅ **Skip Step 1**, because the preprocessed MAT files are already available.
4. ✅ Execute **Step 2** to extract MORGOTH features.
5. ✅ Execute **Step 3** to generate NESI and clinical scale predictions.

> 📌 **Reproducibility Note:** The supplied preprocessed EEG files eliminate the need to repeat EDF preprocessing, allowing researchers to reproduce the feature-extraction and prediction stages directly.

---

## 📌 Summary of Scripts

| Step | Script | Environment | Purpose |
|------|--------|-------------|---------|
| **1** | `STEP1_Preprocessing_EDF_EEGs.py` | `torchenv` | Convert EDF EEG files to MAT files |
| **2** | `STEP2_MORGOTH_Continous_Feature_Extraction_GPU.sh` | `morgoth` | Extract MORGOTH EEG features |
| **3** | `STEP3__Extract_NESI_and_RASS_GCS_CAMS_prediction.py` | `torchenv` | Generate NESI and clinical scale predictions |

---

### 📚 Reference

**MORGOTH Foundation Model**

🔗 https://github.com/bdsp-core/morgoth

For details about MORGOTH installation, dependencies, and EEG feature extraction, please refer to the official MORGOTH repository.

---

**🧠 NESI — Neurophysiologic Encephalopathy Severity Index**

*An EEG-derived framework for quantifying neurological dysfunction and encephalopathy severity.*
