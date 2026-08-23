# 🧠 External Validation

The `cohort_models/ExternalValidation/` directory contains the tools needed to **externally validate the NESI RASS prediction model** on EEG segments of interest. 🚀

The pipeline takes EEG → extracts meaningful representations using **MORGOTH** → predicts **continuous RASS scores** → and finally lets you visualize the predictions alongside EEG and available clinical RASS scores. 📈🧠

## 📂 Directory Structure

```text
cohort_models/
└── ExternalValidation/
    ├── 📁 RASSModel/
    ├── 📁 morgoth/
    ├── 🐍 Continious_RASSPrediction.py
    ├── 🐍 RASSPredictionVisualizationErikaSeg.py
    ├── 📊 RASS_EEG_Prediction_viz.m
    ├── 📊 RASS_EEG_Spectrum_All.m
    └── 📖 Readme.md
```

The `RASSModel/` and `morgoth/` directories contain the model and supporting components required for the external validation pipeline. The Python and MATLAB scripts are used for **RASS prediction and visualization**. 🔬

## 🛠️ Scripts

| 📄 Filename                                                                                                                                                     | 🔍 What it does                                                                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`Continious_RASSPrediction.py`](https://github.com/bdsp-core/NESI/blob/main/cohort_models/ExternalValidation/Continious_RASSPrediction.py)                     | 🧠 Runs **continuous RASS prediction** on the EEG segment of interest. The selected EEG segment is processed through the MORGOTH/NESI pipeline to generate continuous RASS predictions. |
| [`RASSPredictionVisualizationErikaSeg.py`](https://github.com/bdsp-core/NESI/blob/main/cohort_models/ExternalValidation/RASSPredictionVisualizationErikaSeg.py) | 📈 Plots the **predicted continuous RASS trajectory** together with available continuous RASS scores using Python.                                                                      |
| [`RASS_EEG_Prediction_viz.m`](https://github.com/bdsp-core/NESI/blob/main/cohort_models/ExternalValidation/RASS_EEG_Prediction_viz.m)                           | 🧠📊 MATLAB visualization of the **EEG spectrogram + RASS predictions + continuous RASS scores**.                                                                                       |
| [`RASS_EEG_Spectrum_All.m`](https://github.com/bdsp-core/NESI/blob/main/cohort_models/ExternalValidation/RASS_EEG_Spectrum_All.m)                               | 🌈📊 MATLAB visualization of **EEG spectral information together with continuous RASS predictions and continuous RASS scores**.                                                         |

---

# ⚠️ Prerequisite: MORGOTH 🧠

NESI uses **MORGOTH** as the EEG feature-extraction foundation model.

👉 **MORGOTH must be set up before running the NESI external validation pipeline.**

🔗 [MORGOTH Repository](https://github.com/bdsp-core/morgoth)

Follow the MORGOTH installation instructions and make sure its environment is working correctly before running NESI.

MORGOTH transforms the raw EEG into the **591 × 17 feature matrices** used by NESI. 🧩

```text
Raw EEG 🧠
    │
    ▼
 MORGOTH 🔬
    │
    ▼
591 × 17 Features 📊
    │
    ▼
NESI RASS Model 🤖
    │
    ▼
Continuous RASS Prediction 📈
```

---

# ⚙️ Installation

### 1️⃣ Clone NESI

```bash
git clone https://github.com/bdsp-core/NESI.git
cd NESI
```

### 2️⃣ Create the environment 🐍

We recommend using the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate torchenv
```

### 3️⃣ Install the matching PyTorch build 🔥

For systems using CUDA 12.4:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4️⃣ Install remaining dependencies 📦

```bash
pip install -r requirements.txt
```

or:

```bash
pip install -r NESI/requirements.txt
```

### 5️⃣ Verify the environment ✅

```bash
python -c "import torch, numpy, statsmodels, mne; print('Environment OK')"
```

> 💡 **Tip:** Make sure your PyTorch CUDA version matches the CUDA version available on your system. A GPU 🚀 is recommended for EEG feature extraction and RASS prediction.

---

# 🚀 External Validation Workflow

The typical workflow is:

### 1️⃣ Select the EEG segment

Identify the **EEG segment of interest** that you want to evaluate.

⬇️

### 2️⃣ Generate continuous RASS predictions 🤖

Run:

```bash
python cohort_models/ExternalValidation/Continious_RASSPrediction.py
```

This processes the EEG segment through the MORGOTH/NESI pipeline and generates the **continuous RASS predictions**. 📈

⬇️

### 3️⃣ Visualize the predictions 📊

Run:

```bash
python cohort_models/ExternalValidation/RASSPredictionVisualizationErikaSeg.py
```

This creates a Python-based visualization showing the **predicted continuous RASS trajectory** alongside the available continuous RASS scores.

⬇️

### 4️⃣ Explore EEG + RASS in MATLAB 🔬

For EEG and RASS prediction visualization:

```matlab
RASS_EEG_Prediction_viz
```

For EEG spectrum + RASS visualization:

```matlab
RASS_EEG_Spectrum_All
```

These provide a more detailed look at how the **EEG spectral characteristics** relate to the predicted RASS trajectory. 🧠📈

---

# 🗺️ At a Glance

```text
             🧠 EEG Segment
                    │
                    ▼
             🔬 MORGOTH
                    │
                    ▼
             📊 591 × 17
              Features
                    │
                    ▼
             🤖 NESI Model
                    │
                    ▼
          📈 Continuous RASS
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     🐍 Python            📊 MATLAB
   Visualization        Visualization
          │                   │
          ▼                   ▼
   RASS Trajectory     EEG + Spectrum
                         + RASS
```

## 🎯 In Short

**EEG goes in → MORGOTH does the feature extraction → NESI predicts RASS → Python/MATLAB helps you visualize what happened.** 🧠➡️🔬➡️🤖➡️📈

Happy validating! 🚀🧠

