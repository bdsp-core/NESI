# 🧠 External Validation

The `cohort_models/ExternalValidation/` directory contains the tools needed to **externally validate the NESI RASS prediction model** on EEG segments of interest. 🚀

The pipeline takes EEG → extracts meaningful representations using **MORGOTH** → predicts **continuous RASS scores** → and finally lets you visualize the predictions alongside EEG and available clinical RASS scores. 📈🧠

## 📂 Directory Structure

```text
cohort_models/
└── ExternalValidation/
    ├── 📁 RASSModel/
        ├── RESNETGAP_Best_RASS.pth (Weight file of the RASS prediction model)
    ├── 📁 morgoth/
    ├── 🐍 Continious_RASSPrediction.py
    ├── 🐍 RASSPredictionVisualizationErikaSeg.py
    ├── 📊 RASS_EEG_Prediction_viz.m
    ├── 📊 RASS_EEG_Spectrum_All.m
    ├── **Keep your EEG EDF files Here ONLY**
    ├── **Keep your excel metadata file here ONLY**
    └── 📖 Readme.md
```

The `RASSModel/` and `morgoth/` directories contain the model and supporting components required for the external validation pipeline. The Python and MATLAB scripts are used for **RASS prediction and visualization**. 🔬

## 🛠️ Scripts

| 📄 Filename                                                                                                                                                     | 🔍 What it does                                                                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`Continious_RASSPrediction.py`](https://github.com/bdsp-core/NESI/blob/main/cohort_models/ExternalValidation/Continious_RASSPrediction.py)                     | 🧠 Runs **continuous RASS prediction** on the EEG segment of interest. The selected EEG segment is processed through the MORGOTH/NESI pipeline to generate continuous RASS predictions. |
| [`RASSPredictionVisualizationErikaSeg.py`](https://github.com/bdsp-core/NESI/blob/main/cohort_models/ExternalValidation/RASSPredictionVisualizationErikaSeg.py) | 📈 Plots the **predicted continuous RASS trajectory** together with available continuous RASS scores using Python.                                                                      |
| [`RASS_EEG_Prediction_viz.m`](https://github.com/bdsp-core/NESI/blob/main/cohort_models/ExternalValidation/RASS_EEG_Prediction_viz.m),  [`RASS_EEG_Spectrum_All.m`](https://github.com/bdsp-core/NESI/blob/main/cohort_models/ExternalValidation/RASS_EEG_Spectrum_All.m)                           | 🧠📊 MATLAB visualization of the **EEG spectrogram + RASS predictions + continuous RASS scores**.                                                                                       |

---

# 🐍 Environment Setup

The NESI external validation pipeline requires **two separate Conda environments**. We recommend keeping the same naming convention used by the project:

- 🔬 **`morgoth`** — used for EEG feature extraction with MORGOTH
- 🤖 **`torchenv`** — used for the RASS prediction pipeline

Keeping these environments separate helps avoid dependency and version conflicts between MORGOTH and the RASS prediction code.

## 1️⃣ Create the `morgoth` Environment 🔬

The `morgoth` environment should be created and configured by following the installation instructions provided in the MORGOTH repository.

🔗 [MORGOTH Repository](https://github.com/bdsp-core/morgoth)

Please follow the MORGOTH installation procedure exactly as specified in its repository and ensure that the environment is named:

    conda activate morgoth

This environment is responsible for transforming raw EEG into the **591 × 17 feature representation** used by NESI.

> ⚠️ **Important:** Make sure the `morgoth` environment is fully functional before proceeding with the NESI environment.

## 2️⃣ Create the `torchenv` Environment 🤖

The `torchenv` environment is used for the NESI/RASS prediction pipeline.

From the root of the NESI repository (or under this folder, where the environment-related files are present), create the environment using the provided environment file:

    conda env create -f environment.yml

Then activate it:

    conda activate torchenv

Alternatively, if you are setting up the environment manually, install the dependencies listed in the repository's `requirements.txt`:

    pip install -r requirements.txt

For GPU systems using CUDA 12.4, install the corresponding PyTorch build:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Finally, verify that the environment is working:

    python -c "import torch, numpy, statsmodels, mne; print('Environment OK')"


---


