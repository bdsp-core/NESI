# YAMA
**YAMA: Yielding Automated Mortality and Altered mental status prediction from EEG**

## 📌 Overview

YAMA (Yielding Automated Mortality and Altered Mental Status Prediction) is a machine learning framework designed to estimate clinically relevant neurological severity and consciousness scores directly from EEG signals.

The system targets multiple standardized clinical assessment scales, including:

* Glasgow Coma Scale (GCS) for level of consciousness
* Richmond Agitation-Sedation Scale (RASS) for sedation and agitation
* Confusion Assessment Method (CAMS) for delirium detection
* Immune Effector Cell-Associated Neurotoxicity Syndrome (ICANS) grading
* NESI (Neurophysiological EEG-based Severity Index)

By leveraging EEG as an objective, continuous physiological signal, YAMA aims to provide automated and scalable estimation of neurological status, particularly in critically ill patients where bedside assessments may be intermittent, subjective, or infeasible.

The framework supports both dataset-specific and unified modeling approaches, enabling cross-scale learning and generalization across heterogeneous clinical cohorts. This facilitates consistent neurological monitoring and opens avenues for early detection of deterioration and outcome prediction, including mortality risk.

📂 Repository Structure
YAMA/
│── RASS/                  # Codes, Model weights for RASS Prediction Model
│── GCS/                   # Codes, Model weights for GCS Prediction Model
│── ICANS/                 # Codes, Model weights for CAMS Prediction Model
│── CAMS/                  # Codes, Model weights for ICANS Prediction Model
│── NESI/                  # Codes, Model weights for NESI Prediction Model
│── Figures/               # Codes for Figures
│── requirements.txt       # Python dependencies
│── environment.yml        # Conda environment (recommended)
│── README.md              # Project documentation


## ⚙️ Installation

### 🔹 1. Clone the repository

```bash
git clone https://github.com/bdsp-core/YAMA.git
cd YAMA
```

---

### 🔹 2. Create and activate Conda environment

We recommend using the provided environment file for full reproducibility.

```bash
conda env create -f environment.yml
conda activate torchenv
```

---

### 🔹 3. Install PyTorch (CUDA support)

If using GPU acceleration, install the appropriate PyTorch build:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> ⚠️ Ensure the CUDA version matches your system configuration.

---

### 🔹 4. (Optional) Install remaining pip dependencies

If any packages are missing from the conda environment:

```bash
pip install -r requirements.txt
```

---

### 🔹 5. Verify installation

Run the following to confirm the environment is correctly set up:

```bash
python -c "import torch; import numpy; import shap; print('Environment OK')"
```

---

### 🧠 Notes

* Conda environment ensures reproducibility across systems
* GPU support is recommended for training models
* If using Jupyter, register the kernel:

```bash
python -m ipykernel install --user --name torchenv
```
## 📬 Contact

For questions, collaborations, or inquiries, please contact:

- **Arka Roy** – aroy11@bidmc.harvard.edu  
- **Brandon Westover** – bwestove@bidmc.harvard.edu  
