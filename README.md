# NESI — Neurophysiologic Encephalopathy Severity Index

**A unified, continuous, EEG-based index of encephalopathy severity for critical care.**

**🔬 Live interactive demo: https://bdsp-core.github.io/NESI/** — click any point on the PaCMAP embedding to view its 10-minute EEG segment (color by NESI, dominant IIC pattern, or MORGOTH feature).

This repository contains the code and trained model weights for NESI, a continuous
electroencephalography (EEG)-derived measure of acute brain dysfunction. Instead of predicting each
bedside scale separately, NESI treats the major consciousness/encephalopathy scales as noisy
observations of a single shared latent construct and recovers that physiologic dimension directly
from EEG, placing patients on one axis spanning alertness → delirium → sedation → coma.

## 📌 Overview

Disorders of consciousness are assessed at the bedside with multiple, partially redundant scales
that are intermittent, subjective, and often discordant. NESI unifies four of them —
**RASS** (Richmond Agitation–Sedation Scale), **GCS** (Glasgow Coma Scale),
**CAM-S** (Confusion Assessment Method–Severity), and **ICANS** (immune effector cell–associated
neurotoxicity syndrome grading) — onto a single continuous severity scale.

The pipeline has three stages:

1. **Foundation-model features** — each 10-minute EEG segment is passed through the
   [MORGOTH](https://github.com/bdsp-core/morgoth) clinical-EEG foundation model, yielding a
   591×17 matrix of event-level feature activations.
2. **Contrastive encoder** — a supervised-contrastive ResNet compresses that matrix to a 40-D
   embedding.
3. **Pairwise-ranking head** — a Siamese MLP maps the embedding to a single scalar NESI value,
   trained with within-scale and cross-scale ordinal pairs (assembled into a unified dataset,
   SPECTRA) so severity is learned on one scale-agnostic axis.

**Key results** (held-out test set; see the manuscript): NESI agrees strongly with all four scales
(Spearman ρ ≈ 0.72–0.84), shows higher AUROC than GCS for longitudinal in-hospital mortality
(≈0.82 vs 0.76 at 20 h), and tracks propofol effect-site concentration more closely than RASS.

> **Project naming:** this repository was previously named *YAMA*; it is the codebase for the NESI
> project described above. The per-cohort scale models (RASS/GCS/CAM-S/ICANS) are the building
> blocks that feed the unified NESI model.

## 📦 Data & published release

This repo holds **code and model weights only**. The data (EEG segments, MORGOTH activations,
assembled training/validation/test sets, per-cohort metadata, derived tables) are de-identified
(surrogate subject IDs; shifted dates) and distributed under **credentialed access** on BDSP:

- **Published dataset (bdsp.io):** https://bdsp.io/content/1yizvr4l41wmiljc6lou/1.0.0/
- **Credentialed S3:** `s3://bdsp-opendata-credentialed/yama/`
- **DOIs:** version `10.60508/rrbg-ba24` · core `10.60508/65as-th98`
- **Provenance:** `yama/segment_index.csv` (every segment → its continuous source EEG) and
  `yama/source_eeg_files.csv`.

See **[`docs/REPRODUCE.md`](docs/REPRODUCE.md)** for a figure/table → script map and the run order.

**Medication data via OMOP.** The medication exposures (originally pulled from S3 parquet MARs) are
now in the BDSP **OMOP Aurora** database and are far easier to query there
(`BDSPPatientID` == OMOP `person_id`; administrations are `drug_type_concept_id = 38000180`).
A ready-to-run, verified reproduction lives in
**[`NESI/NESI-Medication-Analysis/omop/`](NESI/NESI-Medication-Analysis/omop/)** — it reproduces
Table 2's RASS propofol exposure from OMOP at **3,245 / 6,188 = 52.4%** vs the published
**3,250 / 6,188 = 52.5%** (denominator exact; 5-patient / 0.1-pp difference).

## 📂 Repository structure

```
NESI/
├── cohort_models/      # per-scale building-block models (code + weights)
│   ├── RASS/  GCS/  CAMS/  ICANS/
├── NESI/               # the unified NESI model: training, testing, ablations,
│                       #   embedding atlas, ScaleVsNESI, medication/propofol
│                       #   (Eleveld PK), DeathPrediction (NESI vs GCS), requirements.txt
├── mortality_analysis/ # in-hospital mortality cohort assembly + NESI correlation
├── figures/
│   ├── main/           # final main-text figures (PNG + PDF) + generators (Codes/)
│   └── supplementary_score_prediction/
├── docs/               # REPRODUCE.md (figure/table → script map) + guides
├── requirements.txt
├── environment.yml
└── README.md
```

## ⚠️ Prerequisite: MORGOTH

NESI uses **MORGOTH** as the EEG feature-extraction foundation model. **Set up MORGOTH first:**

🔗 https://github.com/bdsp-core/morgoth

Follow its installation instructions and ensure its environment is functional before installing
NESI; the pipeline depends on MORGOTH to turn raw EEG into the 591×17 feature matrices.

## ⚙️ Installation

```bash
# 1. Clone
git clone https://github.com/bdsp-core/NESI.git
cd NESI

# 2. Create the environment (recommended)
conda env create -f environment.yml
conda activate torchenv

# 3. (GPU) install the matching PyTorch build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining dependencies
pip install -r requirements.txt          # or: pip install -r NESI/requirements.txt

# 5. Verify
python -c "import torch, numpy, statsmodels, mne; print('Environment OK')"
```

> Ensure the CUDA version matches your system. GPU is recommended for training.

## 🔁 Reproducing the paper

1. Request credentialed access to the dataset (link above) and `aws s3 sync` the cohort(s) you need.
2. Set up MORGOTH and this environment.
3. Follow **[`docs/REPRODUCE.md`](docs/REPRODUCE.md)**, which maps each figure and table to the
   script that produces it (training → evaluation → figures), and lists where the trained
   checkpoints and result files live.

## 📑 Citation

If you use NESI, please cite the dataset and the manuscript:

> Roy A, Surrao K, Sun C, Jing J, … Kimchi EY, Zafar SF, Eckhardt CA, Westover MB.
> *Neurophysiologic Encephalopathy Severity Index (NESI) — Data and Code.* Brain Data Science
> Platform (2026). https://doi.org/10.60508/65as-th98

(Manuscript citation to be added on publication.)

## 📬 Contact

- **Arka Roy** — aroy11@bidmc.harvard.edu
- **M. Brandon Westover** — mbwest@stanford.edu

## License

See [`LICENSE.txt`](LICENSE.txt). The accompanying data are governed by the BDSP Credentialed
Health Data License and Data Use Agreement (see the bdsp.io project page).
