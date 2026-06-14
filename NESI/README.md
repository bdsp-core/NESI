## Neurophysiologic Encephalopathy Severity Index  
This repo contains the NESI prediction pipeline and related analysis.

## Method
<img width="4409" height="2640" alt="NESI_method" src="https://github.com/user-attachments/assets/4877c1a1-c5fb-4739-b164-41fc67a7af7f" />
Fig.1: Illustrates (a) the overall framework for predicting the Neurophysiological Encephalopathy Severity Index (NESI); (b) the training strategy of the MORGOTH activation encoder for learning compact embedding representations using a triplet-based contrastive learning approach; and (c) the training process of the NESI prediction head employing a pairwise ranking learning framework.


This repository contains the codes and model weights for the NESI prediction task.

### 📁 Folder Structure
```bash
YAMA/NESI
      ├── model (NESI prediction model)
      │         ├── Training
      |         |   ├── CAMSTraining_Final_Metadata.csv
      |         |   ├── GCSTraining_Final_Metadata.csv
      |         |   ├── ICANSTraining_Final_Metadata.csv
      |         |   ├── RASSTraining_Final_Metadata.csv
      |         |   ├── Train_NESI.py (NESI Prediction head code-Training)
      |         |   ├── Train_TripletEncoder.py (Contrastive Encoder Training code)
      │         ├── Testing
      |         |   ├── NESI_Test_Universal.py
      |         |   ├── NESI_Test_Universal_Fulldataset.py (Morgoth Activations needed to dowloaded from AWS)
      |         |   ├── FAST_NESIvsBeSpoke_plot.py (compare NESI model vs BeSpokes)
      |         ├── AblationStudy 
      |         |   ├── Training/
      |         |   ├── Testing/
      |         |   ├── ModelCheckpoints/
      |         |   └── FiguresAblation/
      │         └── ModelCheckpoints/ (NESI model's checkpoint)    
      ├── FiguresNESI/ (Contains the .png figures)
      ├── Bespoke_models/
      │         ├── Training/ (for RASS, GCS, CAMS, and ICANS Bespokes)
      |         |   ├──RASS/
      |         |        └── RASS_bespoke_Triplet_model_training.py
      |         |        └── RASS_bespoke_BADNESS_PREDICTOR_MODEL.py
      │         ├── Testing/
      │         ├── ModelCheckpoint/
      |         └── Resultts/
      ├── ScaleVsNESI/
      ├── DeathPrediction_NESIvsGCS/
      |         └── model_SeqLR/
      |         └──Results/
      ├── NESI-Medication-Analysis/
      |      ├── README.md                                 
      |      ├── eleveld-pk-analysis/                      (Pipeline 1)
      |      │   ├── path_configs.py
      |      │   ├── (analysis scripts at top level)
      |      │   ├── data/
      |      │   └── outputs/
      |      └── medication-exposure/                      (Pipeline 2)
      |          ├── med_config.py
      |          ├── (analysis scripts at top level)
      |          └── data/
      ├──MORGOTHActivationViz_GroupbyNESI/ (Data and code for NESI grouped Morgoth feature Vizualize)
      |      ├── FAST_NESI_group_morgothactivation.py
      |      ├──NESIbin1_data.py
      |      ├──NESIbin2_data.py
      |      ├──NESIbin3_data.py
      |      ├──NESI_combined_MorgothActivation_bw.png
      ├──MorgothFeatureEmbedding/ (PacMaps of MORGOTH Features)

```
### 📌 Description

## Repository Structure: NESI/

This repository contains the code, models, datasets, and analysis pipelines used for the development, evaluation, and application of the Neurocognitive EEG Severity Index (NESI). The repository is organized into several modules corresponding to model training, benchmarking, downstream clinical analyses, and visualization.

### `model/`

Contains the core NESI model development framework.

* **Training/**: Training scripts and metadata files used to train the NESI contrastive encoder and NESI prediction head using SPECTRA: RASS, GCS, CAMS, and ICANS clinical scales together.
* **Testing/**: Scripts for evaluating trained NESI model on independent datasets and comparing NESI model's performance against bespoke badmess score prediction models that are trained on individual clinical score data groups.
* **AblationStudy/**: Training, testing, checkpoints, and figure generation code used for model ablation experiments.
  <img width="622" height="505" alt="Screenshot 2026-06-14 at 11 03 23 AM" src="https://github.com/user-attachments/assets/97852ffe-b445-41e4-8717-04445ea40b60" />

Supplementary Fig: Baseline NESI prediction pipelines used for ablation analyses to evaluate the effectiveness of incorporating the MORGOTH       activation encoder for NESI prediction. Illustrates (a) the alternative/baseline NESI prediction pipelines without using MORGOTH activation encoder; NESI prediction performance on a common hold-out testing set for the baseline systems trained with (b) median, (c) mean features derived from the MORGOTH activation matrix. 

* **ModelCheckpoints/**: Saved checkpoints of trained NESI models.

### `FiguresNESI/`

Contains publication-quality figures and visualizations generated throughout the NESI project.

### `Bespoke_models/`

Implementation of task-specific (bespoke) models developed for individual clinical scales including RASS, GCS, CAMS, and ICANS.

* **Training/**: Training scripts for individual bespoke models (Contrastive encoder+Badness score predictor head).
* **Testing/**: Evaluation and inference scripts.
* **ModelCheckpoint/**: Saved model weights (for each bespoke models for each dataset).
* **Results/**: Performance metrics and outputs.

### `ScaleVsNESI/`

Analysis scripts and results comparing NESI against individual clinical scale predictions.

### `DeathPrediction_NESIvsGCS/`

Contains survival and mortality prediction experiments comparing NESI-based features against GCS-based approaches.

* **model_SeqLR/**: Sequential logistic regression modeling training code NESI vs GCS.
* **Results/**: Evaluation outputs and figures.

### `NESI-Medication-Analysis/`

Pharmacologic analyses investigating relationships between medication exposure and NESI.

#### `eleveld-pk-analysis/` (Pipeline 1)

Pharmacokinetic modeling pipeline based on the Eleveld framework.

* Configuration files and analysis scripts.
* **data/**: Input datasets.
* **outputs/**: Generated results and figures.

#### `medication-exposure/` (Pipeline 2)

Medication exposure analysis pipeline for quantifying and studying medication administration patterns relative to NESI trajectories.

* Analysis scripts and configuration files.
* **data/**: Input datasets.

### `MORGOTHActivationViz_GroupbyNESI/`

Tools and datasets for visualizing MORGOTH feature activations stratified by NESI severity groups. Includes scripts for generating activation distributions and group-wise feature visualizations.

### `MorgothFeatureEmbedding/`

Visualization and embedding analysis of MORGOTH latent representations, including PacMAP projections and exploratory feature-space analyses.
