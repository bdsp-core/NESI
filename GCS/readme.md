## GCS Prediction Task

This repository contains the codes and CORN-based ordinal regression model weights for the GCS prediction task.
<img width="1279" height="523" alt="Screenshot 2026-06-14 at 12 29 28 PM" src="https://github.com/user-attachments/assets/440f1bd7-9259-4483-b732-bf6e6e1d4d28" />

Supplementary Fig: Illustrates the method for predicting the individual neurological/ neurocognitive score prediction using the EEG recordings.


### 📁 Folder Structure
```bash
YAMA/GCS
      ├── Cohort/
      │     ├── MGH Cohort metadata (csv file)
      │     ├── BWH Cohort metadata (csv file)
      │     ├── Table1_GCS.py (Code for Table 1 for GCS Cohort)
      │     ├── AWSKeyPathDownload_GCSEEG.py (Code to build end to end path to download EEGs from AWS to local machine)
      │     ├── GCSEEG_download_AWS.txt (Ouput from above code containing end to S3 path instrutions to download ecah EEG file)
      ├── EEGPreprocessingDownloadSQA/
      |           ├──EEG_download_10min_GCS_MGB.py (Downloads Session-wise all relevant 10 min EEG segs which has a valid GCS recording: N*19*120000)
      |           ├──Create_individual_eeg10min_GCS.py (Creates 10 min individual EEGs that were downloaded using previous code)
      |           ├──EEG_SQA_based_segment_discard.py (checks SQA and tells which EEGs can be discraded)
      ├── Model/
      │     ├── Training
      |            └── GCSTraining_Final_Metadata.csv (contains 10 min EEG seg filename and corresponding GCS value: metadata)
      |            └── GCS_DL_ResNets_5Fls.py (Ordinal Deep Learning model training code for GCS predection task)
      |            └── GCS_ML_Baselines_5fld.py (SVM, LR, KNN model training code for GCS prediction task)
      │     ├── Testing
      |            └── GCS_Test_DL_ResNets_5fld.py (Testing code for DL models)
      │    
      └── Results5Fld/
      |            └──Results for mean, median Covarraince upper trainagular matrix based SVM, LR, KNN model's result object for 5 fold-CVD
      |            └──Results for MORGOTH Feature matrix 591*17 dim based DL model's Result objects
      |            └──GCS_Results_Compare.py (Compare all model's results for all performance metrics)
      |            └──GCS_EEG_morgoth_activation.py (For each GCS group shows the variation of morgoth features through heatmaps)
      |            └──GCS_Swimmers_plot.py (GCS Cohort's annotation variation over time for each subject)
      └── GCS_Best_DL_model/
      │   ├── ResNetGAP/
      │       ├── RESNETGAP_Best_GCS.pth (best model weight)
      └── FiguresGCS
```
### 📌 Description

## 📌 GCS Module (`YAMA/GCS`)

This module contains the complete pipeline for **GCS (Richmond Agitation-Sedation Scale) prediction using EEG data**, including cohort construction, preprocessing, machine learning/deep learning models, and downstream analysis/visualization.

---

## 📁 Cohort/

Contains cohort-level metadata and preprocessing utilities.

- 📄 MGH Cohort metadata (CSV file)
- 📄 BWH Cohort metadata (CSV file)
- 🧠 `Table1_GCS.py` — Code for Table 1 generation for GCS cohort
- ☁️ `AWSKeyPathDownload_GCSEEG.py` — Builds end-to-end S3 paths to download EEGs from AWS to local machine
- 📜 `GCSEEG_download_AWS.txt` — Output file containing S3 download paths for each EEG file

---

## ⚙️ EEGPreprocessingDownloadSQA/

EEG extraction, preprocessing, and signal quality assessment pipeline.

- ⏱️ `EEG_download_10min_GCS_MGB.py` — Downloads session-wise 10-min EEG segments with valid GCS recordings (N × 19 × 120000)
- 🧩 `Create_individual_eeg10min_GCS.py` — Converts downloaded session EEGs into individual 10-min EEG files
- 🧪 `EEG_SQA_based_segment_discard.py` — Performs signal quality assessment and discards low-quality EEG segments

---

## 🤖 Model/

### 🏋️ Training

- 📊 `GCSTraining_Final_Metadata.csv` — Metadata mapping EEG segments to GCS labels
- 🧠 `GCS_DL_ResNets_5Fls.py` — Ordinal deep learning model (ResNet) for GCS prediction
- 📈 `GCS_ML_Baselines_5fld.py` — Classical ML baselines (SVM, LR, KNN) with 5-fold CV

### 🧪 Testing

- 🔍 `GCS_Test_DL_ResNets_5fld.py` — Evaluation script for trained DL models

---

## 📊 Results5Fld/

Model evaluation outputs and analysis scripts.

- 📦 Contains results for:
  - Covariance-based SVM / LR / KNN models
  - MORGOTH feature-based deep learning models (591 × 17 feature representation)
- 📉 `GCS_Results_Compare.py` — Compares performance metrics across all models
- 🔥 `GCS_EEG_morgoth_activation.py` — Heatmap visualization of MORGOTH feature activations across GCS groups
- 🏊 `GCS_Swimmers_plot.py` — Subject-wise longitudinal GCS trajectory visualization

---

## 🏆 GCS_Best_DL_model/

- 🧠 `ResNetGAP/`
  - 🥇 `RESNETGAP_Best_GCS.pth` — Best trained model checkpoint

---

## 🎨 FiguresGCS/

Contains all publication-quality figures:

- 📊 Model performance plots
- 🔥 Activation heatmaps
- 📈 Cohort distribution plots
- 🏊 Longitudinal GCS trajectory plots
