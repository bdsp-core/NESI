## ICANS Prediction Task

This repository contains the codes and model weights for the ICANS prediction task.

<img width="1279" height="523" alt="Screenshot 2026-06-14 at 12 29 28 PM" src="https://github.com/user-attachments/assets/440f1bd7-9259-4483-b732-bf6e6e1d4d28" />

Supplementary Fig: Illustrates the method for predicting the individual neurological/ neurocognitive score prediction using the EEG recordings.


### 📁 Folder Structure
```bash
YAMA/ICANS
      ├── Cohort/
      │   ├──ICANS_cohort_metadata.csv
      │   ├──Table1_ICANS.py
      ├── model
      │   ├── Training/
      │           └── ICANSTraining_Final_Metadata.csv (contains 10 min EEG seg filename and corresponding ICANS value: metadata)
      |           └── ICANS_DL_ResNets_5Fls.py (Ordinal Deep Learning model training code for ICANS predection task)
      |           └── ICANS_ML_Baselines_5fld.py (SVM, LR, KNN model training code for ICANS prediction task)
      │   └── ModelCheckpoints
      │             ├── KNN_ICANS/
      │             ├── LR_ICANS/
      │             ├── RESNEY_GAPonly/
      │             ├── RESNET_notimeshift/
      │             ├── RESNET_TIMESHIFT/
      │             └── SVM_ICANS/
      └── Results5Fld
      |            └──Results for mean, median Covarraince upper trainagular matrix based SVM, LR, KNN model's result object for 5 fold-CVD
      |            └──Results for MORGOTH Feature matrix 591*17 dim based DL model's Result objects
      |            └──ICANS_Results_Compare.py (Compare all model's results for all performance metrics)
      |            └──ICANS_MORGOTH_Activation_Visualize.py (For each ICANS group shows the variation of morgoth features through heatmaps)      
      └── ICANS_Best_DL_model/
      │   ├── ResNetGAP/
      │       ├── RESNETGAP_Best_ICANS.pth (best model weight)
      └── MorgothActivations/ (Also uploaded in AWS)
      │         ├── BS/
      │         ├── FOCGEN/
      │         ├── IIIC/
      │         ├── NM/
      │         ├── SLEEP/
      │         ├── SLOWING/
      │         └──  SPIKES/
      └── FiguresICANS
```
### 📌 Description

# 🧠 ICANS Module (`YAMA/ICANS`)

---

# 📁 Project Overview

The ICANS module is designed for **predicting ICANS severity from EEG-derived features** using both:

- Classical ML models (SVM, LR, KNN)  
- Deep learning architectures (ResNet-based ordinal regression)

It follows a **5-fold cross-validation framework** and uses:

- Raw EEG-derived 10-minute segments  
- MORGOTH feature representations (591 × 17 feature space)

---

# 📂 Directory Structure (Clean Overview)

---

## 🗂️ Cohort & Metadata

Contains clinical cohort definition and preprocessing scripts.

- **ICANS_cohort_metadata.csv**  
  → Patient-level ICANS labels and demographic metadata  

- **Table1_ICANS.py**  
  → Generates cohort statistics (Table 1 style summary)

---

## 🧪 Model Training Pipeline

---

### 📌 Training Data & Features

- **ICANSTraining_Final_Metadata.csv**  
  → Maps EEG 10-min segments → ICANS labels  
  → Core supervised learning dataset  

---

### 🤖 Deep Learning Models

- **ICANS_DL_ResNets_5Fls.py**  
  → Ordinal deep learning pipeline  
  → ResNet-based architecture  
  → 5-fold cross-validation training  

---

### 📊 Classical Machine Learning Baselines

- **ICANS_ML_Baselines_5fld.py**  
  → Implements:
  - Support Vector Machine (SVM)  
  - Logistic Regression (LR)  
  - K-Nearest Neighbors (KNN)

---

## 💾 Model Checkpoints

Stored best-performing trained models per method:

- KNN_ICANS/  
- LR_ICANS/  
- SVM_ICANS/  
- RESNET_GAPonly/  
- RESNET_notimeshift/  
- RESNET_TIMESHIFT/  

Each folder contains:
- trained model weights  
- fold-wise checkpoints  
- best validation models  

---

## 📊 Results & Evaluation (5-Fold CV)

**Results5Fld/** → Central evaluation hub for all models

Includes:

- Aggregated results (mean/median performance)  
- Covariance-based feature experiments (upper triangular matrices)  
- MORGOTH feature-based DL results (591 × 17 representation)

---

## 🔍 Analysis Scripts

- **ICANS_Results_Compare.py**
  - Compares all models across:
    - Accuracy  
    - AUROC  
    - F1-score  
    - Ordinal metrics  

- **ICANS_MORGOTH_Activation_Visualize.py**
  - Visualizes feature activation patterns per ICANS class  
  - Heatmap-based interpretability of MORGOTH features  

---

## 🧠 Best Deep Learning Model

📦 **ICANS_Best_DL_model/ResNetGAP/**  
- **RESNETGAP_Best_ICANS.pth**  
→ Best-performing ResNet-GAP model checkpoint  

---

## 🔬 MORGOTH Activation Repository

Stored both locally and on AWS.

### Feature Groups:

- BS (Burst Suppression)  
- FOCGEN (Focal Generalized activity)  
- IIIC (ICU EEG classification patterns)  
- NM (Normal/Abnormal morphology)  
- SLEEP (Sleep stages)  
- SLOWING (Generalized/Focal slowing)  
- SPIKES (Epileptiform activity)  

---

### 🧠 Usage

Used for:

- Interpretability analysis  
- Class-wise EEG dynamics  
- Feature saliency across ICANS severity  
