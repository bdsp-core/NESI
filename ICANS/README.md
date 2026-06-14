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

- **Model**: Contains all training-related code and trained model weights.
  - **Training**: Scripts used for model development and training with % fold cross validation.
  - **ModelCheckpoints**: Saved weights of the ML/DL models.
    
- **Results5Fld**: Contains evaluation results from 5-fold cross-validation experiments.
