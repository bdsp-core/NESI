## RASS Prediction Task

This repository contains the codes and CORN-based ordinal regression model weights for the RASS prediction task.

### 📁 Folder Structure
```bash
YAMA/RASS
      ├── Cohort/
      │     ├── MGH Cohort metadata (csv file)
      │     ├── BWH Cohort metadata (csv file)
      │     ├── Table1_RASS.py (Code for Table 1 for RASS Cohort)
      │     ├── AWSKeyPathDownload_RASSEEG.py (Code to build end to end path to download EEGs from AWS to local machine)
      │     ├── RASSEEG_download_AWS.txt (Ouput from above code containing end to S3 path instrutions to download ecah EEG file)
      ├── EEGPreprocessingDownloadSQA/
      |           ├──EEG_download_10min_RASS_MGB.py (Downloads Session-wise all relevant 10 min EEG segs which has a valid RASS recording: N*19*120000)
      |           ├──Create_individual_eeg10min_RASS.py (Creates 10 min individual EEGs that were downloaded using previous code)
      |           ├──EEG_SQA_based_segment_discard.py (checks SQA and tells which EEGs can be discraded)
      ├── Model/
      │     ├── Training
      |            └── RASSTraining_Final_Metadata.csv (contains 10 min EEG seg filename and corresponding RASS value: metadata)
      |            └── RASS_DL_ResNets_5Fls.py (Ordinal Deep Learning model training code for RASS predection task)
      |            └── RASS_ML_Baselines_5fld.py (SVM, LR, KNN model training code for RASS prediction task)
      │     ├── Testing
      |            └── RASS_Test_DL_ResNets_5fld.py (Testing code for DL models)
      │    
      └── Results5Fld/
      |            └──Results for mean, median Covarraince upper trainagular matrix based SVM, LR, KNN model's result object for 5 fold-CVD
      |            └──Results for MORGOTH Feature matrix 591*17 dim based DL model's Result objects
      |            └──RASS_Results_Compare.py (Compare all model's results for all performance metrics)
      |            └──RASS_EEG_morgoth_activation.py (For each RASS group shows the variation of morgoth features through heatmaps)
      |            └──RASS_Swimmers_plot.py (RASS Cohort's annotation variation over time for each subject)
      └── RASS_Best_DL_model/
      │   ├── ResNetGAP/
      │       ├── RESNETGAP_Best_RASS.pth (best model weight)
      └── FiguresRASS
```
### 📌 Description

- **Model**: Contains all training-related code and trained model weights.
  - **Training**: Scripts used for model development and training with 5 fold cross validation.
  - 📌📌**ModelCheckpoints**: This has to be downloaded from the AWS. Due to Space realted constraint; we did not upload the Saved weights of the ML/DL model here in RASS folder of GitHub. Please download them from the AWS
    
- **Results5Fld**: Contains evaluation results from 5-fold cross-validation experiments.

