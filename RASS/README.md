## RASS Prediction Task

This repository contains the codes and model weights for the RASS prediction task.

### 📁 Folder Structure
```bash
YAMA/RASS
      ├── Cohort
      │     ├── MGH Cohort metadata (csv file)
      │     ├── BWH Cohort metadata (csv file)
      ├── Model
      │   ├── Training
      │   ├── Testing
      │    
      └── Results5Fld
      └── RASS_Best_DL_model
      │   ├── ResNetGAP
      │       ├── RESNETGAP_Best_RASS.pth (best model weight)
      └── FiguresRASS
```
### 📌 Description

- **Model**: Contains all training-related code and trained model weights.
  - **Training**: Scripts used for model development and training with 5 fold cross validation.
  - 📌📌**ModelCheckpoints**: This has to be downloaded from the AWS. Due to Space realted constraint; we did not upload the Saved weights of the ML/DL model here in RASS folder of GitHub. Please download them from the AWS
    
- **Results5Fld**: Contains evaluation results from 5-fold cross-validation experiments.

