## ICANS Prediction Task

This repository contains the codes and model weights for the ICANS prediction task.

### 📁 Folder Structure
```bash
YAMA/ICANS
      ├── Model
      │   ├── Training
      │   └── ModelCheckpoints
      │       ├── KNN_ICANS
      │       ├── LR_ICANS
      │       ├── RESNEY_GAPonly
      │       ├── RESNET_notimeshift
      │       ├── RESNET_TIMESHIFT
      │       └── SVM_ICANS
      └── Results5Fld
      └── ICANS_Best_DL_model
      │   ├── ResNetGAP
      │       ├── RESNETGAP_Best_ICANS.pth (best model weight)
      └── MorgothActivations
      │   ├── BS
      │   ├── FOCGEN
      │   ├── IIIC
      │   ├── NM
      │   ├── SLEEP
      │   ├── SLOWING
      │   └──  SPIKES
```
### 📌 Description

- **Model**: Contains all training-related code and trained model weights.
  - **Training**: Scripts used for model development and training with % fold cross validation.
  - **ModelCheckpoints**: Saved weights of the ML/DL models.
    
- **Results5Fld**: Contains evaluation results from 5-fold cross-validation experiments.
