## CAMS Prediction Task

This repository contains the codes and model weights for the CAMS prediction task.

### 📁 Folder Structure
```bash
YAMA/CAMS
      ├── Model
      │   ├── Training
      │   └── ModelCheckpoints
      │       ├── 3class
      │             ├── KNN_CAMS
      │             ├── LR_CAMS
      │             ├── RESNEY_GAPonly
      │             ├── RESNET_notimeshift
      │             ├── RESNET_TIMESHIFT
      │             └── SVM_CAMS
      │       ├── 8class
      │             ├── KNN_CAMS
      │             ├── LR_CAMS
      │             ├── RESNEY_GAPonly
      │             ├── RESNET_notimeshift
      │             ├── RESNET_TIMESHIFT
      │             └── SVM_CAMS
      └── MorgothActivations
      │   ├── BS
      │   ├── FOCGEN
      │   ├── IIIC
      │   ├── NM
      │   ├── SLEEP
      │   ├── SLOWING
      │   └──  SPIKES
      └── Results5Fld
```
### 📌 Description

- **Model**: Contains all training-related code and trained model weights.
  - **Training**: Scripts used for model development and training with 5 fold cross validation.
  - **ModelCheckpoints**: Saved weights of the ML/DL models.
    
- **Results5Fld**: Contains evaluation results from 5-fold cross-validation experiments.
