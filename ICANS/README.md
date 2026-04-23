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

- **ModelTraining5Fld**: Contains all training-related code and trained model weights.
  - **Code**: Scripts used for model development and training.
  - **Model Weights**: Saved trained models.
    - **ML Models**: Traditional machine learning model checkpoints.
    - **DL Model**: Deep learning model checkpoints.

- **Results5Fold**: Contains evaluation results from 5-fold cross-validation experiments.
