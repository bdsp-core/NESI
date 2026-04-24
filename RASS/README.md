## RASS Prediction Task

This repository contains the codes and model weights for the RASS prediction task.

### 📁 Folder Structure
```bash
YAMA/RASS
      ├── Model
      │   ├── Training
      │   └── ModelCheckpoints
      │       ├── LR_RASS
      │       ├── RESNEY_GAPonly
      │       ├── RESNET_notimeshift
      │       ├── RESNET_TIMESHIFT
      └── Results5Fld
      └── RASS_Best_DL_model
      │   ├── ResNetGAP
      │       ├── RESNETGAP_Best_RASS.pth (best model weight)
      └── FiguresRASS
```
### 📌 Description

- **Model**: Contains all training-related code and trained model weights.
  - **Training**: Scripts used for model development and training with 5 fold cross validation.
  - **ModelCheckpoints**: Saved weights of the ML/DL models.
    
- **Results5Fld**: Contains evaluation results from 5-fold cross-validation experiments.

