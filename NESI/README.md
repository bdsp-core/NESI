## NESI Prediction Task

## Method
<img width="4409" height="2640" alt="NESI_method" src="https://github.com/user-attachments/assets/4877c1a1-c5fb-4739-b164-41fc67a7af7f" />
Fig.1: Illustrates (a) the overall framework for predicting the Neurophysiological Encephalopathy Severity Index (NESI); (b) the training strategy of the MORGOTH activation encoder for learning compact embedding representations using a triplet-based contrastive learning approach; and (c) the training process of the NESI prediction head employing a pairwise ranking learning framework.


This repository contains the codes and model weights for the NESI prediction task.

### 📁 Folder Structure
```bash
YAMA/NESI
      ├── model
      │   ├── Training
      │   ├── Testing
      |   ├── AblationStudy
      |   |   ├── Training
      |   |   ├── Testing
      |   |   ├── ModelCheckpoints
      |   |   └── FiguresAblation
      │   └── ModelCheckpoints
      │       
      └── FiguresNESI
      └── BespokeModelCheckpoints
      │   ├── RASS_bespoke.pth
      │   ├── GCS_bespoke.pth
      │   ├── ICANS_bespoke.pth
          └── CAMS_bespoke.pth
```
### 📌 Description

- **Model**: Contains all training-related code and trained model weights.
  - **Training**: Scripts used for model development and training.
  - **Training**: Scripts used for model testing.
  - **ModelCheckpoints**: Saved weights of the NESI-DL models.
