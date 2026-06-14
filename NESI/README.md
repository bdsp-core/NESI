## Neurophysiologic Encephalopathy Severity Index  

## Method
<img width="4409" height="2640" alt="NESI_method" src="https://github.com/user-attachments/assets/4877c1a1-c5fb-4739-b164-41fc67a7af7f" />
Fig.1: Illustrates (a) the overall framework for predicting the Neurophysiological Encephalopathy Severity Index (NESI); (b) the training strategy of the MORGOTH activation encoder for learning compact embedding representations using a triplet-based contrastive learning approach; and (c) the training process of the NESI prediction head employing a pairwise ranking learning framework.


This repository contains the codes and model weights for the NESI prediction task.

### 📁 Folder Structure
```bash
YAMA/NESI
      ├── model (NESI prediction model)
      │   ├── Training
      |   |   ├── CAMSTraining_Final_Metadata.csv
      |   |   ├── GCSTraining_Final_Metadata.csv
      |   |   ├── ICANSTraining_Final_Metadata.csv
      |   |   ├── RASSTraining_Final_Metadata.csv
      |   |   ├── Train_NESI.py (NESI Prediction head code-Training)
      |   |   ├── Train_TripletEncoder.py (Contrastive Encoder Training code)
      │   ├── Testing
      |   |   ├── NESI_Test_Universal.py
      |   |   ├── NESI_Test_Universal_Fulldataset.py (Morgoth Activations needed to dowloaded from AWS)
      |   |   ├── FAST_NESIvsBeSpoke_plot.py (compare NESI model vs BeSpokes)
      |   ├── AblationStudy 
      |   |   ├── Training
      |   |   ├── Testing
      |   |   ├── ModelCheckpoints
      |   |   └── FiguresAblation
      │   └── ModelCheckpoints (NESI model's checkpoint)    
      └── FiguresNESI (Contains the .png figures)
      └── BespokeModelCheckpoints
      │   ├── Training (for RASS, GCS, CAMS, and ICANS Bespokes)
      |   |   ├──RASS
      |   |        └── RASS_bespoke_Triplet_model_training.py
      |   |        └── RASS_bespoke_BADNESS_PREDICTOR_MODEL.py
      │   ├── Testing
      │   ├── ModelCheckpoint
          └── Resultts

```
### 📌 Description

- **Model**: Contains all training-related code and trained model weights.
  - **Training**: Scripts used for model development and training.
  - **Training**: Scripts used for model testing.
  - **ModelCheckpoints**: Saved weights of the NESI-DL models.
