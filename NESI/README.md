## Neurophysiologic Encephalopathy Severity Index  

## Method
<img width="4409" height="2640" alt="NESI_method" src="https://github.com/user-attachments/assets/4877c1a1-c5fb-4739-b164-41fc67a7af7f" />
Fig.1: Illustrates (a) the overall framework for predicting the Neurophysiological Encephalopathy Severity Index (NESI); (b) the training strategy of the MORGOTH activation encoder for learning compact embedding representations using a triplet-based contrastive learning approach; and (c) the training process of the NESI prediction head employing a pairwise ranking learning framework.


This repository contains the codes and model weights for the NESI prediction task.

### 📁 Folder Structure
```bash
YAMA/NESI
      ├── model (NESI prediction model)
      │         ├── Training
      |         |   ├── CAMSTraining_Final_Metadata.csv
      |         |   ├── GCSTraining_Final_Metadata.csv
      |         |   ├── ICANSTraining_Final_Metadata.csv
      |         |   ├── RASSTraining_Final_Metadata.csv
      |         |   ├── Train_NESI.py (NESI Prediction head code-Training)
      |         |   ├── Train_TripletEncoder.py (Contrastive Encoder Training code)
      │         ├── Testing
      |         |   ├── NESI_Test_Universal.py
      |         |   ├── NESI_Test_Universal_Fulldataset.py (Morgoth Activations needed to dowloaded from AWS)
      |         |   ├── FAST_NESIvsBeSpoke_plot.py (compare NESI model vs BeSpokes)
      |         ├── AblationStudy 
      |         |   ├── Training/
      |         |   ├── Testing/
      |         |   ├── ModelCheckpoints/
      |         |   └── FiguresAblation/
      │         └── ModelCheckpoints/ (NESI model's checkpoint)    
      ├── FiguresNESI/ (Contains the .png figures)
      ├── Bespoke_models/
      │         ├── Training/ (for RASS, GCS, CAMS, and ICANS Bespokes)
      |         |   ├──RASS/
      |         |        └── RASS_bespoke_Triplet_model_training.py
      |         |        └── RASS_bespoke_BADNESS_PREDICTOR_MODEL.py
      │         ├── Testing/
      │         ├── ModelCheckpoint/
      |         └── Resultts/
      ├── ScaleVsNESI/
      ├── DeathPrediction_NESIvsGCS/
      |         └── model_SeqLR/
      |         └──Results/
      ├── NESI-Medication-Analysis/
      |      ├── README.md                                 
      |      ├── eleveld-pk-analysis/                      (Pipeline 1)
      |      │   ├── path_configs.py
      |      │   ├── (analysis scripts at top level)
      |      │   ├── data/
      |      │   └── outputs/
      |      └── medication-exposure/                      (Pipeline 2)
      |          ├── med_config.py
      |          ├── (analysis scripts at top level)
      |          └── data/
      ├──MORGOTHActivationViz_GroupbyNESI/ (Data and code for NESI grouped Morgoth feature Vizualize)
      |      ├── FAST_NESI_group_morgothactivation.py
      |      ├──NESIbin1_data.py
      |      ├──NESIbin2_data.py
      |      ├──NESIbin3_data.py
      |      ├──NESI_combined_MorgothActivation_bw.png
      ├──MorgothFeatureEmbedding/ (PacMaps of MORGOTH Features)

```
### 📌 Description

- **Model**: Contains all training-related code and trained model weights.
  - **Training**: Scripts used for model development and training.
  - **Training**: Scripts used for model testing.
  - **ModelCheckpoints**: Saved weights of the NESI-DL models.
