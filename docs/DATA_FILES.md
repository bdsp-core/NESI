# Data files kept outside this repository

Patient- and segment-level files are distributed through the credentialed BDSP release (`s3://bdsp-opendata-credentialed/yama/`), not through GitHub. Set `NESI_DATA_ROOT` to a local copy that mirrors the paths below.

| Path under `NESI_DATA_ROOT` | Bytes | SHA-256 (first 12) | Current S3 location |
|---|---:|---|---|
| `NESI/Bespoke_models/Results/CAMSBespokescore_results.csv` | 44555 | `d9ac8b4e9413` | pending upload |
| `NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv` | 17242871 | `cdff9800becd` | pending upload |
| `NESI/Bespoke_models/Results/GCSBespokescore_results.csv` | 9067678 | `dcfa38eb2995` | pending upload |
| `NESI/Bespoke_models/Results/ICANSBespokescore_results.csv` | 61191 | `45af8769889c` | `yama/NESI/Bespoke_models/Results/ICANSBespokescore_results.csv` |
| `NESI/Bespoke_models/Results/RASSBespokescore_results.csv` | 8087181 | `2b70366d9ecf` | pending upload |
| `NESI/MORGOTHActivationViz_GroupedbyNESI/NESIbin1_data.pkl` | 240984 | `91b25472b911` | pending upload |
| `NESI/MORGOTHActivationViz_GroupedbyNESI/NESIbin2_data.pkl` | 228136 | `8c69f3697688` | pending upload |
| `NESI/MORGOTHActivationViz_GroupedbyNESI/NESIbin3_data.pkl` | 169590 | `5093026a026c` | pending upload |
| `NESI/MORGOTHActivationViz_GroupedbyNESI/UniversalBadnessModelResult_Full.csv` | 14724457 | `5ddb10ff46b8` | pending upload |
| `NESI/MorgothFeatureEmbedding/NESI_pacmap_iiic_data.npz` | 676674 | `9cc7596f06b4` | pending upload |
| `NESI/PacMAP/NESI_pacmap_coords.csv` | 9759818 | `689ed1b0a4e9` | pending upload |
| `NESI/PacMAP/UniversalBadnessModelResult_Full.csv` | 14724457 | `5ddb10ff46b8` | pending upload |
| `NESI/model/Training/CAMSTraining_Final_Metadata.csv` | 57419 | `8c2a1d434e47` | `yama/cohort_models/CAMS/CAMSTraining_Final_Metadata.csv` |
| `NESI/model/Training/GCSTraining_Final_Metadata.csv` | 4762367 | `ea3ff22fdbae` | pending upload |
| `NESI/model/Training/ICANSTraining_Final_Metadata.csv` | 79726 | `97159d374dee` | `yama/cohort_models/ICANS/ICANSTraining_Final_Metadata.csv` |
| `NESI/model/Training/RASSTraining_Final_Metadata.csv` | 4758585 | `43897cfa63d8` | `yama/cohort_models/RASS/RASSTraining_Final_Metadata.csv` |
| `cohort_models/CAMS/Results5Fld/RESNET_5foldNOSHUFFLE_Results_3class.pkl` | 14350 | `969fba079a4c` | pending upload |
| `cohort_models/CAMS/Results5Fld/RESNET_5foldNOSHUFFLE_Results_8class.pkl` | 35390 | `6de76905e17a` | pending upload |
| `cohort_models/CAMS/Results5Fld/RESNET_5foldOnlyGap_Results_3class.pkl` | 14350 | `a864c4ba1ab0` | pending upload |
| `cohort_models/CAMS/Results5Fld/RESNET_5foldOnlyGap_Results_8class.pkl` | 35390 | `7becfcc5f5a4` | pending upload |
| `cohort_models/CAMS/Results5Fld/RESNET_5foldTIMESHUFFLE_Results_3class.pkl` | 14350 | `a5ec7fe8faea` | pending upload |
| `cohort_models/CAMS/Results5Fld/RESNET_5foldTIMESHUFFLE_Results_8class.pkl` | 35390 | `1140393e7fd5` | pending upload |
| `cohort_models/CAMS/model/Training/CAMSTraining_Final_Metadata.csv` | 57419 | `8c2a1d434e47` | `yama/cohort_models/CAMS/CAMSTraining_Final_Metadata.csv` |
| `cohort_models/ExternalValidation/EEG_viewing_codes/sample_EEG_10min.mat` | 18488797 | `d16241812078` | pending upload |
| `cohort_models/ExternalValidation/KIMCHI_FINAL_METADATA.xlsx` | 17164 | `fc9a5518c2e5` | pending upload |
| `cohort_models/GCS/Cohort/BWH_GCS_download_AWS.txt` | 2501932 | `38e120c4bade` | pending upload |
| `cohort_models/GCS/Cohort/GCS_BWH_HarvardEEG_metadata.csv` | 15565873 | `cdfbd95b42da` | `yama/NESI/NESI-Medication-Analysis/medication-exposure/data/GCS_i0001a_HarvardEEG_metadata.csv` |
| `cohort_models/GCS/Cohort/GCS_MGH_HarvardEEG_metadata.csv` | 11330787 | `365b7124c739` | `yama/NESI/NESI-Medication-Analysis/medication-exposure/data/GCS_i0001b_HarvardEEG_metadata.csv` |
| `cohort_models/GCS/Cohort/MGH_GCS_download_AWS.txt` | 2763520 | `ba8586b65820` | pending upload |
| `cohort_models/GCS/Results5Fld/ReSNetBiLSTM_WithShuffle_5fldResults_3classGCS.pkl` | 21282823 | `766a876f2ced` | pending upload |
| `cohort_models/GCS/Results5Fld/ResNetBiLSTM_NoShuffle_5fldResults_3classGCS.pkl` | 21282823 | `4d8744076be9` | pending upload |
| `cohort_models/GCS/Results5Fld/ResNetGAP_5fldResults_3classGCS.pkl` | 21105915 | `4fa8beb399e8` | pending upload |
| `cohort_models/GCS/model/Training/GCSTraining_Final_Metadata.csv` | 4762367 | `ea3ff22fdbae` | pending upload |
| `cohort_models/ICANS/Results5Fld/RESNET_5foldNOSHUFFLE_Results.pkl` | 20542 | `b3219bdd5758` | pending upload |
| `cohort_models/ICANS/Results5Fld/RESNET_5foldOnlyGap_Results.pkl` | 20542 | `d341b5ba8180` | pending upload |
| `cohort_models/ICANS/Results5Fld/RESNET_5foldTIMESHUFFLE_Results.pkl` | 20542 | `001e7bd64422` | pending upload |
| `cohort_models/ICANS/SLEEPHeadbasedSelectEEGSegments/Spreadsheets/DailyICANSData.xlsx` | 1572867 | `0b6d2fa3c7c1` | pending upload |
| `cohort_models/ICANS/SLEEPHeadbasedSelectEEGSegments/Spreadsheets/ICANSFiles.xlsx` | 43536 | `d34ac4013e3f` | pending upload |
| `cohort_models/ICANS/model/Training/ICANSTraining_Final_Metadata.csv` | 79726 | `97159d374dee` | `yama/cohort_models/ICANS/ICANSTraining_Final_Metadata.csv` |
| `cohort_models/RASS/Cohort/RASSEEG_download_AWS.txt` | 4433538 | `efea3d03993f` | pending upload |
| `cohort_models/RASS/Cohort/RASS_BWH_HarvardEEG_metadata.csv` | 12180199 | `cf996f0844f9` | `yama/NESI/NESI-Medication-Analysis/medication-exposure/data/RASS_i0001a_HarvardEEG_metadata.csv` |
| `cohort_models/RASS/Cohort/RASS_MGH_HarvardEEG_metadata.csv` | 12229370 | `9692b3ee3d79` | `yama/NESI/NESI-Medication-Analysis/medication-exposure/data/RASS_i0001b_HarvardEEG_metadata.csv` |
| `cohort_models/RASS/Results5Fld/ReSNetBiLSTM_WithShuffle_5fldResults_6classRASS.pkl` | 20947336 | `25090cea7cd0` | pending upload |
| `cohort_models/RASS/Results5Fld/ResNetBiLSTM_NoShuffle_5fldResults_6classRASS.pkl` | 20947336 | `0493b06d856c` | pending upload |
| `cohort_models/RASS/Results5Fld/ResNetGAP_5fldResults_6classRASS.pkl` | 20947336 | `e92f5bc15f08` | pending upload |
| `cohort_models/RASS/model/Training/RASSTraining_Final_Metadata.csv` | 4758585 | `43897cfa63d8` | `yama/cohort_models/RASS/RASSTraining_Final_Metadata.csv` |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/CAMS_pacmap_data.pkl` | 4337 | `bf80f3cce806` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/GCS_pacmap_data.pkl` | 183730 | `c2da11a8a7c6` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/ICANS_pacmap_data.pkl` | 6401 | `9466920f8fb9` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/RASS_pacmap_data.pkl` | 158994 | `7f9dc1e75b65` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/X_tst_data_global_CAMS_median.pkl` | 71674 | `51b08be21213` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/X_tst_data_global_GCS_median.pkl` | 3121202 | `c45f07870e69` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/X_tst_data_global_ICANS_median.pkl` | 106762 | `6d6fe2451da6` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/X_tst_data_global_RASS_median.pkl` | 2700690 | `278a8ab5dacb` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/Y_tst_NESI_CAMS.pkl` | 4335 | `c4e103464aee` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/Y_tst_NESI_GCS.pkl` | 183728 | `aaba261c2794` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/Y_tst_NESI_ICANS.pkl` | 6399 | `755a6462dc80` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/Y_tst_NESI_RASS.pkl` | 158992 | `c0890556fc84` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/Yraw_tst_global_CAMS.pkl` | 4335 | `bd224908ea09` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/Yraw_tst_global_GCS.pkl` | 183728 | `c0d261dc5cde` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/Yraw_tst_global_ICANS.pkl` | 6399 | `521f6edb2c74` | pending upload |
| `figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/Yraw_tst_global_RASS.pkl` | 158992 | `79bb4e705ec4` | pending upload |
| `figures/supplementary_score_prediction/PacMAP_4OrdinalModels/CAMS_pacmap_data.pkl` | 4361 | `9810c1f42e6b` | pending upload |
| `figures/supplementary_score_prediction/PacMAP_4OrdinalModels/GCS_pacmap_data.pkl` | 183754 | `2714c18ae7e5` | pending upload |
| `figures/supplementary_score_prediction/PacMAP_4OrdinalModels/ICANS_pacmap_data.pkl` | 6425 | `ec4bad79db2a` | pending upload |
| `figures/supplementary_score_prediction/PacMAP_4OrdinalModels/RASS_pacmap_data.pkl` | 159018 | `420ba57eadbf` | pending upload |
| `figures/supplementary_score_prediction/PacMAP_4OrdinalModels/Yraw_tst_global_CAMS.pkl` | 4359 | `51e8541fb6ac` | pending upload |
| `figures/supplementary_score_prediction/PacMAP_4OrdinalModels/Yraw_tst_global_GCS.pkl` | 183752 | `d6d1ff1d2a2b` | pending upload |
| `figures/supplementary_score_prediction/PacMAP_4OrdinalModels/Yraw_tst_global_ICANS.pkl` | 6423 | `fb5aae2479fd` | pending upload |
| `figures/supplementary_score_prediction/PacMAP_4OrdinalModels/Yraw_tst_global_RASS.pkl` | 159016 | `4807a3f2b6e3` | pending upload |
| `mortality_analysis/Cohort/YAMA_FINAL_DEATH_RANDOM_SINGLE_SESSIONS_COHORT.csv` | 4279827 | `a80e0d07c984` | pending upload |
