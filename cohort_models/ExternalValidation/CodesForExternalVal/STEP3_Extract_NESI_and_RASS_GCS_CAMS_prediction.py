#!/usr/bin/env python
# coding: utf-8

# In[62]:


import os
import re
import mne
import sys
import subprocess
mne.set_log_level(verbose='WARNING')

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
from scipy.signal import resample
from scipy.io import loadmat, savemat
import shutil

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*"
)
import logging
logging.disable(logging.CRITICAL)

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["MNE_LOG_LEVEL"] = "ERROR"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import contextlib
@contextlib.contextmanager
def suppress_all_output():
    devnull = os.open(os.devnull, os.O_WRONLY)

    old_stdout = os.dup(1)
    old_stderr = os.dup(2)

    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)

################### HELPER FUNCTIONS ###########################


# ---------------------- MORGOTH-based EEG Feature Extrcation --------------------
def morgoth_10minfea_matrix_stat_for(
    Morgoth_activation_root_path, filename):

    BASE = Morgoth_activation_root_path 

    slowing_folder_loc = BASE / "SLOWING"
    focgen_folder_loc  = BASE / "FOCGEN"
    iiic_folder_loc    = BASE / "IIIC"
    nm_folder_loc      = BASE / "NM"
    bs_folder_loc      = BASE / "BS"
    sleep_folder_loc   = BASE / "SLEEP"
    
        
    file_name_csv = Path(filename).with_suffix(".csv").name

    slowing_fea_sub = pd.read_csv(os.path.join(slowing_folder_loc, file_name_csv))
    slowing_fea_sub = slowing_fea_sub.drop(columns='pred_class')
    slowing_fea_sub_val = slowing_fea_sub.values

    focgen_fea_sub = pd.read_csv(os.path.join(focgen_folder_loc, file_name_csv))
    focgen_fea_sub = focgen_fea_sub.drop(columns='pred_class')
    focgen_fea_sub_val = focgen_fea_sub.values

    iiic_fea_sub = pd.read_csv(os.path.join(iiic_folder_loc, file_name_csv))
    iiic_fea_sub = iiic_fea_sub.drop(columns='pred_class')
    iiic_fea_sub_val = iiic_fea_sub.values


    bs_fea_sub = pd.read_csv(os.path.join(bs_folder_loc, file_name_csv))
    bs_fea_sub_val = bs_fea_sub.values

    nm_fea_sub = pd.read_csv(os.path.join(nm_folder_loc, file_name_csv))
    nm_fea_sub_val = nm_fea_sub.values

    sleep_fea_sub = pd.read_csv(os.path.join(sleep_folder_loc, file_name_csv))
    sleep_fea_sub = sleep_fea_sub.drop(columns='pred_class')
    sleep_fea_sub_val = sleep_fea_sub.values
    
    sub_morgoth_fea = np.concatenate([sleep_fea_sub_val, nm_fea_sub_val,
                                      bs_fea_sub_val, focgen_fea_sub_val,
                                      slowing_fea_sub_val, iiic_fea_sub_val], axis=1)
    
    return sub_morgoth_fea


############################################
#        RASS-GCS-CAMS Cohort MODEL        #
############################################
#-------------------------- ResNet-GAP only model -------------------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # If channels differ → use 1x1 conv for skip
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        return F.relu(out)

class MORGOTH_ResNet1D_onlyGAP_CORAL(nn.Module):
    def __init__(self, num_features, num_classes, filters=None, use_logit=True):
        super().__init__()
        self.use_logit = use_logit
        self.num_classes = num_classes

        if filters is None:
            filters = [64, 128, 128, 256, 256]

        # Initial conv
        self.conv0 = nn.Conv1d(num_features, filters[0], kernel_size=7, padding=3)
        self.bn0 = nn.BatchNorm1d(filters[0])
        self.pool0 = nn.MaxPool1d(kernel_size=2)

        # ResNet blocks
        blocks = []
        in_ch = filters[0]
        for out_ch in filters:
            blocks.append(ResidualBlock1D(in_ch, out_ch))
            blocks.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch
        self.resnet_layers = nn.Sequential(*blocks)

        # GAP but DON'T squeeze yet
        self.gap = nn.AdaptiveAvgPool1d(1)

        
        # Dense before CORAL
        self.dropout = nn.Dropout(0.5)

        # CORAL layer → output K−1
        self.fc2 = nn.Linear(256, num_classes - 1)

    def forward(self, x):
        if self.use_logit:
            eps = 1e-6
            x = torch.log((x + eps) / (1 - x + eps))

        x = x.permute(0, 2, 1)

        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.resnet_layers(x)

        # GAP: (B,C,1)
        x = self.gap(x).squeeze(-1)

        x = self.dropout(x)
        x = self.fc2(x)          # CORAL output
        return x


######################
#      NESI MODEL    #
######################
import torch.nn.functional as F
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # If channels differ → use 1x1 conv for skip
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        return F.relu(out)

class MORGOTH_ResNet1D_onlyGAP(nn.Module):
    def __init__(self, num_features, filters=None, use_logit=True):
        super().__init__()
        self.use_logit = use_logit
        

        if filters is None:
            filters = [64, 128, 128, 256, 256]

        # Initial conv
        self.conv0 = nn.Conv1d(num_features, filters[0], kernel_size=7, padding=3)
        self.bn0 = nn.BatchNorm1d(filters[0])
        self.pool0 = nn.MaxPool1d(kernel_size=2)

        # ResNet blocks
        blocks = []
        in_ch = filters[0]
        for out_ch in filters:
            blocks.append(ResidualBlock1D(in_ch, out_ch))
            blocks.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch
        self.resnet_layers = nn.Sequential(*blocks)

        # GAP but DON'T squeeze yet
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 40)
    def forward(self, x):
        if self.use_logit:
            eps = 1e-6
            x = torch.log((x + eps) / (1 - x + eps))

        x = x.permute(0, 2, 1)

        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.resnet_layers(x)

        # GAP: (B,C,1)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# ------------ NESI score model ----------------
import torch.nn.functional as F
class EEGScoringModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=30, dropout=0.3):
        super(EEGScoringModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)  # scalar badness score
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


########################
#       EXECUTE        #
########################
current = Path(__file__).absolute()
CONTPRED_ROOT = None

for parent in current.parents:
    if parent.name == "ExternalValidation":
        CONTPRED_ROOT = parent
        break

if CONTPRED_ROOT is None:
    raise RuntimeError("ExternalValidation folder not found.")



Morgoth_activation_root_path = CONTPRED_ROOT / "ProductsV2" / "MorgothAcivtaions_Segments"
bs_root = Morgoth_activation_root_path / "BS"

print(f"CONTPRED_ROOT: {CONTPRED_ROOT}")
print(f"BS folder path: {bs_root}")
print(f"BS folder exists: {bs_root.exists()}")
print(f"BS folder is directory: {bs_root.is_dir()}")

if bs_root.is_dir():
    print(f"Total files in BS folder: {len(list(bs_root.iterdir()))}")
    print(f"First 10 files: {[f.name for f in list(bs_root.iterdir())[:10]]}")

csv_files = sorted(bs_root.glob("*.csv"))
print(f"Total activation CSV files found: {len(csv_files)}")

for csv_file in tqdm(csv_files):
    filename = csv_file.name  # Only filename, e.g., <recording>.csv
    edf_filename = csv_file.stem

    try:
        # GET MORGOTH FEATURE EMBEDDING MATRIX --> Nx17 (N dentes totak 10 sec windows with 1 sec slide)
        MORGOTH_fea_mat = morgoth_10minfea_matrix_stat_for(Morgoth_activation_root_path, filename)
        print(f"[SUCCESS] {csv_file.name} | Shape: {MORGOTH_fea_mat.shape}")


        # CONTINIOUS RASS PREDICTION
        WINDOW_SIZE = 591   # 10 min - 10 sec + 1
        STRIDE = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --------------------- Load RASS Model -----------------------------------
        RASS_model_onlygap = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, num_classes=6)
        RASS_model_dir = CONTPRED_ROOT / "DeliriumScoreModel" / "RESNETGAP_Best_RASS.pth"
        RASS_model_onlygap.load_state_dict(torch.load(RASS_model_dir, map_location=device, weights_only=True))
        RASS_model_onlygap = RASS_model_onlygap.to(device)
        RASS_model_onlygap.eval()

        # --------------------- Load GCS Model -----------------------------------
        GCS_model_onlygap = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, num_classes=3)
        GCS_model_dir = CONTPRED_ROOT / "DeliriumScoreModel" / "RESNETGAP_Best_GCS.pth"
        GCS_model_onlygap.load_state_dict(torch.load(GCS_model_dir, map_location=device, weights_only=True))
        GCS_model_onlygap = GCS_model_onlygap.to(device)
        GCS_model_onlygap.eval()

        # --------------------- Load CAMS Model -----------------------------------
        CAMS_model_onlygap = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, num_classes=3)
        CAMS_model_dir = CONTPRED_ROOT / "DeliriumScoreModel" / "RESNETGAP_Best_CAMS.pth"
        CAMS_model_onlygap.load_state_dict(torch.load(CAMS_model_dir, map_location=device, weights_only=True))
        CAMS_model_onlygap = CAMS_model_onlygap.to(device)
        CAMS_model_onlygap.eval()

        # --------------------- Load NESI Model -----------------------------------
        Triplet_model_path = CONTPRED_ROOT / "NESImodel" / "ResNetGAP_BestModel.pth" 
        Tripletmodel_trained = MORGOTH_ResNet1D_onlyGAP(num_features=17)
        Tripletmodel_trained.load_state_dict(
            torch.load(Triplet_model_path,
                       map_location=device,
                       weights_only=True)
        )
        Tripletmodel_trained = Tripletmodel_trained.to(device)
        Tripletmodel_trained.eval()

        NESIScore_model_path = CONTPRED_ROOT / "NESImodel" / "NESI_best_model.pth"
        checkpoint = torch.load(
            NESIScore_model_path,
            map_location=device,
            weights_only=True
        )

        NESIScore_model = EEGScoringModel()
        NESIScore_model.load_state_dict(checkpoint["model_state_dict"])
        NESIScore_model.to(device)
        NESIScore_model.eval()


        Y_CORN_out_rass, Y_pred_rass  = [], []
        Y_CORN_out_gcs, Y_pred_gcs  = [], []
        Y_CORN_out_cams, Y_pred_cams  = [], []
        Y_NESI = []

        with torch.no_grad():
            for start in range(0, MORGOTH_fea_mat.shape[0] - WINDOW_SIZE + 1, STRIDE):

                #--------Get data----------------
                x = MORGOTH_fea_mat[start:start + WINDOW_SIZE]
                x = torch.tensor(x, dtype=torch.float32)
                x = x.unsqueeze(0).to(device)
                
                #--------evaluate using RASS model----------------
                out_rass = RASS_model_onlygap(x)
                preds_rass = corn_label_from_logits(out_rass).float()

                Y_pred_rass.extend(preds_rass.cpu().numpy())
                Y_CORN_out_rass.append(out_rass.cpu().numpy())

                #--------evaluate using GCS model----------------
                out_gcs = GCS_model_onlygap(x)
                preds_gcs = corn_label_from_logits(out_gcs).float()
                Y_pred_gcs.extend(preds_gcs.cpu().numpy())
                Y_CORN_out_gcs.append(out_gcs.cpu().numpy())

                #--------evaluate using CAMS model----------------
                out_cams = CAMS_model_onlygap(x)
                preds_cams = corn_label_from_logits(out_cams).float()
                Y_pred_cams.extend(preds_cams.cpu().numpy())
                Y_CORN_out_cams.append(out_cams.cpu().numpy())

                # -------------- NESI Extrcation --------------------
                latent_nesi = Tripletmodel_trained(x)
                nesi = NESIScore_model(latent_nesi)
                Y_NESI.append(nesi.cpu().numpy())

        predictions_rass = np.array(Y_pred_rass)
        CORN_logits_rass = np.array(Y_CORN_out_rass).squeeze(1)
        print("RASS Prediction shape:", predictions_rass.shape)
        print("RASS CORN last layer logits shape:", CORN_logits_rass.shape)

        predictions_gcs = np.array(Y_pred_gcs)
        CORN_logits_gcs = np.array(Y_CORN_out_gcs).squeeze(1)
        print("GCS Prediction shape:", predictions_gcs.shape)
        print("GCS CORN last layer logits shape:", CORN_logits_gcs.shape)

        predictions_cams = np.array(Y_pred_cams)
        CORN_logits_cams = np.array(Y_CORN_out_cams).squeeze(1)
        print("CAMS Prediction shape:", predictions_cams.shape)
        print("CAMS CORN last layer logits shape:", CORN_logits_cams.shape)

        predictions_NESI = np.array(Y_NESI)
        print("NESI Prediction shape:", predictions_NESI.shape)

        # ---------------- NESI Prediction Save  --------------------------
        df_pred_save_nesi = pd.DataFrame({
            "NESI": predictions_NESI.flatten(),
        })
        nesi_save_path = CONTPRED_ROOT / "ProductsV2" / "NESIPredictions" / f"{edf_filename}_NESI_predictions.csv"
        nesi_save_path.parent.mkdir(parents=True, exist_ok=True)
        df_pred_save_nesi.to_csv(nesi_save_path, index=False)

        print(f"Saved to: {nesi_save_path}")

        # ---------------- RASS predictions and CORN lOGITS Save ----------------
        rass_mapping = {0: -5, 1: -4, 2: -3, 3: -2, 4: -1, 5: 0}
        RASSMappingClass = np.array([rass_mapping[int(pred)] for pred in predictions_rass])

        df_pred_save_rass = pd.DataFrame({
            "modelRASSPredClass": predictions_rass.astype(int),
            "RASSMappingClass": RASSMappingClass.astype(int)
        })

        df_corn_rass = pd.DataFrame(CORN_logits_rass, columns=[f"logit_{i}" for i in range(CORN_logits_rass.shape[1])])
        df_pred_corn_save_rass = pd.concat([df_pred_save_rass, df_corn_rass], axis=1)

        corn_predictions_save_path_rass = CONTPRED_ROOT / "ProductsV2" / "RASSPredictions" / f"{edf_filename}_RASS_predictions.csv"
        corn_predictions_save_path_rass.parent.mkdir(parents=True, exist_ok=True)
        df_pred_corn_save_rass.to_csv(corn_predictions_save_path_rass, index=False)

        print(f"Saved to: {corn_predictions_save_path_rass}")


        # ---------------- GCS predictions and CORN lOGITS Save ----------------
        gcs_mapping = {0: 'Severe', 1: 'Moderate', 2: 'Mild'}
        GCSMappingClass = np.array([gcs_mapping[int(pred)] for pred in predictions_gcs])

        df_pred_save_gcs = pd.DataFrame({
            "modelGCSPredClass": predictions_gcs.astype(int),
            "GCSMappingClass": GCSMappingClass
        })

        df_corn_gcs = pd.DataFrame(CORN_logits_gcs, columns=[f"logit_{i}" for i in range(CORN_logits_gcs.shape[1])])
        df_pred_corn_save_gcs = pd.concat([df_pred_save_gcs, df_corn_gcs], axis=1)

        corn_predictions_save_path_gcs = CONTPRED_ROOT / "ProductsV2" / "GCSPredictions" / f"{edf_filename}_GCS_predictions.csv"
        corn_predictions_save_path_gcs.parent.mkdir(parents=True, exist_ok=True)
        df_pred_corn_save_gcs.to_csv(corn_predictions_save_path_gcs, index=False)

        print(f"Saved to: {corn_predictions_save_path_gcs}")

        # ---------------- CAMS predictions and CORN lOGITS Save ----------------
        cams_mapping = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}
        CAMSMappingClass = np.array([cams_mapping[int(pred)] for pred in predictions_cams])

        df_pred_save_cams = pd.DataFrame({
            "modelCAMSPredClass": predictions_cams.astype(int),
            "CAMSMappingClass": CAMSMappingClass
        })

        df_corn_cams = pd.DataFrame(CORN_logits_cams, columns=[f"logit_{i}" for i in range(CORN_logits_cams.shape[1])])
        df_pred_corn_save_cams = pd.concat([df_pred_save_cams, df_corn_cams], axis=1)

        corn_predictions_save_path_cams = CONTPRED_ROOT / "ProductsV2" / "CAMSPredictions" / f"{edf_filename}_CAMS_predictions.csv"
        corn_predictions_save_path_cams.parent.mkdir(parents=True, exist_ok=True)
        df_pred_corn_save_cams.to_csv(corn_predictions_save_path_cams, index=False)

        print(f"Saved to: {corn_predictions_save_path_cams}")

    except Exception as e:
        print(f"[SKIPPED] {edf_filename} | Reason: {e}")
        continue