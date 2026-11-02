#!/usr/bin/env python
# coding: utf-8

# # **Libraries**

# In[1]:


# ---------------- Standard Libraries ----------------
import os
import pickle
from pathlib import Path
from datetime import datetime
import time
# ---------------- Data Handling ----------------
import numpy as np
import pandas as pd

# ---------------- Visualization ----------------
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Machine Learning (sklearn) ----------------
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score, average_precision_score
)
from scipy.stats import spearmanr, wilcoxon
import scipy.io as sio
import mat73
# ---------------- Deep Learning (PyTorch) ----------------
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torchsummary import summary


# # **Metadata Loading**

# In[2]:
current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")

ICANS_Result_dir = NESI_ROOT / "RevisionRelatedMaterials" / "ICANS_Awake_Segment_Ablation"

df_icans_random_awake = pd.read_csv(ICANS_Result_dir/"RandomAwakeBasedAnalysis"/"RandomAwake_ICANS_Metadata.csv")



# # **Feature Engineering**

# In[3]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def morgoth_output_file_location_RandomAwake(data_group):
    base_paths = {
        "ICANS": ICANS_Result_dir/"RandomAwakeBasedAnalysis"/"MorgothActivations"
    }

    root = base_paths[data_group]

    return {
        "SLOWING": os.path.join(root, "SLOWING"),
        "FOCGEN": os.path.join(root, "FOCGEN"),
        "IIIC": os.path.join(root, "IIIC"),
        "NM": os.path.join(root, "NM"),
        "BS": os.path.join(root, "BS"),
        "SLEEP": os.path.join(root, "SLEEP")
    }


def load_feature(path, filename):
    df = pd.read_csv(os.path.join(path, filename))
    if "pred_class" in df.columns:
        df = df.drop(columns=["pred_class"])
    return df.values


def morgoth_10minfea_matrix(data_frame):
    features = []
    file_names = []

    for row in tqdm(data_frame.itertuples(index=False), total=len(data_frame)):
        fname = row.MorgothOutputFilename
        dataset = row.Dataset

        paths = morgoth_output_file_location_RandomAwake(dataset)

        sleep = load_feature(paths["SLEEP"], fname)
        nm = load_feature(paths["NM"], fname)
        bs = load_feature(paths["BS"], fname)

        focgen = load_feature(paths["FOCGEN"], fname)
        slowing = load_feature(paths["SLOWING"], fname)
        iiic = load_feature(paths["IIIC"], fname)

        subject_feature = np.concatenate(
            [sleep, nm, bs, focgen, slowing, iiic],
            axis=1
        )

        features.append(subject_feature)
        file_names.append(fname)

    X = np.stack(features, axis=0)

    Y_grouped = data_frame["GroupedScore"].to_numpy()
    Y_raw = data_frame["RawScore"].to_numpy()

    dataset_names = data_frame["Dataset"].to_numpy()
    file_names = np.array(file_names)

    print("Grouped Label shape:", Y_grouped.shape)
    print("Raw Label shape:", Y_raw.shape)
    print("Feature matrix shape:", X.shape)
    
    return X, Y_grouped, Y_raw, dataset_names, file_names


# # **NESI Extrcation Functions**

# In[4]:


#-------------------------- ResNet-GAP only model -------------------------------------
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


# ------------ NESI model ----------------
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


# # **Get embeddings from triplet model**

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- LOAD Triplet MODEL ----------
NESI_ROOT = Path('/home/ayush/Desktop/GitHub-YAMA/NESI')
Triplet_model_path = NESI_ROOT / "model" / "ModelCheckpoints" / "ResNetGAP_BestModel.pth" 
Triplet_model_trained = MORGOTH_ResNet1D_onlyGAP(num_features=17)
Triplet_model_trained.load_state_dict(
    torch.load(Triplet_model_path,
               map_location=device,
               weights_only=True)
)
Triplet_model_trained = Triplet_model_trained.to(device)
Triplet_model_trained.eval()

# --------- LOAD NESI MODEL ----------

device = "cuda" if torch.cuda.is_available() else "cpu"

NESI_model_path = NESI_ROOT / "model" / "ModelCheckpoints" / "NESI_best_model.pth"
checkpoint = torch.load(
    NESI_model_path,
    map_location=device,
    weights_only=True
)

NESI_model_score = EEGScoringModel()
NESI_model_score.load_state_dict(checkpoint["model_state_dict"])
NESI_model_score.to(device)

NESI_model_score.eval()


# # **Predict NESI**

# In[5]:


X_data, Ygrouped, Yraw, dataset_names,trn_filenames = morgoth_10minfea_matrix(df_icans_random_awake)

X_data_trch = torch.tensor(X_data, dtype=torch.float32)
Ygrouped_trch = torch.tensor(Ygrouped, dtype=torch.long)
Yraw_trch = torch.tensor(Yraw, dtype=torch.long)
dataset_names = np.array(dataset_names)


# In[6]:


import torch
import numpy as np

device = next(Triplet_model_trained.parameters()).device

Triplet_model_trained.eval()
NESI_model_score.eval()

X_data_trch = torch.as_tensor(X_data_trch, dtype=torch.float32, device=device)

with torch.no_grad():
    embeddings = Triplet_model_trained(X_data_trch)
    NESI_scores = NESI_model_score(embeddings)

embeddings_np = embeddings.cpu().numpy()
NESI_scores_np = NESI_scores.cpu().numpy()

print("Input shape:", tuple(X_data_trch.shape))
print("Embeddings shape:", embeddings_np.shape)
print("NESI scores shape:", NESI_scores_np.shape)


# In[7]:


df_icans_random_awake = df_icans_random_awake.copy()
df_icans_random_awake["NESI_RandomAwake"] = NESI_scores_np
print("DataFrame shape:", df_icans_random_awake.shape)


# In[8]:


df_icans_random_awake.to_csv(ICANS_Result_dir/"RandomAwakeBasedAnalysis"/"RandomAwake_Results.csv",
                            index=False)