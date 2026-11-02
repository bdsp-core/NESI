#!/usr/bin/env python
# coding: utf-8

# # **Libraries**

# In[10]:


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

plt.rcParams.update({
    'font.size': 9,
    'font.weight': 'bold',
    'font.family': 'serif'
})

# ---------------- Machine Learning (sklearn) ----------------
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score, average_precision_score
)
# ---------------- Deep Learning (PyTorch) ----------------
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torchsummary import summary


# ---------------- File Handling ----------------
import h5py
import hdf5storage

# ---------------- Utilities ----------------
from tqdm import tqdm
import joblib
import shap

# ---------------- Custom Function ----------------
def load_pickle(filepath):
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    filepath : str
        Path to the .pkl file.

    Returns
    -------
    obj : any
        The Python object stored in the pickle file.
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj

def extract_pid(filename):
    part = filename.split('_')[0]          
    pid_full = part.split('-')[1]          
    pid = pid_full[5:]                     
    return pid


# # **Load Original NESI result based Metadata**

# In[11]:


import pandas as pd
from itertools import combinations

current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")

NESI_result_path = NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "Final_NESIscores_full_dataset_result.csv"
df_NESI_metadata = pd.read_csv(NESI_result_path)

df = df_NESI_metadata.copy()
df["BDSPPatientID"] = df["BDSPPatientID"].astype("string").str.strip()
df["Split"] = df["Split"].astype("string").str.strip().str.capitalize()
df["Dataset"] = df["Dataset"].astype("string").str.strip().str.upper()
df = df.dropna(subset=["BDSPPatientID", "Split", "Dataset"])
df = df[df["BDSPPatientID"].ne("")]

splits = ["Train", "Val", "Test"]

for dataset in ["RASS", "GCS", "CAMS", "ICANS"]:
    d = df[df["Dataset"] == dataset]
    print(f"\n========== {dataset} DATASET ==========")
    print(f"Total EEG recordings: {len(d)}")
    print(f"Total unique patients: {d['BDSPPatientID'].nunique()}")

    split_ids = {}
    for s in splits:
        ds = d[d["Split"] == s]
        split_ids[s] = set(ds["BDSPPatientID"])
        print(f"{s}: {len(ds)} EEG recordings | {len(split_ids[s])} unique patients")

    print("\nPatient overlap:")
    for s1, s2 in combinations(splits, 2):
        print(f"{s1} vs {s2}: {len(split_ids[s1] & split_ids[s2])} overlapping patients")

    all_three = set.intersection(*split_ids.values())
    overlapping = set().union(*(split_ids[a] & split_ids[b] for a, b in combinations(splits, 2)))

    print(f"Train vs Val vs Test: {len(all_three)} overlapping patients")
    print(f"Total unique patients in multiple splits: {len(overlapping)}")


# # **Feaure Engineering**

# In[12]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def morgoth_output_file_location(data_group):
    base_paths = {
        "RASS": "/home/ayush/Desktop/MGB_RASS_EEGs/RASS_ALL_EEG_10min/MORGOTH_embeddings",
        "GCS": "/home/ayush/Desktop/MGB_GCS_EEGs/MORGOTH_GCS_Embeddings",
        "CAMS": "/home/ayush/Desktop/CAM-S_dataset/MORGOTH_Embeddings",
        "ICANS": "/home/ayush/Desktop/ICANS_dataset/Morgoth_embeddings_for_model"
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

        paths = morgoth_output_file_location(dataset)

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
    Y_raw_transformed = data_frame['TransformedScore'].to_numpy()

    dataset_names = data_frame["Dataset"].to_numpy()
    file_names = np.array(file_names)

    print("Grouped Label shape:", Y_grouped.shape)
    print("Raw Label shape:", Y_raw.shape)
    print("Raw Label transformed shape:", Y_raw_transformed.shape)
    print("Feature matrix shape:", X.shape)
    
    return X, Y_grouped, Y_raw, Y_raw_transformed, dataset_names, file_names


# # **NESI Neural Networks Backbone**

# In[13]:


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


# In[14]:


import torch
import torch.nn as nn
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


# In[16]:


import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings(model_trained, dataloader):
    model_trained.eval()
    all_embeddings = []
    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)
            emb = model_trained(x)
            all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)

def get_scores(model, embeddings, device="cuda", batch_size=256):
    model.eval()
    model.to(device)
    if not torch.is_tensor(embeddings):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    else:
        embeddings = embeddings.detach().float()

    loader = DataLoader(embeddings, batch_size=batch_size, shuffle=False)
    all_scores = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            scores = model(x).squeeze(-1)
            all_scores.append(scores.cpu())
    return torch.cat(all_scores).numpy()


# # **Get the NESI for all 24 Model**

# In[15]:


X, Y_grouped, Y_raw, Y_raw_transformed, dataset_names, file_names = morgoth_10minfea_matrix(df_NESI_metadata)


# In[17]:


X_data_trch = torch.tensor(X, dtype=torch.float32)
X_ds_all = TensorDataset(X_data_trch)
X_dl_all = DataLoader(X_ds_all, batch_size=128, shuffle=False)
assert len(X) == len(df_NESI_metadata), "Mismatch between X and df_NESI_metadata!"


# In[18]:


for i in range(2, 25):
    print(f"\n========== PROCESSING MODEL M{i} ==========")

    TripletModel_trained = MORGOTH_ResNet1D_onlyGAP(num_features=17)
    TripletModel_trained.load_state_dict(
        torch.load(
            NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" / "ModelCheckpoints" / "TripletModels" / f"Triplet_M{i}_BestModel.pth",
            map_location=device,
            weights_only=True
        )
    )
    TripletModel_trained = TripletModel_trained.to(device)
    TripletModel_trained.eval()

    checkpoint = torch.load(
        f"/home/ayush/Desktop/NESI_review_material/24ModelBaseline/ModelCheckpoints/NESIScoreModels/NESIScore_M{i}.pth",
        map_location=device,
        weights_only=True
    )

    NESIScoreModel_trained = EEGScoringModel()
    NESIScoreModel_trained.load_state_dict(checkpoint["model_state_dict"])
    NESIScoreModel_trained = NESIScoreModel_trained.to(device)
    NESIScoreModel_trained.eval()

    nesi_embeddings = get_embeddings(TripletModel_trained, X_dl_all)
    print("NESI embeddings shape:", nesi_embeddings.shape)

    nesi_score = get_scores(
        NESIScoreModel_trained,
        nesi_embeddings,
        device=device
    )
    print("NESI score shape:", nesi_score.shape)

    assert len(nesi_score) == len(df_NESI_metadata), f"Score length mismatch for M{i}"

    df_NESI_metadata[f"NESIScore_FromModel{i}"] = nesi_score

    print(f"Added column: NESIScore_FromModel{i}")

    del TripletModel_trained, NESIScoreModel_trained, nesi_embeddings, nesi_score
    if device == "cuda":
        torch.cuda.empty_cache()

print("\nAll models M2–M24 processed successfully.")
print(f"Final dataframe shape: {df_NESI_metadata.shape}")


# In[21]:


df_NESI_metadata.to_csv(NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" / "Full_Result_NESI_for_24_Models.csv",
                       index=False)

df_NESI_metadata_test = df_NESI_metadata[df_NESI_metadata['Split']=='Test']
df_NESI_metadata.to_csv(NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" / "Test_Result_NESI_for_24_Models.csv",
                       index=False)