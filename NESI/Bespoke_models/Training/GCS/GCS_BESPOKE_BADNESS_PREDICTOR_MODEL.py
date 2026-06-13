#!/usr/bin/env python
# coding: utf-8

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

# ---------------- CORAL (Ordinal Regression) ----------------
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

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

def group_gcs(y):
    if y in [3,4,5,6,7,8]:
        return 2 # Bad Neurological state
    elif y in [9,10,11,12]:
        return 1 # Moderate Neurological state
    elif y in [13,14,15]:
        return 0 # Good Neurological state
    else:
        return None 

def group_rass(y):
    if y in [-5, -4]:
        return 2 # Bad Neurological state
    elif y in [-3, -2]:
        return 1 # Moderate Neurological state
    elif y in [-1, 0]:
        return 0 # Good Neurological state
    else:
        return None 
        
def group_icans(y):
    if y in [0]:
        return 0 # Good Neurological state
    elif y in [1, 2]:
        return 1 # Moderate Neurological state
    elif y in [3, 4]:
        return 2 # Bad Neurological state
    else:
        return None 

def group_cams(x):
    if 0 <= x <= 1:
        return 0 # Good Neurological state
    elif 2 <= x <= 5:
        return 1 # Moderate Neurological state
    else:
        return 2 # Bad Neurological state


# # **Data load across all cohort: RASS, GCS, GCS, ICANS**
#  - **Good neurological state (0):** GCS(13-15)
#  - **Moderate state (1):** GCS (9-12)
#  - **Bad state (2):**  GCS(3-8)

# In[2]:

current = Path(__file__).resolve()
NESI_ROOT = None
for parent in current.parents:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI folder not found")

# ------------------- CAMS - Dataset --------------------------
metadata_path_CAMS = NESI_ROOT / "model" / "Training" / "CAMSTraining_Final_Metadata.csv"
df_CAMS_metadata = pd.read_csv(metadata_path_CAMS)
df_CAMS_metadata["GroupedScore"] = df_CAMS_metadata["CAMS_SF"].apply(group_cams)
df_CAMS_metadata = df_CAMS_metadata.rename(columns={'CAMS_SF': 'RawScore'})
df_CAMS_metadata['Dataset'] = 'CAMS'
df_CAMS_metadata['TransformedScore'] = df_CAMS_metadata['RawScore']
df_CAMS_metadata = df_CAMS_metadata[['BDSPPatientID', 'Dataset', 'MorgothOutputFilename', 'RawScore', 'GroupedScore', 'TransformedScore']]

# ------------------- ICANS - Dataset --------------------------
metadata_path_ICANS = NESI_ROOT / "model" / "Training" / "ICANSTraining_Final_Metadata.csv"
df_ICANS_metadata = pd.read_csv(metadata_path_ICANS)
df_ICANS_metadata["GroupedScore"] = df_ICANS_metadata["ICANS_raw"].apply(group_icans)
df_ICANS_metadata = df_ICANS_metadata.rename(columns={'StudyID':'BDSPPatientID',                                                      
                                                      'ICANS_raw': 'RawScore'})
df_ICANS_metadata['Dataset'] = 'ICANS'
df_ICANS_metadata['TransformedScore'] = df_ICANS_metadata['RawScore']
df_ICANS_metadata = df_ICANS_metadata[['BDSPPatientID', 'Dataset', 'MorgothOutputFilename', 'RawScore', 'GroupedScore', 'TransformedScore']]

# ------------------- GCS - Dataset --------------------------
metadata_path_GCS = NESI_ROOT / "model" / "Training" / "GCSTraining_Final_Metadata.csv"
df_GCS_metadata = pd.read_csv(metadata_path_GCS)
df_GCS_metadata['Filename'] = df_GCS_metadata['Filename'].astype(str) + '.csv'
df_GCS_metadata['GroupedScore'] = df_GCS_metadata['GCS_value'].apply(group_gcs)
df_GCS_metadata['BDSPPatientID'] = df_GCS_metadata['Filename'].apply(extract_pid)
df_GCS_metadata = df_GCS_metadata.rename(columns={'Filename': 'MorgothOutputFilename',
                                                  'GCS_value': 'RawScore'})
df_GCS_metadata['Dataset'] = 'GCS'
df_GCS_metadata['TransformedScore'] = 15-df_GCS_metadata['RawScore']
df_GCS_metadata = df_GCS_metadata[['BDSPPatientID', 'Dataset', 'MorgothOutputFilename', 'RawScore', 'GroupedScore', 'TransformedScore']]

# ------------------- RASS - Dataset --------------------------
metadata_path_RASS = NESI_ROOT / "model" / "Training" / "RASSTraining_Final_Metadata.csv"
df_RASS_metadata=pd.read_csv(metadata_path_RASS)
df_RASS_metadata['Filename'] = df_RASS_metadata['Filename'].str.rstrip("'") + '.csv'
df_RASS_metadata = df_RASS_metadata[
    ~df_RASS_metadata['RASS_value'].isin([1, 2, 3, 4])
]
df_RASS_metadata['GroupedScore'] = df_RASS_metadata['RASS_value'].apply(group_rass)
df_RASS_metadata['BDSPPatientID'] = df_RASS_metadata['Filename'].apply(extract_pid)
df_RASS_metadata['Dataset'] = 'RASS'
df_RASS_metadata = df_RASS_metadata.rename(columns={'Filename': 'MorgothOutputFilename',
                                                    'RASS_value': 'RawScore'})
df_RASS_metadata['TransformedScore'] = -df_RASS_metadata['RawScore']
df_RASS_metadata = df_RASS_metadata[['BDSPPatientID', 'Dataset', 'MorgothOutputFilename', 'RawScore', 'GroupedScore', 'TransformedScore']]

## ------------------ FULL DATASET (RASS- GCS - CAMS - ICANS) -------------------------
df_all_metadata = pd.concat(
    [df_RASS_metadata, df_GCS_metadata, df_CAMS_metadata, df_ICANS_metadata], # ],
    axis=0,
    ignore_index=True
)

df_all_metadata


# # **First create the entire dataset with all clinical scores and then split into subjet indeendent train-val-test set**
# **It is done so, to get the same train-val-test instances for the RASS cohort only that appears in the Universal NESI model**
# **i.e., for fai comparision of test perfromance of bespoke model vs NESI model on the RASS clinical score**

# In[3]:


df_subjects = df_all_metadata[['BDSPPatientID', 'Dataset']].drop_duplicates()

from sklearn.model_selection import train_test_split

train_val, test = train_test_split(
    df_subjects,
    test_size=0.2,
    stratify=df_subjects['Dataset'],
    random_state=40
)

train, val = train_test_split(
    train_val,
    test_size=0.125,
    stratify=train_val['Dataset'],
    random_state=40
)

train_ids = set(train['BDSPPatientID'])
val_ids = set(val['BDSPPatientID'])
test_ids = set(test['BDSPPatientID'])

def assign_split(pid):
    if pid in train_ids:
        return 'Train'
    elif pid in val_ids:
        return 'Val'
    elif pid in test_ids:
        return 'Test'
    else:
        return 'None'

df_all_metadata['Split'] = df_all_metadata['BDSPPatientID'].apply(assign_split)

df_train = df_all_metadata[(df_all_metadata['Split'] == 'Train') & (df_all_metadata['Dataset'] == 'GCS')].reset_index(drop=True)
df_val   = df_all_metadata[(df_all_metadata['Split'] == 'Val') & (df_all_metadata['Dataset'] == 'GCS')].reset_index(drop=True)
df_test  = df_all_metadata[(df_all_metadata['Split'] == 'Test') & (df_all_metadata['Dataset'] == 'GCS')].reset_index(drop=True)

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)
print("Test shape:", df_test.shape)


# # **Feature Engineering**

# In[4]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm

#------ RASS Root Path--------
if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

RASS_ROOT = None
for parent in current.parents:
    if (parent / "RASS").exists():
        RASS_ROOT = parent
        break

if RASS_ROOT is None:
    raise RuntimeError("RASS folder not found")

#------ GCS Root Path--------
if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

GCS_ROOT = None
for parent in current.parents:
    if (parent / "GCS").exists():
        GCS_ROOT = parent
        break

if GCS_ROOT is None:
    raise RuntimeError("GCS folder not found")


#------ CAMS Root Path--------
from pathlib import Path

if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

CAMS_ROOT = None
for parent in current.parents:
    if (parent / "CAMS").exists():
        CAMS_ROOT = parent
        break

if CAMS_ROOT is None:
    raise RuntimeError("CAMS folder not found")



#------ ICANS Root Path--------
if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

ICANS_ROOT = None
for parent in current.parents:
    if (parent / "ICANS").exists():
        ICANS_ROOT = parent
        break

if ICANS_ROOT is None:
    raise RuntimeError("ICANS folder not found")


def morgoth_output_file_location(data_group):
    base_paths = {
        "RASS": RASS_ROOT / "RASS" / "MorgothActivations",
        "GCS": GCS_ROOT / "GCS" / "MorgothActivations",
        "GCS": GCS_ROOT / "GCS" / "MorgothActivations",
        "ICANS": ICANS_ROOT / "ICANS" / "MorgothActivations"
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


# In[5]:


X_train_data, Ygrouped_train, Yraw_train, Yraw_transformed_train, train_dataset_names,trn_filenames = morgoth_10minfea_matrix(df_train)
X_val_data, Ygrouped_val, Yraw_val, Yraw_transformed_val, val_dataset_names, val_filenames = morgoth_10minfea_matrix(df_val)
X_tst_data, Ygrouped_tst, Yraw_tst, Yraw_transformed_tst, tst_dataset_names, tst_filenames = morgoth_10minfea_matrix(df_test)


# In[6]:


X_train_data_trch = torch.tensor(X_train_data, dtype=torch.float32)
X_val_data_trch   = torch.tensor(X_val_data, dtype=torch.float32)
X_tst_data_trch   = torch.tensor(X_tst_data, dtype=torch.float32)

Ygrouped_train_trch = torch.tensor(Ygrouped_train, dtype=torch.long)
Ygrouped_val_trch   = torch.tensor(Ygrouped_val, dtype=torch.long)
Ygrouped_tst_trch   = torch.tensor(Ygrouped_tst, dtype=torch.long)

Yraw_train_trch = torch.tensor(Yraw_train, dtype=torch.long)
Yraw_val_trch   = torch.tensor(Yraw_val, dtype=torch.long)
Yraw_tst_trch   = torch.tensor(Yraw_tst, dtype=torch.long)

train_dataset_names = np.array(train_dataset_names)
val_dataset_names   = np.array(val_dataset_names)
tst_dataset_names   = np.array(tst_dataset_names)


# # **Get the Embeddings from the Frozen Triplet Model**

# In[7]:


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


# In[8]:


device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- LOAD MODEL ----------
model_path_trained = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints_NEW" / "TripletCheckpoint" / "GCS" / "ResNetGAP_GCS_BestModel.pth"

model_trained = MORGOTH_ResNet1D_onlyGAP(num_features=17)
model_trained.load_state_dict(
    torch.load(model_path_trained,
               map_location=device,
               weights_only=True)
)
model_trained = model_trained.to(device)
model_trained.eval()

# --------- CREATE LOADERS ----------
train_ds_all = TensorDataset(X_train_data_trch)
val_ds_all   = TensorDataset(X_val_data_trch)
tst_ds_all   = TensorDataset(X_tst_data_trch)

train_dl_all = DataLoader(train_ds_all, batch_size=128, shuffle=False)
val_dl_all   = DataLoader(val_ds_all, batch_size=128, shuffle=False)
tst_dl_all   = DataLoader(tst_ds_all, batch_size=128, shuffle=False)

# --------- FUNCTION TO EXTRACT EMBEDDINGS ----------
def get_embeddings(model_trained, dataloader):
    all_embeddings = []

    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)
            emb = model_trained(x)
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)

# --------- GET EMBEDDINGS ----------
train_embeddings = get_embeddings(model_trained, train_dl_all)
val_embeddings   = get_embeddings(model_trained, val_dl_all)
tst_embeddings   = get_embeddings(model_trained, tst_dl_all)

print("Train embeddings shape:", train_embeddings.shape)
print("Val embeddings shape:", val_embeddings.shape)
print("Test embeddings shape:", tst_embeddings.shape)


# # **Pair-wise Global Score Preduction Model Training**

# ## **Pairwise data sampling**

# In[9]:


import numpy as np
import random
from tqdm import tqdm


def create_pairs(
    embeddings,
    y_raw_transformed,
    y_grouped,
    dataset_names,
    mode="within_plus_global",
    k_pairs_per_anchor=4,
    hard_ratio=0.45,
    medium_ratio=0.30,
    easy_ratio=0.25,
    dataset_weights=None,
):

    N = len(embeddings)

    embeddings = np.array(embeddings)
    y_raw_transformed = np.array(y_raw_transformed)
    y_grouped = np.array(y_grouped)
    dataset_names = np.array(dataset_names)

    if dataset_weights is None:
        dataset_weights = {
            "RASS": 0.3,
            "GCS": 0.3,
            "GCS": 0.2,
            "ICANS": 0.2,
        }

    def sample_idx(cands):
        return random.choice(cands) if len(cands) > 0 else None

    # group indices
    dataset_to_indices = {}
    for i in range(N):
        d = dataset_names[i]
        dataset_to_indices.setdefault(d, []).append(i)

    pairs = []

    total_anchors = N // 10

    dataset_anchor_counts = {
        d: int(total_anchors * dataset_weights.get(d, 0.25))
        for d in dataset_to_indices.keys()
    }

    # =========================
    # DATASET LOOP (ONLY BAR)
    # =========================
    for dataset, indices in tqdm(dataset_to_indices.items(), desc="Creating Pairs"):

        anchor_count = dataset_anchor_counts.get(dataset, len(indices))
        anchor_indices = random.sample(indices, min(anchor_count, len(indices)))

        # NO inner tqdm here → prevents spam
        for i in anchor_indices:

            emb_i = embeddings[i]

            k = k_pairs_per_anchor

            n_hard = max(1, int(k * hard_ratio))
            n_med = max(1, int(k * medium_ratio))
            n_easy = max(1, k - n_hard - n_med)

            same_dataset_idx = indices
            yi = y_raw_transformed[i]

            # ---------------- HARD ----------------
            for _ in range(n_hard):
                candidates = [
                    j for j in same_dataset_idx
                    if abs(y_raw_transformed[j] - yi) <= 1e-3
                ]

                j = sample_idx(candidates)
                if j is None or j == i:
                    continue

                y = 1 if y_raw_transformed[i] > y_raw_transformed[j] else -1

                pairs.append((emb_i, embeddings[j], y, {
                    "type": "hard_within",
                    "dataset": dataset
                }))

            # ---------------- MEDIUM ----------------
            for _ in range(n_med):
                candidates = [
                    j for j in same_dataset_idx
                    if 0.5 < abs(y_raw_transformed[j] - yi) <= 3
                ]

                j = sample_idx(candidates)
                if j is None or j == i:
                    continue

                y = 1 if y_raw_transformed[i] > y_raw_transformed[j] else -1

                pairs.append((emb_i, embeddings[j], y, {
                    "type": "medium_within",
                    "dataset": dataset
                }))

            # ---------------- EASY ----------------
            for _ in range(n_easy):
                candidates = [
                    j for j in same_dataset_idx
                    if abs(y_raw_transformed[j] - yi) > 3
                ]

                j = sample_idx(candidates)
                if j is None or j == i:
                    continue

                y = 1 if y_raw_transformed[i] > y_raw_transformed[j] else -1

                pairs.append((emb_i, embeddings[j], y, {
                    "type": "easy_within",
                    "dataset": dataset
                }))

            # ---------------- CROSS ----------------
            if mode == "within_plus_global":

                cross_candidates = [
                    j for j in range(N)
                    if dataset_names[j] != dataset
                    and y_grouped[j] != y_grouped[i]
                ]

                for _ in range(max(1, k // 2)):

                    j = sample_idx(cross_candidates)
                    if j is None:
                        continue

                    y = 1 if y_grouped[i] > y_grouped[j] else -1

                    pairs.append((emb_i, embeddings[j], y, {
                        "type": "cross_global",
                        "dataset_i": dataset,
                        "dataset_j": dataset_names[j]
                    }))

    return pairs


# In[10]:


import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        emb_a, emb_b, y, meta = self.pairs[idx]

        emb_a = torch.tensor(emb_a, dtype=torch.float32)
        emb_b = torch.tensor(emb_b, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return emb_a, emb_b, y


# ## **Model building**

# In[11]:


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


# # **Training code for the global score prediction model**

# In[12]:


def pairwise_loss(score_a, score_b, y):
    """
    y = +1 if A worse than B else -1
    """
    return -torch.log(torch.sigmoid(y * (score_a - score_b)) + 1e-8).mean()


# In[13]:


import torch


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=100,
    save_path="./best_model.pth",
    patience=20
):

    best_val_loss = float("inf")
    patience_counter = 0

    model.to(device)

    for epoch in range(epochs):

        # =========================
        # TRAIN
        # =========================
        model.train()
        train_loss_total = 0.0

        for emb_a, emb_b, y in train_loader:

            emb_a = emb_a.to(device)
            emb_b = emb_b.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            score_a = model(emb_a)
            score_b = model(emb_b)

            loss = pairwise_loss(score_a, score_b, y)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        train_loss = train_loss_total / len(train_loader)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for emb_a, emb_b, y in val_loader:

                emb_a = emb_a.to(device)
                emb_b = emb_b.to(device)
                y = y.to(device)

                score_a = model(emb_a)
                score_b = model(emb_b)

                loss = pairwise_loss(score_a, score_b, y)

                val_loss_total += loss.item()

        val_loss = val_loss_total / len(val_loader)

        # =========================
        # LOGGING
        # =========================
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # =========================
        # CHECKPOINT
        # =========================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, save_path)

            print("✔ Saved best model")

        else:
            patience_counter += 1

        # =========================
        # EARLY STOPPING
        # =========================
        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break


# # **Implement the Global score prediction BESPOKE model**

# ## **Dataset prearation for pairwise training and dataloader creation**

# In[14]:


train_pairs = create_pairs(
    embeddings=train_embeddings,
    y_raw_transformed=Yraw_transformed_train,
    y_grouped=Ygrouped_train,
    dataset_names=train_dataset_names,
    mode="within"
)

val_pairs = create_pairs(
    embeddings=val_embeddings,
    y_raw_transformed=Yraw_transformed_val,
    y_grouped=Ygrouped_val,
    dataset_names=val_dataset_names,
    mode="within"
)

train_dataset = PairDataset(train_pairs)
val_dataset = PairDataset(val_pairs)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)


# # **Train the model**

# In[16]:


device = "cuda" if torch.cuda.is_available() else "cpu"

model = EEGScoringModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model_bespoke_bdness_path = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints_NEW" / "BespokeBadnessCheckpoint" / "GCS" / "GCS_Bespoke_Badness_Bet_model.pth"

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=100,
    save_path=model_bespoke_bdness_path,
    patience=20
)


# # **Load the trained model and get the EEG badness score**

# In[17]:


def get_scores(model, embeddings, device="cuda", batch_size=256):
    model.eval()
    model.to(device)
    # ---------------- SAFE CONVERSION ----------------
    if not torch.is_tensor(embeddings):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    else:
        embeddings = embeddings.detach().float()

    loader = torch.utils.data.DataLoader(
        embeddings,
        batch_size=batch_size,
        shuffle=False
    )

    all_scores = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            scores = model(x).squeeze(-1)
            all_scores.append(scores.cpu())

    return torch.cat(all_scores).numpy()


# In[18]:


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    model_bespoke_bdness_path,
    map_location=device,
    weights_only=True
)

model_trained = EEGScoringModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

model.eval()

train_scores = get_scores(model, train_embeddings, device=device)
val_scores   = get_scores(model, val_embeddings, device=device)
test_scores  = get_scores(model, tst_embeddings, device=device)


# # **Plot the EEG badness score**

# In[19]:


import numpy as np
from scipy.stats import spearmanr


def compute_spearman_datasetwise(scores, y_transformed, dataset_names, split_name=""):

    scores = np.array(scores)
    y = np.array(y_transformed)
    datasets = np.array(dataset_names)

    unique_datasets = ["GCS"]

    print(f"\n===== {split_name.upper()} SET SPEARMAN =====")

    results = {}

    for d in unique_datasets:
        mask = datasets == d

        if np.sum(mask) < 2:
            print(f"{d}: Not enough samples")
            results[d] = np.nan
            continue

        rho, _ = spearmanr(scores[mask], y[mask])
        results[d] = rho

        print(f"{d}: Spearman ρ = {rho:.4f}")

    return results


train_spearman_global = compute_spearman_datasetwise(
    train_scores,
    Yraw_transformed_train,
    train_dataset_names,
    split_name="Train"
)

# val_spearman_global = compute_spearman_datasetwise(
#     val_scores_global,
#     Yraw_transformed_val_global,
#     val_dataset_names_global,
#     split_name="Validation"
# )

tst_spearman_global = compute_spearman_datasetwise(
    test_scores,
    Yraw_transformed_tst,
    tst_dataset_names,
    split_name="Test"
)


# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_score_distribution_by_raw(
    predicted_scores,
    y_raw,
    dataset_name="RASS",
    title="Model Score Distribution by Raw Labels"
):

    plt.rcParams['figure.dpi'] = 300
    sns.set_style("whitegrid")

    df = pd.DataFrame({
        "score": np.array(predicted_scores),
        "y_raw": np.array(y_raw)
    })

    df = df.sort_values("y_raw")

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="y_raw",
        y="score",
        hue="y_raw",
        legend=False,
        palette="pastel",
        showfliers=True,
        boxprops={'edgecolor': 'black', 'linewidth': 1.5},
        whiskerprops={'color': 'black', 'linewidth': 1.5},
        capprops={'color': 'black', 'linewidth': 1.5},
        medianprops={'color': 'black', 'linewidth': 2},
        flierprops={
            'markerfacecolor': 'grey',
            'markeredgecolor': 'black',
            'markersize': 4
        }
    )

    ax.set_title(
        f"{dataset_name} Dataset",
        fontweight='bold',
        fontsize=15,
        pad=15
    )

    ax.set_xlabel(
        "Raw Score",
        fontweight='bold',
        fontsize=12
    )

    ax.set_ylabel(
        "Predicted Badness Score",
        fontweight='bold',
        fontsize=12
    )

    # Black border around subplot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_visible(True)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.suptitle(
        title,
        fontsize=18,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


# In[21]:


plot_score_distribution_by_raw(
    predicted_scores=train_scores,
    y_raw=Yraw_train,
    dataset_name="GCS",
    title="GCS Train Set Score Distribution"
)



# In[22]:


plot_score_distribution_by_raw(
    predicted_scores=test_scores,
    y_raw=Yraw_tst,
    dataset_name="GCS",
    title="GCS Train Set Score Distribution"
)


# In[ ]:




