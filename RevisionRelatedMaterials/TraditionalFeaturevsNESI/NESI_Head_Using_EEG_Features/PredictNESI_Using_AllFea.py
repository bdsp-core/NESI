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


# # **Metadata: NESI Cohort and Individual Cohort Loading**

# In[2]:
current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")

FeatureCompare_Root = NESI_ROOT / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI"
NESI_result_path = NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "Final_NESIscores_full_dataset_result.csv"

#------------------ NESI Metadata ---------------------
df_NESI_result = pd.read_csv(NESI_result_path)

df_RASS_from_NESI = df_NESI_result[(df_NESI_result['Dataset']=='RASS')].reset_index(drop=True)
df_GCS_from_NESI = df_NESI_result[(df_NESI_result['Dataset']=='GCS')].reset_index(drop=True)
df_CAMS_from_NESI = df_NESI_result[(df_NESI_result['Dataset']=='CAMS')].reset_index(drop=True)
df_ICANS_from_NESI = df_NESI_result[(df_NESI_result['Dataset']=='ICANS')].reset_index(drop=True)


# # **Load EEG Feature Data: RelativeDelta**

# In[3]:


Feature_dir = NESI_ROOT / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI" / "CohortFeatureData"

#----------------- RELATIVE DELTA -------------------
RASS_RelativeDelta_dict = mat73.loadmat(Feature_dir / "RASS_All_RelativeDelta.mat")
RASS_RelativeDelta = RASS_RelativeDelta_dict['RelativeDelta']

GCS_RelativeDelta_dict = mat73.loadmat(Feature_dir / "GCS_All_RelativeDelta.mat")
GCS_RelativeDelta = GCS_RelativeDelta_dict['RelativeDelta']

CAMS_RelativeDelta_dict = mat73.loadmat(Feature_dir / "CAMS_All_RelativeDelta.mat")
CAMS_RelativeDelta = CAMS_RelativeDelta_dict['RelativeDelta']

ICANS_RelativeDelta_dict = mat73.loadmat(Feature_dir / "ICANS_All_RelativeDelta.mat")
ICANS_RelativeDelta = ICANS_RelativeDelta_dict['RelativeDelta']

# ------------------ DELTA ALPHA ------------------------
RASS_DeltaAlpha_dict = mat73.loadmat(Feature_dir / "RASS_All_DeltaAlpha.mat")
RASS_DeltaAlpha = RASS_DeltaAlpha_dict['DeltaAlpha']

GCS_DeltaAlpha_dict = mat73.loadmat(Feature_dir / "GCS_All_DeltaAlpha.mat")
GCS_DeltaAlpha = GCS_DeltaAlpha_dict['DeltaAlpha']

CAMS_DeltaAlpha_dict = mat73.loadmat(Feature_dir / "CAMS_All_DeltaAlpha.mat")
CAMS_DeltaAlpha = CAMS_DeltaAlpha_dict['DeltaAlpha']

ICANS_DeltaAlpha_dict = mat73.loadmat(Feature_dir / "ICANS_All_DeltaAlpha.mat")
ICANS_DeltaAlpha = ICANS_DeltaAlpha_dict['DeltaAlpha']

# ----------------------- SEF95 --------------------------
RASS_SEF95_dict = mat73.loadmat(Feature_dir / "RASS_All_SEF95.mat")
RASS_SEF95 = RASS_SEF95_dict['SEF95']

GCS_SEF95_dict = mat73.loadmat(Feature_dir / "GCS_All_SEF95.mat")
GCS_SEF95 = GCS_SEF95_dict['SEF95']

CAMS_SEF95_dict = mat73.loadmat(Feature_dir / "CAMS_All_SEF95.mat")
CAMS_SEF95 = CAMS_SEF95_dict['SEF95']

ICANS_SEF95_dict = mat73.loadmat(Feature_dir / "ICANS_All_SEF95.mat")
ICANS_SEF95 = ICANS_SEF95_dict['SEF95']

print('*'*40)
print('SEF95')
for name in ['RASS', 'GCS', 'CAMS', 'ICANS']:
    print(f"{name} SEF95 shape: {globals()[f'{name}_SEF95'].shape}")

print('*'*40)
print('RelativeDelta')
for name in ['RASS', 'GCS', 'CAMS', 'ICANS']:
    print(f"{name} RelativeDelta shape: {globals()[f'{name}_RelativeDelta'].shape}")

print('*'*40)
print('DeltaAlpha')
for name in ['RASS', 'GCS', 'CAMS', 'ICANS']:
    print(f"{name} DeltaAlpha shape: {globals()[f'{name}_DeltaAlpha'].shape}")


# In[4]:


def add_realtivedelta_column(df, realtivedelta, sef95, deltaalpha):
    df = df.copy()
    assert len(df) == realtivedelta.shape[0] == sef95.shape[0] == deltaalpha.shape[0], "Row count mismatch"
    df['RelativeDelta'] = list(realtivedelta)
    df['SEF95'] = list(sef95)
    df['DeltaAlpha'] = list(deltaalpha)
    return df

df_RASS_from_NESI_RelativeDelta = add_realtivedelta_column(
    df_RASS_from_NESI, RASS_RelativeDelta, RASS_SEF95, RASS_DeltaAlpha)

df_GCS_from_NESI_RelativeDelta = add_realtivedelta_column(
    df_GCS_from_NESI, GCS_RelativeDelta, GCS_SEF95, GCS_DeltaAlpha)

df_CAMS_from_NESI_RelativeDelta = add_realtivedelta_column(
    df_CAMS_from_NESI, CAMS_RelativeDelta, CAMS_SEF95, CAMS_DeltaAlpha)

df_ICANS_from_NESI_RelativeDelta = add_realtivedelta_column(
    df_ICANS_from_NESI, ICANS_RelativeDelta, ICANS_SEF95, ICANS_DeltaAlpha)

df_all_metadata = pd.concat([
    df_RASS_from_NESI_RelativeDelta,
    df_GCS_from_NESI_RelativeDelta,
    df_CAMS_from_NESI_RelativeDelta,
    df_ICANS_from_NESI_RelativeDelta
], ignore_index=True)

print("Combined shape:", df_all_metadata.shape)
display(df_all_metadata.head())


# # **Train Test Validation Split Extract**

# In[5]:


df_train = df_all_metadata[df_all_metadata['Split'] == 'Train'].reset_index(drop=True)
df_val   = df_all_metadata[df_all_metadata['Split'] == 'Val'].reset_index(drop=True)
df_test  = df_all_metadata[df_all_metadata['Split'] == 'Test'].reset_index(drop=True)

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)
print("Test shape:", df_test.shape)


# In[6]:


def EEGFeature_10minfea_matrix(data_frame):
    X_relativedelta = np.stack(data_frame['RelativeDelta'].to_numpy())
    X_deltaAlpha = np.stack(data_frame['DeltaAlpha'].to_numpy())
    X_sef95 = np.stack(data_frame['SEF95'].to_numpy())

    X = np.concatenate([X_relativedelta, X_deltaAlpha, X_sef95], axis=1)

    Y_grouped = data_frame["GroupedScore"].to_numpy()
    Y_raw = data_frame["RawScore"].to_numpy()
    Y_raw_transformed = data_frame["TransformedScore"].to_numpy()
    dataset_names = data_frame["Dataset"].to_numpy()

    print("Grouped Label shape:", Y_grouped.shape)
    print("Raw Label shape:", Y_raw.shape)
    print("Raw Label transformed shape:", Y_raw_transformed.shape)
    print("Feature matrix shape:", X.shape)
    print('\n')

    return X, Y_grouped, Y_raw, Y_raw_transformed, dataset_names


# In[7]:


X_train_data, Ygrouped_train, Yraw_train, Yraw_transformed_train, train_dataset_names = EEGFeature_10minfea_matrix(df_train)
X_val_data, Ygrouped_val, Yraw_val, Yraw_transformed_val, val_dataset_names = EEGFeature_10minfea_matrix(df_val)
X_tst_data, Ygrouped_tst, Yraw_tst, Yraw_transformed_tst, tst_dataset_names = EEGFeature_10minfea_matrix(df_test)


# In[8]:


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
            "CAMS": 0.2,
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


# # **Model building**

# In[16]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGScoringModel(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=30, dropout=0.3):
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


# ## **Training code for the global score prediction model**

# In[17]:


def pairwise_loss(score_a, score_b, y):
    """
    y = +1 if A worse than B else -1
    """
    return -torch.log(torch.sigmoid(y * (score_a - score_b)) + 1e-8).mean()


# In[18]:


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


# # **Implement the Global score prediction model**

# ## **Dataset prearation for pairwise training and dataloader creation**

# In[14]:


train_pairs = create_pairs(
    embeddings=X_train_data,
    y_raw_transformed=Yraw_transformed_train,
    y_grouped=Ygrouped_train,
    dataset_names=train_dataset_names,
    mode="within_plus_global"
)

val_pairs = create_pairs(
    embeddings=X_val_data,
    y_raw_transformed=Yraw_transformed_val,
    y_grouped=Ygrouped_val,
    dataset_names=val_dataset_names,
    mode="within_plus_global"
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

# In[19]:


device = "cuda:1" if torch.cuda.is_available() else "cpu"

model = EEGScoringModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=100,
    save_path=NESI_ROOT / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI"/ "ModelCheckpoints" / "All_qEEG_Fea" / "NESIScore_All_qEEG_Fea.pth",
    patience=20
)


# # **Test Model**

# In[20]:


import torch
from pathlib import Path

X_data, Ygrouped, Yraw, Yraw_transformed, dataset_names = EEGFeature_10minfea_matrix(df_all_metadata)
X_data_trch = torch.tensor(X_data, dtype=torch.float32)

model = EEGScoringModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

checkpoint_path = NESI_ROOT / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI"/ "ModelCheckpoints" / "All_qEEG_Fea" / "NESIScore_All_qEEG_Fea.pth"

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Model weights loaded successfully!")
print("Saved epoch:", checkpoint['epoch'])
print("Validation loss:", checkpoint['val_loss'])

model.eval()
with torch.no_grad():
    Y_pred = model(X_data_trch)

print("Prediction shape:", Y_pred.shape)

df_all_metadata['NESI_from3qEEGFea'] = Y_pred.detach().cpu().numpy().reshape(-1)

print(df_all_metadata.shape)
display(df_all_metadata)


# In[21]:


from pathlib import Path

Result_path = NESI_ROOT / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI"/ "Results_NESI_from_qEEGFea"
Result_path.mkdir(parents=True, exist_ok=True)

csv_path = Result_path / "Final_NESI_AllqEEGFea.csv"
df_all_metadata.to_csv(csv_path, index=False)

print(f"Saved successfully: {csv_path}")


# In[22]:


df_all_metadata.columns


# In[ ]:




