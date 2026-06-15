#!/usr/bin/env python
# coding: utf-8

# # **Libraries**

# In[2]:


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


# # **Load the Full Dataset**

# In[4]:

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

print('All metadata loaded!!\n')


# ## **Subject Independent Splitting**

# In[5]:


df_subjects = df_all_metadata[['BDSPPatientID', 'Dataset']].drop_duplicates()

from sklearn.model_selection import train_test_split

train_val, test = train_test_split(
    df_subjects,
    test_size=0.2,
    stratify=df_subjects['Dataset'],
    random_state=42
)

train, val = train_test_split(
    train_val,
    test_size=0.125,
    stratify=train_val['Dataset'],
    random_state=42
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

df_train = df_all_metadata[df_all_metadata['Split'] == 'Train'].reset_index(drop=True)
df_val   = df_all_metadata[df_all_metadata['Split'] == 'Val'].reset_index(drop=True)
df_test  = df_all_metadata[df_all_metadata['Split'] == 'Test'].reset_index(drop=True)

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)
print("Test shape:", df_test.shape)


# # **Helper functions**

# ## **Feature engineering helper**

# In[6]:


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
    if (parent / "NESI").exists():
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
    if (parent / "NESI").exists():
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
    if (parent / "NESI").exists():
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
    if (parent / "NESI").exists():
        ICANS_ROOT = parent
        break

if ICANS_ROOT is None:
    raise RuntimeError("ICANS folder not found")


def morgoth_output_file_location(data_group):
    base_paths = {
        "RASS": RASS_ROOT / "cohort_models" / "RASS" / "MorgothActivations",
        "GCS": GCS_ROOT / "cohort_models" / "GCS" / "MorgothActivations",
        "CAMS": CAMS_ROOT / "cohort_models" / "CAMS" / "MorgothActivations",
        "ICANS": ICANS_ROOT / "cohort_models" / "ICANS" / "MorgothActivations"
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


def morgoth_10minfea_matrix(data_frame, statistic):
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
        
        # Compute the requested statistic
        if statistic == 'mean':
            stat_10min_fea = np.mean(subject_feature, axis=0)
        elif statistic == 'median':
            stat_10min_fea = np.median(subject_feature, axis=0)
        elif statistic == 'cov_upper':
            # Covariance across features (columns)
            cov_matrix = np.cov(subject_feature, rowvar=False)  # shape: (num_features, num_features)
            # Take upper triangle without diagonal
            triu_indices = np.triu_indices(cov_matrix.shape[0], k=1)
            stat_10min_fea = cov_matrix[triu_indices]           # flattened upper triangle
        else:
            raise ValueError("statistic must be 'mean', 'median', or 'cov_upper'")
        
        features.append(stat_10min_fea)
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


# ## **Anchor and Query pair creation helper**
# - **Pairs from same dataset and cross dataset groups (RASS, GCS, CAMS, ICANS)**

# In[7]:


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


# ## **NESI model building helper**

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGScoringModel(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=30, dropout=0.3):
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


# ## **Pairwise ranking loss helper**

# In[10]:


def pairwise_loss(score_a, score_b, y):
    """
    y = +1 if A worse than B else -1
    """
    return -torch.log(torch.sigmoid(y * (score_a - score_b)) + 1e-8).mean()


# ## **Model training loop helper**

# In[11]:


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


# # **Median feature based experiment**

# ## **Feature dataset with median statistic**

# In[12]:


fea_stat1 ='median'

# X_train_data, Ygrouped_train, Yraw_train, Yraw_transformed_train, train_dataset_names,trn_filenames = morgoth_10minfea_matrix(df_train, fea_stat1)
# X_val_data, Ygrouped_val, Yraw_val, Yraw_transformed_val, val_dataset_names, val_filenames = morgoth_10minfea_matrix(df_val, fea_stat1)
X_tst_data, Ygrouped_tst, Yraw_tst, Yraw_transformed_tst, tst_dataset_names, tst_filenames = morgoth_10minfea_matrix(df_test, fea_stat1)

# X_train_data_trch = torch.tensor(X_train_data, dtype=torch.float32)
# X_val_data_trch   = torch.tensor(X_val_data, dtype=torch.float32)
X_tst_data_trch   = torch.tensor(X_tst_data, dtype=torch.float32)

# Ygrouped_train_trch = torch.tensor(Ygrouped_train, dtype=torch.long)
# Ygrouped_val_trch   = torch.tensor(Ygrouped_val, dtype=torch.long)
Ygrouped_tst_trch   = torch.tensor(Ygrouped_tst, dtype=torch.long)

# Yraw_train_trch = torch.tensor(Yraw_train, dtype=torch.long)
# Yraw_val_trch   = torch.tensor(Yraw_val, dtype=torch.long)
Yraw_tst_trch   = torch.tensor(Yraw_tst, dtype=torch.long)

# train_dataset_names = np.array(train_dataset_names)
# val_dataset_names   = np.array(val_dataset_names)
tst_dataset_names   = np.array(tst_dataset_names)

# ## **Median model training**


# ## **Get the badnessscore**

# In[18]:


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


device = "cuda" if torch.cuda.is_available() else "cpu"
median_model_save_path = NESI_ROOT / "model" / "AblationStudy" / "ModelCheckpoints" / "Median_best_model.pth"

checkpoint = torch.load(
    median_model_save_path,
    map_location=device,
    weights_only=True
)

median_model_trained = EEGScoringModel()
median_model_trained.load_state_dict(checkpoint["model_state_dict"])
median_model_trained.to(device)

median_model_trained.eval()

# train_scores_median_model = get_scores(median_model_trained, X_train_data, device=device)
# val_scores_median_model   = get_scores(median_model_trained, X_val_data, device=device)
tst_scores_median_model  = get_scores(median_model_trained, X_tst_data, device=device)


# print('Training NESI shape: ', train_scores_median_model.shape)
# print('Validation NESI shape: ', val_scores_median_model.shape)
print('Testing NESI shape: ', tst_scores_median_model.shape)


# ## **Spearman correlation**

# ### **RASS dataset**

# In[26]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------- mask ----------------
mask_RASS = tst_dataset_names == 'RASS'
Yraw_tst_RASS = Yraw_tst[mask_RASS]
Yraw_transformed_tst_RASS = Yraw_transformed_tst[mask_RASS]
tst_scores_RASS = tst_scores_median_model[mask_RASS]

# ---------------- dataframe ----------------
df_rass = pd.DataFrame({
    "TrueRawRASS": Yraw_tst_RASS,
    "TrueTransformedRASS": Yraw_transformed_tst_RASS,
    "MEDIANModelScore": tst_scores_RASS,
})

# ---------------- extract ----------------
y_true = df_rass["TrueTransformedRASS"].values
y = df_rass["MEDIANModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y[idx])

# ---------------- stats ----------------
global_mean_RASS = np.mean(global_corrs)

global_ci_RASS = np.percentile(global_corrs, [2.5, 97.5])

# ---------------- output ----------------
print("Median MODEL")
print(f"Mean Spearman: {global_mean_RASS:.4f}")
print(f"95% CI: [{global_ci_RASS[0]:.4f}, {global_ci_RASS[1]:.4f}]")


# ### **GCS dataset**

# In[27]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------- mask ----------------
mask_GCS = tst_dataset_names == 'GCS'
Yraw_tst_GCS = Yraw_tst[mask_GCS]
Yraw_transformed_tst_GCS = Yraw_transformed_tst[mask_GCS]
tst_scores_GCS = tst_scores_median_model[mask_GCS]

# ---------------- dataframe ----------------
df_gcs = pd.DataFrame({
    "TrueRawGCS": Yraw_tst_GCS,
    "TrueTransformedGCS": Yraw_transformed_tst_GCS,
    "MEDIANModelScore": tst_scores_GCS,
})

# ---------------- extract ----------------
y_true = df_gcs["TrueTransformedGCS"].values
y = df_gcs["MEDIANModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y[idx])

# ---------------- stats ----------------
global_mean_GCS = np.mean(global_corrs)

global_ci_GCS = np.percentile(global_corrs, [2.5, 97.5])

# ---------------- output ----------------
print("Median MODEL")
print(f"Mean Spearman: {global_mean_GCS:.4f}")
print(f"95% CI: [{global_ci_GCS[0]:.4f}, {global_ci_GCS[1]:.4f}]")


# ### **CAMS dataset**

# In[28]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------- mask ----------------
mask_CAMS = tst_dataset_names == 'CAMS'
Yraw_tst_CAMS = Yraw_tst[mask_CAMS]
Yraw_transformed_tst_CAMS = Yraw_transformed_tst[mask_CAMS]
tst_scores_CAMS = tst_scores_median_model[mask_CAMS]

# ---------------- dataframe ----------------
df_cams = pd.DataFrame({
    "TrueRawCAMS": Yraw_tst_CAMS,
    "TrueTransformedCAMS": Yraw_transformed_tst_CAMS,
    "MEDIANModelScore": tst_scores_CAMS,
})

# ---------------- extract ----------------
y_true = df_cams["TrueTransformedCAMS"].values
y = df_cams["MEDIANModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y[idx])

# ---------------- stats ----------------
global_mean_CAMS = np.mean(global_corrs)

global_ci_CAMS = np.percentile(global_corrs, [2.5, 97.5])

# ---------------- output ----------------
print("Median MODEL")
print(f"Mean Spearman: {global_mean_CAMS:.4f}")
print(f"95% CI: [{global_ci_CAMS[0]:.4f}, {global_ci_CAMS[1]:.4f}]")


# ### **ICANS dataset**

# In[29]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------- mask ----------------
mask_ICANS = tst_dataset_names == 'ICANS'
Yraw_tst_ICANS = Yraw_tst[mask_ICANS]
Yraw_transformed_tst_ICANS = Yraw_transformed_tst[mask_ICANS]
tst_scores_ICANS = tst_scores_median_model[mask_ICANS]

# ---------------- dataframe ----------------
df_icans = pd.DataFrame({
    "TrueRawICANS": Yraw_tst_ICANS,
    "TrueTransformedICANS": Yraw_transformed_tst_ICANS,
    "MEDIANModelScore": tst_scores_ICANS,
})

# ---------------- extract ----------------
y_true = df_icans["TrueTransformedICANS"].values
y = df_icans["MEDIANModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y[idx])

# ---------------- stats ----------------
global_mean_ICANS = np.mean(global_corrs)

global_ci_ICANS = np.percentile(global_corrs, [2.5, 97.5])

# ---------------- output ----------------
print("Median MODEL")
print(f"Mean Spearman: {global_mean_ICANS:.4f}")
print(f"95% CI: [{global_ci_ICANS[0]:.4f}, {global_ci_ICANS[1]:.4f}]")


# ## **Plot the NESI distribution**

# In[38]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_score_distributions_by_raw(
    predicted_scores,
    y_raw,
    dataset_names,
    global_mean_RASS, global_ci_RASS,
    global_mean_GCS, global_ci_GCS,
    global_mean_CAMS, global_ci_CAMS,
    global_mean_ICANS, global_ci_ICANS,
    title=None
):
    plt.rcParams['figure.dpi'] = 150
    sns.set_style("whitegrid")
    
    df = pd.DataFrame({
        "score": np.array(predicted_scores),
        "y_raw": np.array(y_raw),
        "dataset": np.array(dataset_names)
    })

    datasets = ["RASS", "GCS", "CAMS", "ICANS"]

    corr_map = {
        "RASS": (global_mean_RASS, global_ci_RASS),
        "GCS": (global_mean_GCS, global_ci_GCS),
        "CAMS": (global_mean_CAMS, global_ci_CAMS),
        "ICANS": (global_mean_ICANS, global_ci_ICANS)
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes = axes.flatten()

    for idx, d in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["dataset"] == d].copy()

        if sub.empty:
            ax.set_title(f"{d} Dataset\n(No Data)", fontweight='bold')
            continue

        sub = sub.sort_values("y_raw")

        sns.boxplot(
            data=sub, 
            x="y_raw", 
            y="score", 
            hue="y_raw",
            legend=False,
            ax=ax,
            palette="pastel",
            showfliers=True,
            boxprops={'edgecolor': 'black', 'linewidth': 1.1},
            whiskerprops={'color': 'black', 'linewidth': 1.1},
            capprops={'color': 'black', 'linewidth': 1.1},
            medianprops={'color': 'black', 'linewidth': 1.2},
            flierprops={'markerfacecolor': 'grey', 'markeredgecolor': 'black', 'markersize': 4}
        )

        rho, ci = corr_map[d]
        ci_low, ci_high = ci

        ax.set_title(
            f"{d} Dataset\n"
            f"Spearman ρ = {rho:.3f} [{ci_low:.3f}, {ci_high:.3f}]",
            fontweight='bold',
            fontsize=8,
            pad=15
        )

        ax.set_xlabel("Raw Score", fontweight='bold', fontsize=10)
        
        # Only show ylabel for the first plot
        if idx == 0:
            ax.set_ylabel("Predicted NESI", fontweight='bold', fontsize=10)
        else:
            ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_visible(True)
            spine.set_linewidth(1.2)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight('bold')

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1)
    plt.tight_layout()
    plt.show()


# In[39]:


plot_score_distributions_by_raw(
    tst_scores_median_model ,
    Yraw_tst,
    tst_dataset_names,
    global_mean_RASS, global_ci_RASS,
    global_mean_GCS, global_ci_GCS,
    global_mean_CAMS, global_ci_CAMS,
    global_mean_ICANS, global_ci_ICANS,
    title='NESI Distribution using median-feature-based NESI head'
)

#########################################################################
# # **Mean feature-based exprimentation**
#########################################################################

# In[40]:


fea_stat2 ='mean'

# X_train_data, Ygrouped_train, Yraw_train, Yraw_transformed_train, train_dataset_names,trn_filenames = morgoth_10minfea_matrix(df_train, fea_stat2)
# X_val_data, Ygrouped_val, Yraw_val, Yraw_transformed_val, val_dataset_names, val_filenames = morgoth_10minfea_matrix(df_val, fea_stat2)
X_tst_data, Ygrouped_tst, Yraw_tst, Yraw_transformed_tst, tst_dataset_names, tst_filenames = morgoth_10minfea_matrix(df_test, fea_stat2)

# X_train_data_trch = torch.tensor(X_train_data, dtype=torch.float32)
# X_val_data_trch   = torch.tensor(X_val_data, dtype=torch.float32)
X_tst_data_trch   = torch.tensor(X_tst_data, dtype=torch.float32)

# Ygrouped_train_trch = torch.tensor(Ygrouped_train, dtype=torch.long)
# Ygrouped_val_trch   = torch.tensor(Ygrouped_val, dtype=torch.long)
Ygrouped_tst_trch   = torch.tensor(Ygrouped_tst, dtype=torch.long)

# Yraw_train_trch = torch.tensor(Yraw_train, dtype=torch.long)
# Yraw_val_trch   = torch.tensor(Yraw_val, dtype=torch.long)
Yraw_tst_trch   = torch.tensor(Yraw_tst, dtype=torch.long)

# train_dataset_names = np.array(train_dataset_names)
# val_dataset_names   = np.array(val_dataset_names)
tst_dataset_names   = np.array(tst_dataset_names)

# ## **Mean model training**

# ## **Get badness scores**


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


device = "cuda" if torch.cuda.is_available() else "cpu"
mean_model_save_path = NESI_ROOT / "model" / "AblationStudy" / "ModelCheckpoints" / "Mean_best_model.pth"

checkpoint = torch.load(
    mean_model_save_path,
    map_location=device,
    weights_only=True
)

mean_model_trained = EEGScoringModel()
mean_model_trained.load_state_dict(checkpoint["model_state_dict"])
mean_model_trained.to(device)

mean_model_trained.eval()

# train_scores_mean_model = get_scores(mean_model_trained, X_train_data, device=device)
# val_scores_mean_model   = get_scores(mean_model_trained, X_val_data, device=device)
tst_scores_mean_model  = get_scores(mean_model_trained, X_tst_data, device=device)


# print('Training NESI shape: ', train_scores_mean_model.shape)
# print('Validation NESI shape: ', val_scores_mean_model.shape)
print('Testing NESI shape: ', tst_scores_mean_model.shape)


# ## **Spearman correlation**

# In[44]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------- mask ----------------
mask_RASS = tst_dataset_names == 'RASS'
Yraw_tst_RASS = Yraw_tst[mask_RASS]
Yraw_transformed_tst_RASS = Yraw_transformed_tst[mask_RASS]
tst_scores_RASS = tst_scores_mean_model[mask_RASS]

# ---------------- dataframe ----------------
df_rass = pd.DataFrame({
    "TrueRawRASS": Yraw_tst_RASS,
    "TrueTransformedRASS": Yraw_transformed_tst_RASS,
    "MEDIANModelScore": tst_scores_RASS,
})

# ---------------- extract ----------------
y_true = df_rass["TrueTransformedRASS"].values
y = df_rass["MEDIANModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y[idx])

# ---------------- stats ----------------
global_mean_RASS = np.mean(global_corrs)

global_ci_RASS = np.percentile(global_corrs, [2.5, 97.5])

# ---------------- output ----------------
print("Median MODEL")
print(f"Mean Spearman: {global_mean_RASS:.4f}")
print(f"95% CI: [{global_ci_RASS[0]:.4f}, {global_ci_RASS[1]:.4f}]")


# In[45]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------- mask ----------------
mask_GCS = tst_dataset_names == 'GCS'
Yraw_tst_GCS = Yraw_tst[mask_GCS]
Yraw_transformed_tst_GCS = Yraw_transformed_tst[mask_GCS]
tst_scores_GCS = tst_scores_mean_model[mask_GCS]

# ---------------- dataframe ----------------
df_gcs = pd.DataFrame({
    "TrueRawGCS": Yraw_tst_GCS,
    "TrueTransformedGCS": Yraw_transformed_tst_GCS,
    "MEDIANModelScore": tst_scores_GCS,
})

# ---------------- extract ----------------
y_true = df_gcs["TrueTransformedGCS"].values
y = df_gcs["MEDIANModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y[idx])

# ---------------- stats ----------------
global_mean_GCS = np.mean(global_corrs)

global_ci_GCS = np.percentile(global_corrs, [2.5, 97.5])

# ---------------- output ----------------
print("Median MODEL")
print(f"Mean Spearman: {global_mean_GCS:.4f}")
print(f"95% CI: [{global_ci_GCS[0]:.4f}, {global_ci_GCS[1]:.4f}]")


# In[46]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------- mask ----------------
mask_CAMS = tst_dataset_names == 'CAMS'
Yraw_tst_CAMS = Yraw_tst[mask_CAMS]
Yraw_transformed_tst_CAMS = Yraw_transformed_tst[mask_CAMS]
tst_scores_CAMS = tst_scores_mean_model[mask_CAMS]

# ---------------- dataframe ----------------
df_cams = pd.DataFrame({
    "TrueRawCAMS": Yraw_tst_CAMS,
    "TrueTransformedCAMS": Yraw_transformed_tst_CAMS,
    "MEDIANModelScore": tst_scores_CAMS,
})

# ---------------- extract ----------------
y_true = df_cams["TrueTransformedCAMS"].values
y = df_cams["MEDIANModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y[idx])

# ---------------- stats ----------------
global_mean_CAMS = np.mean(global_corrs)

global_ci_CAMS = np.percentile(global_corrs, [2.5, 97.5])

# ---------------- output ----------------
print("Median MODEL")
print(f"Mean Spearman: {global_mean_CAMS:.4f}")
print(f"95% CI: [{global_ci_CAMS[0]:.4f}, {global_ci_CAMS[1]:.4f}]")


# In[47]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------- mask ----------------
mask_ICANS = tst_dataset_names == 'ICANS'
Yraw_tst_ICANS = Yraw_tst[mask_ICANS]
Yraw_transformed_tst_ICANS = Yraw_transformed_tst[mask_ICANS]
tst_scores_ICANS = tst_scores_mean_model[mask_ICANS]

# ---------------- dataframe ----------------
df_icans = pd.DataFrame({
    "TrueRawICANS": Yraw_tst_ICANS,
    "TrueTransformedICANS": Yraw_transformed_tst_ICANS,
    "MEDIANModelScore": tst_scores_ICANS,
})

# ---------------- extract ----------------
y_true = df_icans["TrueTransformedICANS"].values
y = df_icans["MEDIANModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y[idx])

# ---------------- stats ----------------
global_mean_ICANS = np.mean(global_corrs)

global_ci_ICANS = np.percentile(global_corrs, [2.5, 97.5])

# ---------------- output ----------------
print("Median MODEL")
print(f"Mean Spearman: {global_mean_ICANS:.4f}")
print(f"95% CI: [{global_ci_ICANS[0]:.4f}, {global_ci_ICANS[1]:.4f}]")


# ## **Plot the NESI distribution**

# In[48]:




plot_score_distributions_by_raw(
    tst_scores_mean_model ,
    Yraw_tst,
    tst_dataset_names,
    global_mean_RASS, global_ci_RASS,
    global_mean_GCS, global_ci_GCS,
    global_mean_CAMS, global_ci_CAMS,
    global_mean_ICANS, global_ci_ICANS,
    title='NESI Distribution using mean-feature-based NESI head'
)