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


# # **Data load across all cohort: RASS, GCS, CAMS, ICANS**
#  - **Good neurological state (0):** RASS(-1, 0)
#  - **Moderate state (1):** RASS (-3, -2)
#  - **Bad state (2):**  RASS(-5, -4)

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

df_train = df_all_metadata[(df_all_metadata['Split'] == 'Train') & (df_all_metadata['Dataset'] == 'RASS')].reset_index(drop=True)
df_val   = df_all_metadata[(df_all_metadata['Split'] == 'Val') & (df_all_metadata['Dataset'] == 'RASS')].reset_index(drop=True)
df_test  = df_all_metadata[(df_all_metadata['Split'] == 'Test') & (df_all_metadata['Dataset'] == 'RASS')].reset_index(drop=True)

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)
print("Test shape:", df_test.shape)


# In[4]:


import matplotlib.pyplot as plt

# Define pastel colors and the label mapping
pastel_colors = ['#ffb3ba', '#ffdfba', '#ffffba'] # Pink, Orange, Yellow pastels
label_map = {'0': 'Good', '1': 'Moderate', '2': 'Worst'}

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

dfs = [df_train, df_val, df_test]
titles = ["Train", "Validation", "Test"]

for i, (df, title) in enumerate(zip(dfs, titles)):
    # Get counts and ensure index is string for categorical plotting
    counts = df['GroupedScore'].value_counts().sort_index()
    x_labels = [label_map.get(str(idx), str(idx)) for idx in counts.index]
    
    # Plot bars with black edges and pastel colors
    bars = axs[i].bar(x_labels, counts.values, 
                      color=pastel_colors[:len(counts)], 
                      edgecolor='black')
    
    # Add numbers above the bars
    for bar in bars:
        height = bar.get_height()
        axs[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Formatting
    axs[i].set_title(title, fontsize=14)
    axs[i].set_xlabel("GroupedScore (State)", fontsize=10)
    axs[i].set_ylabel("No of 10 min segments in this class", fontsize=10)

plt.tight_layout()
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd

# Define colors and dataset order
dataset_order = ['GCS', 'RASS', 'ICANS', 'CAMS'] 
# Using a pastel palette
pastel_colors = ['#ffb3ba', '#baffc9', '#bae1ff', '#ffffba'] 
label_map = {0: 'Good', 1: 'Moderate', 2: 'Worst'}

fig, axs = plt.subplots(1, 3, figsize=(15, 6))

dfs = [df_train, df_val, df_test]
titles = ["Train", "Validation", "Test"]

for i, (df, title) in enumerate(zip(dfs, titles)):
    # Prepare data: Group by Score and Dataset, then pivot
    plot_data = df.groupby(['GroupedScore', 'Dataset']).size().unstack(fill_value=0)
    
    # Ensure all classes (0,1,2) and datasets are present for consistent plotting
    plot_data = plot_data.reindex(index=[0, 1, 2], columns=dataset_order, fill_value=0)
    
    x_labels = [label_map[idx] for idx in plot_data.index]
    bottom = pd.Series([0, 0, 0], index=plot_data.index)

    # Plot each dataset layer
    for j, dataset in enumerate(dataset_order):
        counts = plot_data[dataset]
        bars = axs[i].bar(x_labels, counts, bottom=bottom, 
                          color=pastel_colors[j], edgecolor='black', label=dataset)
        bottom += counts

    # Add total count numbers above the stacked bars
    for idx, total_height in enumerate(bottom):
        axs[i].text(idx, total_height + 0.1, f'{int(total_height)}', 
                    ha='center', va='bottom', fontweight='bold')

    # Formatting
    axs[i].set_title(title, fontsize=14, pad=20)
    axs[i].set_xlabel("GroupedScore", fontsize=10)
    axs[i].set_ylabel("No of 10 min segments in this class", fontsize=10)

# Create legend above the 2nd subfigure head
# We use the handles from one of the axes
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=4, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for the top legend
plt.show()


# # **Feature Engineering**

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
        "CAMS": CAMS_ROOT / "CAMS" / "MorgothActivations",
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
    

    dataset_names = data_frame["Dataset"].to_numpy()
    file_names = np.array(file_names)

    print("Grouped Label shape:", Y_grouped.shape)
    print("Raw Label shape:", Y_raw.shape)
    
    print("Feature matrix shape:", X.shape)
    
    return X, Y_grouped, Y_raw, dataset_names, file_names


# In[7]:


X_train_data, Ygrouped_train, Yraw_train, train_dataset_names,trn_filenames = morgoth_10minfea_matrix(df_train)
X_val_data, Ygrouped_val, Yraw_val, val_dataset_names, val_filenames = morgoth_10minfea_matrix(df_val)
X_tst_data, Ygrouped_tst, Yraw_tst, tst_dataset_names, tst_filenames = morgoth_10minfea_matrix(df_test)


# # **Triplet Set Creation**

# In[8]:


import torch
from torch.utils.data import Dataset
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, X, y_grouped, y_raw, y_dataset):
        self.X = X
        self.y_grouped = y_grouped
        self.y_raw = y_raw
        self.y_dataset = np.array(y_dataset)

        self.unique_datasets = np.unique(self.y_dataset)

        # dataset → indices
        self.dataset_indices = {
            d: np.where(self.y_dataset == d)[0]
            for d in self.unique_datasets
        }

        # label → indices (FIXED)
        self.label_indices = {
            int(l.item()): torch.where(self.y_grouped == l)[0]
            for l in torch.unique(self.y_grouped)
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        # -------------------------
        # 1. BALANCED ANCHOR SAMPLING
        # -------------------------
        anchor_dataset = np.random.choice(self.unique_datasets)
        anchor_pool = self.dataset_indices[anchor_dataset]
        anchor_idx = np.random.choice(anchor_pool)

        anchor_x = self.X[anchor_idx]
        anchor_g = self.y_grouped[anchor_idx]
        anchor_r = self.y_raw[anchor_idx]
        anchor_d = self.y_dataset[anchor_idx]

        # -------------------------
        # 2. POSITIVE (same label)
        # -------------------------
        pos_pool = self.label_indices[int(anchor_g.item())]

        # avoid picking same index (optional but better)
        if len(pos_pool) > 1:
            pos_pool = pos_pool[pos_pool != anchor_idx]

        pos_idx = pos_pool[torch.randint(len(pos_pool), (1,)).item()]

        pos_x = self.X[pos_idx]
        pos_g = self.y_grouped[pos_idx]
        pos_r = self.y_raw[pos_idx]
        pos_d = self.y_dataset[pos_idx]

        # -------------------------
        # 3. NEGATIVE (different label)
        # -------------------------
        all_labels = list(self.label_indices.keys())
        neg_labels = [l for l in all_labels if l != int(anchor_g.item())]

        neg_label = np.random.choice(neg_labels)
        neg_pool = self.label_indices[neg_label]

        neg_idx = neg_pool[torch.randint(len(neg_pool), (1,)).item()]

        neg_x = self.X[neg_idx]
        neg_g = self.y_grouped[neg_idx]
        neg_r = self.y_raw[neg_idx]
        neg_d = self.y_dataset[neg_idx]

        return (
            anchor_x, pos_x, neg_x,
            anchor_g, pos_g, neg_g,
            anchor_r, pos_r, neg_r,
            anchor_d, pos_d, neg_d
        )


# # **Dataloader**

# In[10]:


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

tst_dataset = TripletDataset(
    X_tst_data_trch,
    Ygrouped_tst_trch,
    Yraw_tst_trch,
    tst_dataset_names
)

train_dataset = TripletDataset(
    X_train_data_trch,
    Ygrouped_train_trch,
    Yraw_train_trch,
    train_dataset_names
)

val_dataset = TripletDataset(
    X_val_data_trch,
    Ygrouped_val_trch,
    Yraw_val_trch,
    val_dataset_names
)

print(f"Size of trainset : {len(train_dataset)}")
print(f"Size of validset : {len(val_dataset)}")
print(f"Size of Testset : {len(tst_dataset)}")

bs = 128
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_dataloader   = DataLoader(val_dataset, batch_size=bs, shuffle=False)
tst_dataloader   = DataLoader(tst_dataset, batch_size=bs, shuffle=False)


# # **Triplet set visualization**

# In[11]:


import matplotlib.pyplot as plt
import numpy as np

def plot_triplet(
    anchor_x, pos_x, neg_x,
    anchor_g, pos_g, neg_g,
    anchor_r, pos_r, neg_r,
    anchor_d, pos_d, neg_d
):

    def prep(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.array(x)

    anchor_x = prep(anchor_x)
    pos_x = prep(pos_x)
    neg_x = prep(neg_x)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    triplets = [
        (anchor_x, "Anchor", anchor_g, anchor_d, anchor_r, axs[0]),
        (pos_x, "Positive", pos_g, pos_d, pos_r, axs[1]),
        (neg_x, "Negative", neg_g, neg_d, neg_r, axs[2]),
    ]

    for x, role, g, d, r, ax in triplets:

        im = ax.imshow(x, aspect='auto', cmap='Blues')
        label_map = {
            0: "Good Neuro state",
            1: "Moderate Neuro state",
            2: "Bad Neuro state"
        }

        ax.set_title(
            f"{role}: Group {g} ({label_map.get(int(g), 'Unknown')})\nDataset: {d}, Raw: {r}",
            fontsize=10
        )

        ax.set_xlabel("Feature dimension")
        ax.set_ylabel("Time / segments")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# In[12]:


batch = next(iter(train_dataloader))

(
    anchor_x, pos_x, neg_x,
    anchor_g, pos_g, neg_g,
    anchor_r, pos_r, neg_r,
    anchor_d, pos_d, neg_d
) = batch

for i in range(len(anchor_x)):

    plot_triplet(
        anchor_x[i], pos_x[i], neg_x[i],
        anchor_g[i], pos_g[i], neg_g[i],
        anchor_r[i], pos_r[i], neg_r[i],
        anchor_d[i], pos_d[i], neg_d[i]
    )


# # **Model development (classe defenitions)**

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

class MORGOTH_ResNet1D_onlyGAP_CORAL(nn.Module):
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


# # **Contrastive Training Loop**

# In[14]:


import os
import torch
import copy
import numpy as np

def Training_triplet_function(
    model,
    lr,
    n_epochs,
    train_dl,
    valid_dl,
    save_loc,
    model_prefix,
    margin,
    device="cuda"
):

    patience = 20
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    model = model.to(device)
    
    os.makedirs(save_loc, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

    best_model_wts = copy.deepcopy(model.state_dict())

    trainloss_history = []
    valloss_history = []

    print("Start Triplet Model Training")

    for epoch in range(1, n_epochs + 1):

        # ---------------- TRAIN ----------------
        model.train()
        running_loss = 0.0

        for batch in train_dl:

            (
                anchor_x, pos_x, neg_x,
                _, _, _,
                _, _, _,
                _, _, _
            ) = batch

            anchor_x = anchor_x.to(device)
            pos_x = pos_x.to(device)
            neg_x = neg_x.to(device)

            opt.zero_grad()

            anchor_emb = model(anchor_x)
            pos_emb = model(pos_x)
            neg_emb = model(neg_x)

            loss = criterion(anchor_emb, pos_emb, neg_emb)

            loss.backward()
            opt.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dl)
        trainloss_history.append(train_loss)

        # ---------------- VALIDATION ----------------
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in valid_dl:

                (
                    anchor_x, pos_x, neg_x,
                    _, _, _,
                    _, _, _,
                    _, _, _
                ) = batch

                anchor_x = anchor_x.to(device)
                pos_x = pos_x.to(device)
                neg_x = neg_x.to(device)

                anchor_emb = model(anchor_x)
                pos_emb = model(pos_x)
                neg_emb = model(neg_x)

                val_loss = criterion(anchor_emb, pos_emb, neg_emb)

                running_val_loss += val_loss.item()

        val_loss = running_val_loss / len(valid_dl)
        valloss_history.append(val_loss)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ---------------- EARLY STOPPING ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            best_model_wts = copy.deepcopy(model.state_dict())

            best_path = os.path.join(save_loc, f"{model_prefix}_BestModel.pth")

            if os.path.exists(best_path):
                os.remove(best_path)

            torch.save(best_model_wts, best_path)

            print(f"Saved best model at epoch {epoch}")

        else:
            epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    final_path = os.path.join(save_loc, f"{model_prefix}_valLoss_{best_val_loss:.6f}.pth")

    torch.save(best_model_wts, final_path)

    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved at: {final_path}")

    return trainloss_history, valloss_history, final_path


# # **Train model**

# In[15]:


tic = time.perf_counter()
model_onlygap = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17)
base_model_dir = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints_NEW" / "TripletCheckpoint" / "RASS" / 


trainloss_history, valloss_history, final_path = Training_triplet_function(
    model= model_onlygap,
    lr=0.0005,
    n_epochs=200,
    train_dl= train_dataloader,
    valid_dl= val_dataloader,
    save_loc=base_model_dir,
    model_prefix='ResNetGAP_RASS',
    margin=0.2,
    device="cuda"
)
toc = time.perf_counter()
print('='*40)
print(f"Elapsed time: {(toc - tic)/60:0.4f} mins")


# # **Embedding Extrcation process and visulaize them**

# ## **Get embeddings**

# In[16]:


device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- LOAD MODEL ----------
model_path_trained = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints_NEW" / "TripletCheckpoint" / "RASS" / "ResNetGAP_RASS_BestModel.pth"

model_trained = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17)
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
from tqdm import tqdm
import torch

def get_embeddings(model_trained, dataloader):
    all_embeddings = []

    model_trained.eval()

    with torch.no_grad():
        for (x,) in tqdm(dataloader, desc="Extracting embeddings"):
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


# # **Get PacMAPs**

# In[17]:


import pacmap
def compute_pacmap(embeddings, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=6.0):
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    emb = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio
    )
    
    Y_pacmap = emb.fit_transform(embeddings, init="pca")
    
    return Y_pacmap

# --------- USAGE ----------
train_pacmap = compute_pacmap(train_embeddings)
val_pacmap   = compute_pacmap(val_embeddings)
tst_pacmap   = compute_pacmap(tst_embeddings)

print("Train PaCMAP shape:", train_pacmap.shape)
print("Val PaCMAP shape:", val_pacmap.shape)
print("Test PaCMAP shape:", tst_pacmap.shape)


# # **Plot PacMAPs**

# In[18]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------- COLOR GRADIENTS ----------------
greens = [
    "yellowgreen", "greenyellow",
    "lawngreen", "palegreen",
    "lightgreen", "mediumspringgreen",
    "lime", "limegreen"
]

pinks = [
    "lavenderblush", "lightpink",
    "pink", "hotpink",
    "deeppink", "orchid",
    "mediumorchid", "plum",
    "violet", "mediumvioletred",
    "darkmagenta", "magenta"
]

blues = [
    "powderblue", "lightblue",
    "deepskyblue", "skyblue",
    "steelblue", "dodgerblue",
    "cyan", "turquoise",
    "darkturquoise", "mediumturquoise",
    "paleturquoise", "royalblue"
]

# ---------------- VALUE MAPS ----------------
value_maps = {
    0: {  # GOOD
        #"GCS": [13, 14, 15],
        "RASS": [-1, 0],
        # "ICANS": [0],
        # "CAMS": [0, 1]
    },
    1: {  # MODERATE
        # "GCS": [9, 10, 11, 12],
        "RASS": [-3, -2],
        # "ICANS": [1, 2],
        # "CAMS": [2, 3, 4, 5]
    },
    2: {  # BAD
        # "GCS": [3, 4, 5, 6, 7, 8],
        "RASS": [-5, -4],
        # "ICANS": [3, 4],
        # "CAMS": [6, 7]
    }
}

# ---------------- MAIN FUNCTION ----------------
def plot_pacmap_granular(emb_2d, y_grouped, y_raw, dataset_names, title="PaCMAP"):

    if isinstance(y_grouped, torch.Tensor):
        y_grouped = y_grouped.detach().cpu().numpy()
    if isinstance(y_raw, torch.Tensor):
        y_raw = y_raw.detach().cpu().numpy()
    if isinstance(dataset_names, torch.Tensor):
        dataset_names = dataset_names.detach().cpu().numpy()

    emb_2d = np.array(emb_2d)
    y_grouped = np.array(y_grouped)
    y_raw = np.array(y_raw)
    dataset_names = np.array(dataset_names)

    # ---------------- COLOR ASSIGNMENT ----------------
    gradients = {0: greens, 1: pinks, 2: blues}
    color_map = {}

    for cls in [0, 1, 2]:
        for dname, values in value_maps[cls].items():

            n = len(values)
            color_list = gradients[cls]

            for i, v in enumerate(values):
                idx = int(i * (len(color_list) - 1) / max(n - 1, 1))
                color_map[(cls, dname, v)] = color_list[idx]

    # ---------------- PLOT ----------------
    plt.figure(figsize=(12, 8))

    legend_handles = []
    seen = set()

    unique_points = sorted(set(zip(y_grouped, y_raw, dataset_names)))

    for g, r, dname in unique_points:

        idx = (
            (y_grouped == g) &
            (y_raw == r) &
            (dataset_names == dname)
        )

        if np.sum(idx) == 0:
            continue

        face_color = color_map.get((g, dname, r), "gray")

        edge_color = {
            0: "darkgreen",
            1: "magenta",
            2: "blue"
        }[g]

        plt.scatter(
            emb_2d[idx, 0],
            emb_2d[idx, 1],
            s=10,
            alpha=0.85,
            facecolors=face_color,
            edgecolors=edge_color,
            linewidths=0.8
        )

        # ---------------- FIXED LEGEND ----------------
        group_name = {
            0: "Good",
            1: "Moderate",
            2: "Bad"
        }[g]

        label = f"{group_name} ({dname} = {int(r)})"

        if label not in seen:
            legend_handles.append(
                mpl.lines.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    label=label,
                    markerfacecolor=face_color,
                    markeredgecolor=edge_color,
                    markersize=5
                )
            )
            seen.add(label)

    plt.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=7,
        fontsize=8
    )

    plt.title(title, pad=70)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ## **Train static PacMAPS**

# In[19]:


plot_pacmap_granular(train_pacmap, 
                     Ygrouped_train_trch,
                     Yraw_train_trch,
                     train_dataset_names,
                     title="Train RASS BESPOKE PaCMAP")


# ## **Validation PacMAPs**

# In[20]:


plot_pacmap_granular(val_pacmap, 
                     Ygrouped_val_trch,
                     Yraw_val_trch,
                     val_dataset_names,
                     title="Validation RASS BESPOKE PaCMAP")


# ## **Test static PacMAPs**

# In[21]:


plot_pacmap_granular(tst_pacmap, 
                     Ygrouped_tst_trch,
                     Yraw_tst_trch,
                     tst_dataset_names,
                     title="Test RASS BESPOKE PaCMAP") 


# In[ ]:




