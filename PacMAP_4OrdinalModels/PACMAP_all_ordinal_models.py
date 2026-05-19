#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
from datetime import datetime
from pathlib import Path
import pacmap

import h5py
import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from tqdm import tqdm

from coral_pytorch.dataset import (
    corn_label_from_logits
)
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import corn_loss

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


# In[ ]:





# In[2]:


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
        x_gap = self.gap(x).squeeze(-1)

        x = self.dropout(x_gap)
        x = self.fc2(x)          # CORAL output
        return x, x_gap


# In[ ]:





# In[23]:


# ------------------- CAMS - Dataset --------------------------
from pathlib import Path

if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

NESI_ROOT = None

for parent in current.parents:
    if (parent / "NESI").exists():
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI folder not found")

metadata_path_CAMS = NESI_ROOT /  "NESI" / "model" / "Training" / "CAMSTraining_Final_Metadata.csv"
df_CAMS_metadata = pd.read_csv(metadata_path_CAMS)
df_CAMS_metadata["GroupedScore"] = df_CAMS_metadata["CAMS_SF"].apply(group_cams)
df_CAMS_metadata = df_CAMS_metadata.rename(columns={'CAMS_SF': 'RawScore'})
df_CAMS_metadata['Dataset'] = 'CAMS'
df_CAMS_metadata['TransformedScore'] = df_CAMS_metadata['RawScore']
df_CAMS_metadata = df_CAMS_metadata[['BDSPPatientID', 'Dataset', 'MorgothOutputFilename', 'RawScore', 'GroupedScore', 'TransformedScore']]

# ------------------- GCS - Dataset --------------------------
metadata_path_GCS = NESI_ROOT /  "NESI" / "model" / "Training" / "GCSTraining_Final_Metadata.csv"
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
metadata_path_RASS = NESI_ROOT /  "NESI" / "model" / "Training" / "RASSTraining_Final_Metadata.csv"
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

# ------------------- ICANS - Dataset --------------------------
metadata_path_ICANS = NESI_ROOT /  "NESI" / "model" / "Training" / "ICANSTraining_Final_Metadata.csv"
df_ICANS_metadata = pd.read_csv(metadata_path_ICANS)
df_ICANS_metadata["GroupedScore"] = df_ICANS_metadata["ICANS_raw"].apply(group_icans)
df_ICANS_metadata = df_ICANS_metadata.rename(columns={'StudyID':'BDSPPatientID',                                                      
                                                      'ICANS_raw': 'RawScore'})
df_ICANS_metadata['Dataset'] = 'ICANS'
df_ICANS_metadata['TransformedScore'] = df_ICANS_metadata['RawScore']
df_ICANS_metadata = df_ICANS_metadata[['BDSPPatientID', 'Dataset', 'MorgothOutputFilename', 'RawScore', 'GroupedScore', 'TransformedScore']]


# # **Helper function**

# ## **Test metdata creation**

# In[4]:


def get_test_metedata_cohort(cohort_metadata):
    df_subjects = cohort_metadata[['BDSPPatientID', 'Dataset']].drop_duplicates()
    
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
    
    cohort_metadata['Split'] = cohort_metadata['BDSPPatientID'].apply(assign_split)
    
    df_train = cohort_metadata[cohort_metadata['Split'] == 'Train'].reset_index(drop=True)
    df_val   = cohort_metadata[cohort_metadata['Split'] == 'Val'].reset_index(drop=True)
    df_test  = cohort_metadata[cohort_metadata['Split'] == 'Test'].reset_index(drop=True)
    
    print("Train shape:", df_train.shape)
    print("Val shape:", df_val.shape)
    print("Test shape:", df_test.shape)
    print('\n')
    return df_test 


# ## **Feature Engineering**

# In[5]:


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
    Y_raw_transformed = data_frame['TransformedScore'].to_numpy()

    dataset_names = data_frame["Dataset"].to_numpy()
    file_names = np.array(file_names)

    print("Grouped Label shape:", Y_grouped.shape)
    print("Raw Label shape:", Y_raw.shape)
    print("Raw Label transformed shape:", Y_raw_transformed.shape)
    print("Feature matrix shape:", X.shape)
    
    return X, Y_grouped, Y_raw, Y_raw_transformed, dataset_names, file_names


# ## **Model latent output heper**

# In[19]:


def Score_model_load(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model_trained = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, num_classes=num_classes)
    model_trained.load_state_dict(
        torch.load(model_path,
                   map_location=device,
                   weights_only=True)
    )
    model_trained = model_trained.to(device)
    model_trained.eval()
    print('Model Loaded!!')
    return model_trained

def get_corn_model_output(model_trained, X_tst_data_global):
    # ---------------------------------------------------
    # Prepare test data
    # ---------------------------------------------------
    X_tst_data_global = torch.tensor(X_tst_data_global, dtype=torch.float32)
    
    batch_size = 256
    device = next(model_trained.parameters()).device

    # ---------------------------------------------------
    # Collect Latents and Predictions
    # ---------------------------------------------------
    test_latents = []
    Y_pred = [] 
    model_trained.eval()
    with torch.no_grad():
        for i in range(0, len(X_tst_data_global), batch_size):
            x_batch = X_tst_data_global[i:i+batch_size].to(device)
    
            out, gaps = model_trained(x_batch)   # (N, K-1) logits
            preds = corn_label_from_logits(out).float()

            Y_pred.extend(preds.cpu().numpy())
            test_latents.extend(gaps.cpu().numpy())
    
    Y_pred = np.array(Y_pred)
    test_latents = np.array(test_latents)
    return Y_pred, test_latents


# ## **PACMAP helper**

# In[17]:


def pacmap_embeddings(latent_all):
    emb = pacmap.PaCMAP(n_components=2, n_neighbors=10,MN_ratio=0.5, FP_ratio=6.0)
    Y_pacmap = emb.fit_transform(latent_all, init="pca")
    return Y_pacmap


# # **Execute**

# In[ ]:


#-------------- RASS dataset -----------------------
RASS_model_path = RASS_ROOT / "RASS" / "RASS_Best_DL_model" / "ResNetGAP" / "RESNETGAP_Best_RASS.pth" 
RASS_model_trained = Score_model_load(RASS_model_path, 6)
df_test_RASS  = get_test_metedata_cohort(df_RASS_metadata)

X_tst_data_global_RASS, Ygrp_tst_global_RASS, Yraw_tst_global_RASS, _, tst_dataset_names_global, _ = morgoth_10minfea_matrix(df_test_RASS)

RASS_test_preds, RASS_test_latents = get_corn_model_output(RASS_model_trained, X_tst_data_global_RASS)   
RASS_pacmap_data = pacmap_embeddings(RASS_test_latents)
print("\nCORN model GAP latent shape:", RASS_test_latents.shape)
print("\nRASS CORN model PACMAP shape:", RASS_pacmap_data.shape)
print("True labels shape:", np.array(Yraw_tst_global_RASS).shape)


# In[107]:


#-------------- GCS dataset -----------------------
GCS_model_path = GCS_ROOT / "GCS" / "GCS_Best_DL_model" / "ResNetGAP" / "RESNETGAP_Best_GCS.pth" 
GCS_model_trained = Score_model_load(GCS_model_path, 3)
df_test_GCS  = get_test_metedata_cohort(df_GCS_metadata)

X_tst_data_global_GCS, Ygrp_tst_global_GCS, Yraw_tst_global_GCS, _, tst_dataset_names_global, _ = morgoth_10minfea_matrix(df_test_GCS)

GCS_test_preds, GCS_test_latents = get_corn_model_output(GCS_model_trained, X_tst_data_global_GCS)   
GCS_pacmap_data = pacmap_embeddings(GCS_test_latents)
print("\nCORN model GAP latent shape:", GCS_test_latents.shape)
print("\nGCS CORN model PACMAP shape:", GCS_pacmap_data.shape)
print("True labels shape:", np.array(Yraw_tst_global_GCS).shape)


# In[45]:


#-------------- CAMS dataset -----------------------
CAMS_model_path = CAMS_ROOT / "CAMS" / "CAMS_Best_DL_model" / "ResNetGAP" / "RESNETGAP_Best_CAMS.pth" 
CAMS_model_trained = Score_model_load(CAMS_model_path, 3)
df_test_CAMS  = df_CAMS_metadata

X_tst_data_global_CAMS, Ygrp_tst_global_CAMS, Yraw_tst_global_CAMS, _, tst_dataset_names_global, _ = morgoth_10minfea_matrix(df_test_CAMS)

CAMS_test_preds, CAMS_test_latents = get_corn_model_output(CAMS_model_trained, X_tst_data_global_CAMS)  
CAMS_pacmap_data = pacmap_embeddings(CAMS_test_latents)

print("\nCAMS CORN model GAP latent shape:", CAMS_test_latents.shape)
print("\nCAMS CORN model PACMAP shape:", CAMS_pacmap_data.shape)
print("True labels shape:", np.array(Yraw_tst_global_CAMS).shape)

#-------------- ICANS dataset -----------------------
ICANS_model_path = ICANS_ROOT / "ICANS" / "ICANS_Best_DL_model" / "ResNetGAP" / "RESNETGAP_Best_ICANS.pth" 
ICANS_model_trained = Score_model_load(ICANS_model_path, 3)
df_test_ICANS  = df_ICANS_metadata

X_tst_data_global_ICANS, Ygrp_tst_global_ICANS, Yraw_tst_global_ICANS, _, tst_dataset_names_global, _ = morgoth_10minfea_matrix(df_test_ICANS)

ICANS_test_preds, ICANS_test_latents = get_corn_model_output(ICANS_model_trained, X_tst_data_global_ICANS)    
ICANS_pacmap_data = pacmap_embeddings(ICANS_test_latents)

print("\nCORN model GAP latent shape:", ICANS_test_latents.shape)
print("\nICANS CORN model PACMAP shape:", ICANS_pacmap_data.shape)
print("True labels shape:", np.array(Yraw_tst_global_ICANS).shape)


# # **Pacmaps**

from pathlib import Path

if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

PACMAP_ROOT = None

for parent in current.parents:
    if (parent / "PacMAP_4OrdinalModels").exists():
        PACMAP_ROOT = parent
        break

if PACMAP_ROOT is None:
    raise RuntimeError("PacMAP_4OrdinalModels folder not found")

# ## **CAMS PacMAPS**

# In[86]:


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_cams_pacmap_fixed_colors(emb, y_raw, title="PaCMAP Embedding", save_path=None, figsz=(11,7), dpi=300):
    emb = np.array(emb)
    y_raw = np.array(y_raw)
    
    group_info = {
        0: ('limegreen', 'darkgreen', 'No/Mild'), 
        1: ('mediumseagreen', 'darkgreen', 'No/Mild'),
        2: ('khaki', 'darkgoldenrod', 'Moderate'),
        3: ('palegoldenrod', 'darkgoldenrod', 'Moderate'),
        4: ('darkkhaki', 'darkgoldenrod', 'Moderate'),
        5: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        6: ('salmon', 'darkred', 'Severe'), 
        7: ('red', 'darkred', 'Severe')
    }
    
    fig, ax = plt.subplots(figsize=figsz, dpi=dpi)
    legend_elements = []
    
    for raw_val in range(8):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0: 
            continue
            
        face_c, edge_c, group_name = group_info[raw_val]
        
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=1.0,
            zorder=2
        )
        
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"CAMS {raw_val} ({group_name})"
            )
        )

    # ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel("Dimension 1", labelpad=8, fontweight='normal')
    ax.set_ylabel("Dimension 2", labelpad=8, fontweight='normal')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    ax.spines['left'].set_edgecolor('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['bottom'].set_linewidth(1.0)
        
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="CAM-S scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=4
    )
    
    plt.tight_layout()
    
    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        
    plt.show()


# In[87]:


plot_cams_pacmap_fixed_colors(
    emb=CAMS_pacmap_data,
    y_raw=Yraw_tst_global_CAMS,
    title="PACMAP Embedding",
    save_path=PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "CAMS_PacMAP.png",
    figsz=(7,6),
    dpi=210
)


# ## **GCS PacMAPS**

# In[128]:


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_gcs_pacmap_fixed_colors(emb, y_raw, title="PaCMAP Embedding", save_path=None, figsz=(11,7),dpi=300):
    emb = np.array(emb)
    y_raw = np.array(y_raw)
    
    group_info = {
        3: ('red', 'darkred', 'Severe'), 
        4: ('salmon', 'darkred', 'Severe'),
        5: ('tomato', 'darkred', 'Severe'), 
        6: ('darksalmon', 'darkred', 'Severe'),
        7: ('indianred', 'darkred', 'Severe'), 
        8: ('lightcoral', 'darkred', 'Severe'),
        
        9: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        10: ('darkkhaki', 'darkgoldenrod', 'Moderate'),
        11: ('palegoldenrod', 'darkgoldenrod', 'Moderate'),
        12: ('khaki', 'darkgoldenrod', 'Moderate'),

        13: ('mediumseagreen', 'darkgreen', 'No/Mild'), 
        14: ('limegreen', 'darkgreen', 'No/Mild'),
        15: ('lime', 'darkgreen', 'No/Mild'),
    }
    
    fig, ax = plt.subplots(figsize=figsz, dpi=dpi)
    legend_elements = []
    
    for raw_val in range(3, 16):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0: 
            continue
            
        face_c, edge_c, group_name = group_info[raw_val]
        
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=0.5,
            zorder=2
        )
        
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"GCS {raw_val} ({group_name})"
            )
        )

    ax.set_xlabel("Dimension 1", labelpad=8, fontweight='normal')
    ax.set_ylabel("Dimension 2", labelpad=8, fontweight='normal')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    ax.spines['left'].set_edgecolor('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['bottom'].set_linewidth(1.0)
        
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    # ---------------------------------------------------
    # 🔥 LEGEND FIX (STRICT COLUMN CONTROL)
    # ---------------------------------------------------

    def H(label):
        for h in legend_elements:
            if h.get_label() == label:
                return h
        return Line2D([], [], alpha=0)

    ordered_labels = [
        # Column 1 (3–6 Severe)
        "GCS 3 (Severe)", "GCS 4 (Severe)", "GCS 5 (Severe)", "GCS 6 (Severe)",
        
        # Column 2 (7–8)
        "GCS 7 (Severe)", "GCS 8 (Severe)", None, None,
        
        # Column 3 (9–12 Moderate)
        "GCS 9 (Moderate)", "GCS 10 (Moderate)", "GCS 11 (Moderate)", "GCS 12 (Moderate)",
        
        # Column 4 (13–15 No/Mild)
        "GCS 13 (No/Mild)", "GCS 14 (No/Mild)", "GCS 15 (No/Mild)", None
    ]

    ordered_handles = [
        H(lbl) if lbl is not None else Line2D([], [], alpha=0)
        for lbl in ordered_labels
    ]

    ax.legend(
        handles=ordered_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="GCS scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=4
    )

    plt.tight_layout()

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()


# In[129]:


plot_gcs_pacmap_fixed_colors(
    emb=GCS_pacmap_data,
    y_raw=Yraw_tst_global_GCS,
    title="PACMAP Embedding",
    save_path=PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "GCS_PacMAP.png",
    figsz=(7,6),
    dpi=210
)


# ## **RASS PacMAPS**

# In[82]:


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_rass_pacmap_fixed_colors(emb, y_raw, title="PaCMAP Embedding", save_path=None, figsz=(11, 7),dpi=300):
    emb = np.array(emb)
    y_raw = np.array(y_raw)
    
    # Structural bug fixed here for -5 (removed the trailing duplicate string element)
    group_info = {
         0: ('limegreen', 'darkgreen', 'No/Mild'), 
        -1: ('mediumseagreen', 'darkgreen', 'No/Mild'),
        -2: ('gold', 'darkgoldenrod', 'Moderate'),
        -3: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        -4: ('salmon', 'darkred', 'Severe'), 
        -5: ('red', 'darkred', 'Severe')
    }
    
    fig, ax = plt.subplots(figsize=figsz, dpi=dpi)
    legend_elements = []
    
    rass_classes_ordered = [-5, -4, -3, -2, -1, 0]
    
    for raw_val in rass_classes_ordered:
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0: 
            continue
            
        face_c, edge_c, group_name = group_info[raw_val]
        
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=0.5,
            zorder=2
        )
        
        # Increased markersize slightly to 8 so the edge outlines fit nicely in the legend layout
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"RASS {raw_val} ({group_name})"
            )
        )

    # ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel("Dimension 1", labelpad=8, fontweight='bold')
    ax.set_ylabel("Dimension 2", labelpad=8, fontweight='bold')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    ax.spines['left'].set_edgecolor('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['bottom'].set_linewidth(1.0)
        
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="RASS scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=3
    )
    
    plt.tight_layout()
    
    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        
    plt.show()


# In[83]:


plot_rass_pacmap_fixed_colors(
    emb=RASS_pacmap_data,
    y_raw=Yraw_tst_global_RASS,
    title="RASS cohort PACMAP embedding",
    save_path=PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "RASS_PacMAP.png",
    figsz=(7,6),
    dpi=210
)


# ## **ICANS Pacmaps**

# In[84]:


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_ICANS_pacmap_fixed_colors(emb, y_raw, title="PaCMAP Embedding", save_path=None, figsz=(11, 7), dpi=300):
    emb = np.array(emb)
    y_raw = np.array(y_raw)
    
    group_info = {
        0: ('limegreen', 'darkgreen', 'No/Mild'), 
        1: ('gold', 'darkgoldenrod', 'Moderate'),
        2: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        3: ('salmon', 'darkred', 'Severe'), 
        4: ('red', 'darkred', 'Severe')
    }
    
    fig, ax = plt.subplots(figsize=figsz, dpi=dpi)

    legend_elements = []

    for raw_val in range(8):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=1.0,
            zorder=2
        )

        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"ICANS {raw_val} ({group_name})"
            )
        )

    # ---------------------------------------------------
    # 🔥 LEGEND FIX (FORCED COLUMN GROUPING)
    # ---------------------------------------------------

    def get_handle(label):
        for h in legend_elements:
            if h.get_label() == label:
                return h
        return None

    ordered_labels = [
        "ICANS 0 (No/Mild)",   None,                None,
        "ICANS 1 (Moderate)",  "ICANS 2 (Moderate)", None,
        "ICANS 3 (Severe)",    "ICANS 4 (Severe)",   None
    ]

    ordered_handles = [
        get_handle(l) if l is not None else Line2D([], [], alpha=0)
        for l in ordered_labels
    ]

    ax.legend(
        handles=ordered_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        title="ICANS scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=3,
        handletextpad=0.6,
        columnspacing=1.5
    )

    # ---------------------------------------------------
    # keep EVERYTHING else unchanged
    # ---------------------------------------------------

    ax.set_xlabel("Dimension 1", labelpad=8, fontweight='bold')
    ax.set_ylabel("Dimension 2", labelpad=8, fontweight='bold')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    ax.spines['left'].set_edgecolor('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['bottom'].set_linewidth(1.0)
        
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    plt.tight_layout()

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()


# In[85]:


plot_ICANS_pacmap_fixed_colors(
    emb=ICANS_pacmap_data,
    y_raw=Yraw_tst_global_ICANS,
    title="ICANS cohort PACMAP embedding",
    save_path=PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "ICANS_PacMAP.png",
    figsz=(7,6),
    dpi=210
)

# # **Save the data**
import pickle
import os

def save_pkl(data, save_path):
    """
    Save any Python object as a pickle file.

    Parameters
    ----------
    data : any
        Data/object to save
    save_path : str
        Full path including filename ending with .pkl
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved: {save_path}")

save_pkl(RASS_pacmap_data, PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "RASS_pacmap_data.pkl")
save_pkl(Yraw_tst_global_RASS, PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "Yraw_tst_global_RASS.pkl")

save_pkl(CAMS_pacmap_data, PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "CAMS_pacmap_data.pkl")
save_pkl(Yraw_tst_global_CAMS, PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "Yraw_tst_global_CAMS.pkl")

save_pkl(GCS_pacmap_data, PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "GCS_pacmap_data.pkl")
save_pkl(Yraw_tst_global_GCS, PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "Yraw_tst_global_GCS.pkl")

save_pkl(ICANS_pacmap_data, PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "ICANS_pacmap_data.pkl")
save_pkl(Yraw_tst_global_ICANS, PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "Yraw_tst_global_ICANS.pkl")



