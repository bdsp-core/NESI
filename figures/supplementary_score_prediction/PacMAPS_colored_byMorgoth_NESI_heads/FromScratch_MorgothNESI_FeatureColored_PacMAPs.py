#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# # **Helper function**

# In[144]:


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


def save_pickle(obj, save_path):
    """
    Save any Python object as a pickle file.

    Parameters
    ----------
    obj : any
        Object to save.
    save_path : str
        Output .pkl file path.
    """

    dir_name = os.path.dirname(save_path)

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {save_path}")

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


# In[131]:


Supplementray_fig_Root = Path('/home/ayush/Desktop/GitHub-YAMA')
RASS_ROOT = Path('/home/ayush/Desktop/GitHub-YAMA')
GCS_ROOT = Path('/home/ayush/Desktop/GitHub-YAMA')
CAMS_ROOT = Path('/home/ayush/Desktop/GitHub-YAMA')
ICANS_ROOT = Path('/home/ayush/Desktop/GitHub-YAMA')


# In[5]:


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


# In[69]:


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
    Y_NESI = data_frame["NESI"].to_numpy()
    
    print("Grouped Label shape:", Y_grouped.shape)
    print("Raw Label shape:", Y_raw.shape)
    print("Raw Label transformed shape:", Y_raw_transformed.shape)
    print("NESI shape:", Y_NESI.shape)
    print("Feature matrix shape:", X.shape)
    
    return X, Y_grouped, Y_raw, Y_raw_transformed, dataset_names, file_names, Y_NESI


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


# # **RASS PacMAPs**

# In[70]:


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



# In[10]:


# ------------------- RASS - Dataset --------------------------
metadata_path_RASS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "RASS_metadata_with_NESI.csv"
df_RASS_metadata=pd.read_csv(metadata_path_RASS)

#-------------- RASS dataset -----------------------
RASS_model_path = RASS_ROOT / "cohort_models" / "RASS" / "RASS_Best_DL_model" / "ResNetGAP" / "RESNETGAP_Best_RASS.pth" 
RASS_model_trained = Score_model_load(RASS_model_path, 6)
df_test_RASS  = get_test_metedata_cohort(df_RASS_metadata)

X_tst_data_global_RASS, Ygrp_tst_global_RASS, Yraw_tst_global_RASS, _, tst_dataset_names_global, _, Y_tst_NESI_RASS = morgoth_10minfea_matrix(df_test_RASS)

RASS_test_preds, RASS_test_latents = get_corn_model_output(RASS_model_trained, X_tst_data_global_RASS)   
RASS_pacmap_data = pacmap_embeddings(RASS_test_latents)
print("\nCORN model GAP latent shape:", RASS_test_latents.shape)
print("\nRASS CORN model PACMAP shape:", RASS_pacmap_data.shape)
print("True labels shape:", np.array(Yraw_tst_global_RASS).shape)
print("NESI shape:", np.array(Y_tst_NESI_RASS).shape)

plot_rass_pacmap_fixed_colors(
    emb=RASS_pacmap_data,
    y_raw=Yraw_tst_global_RASS,
    title="RASS cohort PACMAP embedding",
    save_path=None,
    figsz=(7,6),
    dpi=210
)


# In[67]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

def plot_rass_and_feature_pacmaps(
    emb,
    y_raw,
    X_features,
    feature_names,
    Y_tst_NESI_RASS=None,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=300
):

    emb = np.asarray(emb)
    y_raw = np.asarray(y_raw)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[10, 6, 6, 6],
        hspace=0.25,
        wspace=0.10
    )

    #####################################################################
    # TOP PANEL (UNCHANGED)
    #####################################################################
    ax_main = fig.add_subplot(gs[0, 1:5])

    group_info = {
         0: ('limegreen', 'darkgreen', 'No/Mild'),
        -1: ('mediumseagreen', 'darkgreen', 'No/Mild'),
        -2: ('gold', 'darkgoldenrod', 'Moderate'),
        -3: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        -4: ('salmon', 'darkred', 'Severe'),
        -5: ('red', 'darkred', 'Severe')
    }

    legend_elements = []
    rass_classes_ordered = [-5, -4, -3, -2, -1, 0]

    for raw_val in rass_classes_ordered:

        mask = (y_raw == raw_val)

        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax_main.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=0.5,
            zorder=2,
            rasterized=True
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
                label=f"RASS {raw_val} ({group_name})"
            )
        )

    ax_main.set_xlabel("Dimension 1", fontweight='bold')
    ax_main.set_ylabel("Dimension 2", fontweight='bold')

    for spine in ['top', 'right']:
        ax_main.spines[spine].set_visible(False)

    ax_main.grid(True, linestyle='--', alpha=0.3)

    ax_main.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="RASS scale detailed key",
        frameon=True,
        fontsize=12,
        title_fontsize=12,
        ncol=3
    )

    #####################################################################
    # FEATURE PANELS (3 × 6) WITH INDIVIDUAL COLORBARS
    #####################################################################

    cmap_all = 'Blues'

    for i in range(18):

        row = 1 + (i // 6)
        col = i % 6

        ax = fig.add_subplot(gs[row, col])

        # ---------------------------
        # NESI PANEL
        # ---------------------------
        if i == 17:

            if plot_nesi and Y_tst_NESI_RASS is not None:

                sc = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=Y_tst_NESI_RASS,
                    cmap=cmap_all,
                    s=4,
                    alpha=1.0,
                    rasterized=True
                )

                ax.set_title("NESI", fontsize=10, fontweight='bold')

                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=9)
            else:
                ax.axis('off')

        elif i < X_features.shape[1]:

            vals = X_features[:, i]

            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=vals,
                cmap=cmap_all,
                s=4,
                alpha=1.0,
                rasterized=True
            )

            ax.set_title(feature_names[i], fontsize=10, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=9)

        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# In[145]:


median_feature_RASS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "X_tst_data_global_RASS_median.pkl"
save_pickle(X_tst_data_global_RASS_median, median_feature_RASS_savepath)

Y_raw_RASS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Yraw_tst_global_RASS.pkl"
save_pickle(Yraw_tst_global_RASS, Y_raw_RASS_savepath)

pacmap_RASS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "RASS_pacmap_data.pkl"
save_pickle(RASS_pacmap_data, pacmap_RASS_savepath)

NESI_RASS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Y_tst_NESI_RASS.pkl"
save_pickle(Y_tst_NESI_RASS, NESI_RASS_savepath)




# In[139]:


X_tst_data_global_RASS_median = np.median(
    X_tst_data_global_RASS,
    axis=1
)

feature_names = [
    'Awake','N1','N2',
    'Normal/Abnormal',
    'Burst/No Burst',
    'No Spike','Focal Spike','Generalized Spike',
    'No Slowing','Focal Slowing','Generalized Slowing',
    'Other','Seizure','LPD','GPD','LRDA','GRDA'
]

fig_RASS = plot_rass_and_feature_pacmaps(
    emb=RASS_pacmap_data,
    y_raw=Yraw_tst_global_RASS,
    X_features=X_tst_data_global_RASS_median,
    feature_names=feature_names,
    Y_tst_NESI_RASS=Y_tst_NESI_RASS,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=300
)

savefig_path_RASS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "RASS_new_pacmap.png"
fig_RASS.savefig(savefig_path_RASS, dpi=600, bbox_inches="tight")
plt.show()


# # **GCS PacMAPs**

# In[104]:


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


# In[105]:


# ------------------- GCS - Dataset --------------------------
metadata_path_GCS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "GCS_metadata_with_NESI.csv"
df_GCS_metadata = pd.read_csv(metadata_path_GCS)


#-------------- GCS dataset -----------------------
GCS_model_path = GCS_ROOT / "cohort_models" / "GCS" / "GCS_Best_DL_model" / "ResNetGAP" / "RESNETGAP_Best_GCS.pth" 
GCS_model_trained = Score_model_load(GCS_model_path, 3)
df_test_GCS  = get_test_metedata_cohort(df_GCS_metadata)

X_tst_data_global_GCS, Ygrp_tst_global_GCS, Yraw_tst_global_GCS, _, tst_dataset_names_global, _, Y_tst_NESI_GCS = morgoth_10minfea_matrix(df_test_GCS)

GCS_test_preds, GCS_test_latents = get_corn_model_output(GCS_model_trained, X_tst_data_global_GCS)   
GCS_pacmap_data = pacmap_embeddings(GCS_test_latents)
print("\nCORN model GAP latent shape:", GCS_test_latents.shape)
print("\nGCS CORN model PACMAP shape:", GCS_pacmap_data.shape)
print("True labels shape:", np.array(Yraw_tst_global_GCS).shape)
print("NESI shape:", np.array(Y_tst_NESI_GCS).shape)

plot_gcs_pacmap_fixed_colors(
    emb=GCS_pacmap_data,
    y_raw=Yraw_tst_global_GCS,
    title="GCS cohort PACMAP embedding",
    save_path=None,
    figsz=(7,6),
    dpi=210
)


# In[108]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

def plot_gcs_and_feature_pacmaps(
    emb,
    y_raw,
    X_features,
    feature_names,
    Y_tst_NESI_GCS=None,
    plot_nesi=False,
    figsize=(24, 20),
    dpi=300
):

    emb = np.asarray(emb)
    y_raw = np.asarray(y_raw)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[10, 6, 6, 6],
        hspace=0.25,
        wspace=0.10
    )

    #####################################################################
    # TOP: GCS MAIN PLOT (same styling as your function)
    #####################################################################
    ax_main = fig.add_subplot(gs[0, 1:5])

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

    legend_elements = []

    for raw_val in range(3, 16):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax_main.scatter(
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

    ax_main.set_xlabel("Dimension 1", fontweight='bold')
    ax_main.set_ylabel("Dimension 2", fontweight='bold')
    ax_main.grid(True, linestyle='--', alpha=0.3)

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
    ax_main.legend(
        handles=ordered_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="GCS scale detailed key",
        frameon=True,
        fontsize=12,
        title_fontsize=12,
        ncol=4
    )
    for spine in ['top', 'right', ]:
        ax_main.spines[spine].set_visible(False)
    
    ax_main.spines['left'].set_visible(True)
    ax_main.spines['left'].set_edgecolor('black')
    ax_main.spines['left'].set_linewidth(1.2)

    ax_main.spines['bottom'].set_visible(True)
    ax_main.spines['bottom'].set_edgecolor('black')
    ax_main.spines['bottom'].set_linewidth(1.2)
    #####################################################################
    # FEATURE GRID (3 × 6 = 18 SLOTS)
    #####################################################################

    cmap_all = 'Blues'

    for i in range(18):

        row = 1 + (i // 6)
        col = i % 6

        ax = fig.add_subplot(gs[row, col])

        # -----------------------------
        # LAST SLOT = NESI (optional)
        # -----------------------------
        if i == 17:

            if plot_nesi and Y_tst_NESI_GCS is not None:

                sc = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=Y_tst_NESI_GCS,
                    cmap=cmap_all,
                    s=4,
                    alpha=1.0,
                    rasterized=True
                )

                ax.set_title("NESI", fontsize=10, fontweight='bold')

                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=9)

            else:
                ax.axis('off')

        # -----------------------------
        # FEATURE PANELS
        # -----------------------------
        elif i < X_features.shape[1]:

            vals = X_features[:, i]

            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=vals,
                cmap=cmap_all,
                s=4,
                alpha=1.0,
                rasterized=True
            )

            ax.set_title(feature_names[i], fontsize=10, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=9)

        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# In[146]:


median_feature_GCS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "X_tst_data_global_GCS_median.pkl"
save_pickle(X_tst_data_global_GCS_median, median_feature_GCS_savepath)

Y_raw_GCS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Yraw_tst_global_GCS.pkl"
save_pickle(Yraw_tst_global_GCS, Y_raw_GCS_savepath)

pacmap_GCS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "GCS_pacmap_data.pkl"
save_pickle(GCS_pacmap_data, pacmap_GCS_savepath)

NESI_GCS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Y_tst_NESI_GCS.pkl"
save_pickle(Y_tst_NESI_GCS, NESI_GCS_savepath)




# In[141]:


X_tst_data_global_GCS_median = np.median(
    X_tst_data_global_GCS,
    axis=1
)

feature_names = [
    'Awake','N1','N2',
    'Normal/Abnormal',
    'Burst/No Burst',
    'No Spike','Focal Spike','Generalized Spike',
    'No Slowing','Focal Slowing','Generalized Slowing',
    'Other','Seizure','LPD','GPD','LRDA','GRDA'
]

fig_GCS = plot_gcs_and_feature_pacmaps(
    emb=GCS_pacmap_data,
    y_raw=Yraw_tst_global_GCS,
    X_features=X_tst_data_global_GCS_median,
    feature_names=feature_names,
    Y_tst_NESI_GCS=Y_tst_NESI_GCS,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=300
)

savefig_path_GCS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "GCS_new_pacmap.png"
fig_GCS.savefig(savefig_path_GCS, dpi=600, bbox_inches="tight")
plt.show()


# # **CAMS PacMAPs**

# In[119]:


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




# In[125]:


# ------------------- CAMS - Dataset --------------------------
metadata_path_CAMS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "CAMS_metadata_with_NESI.csv"
df_CAMS_metadata = pd.read_csv(metadata_path_CAMS)


#-------------- CAMS dataset -----------------------
CAMS_model_path = CAMS_ROOT / "cohort_models" / "CAMS" / "CAMS_Best_DL_model" / "ResNetGAP" / "RESNETGAP_Best_CAMS.pth" 
CAMS_model_trained = Score_model_load(CAMS_model_path, 3)
df_test_CAMS  = df_CAMS_metadata

X_tst_data_global_CAMS, Ygrp_tst_global_CAMS, Yraw_tst_global_CAMS, _, tst_dataset_names_global, _, Y_tst_NESI_CAMS = morgoth_10minfea_matrix(df_test_CAMS)

CAMS_test_preds, CAMS_test_latents = get_corn_model_output(CAMS_model_trained, X_tst_data_global_CAMS)  
CAMS_pacmap_data = pacmap_embeddings(CAMS_test_latents)

print("\nCAMS CORN model GAP latent shape:", CAMS_test_latents.shape)
print("\nCAMS CORN model PACMAP shape:", CAMS_pacmap_data.shape)
print("True labels shape:", np.array(Yraw_tst_global_CAMS).shape)
print("NESI shape:", np.array(Y_tst_NESI_CAMS).shape)


plot_cams_pacmap_fixed_colors(
    emb=CAMS_pacmap_data,
    y_raw=Yraw_tst_global_CAMS,
    title="PACMAP Embedding",
    save_path=None,
    figsz=(7,6),
    dpi=210
)


# In[126]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import os

def plot_cams_and_feature_pacmaps(
    emb,
    y_raw,
    X_features,
    feature_names,
    Y_tst_NESI_CAMS=None,
    plot_nesi=False,
    figsize=(24, 20),
    dpi=300
):

    emb = np.asarray(emb)
    y_raw = np.asarray(y_raw)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[10, 6, 6, 6],
        hspace=0.25,
        wspace=0.10
    )

    #####################################################################
    # TOP: CAM-S MAIN PLOT (UNCHANGED STYLE)
    #####################################################################
    ax_main = fig.add_subplot(gs[0, 1:5])

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

    legend_elements = []

    for raw_val in range(8):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax_main.scatter(
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

    ax_main.set_xlabel("Dimension 1", fontweight='normal')
    ax_main.set_ylabel("Dimension 2", fontweight='normal')

    for spine in ['top', 'right']:
        ax_main.spines[spine].set_visible(False)

    ax_main.spines['left'].set_edgecolor('black')
    ax_main.spines['left'].set_linewidth(1.0)
    ax_main.spines['bottom'].set_edgecolor('black')
    ax_main.spines['bottom'].set_linewidth(1.0)

    ax_main.grid(True, linestyle='--', alpha=0.3)

    ax_main.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="CAM-S scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=4
    )

    #####################################################################
    # FEATURE GRID (3 × 6 = 18 SLOTS)
    #####################################################################

    cmap_all = 'Blues'

    for i in range(18):

        row = 1 + (i // 6)
        col = i % 6

        ax = fig.add_subplot(gs[row, col])

        # -----------------------------
        # NESI SLOT
        # -----------------------------
        if i == 17:

            if plot_nesi and Y_tst_NESI_CAMS is not None:

                sc = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=Y_tst_NESI_CAMS,
                    cmap=cmap_all,
                    s=4,
                    alpha=1.0,
                    rasterized=True
                )

                ax.set_title("NESI", fontsize=10, fontweight='bold')

                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=9)

            else:
                ax.axis('off')

        # -----------------------------
        # FEATURE PANELS
        # -----------------------------
        elif i < X_features.shape[1]:

            vals = X_features[:, i]

            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=vals,
                cmap=cmap_all,
                s=4,
                alpha=1.0,
                rasterized=True
            )

            ax.set_title(feature_names[i], fontsize=10, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=9)

        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# In[147]:


median_feature_CAMS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "X_tst_data_global_CAMS_median.pkl"
save_pickle(X_tst_data_global_CAMS_median, median_feature_CAMS_savepath)

Y_raw_CAMS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Yraw_tst_global_CAMS.pkl"
save_pickle(Yraw_tst_global_CAMS, Y_raw_CAMS_savepath)

pacmap_CAMS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "CAMS_pacmap_data.pkl"
save_pickle(CAMS_pacmap_data, pacmap_CAMS_savepath)

NESI_CAMS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Y_tst_NESI_CAMS.pkl"
save_pickle(Y_tst_NESI_CAMS, NESI_CAMS_savepath)




# In[142]:


X_tst_data_global_CAMS_median = np.median(
    X_tst_data_global_CAMS,
    axis=1
)

feature_names = [
    'Awake','N1','N2',
    'Normal/Abnormal',
    'Burst/No Burst',
    'No Spike','Focal Spike','Generalized Spike',
    'No Slowing','Focal Slowing','Generalized Slowing',
    'Other','Seizure','LPD','GPD','LRDA','GRDA'
]

fig_CAMS = plot_cams_and_feature_pacmaps(
    emb=CAMS_pacmap_data,
    y_raw=Yraw_tst_global_CAMS,
    X_features=X_tst_data_global_CAMS_median,
    feature_names=feature_names,
    Y_tst_NESI_CAMS=Y_tst_NESI_CAMS,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=300
)
savefig_path_CAMS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "CAMS_new_pacmap.png"
fig_CAMS.savefig(savefig_path_CAMS, dpi=600, bbox_inches="tight")
plt.show()


# # **ICANS PacMAPs**

# In[134]:


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


# In[135]:


# ------------------- ICANS - Dataset --------------------------
metadata_path_ICANS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "ICANS_metadata_with_NESI.csv"
df_ICANS_metadata = pd.read_csv(metadata_path_ICANS)


#-------------- ICANS dataset -----------------------
ICANS_model_path = ICANS_ROOT / "cohort_models" / "ICANS" / "ICANS_Best_DL_model" / "ResNetGAP" / "RESNETGAP_Best_ICANS.pth" 
ICANS_model_trained = Score_model_load(ICANS_model_path, 3)
df_test_ICANS  = df_ICANS_metadata

X_tst_data_global_ICANS, Ygrp_tst_global_ICANS, Yraw_tst_global_ICANS, _, tst_dataset_names_global, _, Y_tst_NESI_ICANS = morgoth_10minfea_matrix(df_test_ICANS)

ICANS_test_preds, ICANS_test_latents = get_corn_model_output(ICANS_model_trained, X_tst_data_global_ICANS)  
ICANS_pacmap_data = pacmap_embeddings(ICANS_test_latents)

print("\nICANS CORN model GAP latent shape:", ICANS_test_latents.shape)
print("\nICANS CORN model PACMAP shape:", ICANS_pacmap_data.shape)
print("True labels shape:", np.array(Yraw_tst_global_ICANS).shape)
print("NESI shape:", np.array(Y_tst_NESI_ICANS).shape)

plot_ICANS_pacmap_fixed_colors(
    emb=ICANS_pacmap_data,
    y_raw=Yraw_tst_global_ICANS,
    title="ICANS cohort PACMAP embedding",
    save_path=None,
    figsz=(7,6),
    dpi=210
)


# In[136]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import os

def plot_ICANS_and_feature_pacmaps(
    emb,
    y_raw,
    X_features,
    feature_names,
    Y_tst_NESI_ICANS=None,
    plot_nesi=False,
    figsize=(24, 20),
    dpi=300
):

    emb = np.asarray(emb)
    y_raw = np.asarray(y_raw)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[10, 6, 6, 6],
        hspace=0.25,
        wspace=0.10
    )

    #####################################################################
    # TOP: ICANS MAIN PLOT (UNCHANGED STYLE)
    #####################################################################
    ax_main = fig.add_subplot(gs[0, 1:5])

    group_info = {
        0: ('limegreen', 'darkgreen', 'No/Mild'),
        1: ('gold', 'darkgoldenrod', 'Moderate'),
        2: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        3: ('salmon', 'darkred', 'Severe'),
        4: ('red', 'darkred', 'Severe')
    }

    legend_elements = []

    for raw_val in range(5):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax_main.scatter(
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

    # -------------------------
    # EXACT ICANS LEGEND STYLE
    # -------------------------
    def get_handle(label):
        for h in legend_elements:
            if h.get_label() == label:
                return h
        return None

    ordered_labels = [
        "ICANS 0 (No/Mild)", None, None,
        "ICANS 1 (Moderate)", "ICANS 2 (Moderate)", None,
        "ICANS 3 (Severe)", "ICANS 4 (Severe)", None
    ]

    ordered_handles = [
        get_handle(l) if l is not None else Line2D([], [], alpha=0)
        for l in ordered_labels
    ]

    ax_main.legend(
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

    ax_main.set_xlabel("Dimension 1", fontweight='bold')
    ax_main.set_ylabel("Dimension 2", fontweight='bold')

    for spine in ['top', 'right']:
        ax_main.spines[spine].set_visible(False)

    ax_main.spines['left'].set_edgecolor('black')
    ax_main.spines['left'].set_linewidth(1.0)
    ax_main.spines['bottom'].set_edgecolor('black')
    ax_main.spines['bottom'].set_linewidth(1.0)

    ax_main.grid(True, linestyle='--', alpha=0.3)

    #####################################################################
    # FEATURE GRID (3 × 6 = 18 SLOTS)
    #####################################################################

    cmap_all = 'Blues'

    for i in range(18):

        row = 1 + (i // 6)
        col = i % 6

        ax = fig.add_subplot(gs[row, col])

        # -------------------------
        # NESI SLOT
        # -------------------------
        if i == 17:

            if plot_nesi and Y_tst_NESI_ICANS is not None:

                sc = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=Y_tst_NESI_ICANS,
                    cmap=cmap_all,
                    s=4,
                    alpha=1.0,
                    rasterized=True
                )

                ax.set_title("NESI", fontsize=10, fontweight='bold')

                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=9)

            else:
                ax.axis('off')

        # -------------------------
        # FEATURES
        # -------------------------
        elif i < X_features.shape[1]:

            vals = X_features[:, i]

            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=vals,
                cmap=cmap_all,
                s=4,
                alpha=1.0,
                rasterized=True
            )

            ax.set_title(feature_names[i], fontsize=10, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=9)

        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# In[148]:


median_feature_ICANS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "X_tst_data_global_ICANS_median.pkl"
save_pickle(X_tst_data_global_ICANS_median, median_feature_ICANS_savepath)

Y_raw_ICANS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Yraw_tst_global_ICANS.pkl"
save_pickle(Yraw_tst_global_ICANS, Y_raw_ICANS_savepath)

pacmap_ICANS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "ICANS_pacmap_data.pkl"
save_pickle(ICANS_pacmap_data, pacmap_ICANS_savepath)

NESI_ICANS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Y_tst_NESI_ICANS.pkl"
save_pickle(Y_tst_NESI_ICANS, NESI_ICANS_savepath)




# In[143]:


X_tst_data_global_ICANS_median = np.median(
    X_tst_data_global_ICANS,
    axis=1
)

feature_names = [
    'Awake','N1','N2',
    'Normal/Abnormal',
    'Burst/No Burst',
    'No Spike','Focal Spike','Generalized Spike',
    'No Slowing','Focal Slowing','Generalized Slowing',
    'Other','Seizure','LPD','GPD','LRDA','GRDA'
]

fig_ICANS = plot_ICANS_and_feature_pacmaps(
    emb=ICANS_pacmap_data,
    y_raw=Yraw_tst_global_ICANS,
    X_features=X_tst_data_global_ICANS_median,
    feature_names=feature_names,
    Y_tst_NESI_ICANS=Y_tst_NESI_ICANS,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=300
)
savefig_path_ICANS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "ICANS_new_pacmap.png"
fig_ICANS.savefig(savefig_path_ICANS, dpi=600, bbox_inches="tight")
plt.show()


# In[ ]:





# In[ ]:




