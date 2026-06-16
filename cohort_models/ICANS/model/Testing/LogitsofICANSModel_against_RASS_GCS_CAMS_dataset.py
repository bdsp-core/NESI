#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
from datetime import datetime
from pathlib import Path
import warnings

import h5py
import hdf5storage
import matplotlib.pyplot as plt
import matplotlib as mpl
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
        x = self.gap(x).squeeze(-1)

        x = self.dropout(x)
        x = self.fc2(x)          # CORAL output
        return x


# In[3]:


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


metadata_path_CAMS = NESI_ROOT / "NESI" / "model" / "Training" / "CAMSTraining_Final_Metadata.csv"
df_CAMS_metadata = pd.read_csv(metadata_path_CAMS)
df_CAMS_metadata["GroupedScore"] = df_CAMS_metadata["CAMS_SF"].apply(group_cams)
df_CAMS_metadata = df_CAMS_metadata.rename(columns={'CAMS_SF': 'RawScore'})
df_CAMS_metadata['Dataset'] = 'CAMS'
df_CAMS_metadata['TransformedScore'] = df_CAMS_metadata['RawScore']
df_CAMS_metadata = df_CAMS_metadata[['BDSPPatientID', 'Dataset', 'MorgothOutputFilename', 'RawScore', 'GroupedScore', 'TransformedScore']]

# ------------------- GCS - Dataset --------------------------
metadata_path_GCS = NESI_ROOT / "NESI" / "model" / "Training" / "GCSTraining_Final_Metadata.csv"
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
metadata_path_RASS = NESI_ROOT / "NESI" / "model" / "Training" / "RASSTraining_Final_Metadata.csv"
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


# In[5]:


# # **Feature Engineering**
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

print(ICANS_ROOT)

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

    print("Grouped Label shape:", Y_grouped.shape)
    print("Raw Label shape:", Y_raw.shape)
    print("Raw Label transformed shape:", Y_raw_transformed.shape)
    print("Feature matrix shape:", X.shape)
    
    return X, Y_grouped, Y_raw, Y_raw_transformed, dataset_names, file_names


# # **Helper functions**

# ## **Model loading helper**

# In[6]:


def Score_model_load(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model_trained = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, num_classes=3)
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
    # Collect CORN LOGITS (K-1 outputs)
    # ---------------------------------------------------
    test_CORN_logits = []   
    model_trained.eval()
    with torch.no_grad():
        for i in range(0, len(X_tst_data_global), batch_size):
            x_batch = X_tst_data_global[i:i+batch_size].to(device)
    
            out = model_trained(x_batch)   # (N, K-1) logits
    
            test_CORN_logits.extend(out.cpu().numpy())
    
    test_CORN_logits = np.array(test_CORN_logits)
    return test_CORN_logits


# ## **CORN logit decoding helper**

# In[7]:


# ---------------------------------------------------
# CORN decoding function (FIXED)
# ---------------------------------------------------
def corn_to_class_probs(corn_logits, num_classes: int):
    """
    CORN logits → class probabilities → class logits (log-odds)
    """

    if isinstance(corn_logits, np.ndarray):
        corn_logits = torch.from_numpy(corn_logits)

    corn_logits = corn_logits.float()

    if corn_logits.dim() == 1:
        corn_logits = corn_logits.unsqueeze(0)

    # ---------------------------------------------------
    # IMPORTANT: convert logits → P(y > k)
    # ---------------------------------------------------
    corn_probs = torch.sigmoid(corn_logits)

    N = corn_probs.size(0)
    device = corn_probs.device

    class_probs = torch.zeros((N, num_classes), device=device)

    # Class 0
    class_probs[:, 0] = 1 - corn_probs[:, 0]

    # Middle classes
    for k in range(1, num_classes - 1):
        class_probs[:, k] = corn_probs[:, k - 1] - corn_probs[:, k]

    # Last class
    class_probs[:, -1] = corn_probs[:, -1]

    # ---------------------------------------------------
    # stability clamp before logit transform
    # ---------------------------------------------------
    eps = 1e-7
    class_probs = torch.clamp(class_probs, eps, 1 - eps)

    class_logits = torch.log(class_probs / (1 - class_probs))

    return class_probs, class_logits

    
def clean_logit_outliers(logits, y_true, iqr_multiplier=3.0):
    """
    Removes rows where the logits are extreme outliers relative to their true class.
    
    Args:
        logits: (N, K) numpy array or torch.Tensor
        y_true: (N,) numpy array or torch.Tensor
        iqr_multiplier: float, standard is 1.5. Higher (e.g., 3.0) keeps more outliers.
        
    Returns:
        logits_cleaned: Filtered logits array
        y_true_cleaned: Filtered true labels array
    """
    # Ensure numpy arrays
    if hasattr(logits, 'detach'): logits = logits.detach().cpu().numpy()
    if hasattr(y_true, 'detach'): y_true = y_true.detach().cpu().numpy()
        
    classes = np.unique(y_true)
    num_features = logits.shape[1]
    
    # Start with a mask of all True (keep everything)
    keep_mask = np.ones(len(y_true), dtype=bool)
    
    # Calculate IQR threshold per class, per logit dimension
    for c in classes:
        class_mask = (y_true == c)
        
        for i in range(num_features):
            subset = logits[class_mask, i]
            
            q25, q75 = np.percentile(subset, [25, 75])
            iqr = q75 - q25
            
            lower_bound = q25 - (iqr_multiplier * iqr)
            upper_bound = q75 + (iqr_multiplier * iqr)
            
            # Find outliers within this class/dimension
            outlier_mask = (logits[:, i] < lower_bound) | (logits[:, i] > upper_bound)
            
            # If it's an outlier for this specific class group, mark it for removal
            keep_mask[class_mask & outlier_mask] = False
            
    print(np.sum(~keep_mask), "extreme outlier rows dropped out of", len(y_true))
    return logits[keep_mask], y_true[keep_mask]


# ## **Ploting helper**

# In[14]:


import os
import warnings
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_corn_logits_boxplot(
    logits,
    y_true,
    xlabel_title,       
    num_classes=None,   
    figsize=(16, 5),
    fontsize=12,
    dpi=300,
    save_path=None
):
    warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    y_true = np.array(y_true)

    if num_classes is None:
        num_classes = logits.shape[1]

    classes = np.sort(np.unique(y_true))

    ylabels = [
        "Logits for Mild \nICANS (ICANS 0)",
        "Logits for Moderate \nICANS (ICANS 1-2)",
        "Logits for Severe \nICANS (ICANS 3-4)"
    ]

    pastel_colors = [
        "#A8E6CF",  
        "#FFD3B6",  
        "#FFAAA5"   
    ]

    fig, axes = plt.subplots(1, num_classes, figsize=figsize, sharex=True, dpi=dpi)

    if num_classes == 1:
        axes = [axes]

    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })

    flierprops = dict(
        marker='o', 
        markersize=2, 
        markeredgecolor='gray', 
        alpha=0.3, 
        linestyle='none'
    )

    medianprops = dict(
        color='black', 
        linewidth=1
    )

    for i in range(num_classes):
        data = []
        for c in classes:
            class_data = logits[y_true == c, i]
            data.append(class_data[~np.isnan(class_data)])

        bp = axes[i].boxplot(
            data, 
            tick_labels=classes, 
            patch_artist=True, 
            showfliers=True, 
            flierprops=flierprops,
            medianprops=medianprops,
            widths=0.7
        )

        current_color = pastel_colors[i % len(pastel_colors)]
        for box in bp['boxes']:
            box.set(facecolor=current_color, edgecolor='black', linewidth=1)
            
        for whisker in bp['whiskers']:
            whisker.set(color='#333333', linewidth=1)
        for cap in bp['caps']:
            cap.set(color='#333333', linewidth=1)

        axes[i].set_xlabel(xlabel_title, labelpad=8, fontsize=fontsize)
        
        if i < len(ylabels):
            axes[i].set_ylabel(ylabels[i], labelpad=8, fontsize=fontsize)
        else:
            axes[i].set_ylabel("Logit Value", labelpad=8, fontsize=fontsize)
            
        axes[i].tick_params(axis='both', which='major', labelsize=fontsize)
        
        axes[i].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()


# # **Test on RASS dataset**

# In[13]:


ICANS_model_path = ICANS_ROOT / "cohort_models" / "ICANS" / "ICANS_Best_DL_model" / "ResNetGAP" / "RESNETGAP_Best_ICANS.pth" 
ICANS_model_trained = Score_model_load(ICANS_model_path)

print('+'*80)
print('        Working on RASS Dataset')
print('+'*80)
df_test_RASS  = get_test_metedata_cohort(df_RASS_metadata)

X_tst_data_global, _, Yraw_tst_global, _, tst_dataset_names_global, _ = morgoth_10minfea_matrix(df_test_RASS)

RASS_test_CORN_logits = get_corn_model_output(ICANS_model_trained, X_tst_data_global)    
print("\nCORN model logits shape:", RASS_test_CORN_logits.shape)
print("True labels shape:", np.array(Yraw_tst_global).shape)

RASS_data_ICANS3cls_probs, RASS_data_ICANS3cls_logits = \
    corn_to_class_probs(RASS_test_CORN_logits, num_classes=3)


print("\nClass probs shape:", RASS_data_ICANS3cls_probs.shape)
print("Class logits shape:", RASS_data_ICANS3cls_logits.shape)


# Clean the dataset (multiplier=3.0 keeps some outliers, drops the extreme ones)
RASS_logits_clean, RASS_y_true_clean = clean_logit_outliers(
    RASS_data_ICANS3cls_logits, 
    Yraw_tst_global, 
    iqr_multiplier=3.0
)

# Run using your cleaned subset data
plot_corn_logits_boxplot(
    RASS_logits_clean,
    RASS_y_true_clean,
    num_classes=3,
    xlabel_title='True RASS scale',
    figsize=(7,4),
    fontsize=8,
    dpi=210,
    save_path=None
)


# # **Test on GCS dataset**

# In[16]:
print('+'*80)
print('        Working on GCS Dataset')
print('+'*80)
df_test_GCS  = get_test_metedata_cohort(df_GCS_metadata)

X_tst_data_global, _, Yraw_tst_global, _, tst_dataset_names_global, _ = morgoth_10minfea_matrix(df_test_GCS)

GCS_test_CORN_logits = get_corn_model_output(ICANS_model_trained, X_tst_data_global)    
print("\nCORN model logits shape:", GCS_test_CORN_logits.shape)
print("True labels shape:", np.array(Yraw_tst_global).shape)

GCS_data_ICANS3cls_probs, GCS_data_ICANS3cls_logits = \
    corn_to_class_probs(GCS_test_CORN_logits, num_classes=3)


print("\nClass probs shape:", GCS_data_ICANS3cls_probs.shape)
print("Class logits shape:", GCS_data_ICANS3cls_logits.shape)


# Clean the dataset (multiplier=3.0 keeps some outliers, drops the extreme ones)
GCS_logits_clean, GCS_y_true_clean = clean_logit_outliers(
    GCS_data_ICANS3cls_logits, 
    Yraw_tst_global, 
    iqr_multiplier=3.0
)

# Run using your cleaned subset data
plot_corn_logits_boxplot(
    GCS_logits_clean,
    GCS_y_true_clean,
    num_classes=3,
    xlabel_title='True GCS scale',
    figsize=(7,4),
    fontsize=8,
    dpi=210,
    save_path=None
)


# ## **Test on CAMS dataset**

# In[21]:
print('+'*80)
print('        Working on CAMS Dataset')
print('+'*80)
df_test_CAMS  = get_test_metedata_cohort(df_CAMS_metadata)

X_tst_data_global, _, Yraw_tst_global, _, tst_dataset_names_global, _ = morgoth_10minfea_matrix(df_test_CAMS)

CAMS_test_CORN_logits = get_corn_model_output(ICANS_model_trained, X_tst_data_global)    
print("\nCORN model logits shape:", CAMS_test_CORN_logits.shape)
print("True labels shape:", np.array(Yraw_tst_global).shape)

CAMS_data_ICANS3cls_probs, CAMS_data_ICANS3cls_logits = \
    corn_to_class_probs(CAMS_test_CORN_logits, num_classes=3)


print("\nClass probs shape:", CAMS_data_ICANS3cls_probs.shape)
print("Class logits shape:", CAMS_data_ICANS3cls_logits.shape)


# Clean the dataset (multiplier=3.0 keeps some outliers, drops the extreme ones)
CAMS_logits_clean, CAMS_y_true_clean = clean_logit_outliers(
    CAMS_data_ICANS3cls_logits, 
    Yraw_tst_global, 
    iqr_multiplier=3.0
)

# Run using your cleaned subset data
plot_corn_logits_boxplot(
    CAMS_logits_clean,
    CAMS_y_true_clean,
    num_classes=3,
    xlabel_title='True CAMS scale',
    figsize=(7,4),
    fontsize=8,
    dpi=210,
    save_path=None
)


# # **Consolidate all result in a single figure**

# In[27]:
def plot_all_scales_boxplot_grid(
    data_dict,
    figsize=(16, 12),
    fontsize=11,
    dpi=300,
    save_path=None
):
    warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)

    ylabels = [
        "Logits for Mild \nICANS (ICANS 0)",
        "Logits for Moderate \nICANS (ICANS 1-2)",
        "Logits for Severe \nICANS (ICANS 3-4)"
    ]

    pastel_colors = [
        "#A8E6CF",  
        "#FFD3B6",  
        "#FFAAA5"   
    ]

    scales_order = ['RASS', 'GCS', 'CAMS']
    num_rows = 3
    num_cols = len(scales_order)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=dpi)

    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize
    })

    flierprops = dict(
        marker='o', 
        markersize=2, 
        markeredgecolor='gray', 
        alpha=0.3, 
        linestyle='none'
    )

    medianprops = dict(
        color='black', 
        linewidth=1
    )

    for col_idx, scale_name in enumerate(scales_order):
        logits = data_dict[scale_name]['logits']
        y_true = data_dict[scale_name]['y_true']
        xlabel_title = data_dict[scale_name]['xlabel']

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        y_true = np.array(y_true)

        classes = np.sort(np.unique(y_true))

        for row_idx in range(num_rows):
            ax = axes[row_idx, col_idx]
            
            data = []
            for c in classes:
                class_data = logits[y_true == c, row_idx]
                data.append(class_data[~np.isnan(class_data)])

            bp = ax.boxplot(
                data, 
                tick_labels=classes, 
                patch_artist=True, 
                showfliers=True, 
                flierprops=flierprops,
                medianprops=medianprops,
                widths=0.7
            )

            current_color = pastel_colors[row_idx]
            for box in bp['boxes']:
                box.set(facecolor=current_color, edgecolor='black', linewidth=0.6)
                
            for whisker in bp['whiskers']:
                whisker.set(color='#333333', linewidth=1)
            for cap in bp['caps']:
                cap.set(color='#333333', linewidth=1)

            if row_idx == 2:
                ax.set_xlabel(xlabel_title, labelpad=8, fontsize=fontsize)

            if col_idx == 0:
                ax.set_ylabel(ylabels[row_idx], labelpad=8, fontsize=fontsize)

            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.tick_params(axis='x', labelrotation=45)

            ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()


# In[28]:


dataset_payload = {
    'RASS': {
        'logits': RASS_logits_clean,
        'y_true': RASS_y_true_clean,
        'xlabel': 'True RASS scale'
    },
    'GCS': {
        'logits': GCS_logits_clean,
        'y_true': GCS_y_true_clean,
        'xlabel': 'True GCS scale'
    },
    'CAMS': {
        'logits': CAMS_logits_clean,
        'y_true': CAMS_y_true_clean,
        'xlabel': 'True CAMS scale'
    }
}

ICANS_model_test_saving_path = ICANS_ROOT / "cohort_models" / "ICANS" / "Figures" / "LogitsAgainstOtherScalesvsICANSModel.png"
plot_all_scales_boxplot_grid(
    dataset_payload,
    figsize=(7,5),
    fontsize=8,
    dpi=210,
    save_path=ICANS_model_test_saving_path
)


# In[ ]:




