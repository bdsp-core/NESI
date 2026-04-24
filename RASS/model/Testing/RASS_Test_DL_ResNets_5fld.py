#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
from torchsummary import summary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.layers import CoralLayer
from datetime import datetime
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import h5py
import hdf5storage
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import seaborn as sns
plt.rcParams.update({
    'font.size': 9,
    'font.weight': 'bold',
    'font.family': 'serif'
})
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)
from tqdm import tqdm
import pickle

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


# # **Helper functions**

# ## **Model classes**

# In[2]:


#-------------------------- ResNet-GAP-BiLSTM -------------------------------------
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

class MORGOTH_ResNet1D_GAP_BiLSTM_CORAL(nn.Module):
    def __init__(self, num_features, num_classes, filters=None, use_logit=True, 
                 lstm_hidden=128, lstm_layers=1, bidirectional=False):
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

        # -------------------------
        #       LSTM layer
        # -------------------------
        lstm_input = filters[-1]
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Dense before CORAL
        self.fc1 = nn.Linear(lstm_out_dim, 128)
        self.dropout = nn.Dropout(0.5)

        # CORAL layer → output K−1
        self.fc2 = nn.Linear(128, num_classes - 1)

    def forward(self, x):
        if self.use_logit:
            eps = 1e-6
            x = torch.log((x + eps) / (1 - x + eps))

        x = x.permute(0, 2, 1)

        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.resnet_layers(x)

        # GAP: (B,C,1)
        x = self.gap(x)

        # Prepare for LSTM: (B,1,C)
        x = x.permute(0, 2, 1)

        # LSTM returns (B,1,H)
        out_lstm, _ = self.lstm(x)
        x = out_lstm[:, -1, :]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)          # CORAL output
        return x

# --------- NEW SUGGESTIONS FROM Brandon -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- ResNet Block -----------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
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
        return F.relu(out + identity)

# ---------------- ResNet-LSTM  with or without Time shuffle BiLSTM-----------------
class MORGOTH_ResNet1D_BiLSTM_TimeShuffle(nn.Module):
    def __init__(self, num_features, num_classes, filters=None, use_logit=True, 
                 lstm_hidden=128, lstm_layers=1, bidirectional=False, time_shuffle=False):
        super().__init__()
        self.use_logit = use_logit
        self.time_shuffle = time_shuffle
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

        # LSTM layer
        lstm_input = filters[-1]
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Dense layers before CORAL
        self.fc1 = nn.Linear(lstm_out_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes - 1)

    def forward(self, x):
        # Optional logit transform
        if self.use_logit:
            eps = 1e-6
            x = torch.log((x + eps) / (1 - x + eps))

        # Permute to (batch, features, time) for Conv1d
        x = x.permute(0, 2, 1)

        # Conv + ResNet
        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.resnet_layers(x)  # (B, C, T)

        # Permute to (batch, time, features) for LSTM
        x = x.permute(0, 2, 1)  # (B, T, C)

        # Time-shuffle ablation (if enabled)
        if self.time_shuffle:
            idx = torch.randperm(x.size(1))
            x = x[:, idx, :]  # shuffle time steps per batch

        # LSTM
        out_lstm, _ = self.lstm(x)
        x = out_lstm[:, -1, :]  # take last hidden state

        # Dense + CORAL
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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


# # **Feature engineeirng**

# In[3]:
current = Path(__file__).resolve()
RASS_ROOT = None
for parent in current.parents:
    if parent.name == "RASS":
        RASS_ROOT = parent
        break

if RASS_ROOT is None:
    raise RuntimeError("RASS folder not found")

# ------------------------------- EEG feature statistics collection -------------------------
def morgoth_10minfea_matrix_stat_for(data_frame):
    BASE = RASS_ROOT / "MorgothActivations"

    slowing_folder_loc = BASE / "SLOWING"
    focgen_folder_loc  = BASE / "FOCGEN"
    iiic_folder_loc    = BASE / "IIIC"
    nm_folder_loc      = BASE / "NM"
    bs_folder_loc      = BASE / "BS"
    sleep_folder_loc   = BASE / "SLEEP"

    filenames = data_frame['Filename'].tolist()
    all_subject_features = []; all_file_names =[];
    for file_name in tqdm(filenames, desc="Processing subjects"):
        file_name_csv = file_name + '.csv'

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
        
    
        all_subject_features.append(sub_morgoth_fea)
        all_file_names.append(file_name)

    all_subject_features_matrix = np.stack(all_subject_features, axis=0)
    train_labels=data_frame['RASS_grouped'].values .flatten() 
    all_file_names=np.array(all_file_names)
    
    print('Label size==> ',train_labels.shape)
    print("Stacked feature matrix shape:", all_subject_features_matrix.shape)
    return all_subject_features_matrix, train_labels, all_file_names

# ------------------------- Torch Dataset Creation -----------------------------------------
def create_datasets(X_tst, Y_tst):    
    X_tst = torch.tensor(X_tst, dtype=torch.float32) 
    Y_tst = torch.tensor(Y_tst, dtype=torch.float32) 

    test_ds= TensorDataset(X_tst, Y_tst)
    return test_ds
    
def create_loaders(test_ds, bs, jobs=2):
    test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=jobs)
    return test_dl


# # **K-fold crossvalidation**

# In[4]:


import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def RESNet_kfold_stratified_group(df_RASS_metadata, base_model_dir , dl_model, n_splits=5, bs=32):
    """
    Stratified group k-fold cross-validation (~15% test per fold) 
    ensuring no subject (BDSPPatientID) appears in both train and test.
    
    Args:
        df_RASS_metadata: DataFrame with at least ['BDSPPatientID', 'RASS_grouped']
        n_splits: number of folds (default 7 → ~15% test per fold)
        bs: batch size
    Returns:
        fold_results: list of dicts containing predictions, logits, metrics per fold
    """
    
    print(base_model_dir)
    
    if os.path.exists(base_model_dir):
        print("Path exists")
    else:
        print("Path does NOT exist")
    
    fold_results = []
    
    # Unique subjects and labels
    df_subjects = df_RASS_metadata[['BDSPPatientID','RASS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['RASS_grouped'].values
    groups = df_subjects['BDSPPatientID'].values  # important to prevent overlap
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(unique_studies, labels, groups=groups)):
        print(f"{'='*30} Fold {fold+1} {'='*30}")
        
        # Get train/test subjects
        train_subs = unique_studies[train_idx]
        test_subs = unique_studies[test_idx]
        
        # Check for overlap
        overlap = set(train_subs).intersection(set(test_subs))
        if len(overlap) == 0:
            print("✅ No overlap between train and test subjects")
        else:
            print(f"❌ Overlap detected: {overlap}")
        
        
        # Assign split in metadata
        df_RASS_metadata['Split'] = df_RASS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        df_RASS_test  = df_RASS_metadata[df_RASS_metadata['Split']=='Test'].reset_index(drop=True)
        
        # -------- FEATURES --------
        X_test_fea, Y_test, TestFileNames   = morgoth_10minfea_matrix_stat_for(df_RASS_test)
        
        # -------- DATA LOADERS --------
        print("Preparing torch datasets...")
        tst_ds = create_datasets(X_test_fea, Y_test)
        tst_dl = create_loaders(tst_ds, bs)
        
        # -------- LOAD BEST MODEL --------
        model = dl_model.to(device)
        model_prefix = f'RESNETBILSTM_fold_{fold}'
        model_path = f"{base_model_dir}/{model_prefix}_best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        # -------- INFERENCE --------
        Y_true, Y_pred  = [], []
        logits_list, out_all = [], []
        
        with torch.no_grad():
            for x_tst, y_tst in tst_dl:
                x_tst = x_tst.to(device)
                out = model(x_tst)
                preds = corn_label_from_logits(out).float()
                
                Y_pred.extend(preds.cpu().numpy())
                Y_true.extend(y_tst.numpy())                
                
                logits_list.extend(out.cpu().numpy())
                out_all.append(out.cpu().numpy())
        
        # -------- FORMAT OUTPUT --------
        Y_true = np.array(Y_true)
        Y_pred = np.array(Y_pred)
        
        logits_list = np.array(logits_list)
        out_all = np.concatenate(out_all, axis=0)
        
        # -------- METRICS --------
        acc  = accuracy_score(Y_test, Y_pred)
        one_off_acc = np.mean(np.abs(Y_test - Y_pred) <= 1)        
        pre_macro = precision_score(Y_test, Y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(Y_test, Y_pred, average='macro', zero_division=0)
        f1_macro  = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
        
        pre_micro = precision_score(Y_test, Y_pred, average='micro', zero_division=0)
        rec_micro = recall_score(Y_test, Y_pred, average='micro', zero_division=0)
        f1_micro  = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
        
        fold_results.append({
            "fold": fold,
            "Y_true": Y_true,
            "Y_pred": Y_pred,
            "Test_filenames": TestFileNames,
            "logits": logits_list,
            "out": out_all,
            'accuracy': acc,
            '1-level difference accuracy': one_off_acc,
            'precision_macro': pre_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_micro': pre_micro,
            'recall_micro': rec_micro,
            'f1_micro': f1_micro,
        })
        
        # Print metrics
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro -> Precision: {pre_macro:.4f}, Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}")
        print(f"Micro -> Precision: {pre_micro:.4f}, Recall: {rec_micro:.4f}, F1: {f1_micro:.4f}")
        print("="*100)
    
    return fold_results


# # **RASS Metadata**

# In[5]:


metadata_path_RASS = RASS_ROOT / "model" / "Training" / "RASSTraining_Final_Metadata.csv"

df_RASS_metadata=pd.read_csv(metadata_path_RASS)
df_RASS_metadata['Filename'] = df_RASS_metadata['Filename'].str.rstrip("'")
df_RASS_metadata = df_RASS_metadata[
    ~df_RASS_metadata['RASS_value'].isin([1, 2, 3, 4])
]

df_RASS_metadata['RASS_grouped'] = df_RASS_metadata['RASS_value'] + 5
def extract_pid(filename):
    part = filename.split('_')[0]          
    pid_full = part.split('-')[1]          
    pid = pid_full[5:]                     
    return pid

# Apply to all rows and create new column for BDSPPatientID
df_RASS_metadata['BDSPPatientID'] = df_RASS_metadata['Filename'].apply(extract_pid)
df_RASS_metadata = df_RASS_metadata[['BDSPPatientID', 'Filename', 'RASS_value', 'RASS_grouped']]

# Clean minimal style
sns.set_style("white", {'axes.grid': False})
fig, axes = plt.subplots(1, 1, figsize=(14, 6))

# Labels
group_labels = {0: 'RASS -5', 1: 'RASS -4', 2: 'RASS -3', 3: 'RASS -2', 3: 'RASS -1', 4: 'RASS 0'}

# ----------- Plot 1: RASS_value -----------
sns.histplot(
    data=df_RASS_metadata,
    x='RASS_value',
    bins=range(-5, 2),
    color='lightgray',
    edgecolor='black',
    alpha=1,
    ax=axes,
    discrete=True,
    shrink=0.8
)

axes.set_xticks(range(-5, 1))
axes.set_title('RASS Value Distribution', fontsize=15, fontweight='bold', pad=15)
axes.set_xlabel('RASS Value (Score)', fontsize=12)
axes.set_ylabel('Number of 10 min EEG Segments', fontsize=12)
sns.despine(ax=axes)

# Annotate
for p in axes.patches:
    if p.get_height() > 0:
        axes.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            fontsize=10,
            xytext=(0, 6),
            textcoords='offset points'
        )


# # **ResNet+No Time shuffled BILSTM**

# In[6]:


model_normal = MORGOTH_ResNet1D_BiLSTM_TimeShuffle(num_features=17, 
                                                   num_classes=df_RASS_metadata['RASS_value'].nunique(), 
                                                   time_shuffle=False)
base_model_dir = RASS_ROOT / "model" / "ModelCheckpoints" / "Ordinal" / "RESNET_notimeshuffle"

RESNet_results_notimeshuffle = RESNet_kfold_stratified_group(df_RASS_metadata, base_model_dir, model_normal,
                                                        n_splits=5, bs=128)


# # **ResNet+Time shuffled BiLSTM**

# In[7]:


# Time-shuffled model
model_shuffle = MORGOTH_ResNet1D_BiLSTM_TimeShuffle(num_features=17, 
                                                    num_classes=df_RASS_metadata['RASS_value'].nunique(), 
                                                    time_shuffle=True)
base_model_dir = RASS_ROOT / "model" / "ModelCheckpoints" / "Ordinal" / "RESNET_TIMESHIFT"

RESNet_results_withtimeshuffle = RESNet_kfold_stratified_group(df_RASS_metadata, base_model_dir, model_shuffle, 
                                                        n_splits=5, bs=128)


# # **ResNet+GAP**

# In[8]:


# -------------------------------RESNET only GAP----------------------
model_onlygap = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, 
                                               num_classes=df_RASS_metadata['RASS_value'].nunique())

base_model_dir = RASS_ROOT / "model" / "ModelCheckpoints" / "Ordinal" / "RESNET_GAPonly"

RESNet_results_gaponly = RESNet_kfold_stratified_group(df_RASS_metadata, base_model_dir, model_onlygap, 
                                                        n_splits=5, bs=32)


# # **Save Results**

# In[9]:


import pickle
import os

def save_to_pickle(obj, filepath, overwrite=True):
    """
    Save any Python object to a pickle file.

    Args:
        obj: Python object to save
        filepath: Full path to save the pickle file
        overwrite: If False, prevents overwriting existing file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Check overwrite condition
    if not overwrite and os.path.exists(filepath):
        raise FileExistsError(f"{filepath} already exists and overwrite=False")

    # Save
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    print(f"Saved successfully to: {filepath}")


# In[10]:


# save_to_pickle(
#     RESNet_results_gaponly,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/Ordinal/ResNetGAP_5fldResults_6classRASS.pkl'
# )


# In[11]:


# save_to_pickle(
#     RESNet_results_notimeshuffle,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/Ordinal/ResNetBiLSTM_NoShuffle_5fldResults_6classRASS.pkl'
# )


# In[12]:


# save_to_pickle(
#     RESNet_results_withtimeshuffle,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/Ordinal/ReSNetBiLSTM_WithShuffle_5fldResults_6classRASS.pkl'
# )