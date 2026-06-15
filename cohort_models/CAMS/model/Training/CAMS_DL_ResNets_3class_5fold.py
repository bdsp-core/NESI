#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# ## **Model functions**

# In[49]:


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


# ## **Feature Engineering**

# In[50]:

current = Path(__file__).resolve()

CAMS_ROOT = None
for parent in current.parents:
    if parent.name == "CAMS":
        CAMS_ROOT = parent
        break

if CAMS_ROOT is None:
    raise RuntimeError("CAMS folder not found")


metadata_path = CAMS_ROOT / "model" / "Training" / "CAMSTraining_Final_Metadata.csv"

df_CAMS_metadata = pd.read_csv(metadata_path)

# ------------------------------- EEG feature statistics collection -------------------------
def morgoth_10minfea_matrix_stat_for(data_frame):
    filenames = data_frame['MorgothOutputFilename'].tolist()
    all_subject_features = []; all_file_names =[];

    BASE = CAMS_ROOT / "MorgothActivations"

    slowing_folder_loc = BASE / "SLOWING"
    focgen_folder_loc  = BASE / "FOCGEN"
    iiic_folder_loc    = BASE / "IIIC"
    nm_folder_loc      = BASE / "NM"
    bs_folder_loc      = BASE / "BS"
    sleep_folder_loc   = BASE / "SLEEP"
    
    for file_name in tqdm(filenames, desc="Processing subjects"):
        
        file_name_csv = file_name

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
        
        # if statistic == 'mean':
        #     stat_10min_fea = np.mean(sub_morgoth_fea, axis=0) 
        # elif statistic == 'median':
        #     stat_10min_fea = np.median(sub_morgoth_fea, axis=0)

        all_subject_features.append(sub_morgoth_fea)
        all_file_names.append(file_name)

    all_subject_features_matrix = np.stack(all_subject_features, axis=0)
    train_labels=data_frame['CAMS_grouped'].values .flatten()  
    print('Label size==> ',train_labels.shape)
    print("Stacked feature matrix shape:", all_subject_features_matrix.shape)
    return all_subject_features_matrix, train_labels


# ------------------------- Torch Dataset Creation -----------------------------------------
def create_datasets(X_tr, Y_tr, X_tst, Y_tst):
    
    X_tr, X_tst = [torch.tensor(arr, dtype=torch.float32) for arr in (X_tr, X_tst)]
    y_tr, y_tst = [torch.tensor(arr, dtype=torch.float32) for arr in (Y_tr, Y_tst)]

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds= TensorDataset(X_tst, y_tst)
    return train_ds, test_ds

def create_loaders(train_ds, test_ds, bs, jobs=2):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, test_dl


# ## **Training function**

# In[51]:


# -------------------------- Model training function --------------------------------------------
import torch
import numpy as np
import copy

def Training_function(model, lr, n_epochs, train_dl, valid_dl, save_loc, Y_test, model_prefix):

    # Early stopping parameters
    patience = 20
    best_val_acc = 0.0
    epochs_no_improve = 0

    NUM_CLASSES = len(np.unique(Y_test))

    trainloss_history = []
    valacc_history = []
    valloss_history = []
    trainacc_history = []

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Store best weights
    best_model_wts = copy.deepcopy(model.state_dict())

    print('Start model training')

    for epoch in range(1, n_epochs + 1):

        # -------- TRAIN --------
        model.train()
        correct, total = 0, 0
        running_loss = 0.0

        for x_tr_batch, y_tr_batch in train_dl:
            x_tr_batch = x_tr_batch.cuda()
            y_tr_batch = y_tr_batch.cuda()

            out_tr = model(x_tr_batch)
            loss = corn_loss(out_tr, y_tr_batch, NUM_CLASSES)

            preds_tr = corn_label_from_logits(out_tr).float()

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            total += y_tr_batch.size(0)
            correct += (preds_tr == y_tr_batch).sum().item()

        train_loss = running_loss / len(train_dl)
        train_acc = correct / total

        trainloss_history.append(train_loss)
        trainacc_history.append(train_acc)

        # -------- VALIDATION --------
        model.eval()
        correct, total = 0, 0
        running_val_loss = 0.0

        with torch.no_grad():
            for x_val_batch, y_val_batch in valid_dl:
                x_val_batch = x_val_batch.cuda()
                y_val_batch = y_val_batch.cuda()

                out_vl = model(x_val_batch)
                valid_loss = corn_loss(out_vl, y_val_batch, NUM_CLASSES)

                preds_vl = corn_label_from_logits(out_vl).float()

                running_val_loss += valid_loss.item()
                total += y_val_batch.size(0)
                correct += (preds_vl == y_val_batch).sum().item()

        valid_loss = running_val_loss / len(valid_dl)
        valid_acc = correct / total

        valloss_history.append(valid_loss)
        valacc_history.append(valid_acc)

        print(f'Epoch: {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {valid_acc:.2%}')

        # -------- EARLY STOPPING --------
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            epochs_no_improve = 0

            # Save best weights in memory
            best_model_wts = copy.deepcopy(model.state_dict())

            print(f'New best model found at epoch {epoch} with val acc: {valid_acc:.4f}')

        else:
            epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break

    # -------- SAVE ONLY BEST MODEL --------
    save_path = f"{save_loc.rstrip('/')}/{model_prefix}_best_model.pth"
    torch.save(best_model_wts, save_path)

    print(f'Best model saved at: {save_path}')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    print("-" * 20)


# ## **K-fold cross validation function**

# In[52]:


import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def RESNet_kfold_stratified_group(df_CAMS_metadata, base_model_dir , dl_model, n_splits=5, bs=32):
    """
    Stratified group k-fold cross-validation (~15% test per fold) 
    ensuring no subject (BDSPPatientID) appears in both train and test.
    
    Args:
        df_CAMS_metadata: DataFrame with at least ['BDSPPatientID', 'CAMS_SF']
        n_splits: number of folds (default 7 → ~15% test per fold)
        bs: batch size
    Returns:
        fold_results: list of dicts containing predictions, logits, metrics per fold
    """
    
    print(base_model_dir)
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    # Unique subjects and labels
    df_subjects = df_CAMS_metadata[['BDSPPatientID','CAMS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['CAMS_grouped'].values
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
        df_CAMS_metadata['Split'] = df_CAMS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        
        df_CAMS_train = df_CAMS_metadata[df_CAMS_metadata['Split']=='Train'].reset_index(drop=True)
        df_CAMS_test  = df_CAMS_metadata[df_CAMS_metadata['Split']=='Test'].reset_index(drop=True)
        
        # -------- FEATURES --------
        X_train_fea, Y_train = morgoth_10minfea_matrix_stat_for(df_CAMS_train)
        X_test_fea, Y_test   = morgoth_10minfea_matrix_stat_for(df_CAMS_test)
        
        # -------- DATA LOADERS --------
        print("Preparing torch datasets...")
        trn_ds, tst_ds = create_datasets(X_train_fea, Y_train, X_test_fea, Y_test)
        trn_dl, tst_dl = create_loaders(trn_ds, tst_ds, bs)
        
        # -------- MODEL INIT --------
        model = dl_model.to(device)
        
        model_prefix = f'RESNETBILSTM_fold_{fold}'
        
        # -------- TRAINING --------
        Training_function(
            model=model,
            lr=0.005,
            n_epochs=200,
            train_dl=trn_dl,
            valid_dl=tst_dl,
            save_loc=base_model_dir,
            Y_test=Y_test,
            model_prefix=model_prefix
        )
        
        # -------- LOAD BEST MODEL --------
        model_path = f"{base_model_dir}/{model_prefix}_best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # -------- INFERENCE --------
        Y_true, Y_pred = [], []
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
            "logits": logits_list,
            "out": out_all,
            'accuracy': acc,
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


# # **Load CAMS Training Metdata**

# In[6]:

current = Path(__file__).resolve()

CAMS_ROOT = None
for parent in current.parents:
    if parent.name == "CAMS":
        CAMS_ROOT = parent
        break

if CAMS_ROOT is None:
    raise RuntimeError("CAMS folder not found")


metadata_path = CAMS_ROOT / "model" / "Training" / "CAMSTraining_Final_Metadata.csv"

df_CAMS_metadata = pd.read_csv(metadata_path)
print('Total Unique subject ==> '+str(df_CAMS_metadata['BDSPPatientID'].nunique()))


def map_cams_to_bins(x):
    if 0<=x <=1:
        return 0  # No delirium
    elif 2 <= x <= 5:
        return 1  # Moderate
    else:
        return 2  # Severe

# Apply to create a new column
df_CAMS_metadata["CAMS_grouped"] = df_CAMS_metadata["CAMS_SF"].apply(map_cams_to_bins)

import matplotlib.pyplot as plt

# Figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# -------------------------
# Plot 1: Original CAMS_SF
# -------------------------
counts_s = df_CAMS_metadata['CAMS_SF'].value_counts().sort_index()
axes[0].bar(counts_s.index, counts_s.values, 
            color='lightgray', edgecolor='black', width=0.6)

# Add numbers above bars
for i, v in zip(counts_s.index, counts_s.values):
    axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom')

axes[0].set_xlabel('CAMS_SF Score')
axes[0].set_ylabel('Number of EEG segments')
axes[0].set_title('Original CAMS_SF Distribution')
axes[0].set_xticks(range(0, 8))

# -------------------------
# Plot 2: Grouped CAMS
# -------------------------
counts_g = df_CAMS_metadata['CAMS_grouped'].value_counts().sort_index()
axes[1].bar(counts_g.index, counts_g.values, 
            color='lightgray', edgecolor='black', width=0.6)

# Add numbers above bars
for i, v in zip(counts_g.index, counts_g.values):
    axes[1].text(i, v + 0.5, str(v), ha='center', va='bottom')

axes[1].set_xlabel('CAMS Grouped (0–3)')
axes[1].set_ylabel('Number of EEG segments')
axes[1].set_title('Grouped CAMS Distribution \n (0-1:No/Mild Delirium; 2-5: ModerateDelirium; 6-7: Severe Delirium)')
axes[1].set_xticks(range(0, 3))

plt.tight_layout()
plt.show()


# # **ResNet+No Time shuffled BILSTM**

# In[7]:

current = Path(__file__).resolve()

CAMS_ROOT = None
for parent in current.parents:
    if parent.name == "CAMS":
        CAMS_ROOT = parent
        break

if CAMS_ROOT is None:
    raise RuntimeError("CAMS folder not found")

model_normal = MORGOTH_ResNet1D_BiLSTM_TimeShuffle(num_features=17, 
                                                   num_classes=df_CAMS_metadata['CAMS_grouped'].nunique(), 
                                                   time_shuffle=False)
base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "RESNET_notimeshift"

RESNet_results_notimeshuffle = RESNet_kfold_stratified_group(df_CAMS_metadata, base_model_dir, model_normal,
                                                        n_splits=5, bs=32)


# # **ResNet+Time shuffled BILSTM**

# In[8]:


# Time-shuffled model
model_shuffle = MORGOTH_ResNet1D_BiLSTM_TimeShuffle(num_features=17, num_classes=df_CAMS_metadata['CAMS_grouped'].nunique(), time_shuffle=True)
base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "RESNET_TIMESHIFT"

RESNet_results_withtimeshuffle = RESNet_kfold_stratified_group(df_CAMS_metadata, base_model_dir, model_shuffle, 
                                                        n_splits=5, bs=32)


# # **ResNet with GAP only**

# In[9]:


# -------------------------------RESNET only GAP----------------------
model_onlygap = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, num_classes=df_CAMS_metadata['CAMS_grouped'].nunique())

base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "RESNET_GAPonly"

RESNet_results_gaponly = RESNet_kfold_stratified_group(df_CAMS_metadata, base_model_dir, model_onlygap, 
                                                        n_splits=5, bs=32)


# # **Save Results**

# In[10]:


# import pickle

# # Save to pickle file
# with open('/home/ayush/Desktop/CAM-S_dataset/RESULTS/RESNET_5foldOnlyGap_Results_3class.pkl', 'wb') as f:
#     pickle.dump(RESNet_results_gaponly, f)


# # In[11]:


# import pickle

# # Save to pickle file
# with open('/home/ayush/Desktop/CAM-S_dataset/RESULTS/RESNET_5foldNOSHUFFLE_Results_3class.pkl', 'wb') as f:
#     pickle.dump(RESNet_results_notimeshuffle, f)


# # In[14]:


# import pickle

# # Save to pickle file
# with open('/home/ayush/Desktop/CAM-S_dataset/RESULTS/RESNET_5foldTIMESHUFFLE_Results_3class.pkl', 'wb') as f:
#     pickle.dump(RESNet_results_withtimeshuffle, f)


