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


# # **Load all cohort metadata**

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
print('\n')


# # **Feature Engineering**

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

print(ICANS_ROOT)

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

# X_train_data_global, Ygrouped_train_global, Yraw_train_global, Yraw_transformed_train_global, train_dataset_names_global,trn_filenames_global = morgoth_10minfea_matrix(df_train)
# X_val_data_global, Ygrouped_val_global, Yraw_val_global, Yraw_transformed_val_global, val_dataset_names_global, val_filenames_global = morgoth_10minfea_matrix(df_val)
X_tst_data_global, Ygrouped_tst_global, Yraw_tst_global, Yraw_transformed_tst_global, tst_dataset_names_global, tst_filenames_global = morgoth_10minfea_matrix(df_test)

# X_train_data_global_trch = torch.tensor(X_train_data_global, dtype=torch.float32)
# X_val_data_global_trch   = torch.tensor(X_val_data_global, dtype=torch.float32)
X_tst_data_global_trch   = torch.tensor(X_tst_data_global, dtype=torch.float32)

# Ygrouped_train_global_trch = torch.tensor(Ygrouped_train_global, dtype=torch.long)
# Ygrouped_val_global_trch   = torch.tensor(Ygrouped_val_global, dtype=torch.long)
Ygrouped_tst_global_trch   = torch.tensor(Ygrouped_tst_global, dtype=torch.long)

# Yraw_train_global_trch = torch.tensor(Yraw_train_global, dtype=torch.long)
# Yraw_val_global_trch   = torch.tensor(Yraw_val_global, dtype=torch.long)
Yraw_tst_global_trch   = torch.tensor(Yraw_tst_global, dtype=torch.long)

# train_dataset_names_global = np.array(train_dataset_names_global)
# val_dataset_names_global   = np.array(val_dataset_names_global)
tst_dataset_names_global   = np.array(tst_dataset_names_global)


# # **Triplet model & NESI model backbone for embedding extrcation**

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


# ------------ NESI model ----------------
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


# # **Get embeddings from triplet model**

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- LOAD MODEL ----------
Triplet_model_path = NESI_ROOT / "model" / "ModelCheckpoints" / "ResNetGAP_BestModel.pth" 
model_trained = MORGOTH_ResNet1D_onlyGAP(num_features=17)
model_trained.load_state_dict(
    torch.load(Triplet_model_path,
               map_location=device,
               weights_only=True)
)
model_trained = model_trained.to(device)
model_trained.eval()

# --------- CREATE LOADERS ----------
# train_ds_all_global = TensorDataset(X_train_data_global_trch)
# val_ds_all_global   = TensorDataset(X_val_data_global_trch)
tst_ds_all_global   = TensorDataset(X_tst_data_global_trch)

# train_dl_all_global = DataLoader(train_ds_all_global, batch_size=128, shuffle=False)
# val_dl_all_global   = DataLoader(val_ds_all_global, batch_size=128, shuffle=False)
tst_dl_all_global   = DataLoader(tst_ds_all_global, batch_size=128, shuffle=False)

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
# train_embeddings_global = get_embeddings(model_trained, train_dl_all_global)
# val_embeddings_global = get_embeddings(model_trained, val_dl_all_global)
tst_embeddings_global = get_embeddings(model_trained, tst_dl_all_global)

# print("Train embeddings shape:", train_embeddings_global.shape)
# print("Val embeddings shape:", val_embeddings_global.shape)
print("Test embeddings shape:", tst_embeddings_global.shape)
print('\n')

# # **Get NESI**

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

NESI_model_path = NESI_ROOT / "model" / "ModelCheckpoints" / "NESI_best_model.pth"
checkpoint = torch.load(
    NESI_model_path,
    map_location=device,
    weights_only=True
)

model_score = EEGScoringModel()
model_score.load_state_dict(checkpoint["model_state_dict"])
model_score.to(device)

model_score.eval()

# train_scores_global = get_scores(model_score, train_embeddings_global, device=device)
# val_scores_global   = get_scores(model_score, val_embeddings_global, device=device)
tst_scores_global  = get_scores(model_score, tst_embeddings_global, device=device)

# print("Train EEG NESI shape:", train_scores_global.shape)
# print("Val EEG NESI shape:", val_scores_global.shape)
print("Test EEG NESI shape:", tst_scores_global.shape)
print('\n')

# # **Plot distribution of NESI vs true raw scores from each cohort**

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_score_distributions_by_raw(
    predicted_scores,
    y_raw,
    dataset_names,
    title="Model Score Distribution by Raw Labels"
):
    # High DPI for crisp text and lines
    plt.rcParams['figure.dpi'] = 300
    sns.set_style("whitegrid")
    
    df = pd.DataFrame({
        "score": np.array(predicted_scores),
        "y_raw": np.array(y_raw),
        "dataset": np.array(dataset_names)
    })

    datasets = ["RASS", "GCS", "CAMS", "ICANS"]
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), dpi=150)
    axes = axes.flatten()

    for idx, d in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["dataset"] == d].copy()

        if sub.empty:
            ax.set_title(f"{d} Dataset\n(No Data)", fontweight='bold')
            continue

        sub = sub.sort_values("y_raw")

        # Create the boxplot with strict black edge/line colors
        sns.boxplot(
            data=sub, 
            x="y_raw", 
            y="score", 
            hue="y_raw",          # Assigned hue to fix the FutureWarning
            legend=False,         # Hide legend as x-axis handles labels
            ax=ax,
            palette="pastel",
            showfliers=True,      # Show outliers as requested
            # Setting all line components to black
            boxprops={'edgecolor': 'black', 'linewidth': 1.5},
            whiskerprops={'color': 'black', 'linewidth': 1.5},
            capprops={'color': 'black', 'linewidth': 1.5},
            medianprops={'color': 'black', 'linewidth': 2},
            flierprops={'markerfacecolor': 'grey', 'markeredgecolor': 'black', 'markersize': 4}
        )

        # Bold formatting
        ax.set_title(f"{d} Dataset", fontweight='bold', fontsize=15, pad=15)
        ax.set_xlabel("Raw Score", fontweight='bold', fontsize=12)
        if idx == 0:
            ax.set_ylabel("Predicted NESI", fontweight='bold', fontsize=12)
        else:
            ax.set_ylabel("")
        
        # Adding a black border (spine) around the entire subplot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_visible(True)
            spine.set_linewidth(1.2)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight('bold')

    plt.suptitle(title, fontsize=20, fontweight='bold', y=1)
    plt.tight_layout()
    plt.show()

# plot_score_distributions_by_raw(
#     predicted_scores=train_scores_global,
#     y_raw=Yraw_train_global,
#     dataset_names=train_dataset_names_global,
#     title="Train Set NESI Distribution"
# )
# plot_score_distributions_by_raw(
#     predicted_scores=val_scores_global,
#     y_raw=Yraw_val_global,
#     dataset_names=val_dataset_names_global,
#     title="Validation Set NESI Distribution"
# )
plot_score_distributions_by_raw(
    predicted_scores=tst_scores_global,
    y_raw=Yraw_tst_global,
    dataset_names=tst_dataset_names_global,
    title="Test Set NESI Distribution"
)

## Bespoke Performance
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

def get_corn_logits(model, X, device="cuda", batch_size=128):
    """
    Get CORN logits from model.

    Args:
        model: trained CORN model
        X: torch tensor (N, D)
        device: 'cuda' or 'cpu'
        batch_size: batch size

    Returns:
        logits: torch tensor (N, K-1)
    """

    # ---- safety (avoid your earlier warning) ----
    if isinstance(X, torch.Tensor):
        X_tensor = X.clone().detach()
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()
    model.to(device)

    all_logits = []

    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)

            logits = model(x_batch)   # (batch, K-1)

            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)

    return all_logits

def rass_badness_from_logits(logits: torch.Tensor):
    """
    Computes CORN-based badness score.

    Args:
        logits: Tensor of shape (K-1,) or (batch, K-1)
                where K = number of ordinal classes

    Returns:
        badness score (same shape reduced over thresholds)
    """

    # Step 1: convert logits -> probabilities P(y > k)
    probs = torch.sigmoid(logits)

    # Step 2: expected ordinal value
    # E[y] = sum_k P(y > k)
    expected_y = torch.sum(probs, dim=-1)

    # Step 3: badness = (K-1) - E[y]
    K_minus_1 = logits.shape[-1]  # since logits are K-1 outputs
    badness = K_minus_1 - expected_y

    return badness

def cams_badness_from_logits(logits: torch.Tensor):
    """
    Compute badness score for CAMS (3-class CORN model).

    Args:
        logits: Tensor of shape (N, 2) or (2,)
                CORN outputs [z0, z1]

    Returns:
        badness: Tensor of shape (N,) or scalar
                 Range: 0 (mild) → 2 (severe)
    """

    # Convert logits → probabilities
    probs = torch.sigmoid(logits)

    # Expected severity (no inversion needed)
    badness = torch.sum(probs, dim=-1)

    return badness

def gcs_badness_from_logits(logits: torch.Tensor):
    """
    GCS CORN → badness score

    Args:
        logits: (N, 2)

    Returns:
        badness: (N,)
                 Range: 0 (mild) → 2 (severe)
    """

    probs = torch.sigmoid(logits)
    expected = torch.sum(probs, dim=-1)

    badness = 2.0 - expected   # <-- inversion
    return badness

def icans_badness_from_logits(logits: torch.Tensor):
    """
    Compute badness score for ICANS (3-class CORN model).

    Args:
        logits: Tensor of shape (N, 2) or (2,)
                CORN outputs [z0, z1]

    Returns:
        badness: Tensor of shape (N,) or scalar
                 Range: 0 (mild) → 2 (severe)
    """

    # Convert logits → probabilities
    probs = torch.sigmoid(logits)

    # Expected severity (no inversion needed)
    badness = torch.sum(probs, dim=-1)

    return badness

### RASS Bespoke
df_train_RASS = df_train[df_train['Dataset'] =='RASS'].reset_index(drop=True)
df_val_RASS = df_val[df_val['Dataset'] =='RASS'].reset_index(drop=True)
df_test_RASS = df_test[df_test['Dataset'] =='RASS'].reset_index(drop=True)

# X_train_data_RASS, Ygrouped_train_RASS, Yraw_train_RASS, Yraw_transformed_train_RASS, train_dataset_names_RASS,trn_filenames = morgoth_10minfea_matrix(df_train_RASS)
# X_val_data_RASS, Ygrouped_val_RASS, Yraw_val_RASS, Yraw_transformed_val_RASS, val_dataset_names_RASS, val_filenames = morgoth_10minfea_matrix(df_val_RASS)
X_tst_data_RASS, Ygrouped_tst_RASS, Yraw_tst_RASS, Yraw_transformed_tst_RASS, tst_dataset_names_RASS, tst_filenames = morgoth_10minfea_matrix(df_test_RASS)

# X_train_data_RASS_trch = torch.tensor(X_train_data_RASS, dtype=torch.float32)
# X_val_data_RASS_trch   = torch.tensor(X_val_data_RASS, dtype=torch.float32)
X_tst_data_RASS_trch   = torch.tensor(X_tst_data_RASS, dtype=torch.float32)

# Ygrouped_train_RASS_trch = torch.tensor(Ygrouped_train_RASS, dtype=torch.long)
# Ygrouped_val_RASS_trch   = torch.tensor(Ygrouped_val_RASS, dtype=torch.long)
Ygrouped_tst_RASS_trch   = torch.tensor(Ygrouped_tst_RASS, dtype=torch.long)

# Yraw_train_RASS_trch = torch.tensor(Yraw_train_RASS, dtype=torch.long)
# Yraw_val_RASS_trch   = torch.tensor(Yraw_val_RASS, dtype=torch.long)
Yraw_tst_RASS_trch   = torch.tensor(Yraw_tst_RASS, dtype=torch.long)

# train_dataset_names_RASS = np.array(train_dataset_names_RASS)
# val_dataset_names_RASS   = np.array(val_dataset_names_RASS)
tst_dataset_names_RASS   = np.array(tst_dataset_names_RASS)

model_RASS = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, 
                                               num_classes=6)
        
model_path_RASS = NESI_ROOT / "BespokeModelCheckpoints" / "RASS_bespoke.pth"
model_RASS.load_state_dict(torch.load(model_path_RASS, map_location=device, weights_only=True))
model_RASS.eval()

# train_logits_RASS = get_corn_logits(model_RASS, X_train_data_RASS_trch, device)
# val_logits_RASS   = get_corn_logits(model_RASS, X_val_data_RASS_trch, device)
tst_logits_RASS  = get_corn_logits(model_RASS, X_tst_data_RASS_trch, device)

# print('RASS CORN model Train-set logit size:', train_logits_RASS.shape)
# print('RASS CORN model Val-set logit size:', val_logits_RASS.shape)
print('RASS CORN model Test-set logit size:', tst_logits_RASS.shape)

# train_badness_RASS = rass_badness_from_logits(train_logits_RASS)
# val_badness_RASS   = rass_badness_from_logits(val_logits_RASS)
tst_badness_RASS  = rass_badness_from_logits(tst_logits_RASS)

# print("Train EEG Badness score shape:", train_badness_RASS.shape)
# print("Val EEG Badness score shape:", val_badness_RASS.shape)
print("Test EEG Badness score shape:", tst_badness_RASS.shape)
print('\n')

### GCS Bespoke
# df_train_GCS = df_train[df_train['Dataset'] =='GCS'].reset_index(drop=True)
# df_val_GCS = df_val[df_val['Dataset'] =='GCS'].reset_index(drop=True)
df_test_GCS = df_test[df_test['Dataset'] =='GCS'].reset_index(drop=True)

# X_train_data_GCS, Ygrouped_train_GCS, Yraw_train_GCS, Yraw_transformed_train_GCS, train_dataset_names_GCS,trn_filenames = morgoth_10minfea_matrix(df_train_GCS)
# X_val_data_GCS, Ygrouped_val_GCS, Yraw_val_GCS, Yraw_transformed_val_GCS, val_dataset_names_GCS, val_filenames = morgoth_10minfea_matrix(df_val_GCS)
X_tst_data_GCS, Ygrouped_tst_GCS, Yraw_tst_GCS, Yraw_transformed_tst_GCS, tst_dataset_names_GCS, tst_filenames = morgoth_10minfea_matrix(df_test_GCS)

# X_train_data_GCS_trch = torch.tensor(X_train_data_GCS, dtype=torch.float32)
# X_val_data_GCS_trch   = torch.tensor(X_val_data_GCS, dtype=torch.float32)
X_tst_data_GCS_trch   = torch.tensor(X_tst_data_GCS, dtype=torch.float32)

# Ygrouped_train_GCS_trch = torch.tensor(Ygrouped_train_GCS, dtype=torch.long)
# Ygrouped_val_GCS_trch   = torch.tensor(Ygrouped_val_GCS, dtype=torch.long)
Ygrouped_tst_GCS_trch   = torch.tensor(Ygrouped_tst_GCS, dtype=torch.long)

# Yraw_train_GCS_trch = torch.tensor(Yraw_train_GCS, dtype=torch.long)
# Yraw_val_GCS_trch   = torch.tensor(Yraw_val_GCS, dtype=torch.long)
Yraw_tst_GCS_trch   = torch.tensor(Yraw_tst_GCS, dtype=torch.long)

# train_dataset_names_GCS = np.array(train_dataset_names_GCS)
# val_dataset_names_GCS   = np.array(val_dataset_names_GCS)
tst_dataset_names_GCS   = np.array(tst_dataset_names_GCS)


model_GCS = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, 
                                               num_classes=3)
        
model_path_GCS = NESI_ROOT / "BespokeModelCheckpoints" / "GCS_bespoke.pth"
model_GCS.load_state_dict(torch.load(model_path_GCS, map_location=device, weights_only=True))
model_GCS.eval()

# train_logits_GCS = get_corn_logits(model_GCS, X_train_data_GCS_trch, device)
# val_logits_GCS   = get_corn_logits(model_GCS, X_val_data_GCS_trch, device)
tst_logits_GCS  = get_corn_logits(model_GCS, X_tst_data_GCS_trch, device)

# print('GCS CORN model Train-set logit size:', train_logits_GCS.shape)
# print('GCS CORN model Val-set logit size:', val_logits_GCS.shape)
print('GCS CORN model Test-set logit size:', tst_logits_GCS.shape)

# train_badness_GCS = gcs_badness_from_logits(train_logits_GCS)
# val_badness_GCS   = gcs_badness_from_logits(val_logits_GCS)
tst_badness_GCS  = gcs_badness_from_logits(tst_logits_GCS)

# print("Train EEG Badness score shape:", train_badness_GCS.shape)
# print("Val EEG Badness score shape:", val_badness_GCS.shape)
print("Test EEG Badness score shape:", tst_badness_GCS.shape)
print('\n')

### CAMS Bespoke
# df_train_CAMS = df_train[df_train['Dataset'] =='CAMS'].reset_index(drop=True)
# df_val_CAMS = df_val[df_val['Dataset'] =='CAMS'].reset_index(drop=True)
df_test_CAMS = df_test[df_test['Dataset'] =='CAMS'].reset_index(drop=True)

# X_train_data_CAMS, Ygrouped_train_CAMS, Yraw_train_CAMS, Yraw_transformed_train_CAMS, train_dataset_names_CAMS,trn_filenames = morgoth_10minfea_matrix(df_train_CAMS)
# X_val_data_CAMS, Ygrouped_val_CAMS, Yraw_val_CAMS, Yraw_transformed_val_CAMS, val_dataset_names_CAMS, val_filenames = morgoth_10minfea_matrix(df_val_CAMS)
X_tst_data_CAMS, Ygrouped_tst_CAMS, Yraw_tst_CAMS, Yraw_transformed_tst_CAMS, tst_dataset_names_CAMS, tst_filenames = morgoth_10minfea_matrix(df_test_CAMS)

# X_train_data_CAMS_trch = torch.tensor(X_train_data_CAMS, dtype=torch.float32)
# X_val_data_CAMS_trch   = torch.tensor(X_val_data_CAMS, dtype=torch.float32)
X_tst_data_CAMS_trch   = torch.tensor(X_tst_data_CAMS, dtype=torch.float32)

# Ygrouped_train_CAMS_trch = torch.tensor(Ygrouped_train_CAMS, dtype=torch.long)
# Ygrouped_val_CAMS_trch   = torch.tensor(Ygrouped_val_CAMS, dtype=torch.long)
Ygrouped_tst_CAMS_trch   = torch.tensor(Ygrouped_tst_CAMS, dtype=torch.long)

# Yraw_train_CAMS_trch = torch.tensor(Yraw_train_CAMS, dtype=torch.long)
# Yraw_val_CAMS_trch   = torch.tensor(Yraw_val_CAMS, dtype=torch.long)
Yraw_tst_CAMS_trch   = torch.tensor(Yraw_tst_CAMS, dtype=torch.long)

# train_dataset_names_CAMS = np.array(train_dataset_names_CAMS)
# val_dataset_names_CAMS   = np.array(val_dataset_names_CAMS)
tst_dataset_names_CAMS   = np.array(tst_dataset_names_CAMS)

model_CAMS = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, 
                                               num_classes=3)
        
model_path_CAMS = NESI_ROOT / "BespokeModelCheckpoints" / "CAMS_bespoke.pth"
model_CAMS.load_state_dict(torch.load(model_path_CAMS, map_location=device, weights_only=True))
model_CAMS.eval()

# train_logits_CAMS = get_corn_logits(model_CAMS, X_train_data_CAMS_trch, device)
# val_logits_CAMS   = get_corn_logits(model_CAMS, X_val_data_CAMS_trch, device)
tst_logits_CAMS  = get_corn_logits(model_CAMS, X_tst_data_CAMS_trch, device)

# print('CAMS CORN model Train-set logit size:', train_logits_CAMS.shape)
# print('CAMS CORN model Val-set logit size:', val_logits_CAMS.shape)
print('CAMS CORN model Test-set logit size:', tst_logits_CAMS.shape)

# train_badness_CAMS = cams_badness_from_logits(train_logits_CAMS)
# val_badness_CAMS   = cams_badness_from_logits(val_logits_CAMS)
tst_badness_CAMS  = cams_badness_from_logits(tst_logits_CAMS)

# print("Train CAMS EEG Badness score shape:", train_badness_CAMS.shape)
# print("Val CAMS EEG Badness score shape:", val_badness_CAMS.shape)
print("Test CAMS EEG Badness score shape:", tst_badness_CAMS.shape)
print('\n')

#### ICANS Bespoke
# df_train_ICANS = df_train[df_train['Dataset'] =='ICANS'].reset_index(drop=True)
# df_val_ICANS = df_val[df_val['Dataset'] =='ICANS'].reset_index(drop=True)
df_test_ICANS = df_test[df_test['Dataset'] =='ICANS'].reset_index(drop=True)

# X_train_data_ICANS, Ygrouped_train_ICANS, Yraw_train_ICANS, Yraw_transformed_train_ICANS, train_dataset_names_ICANS,trn_filenames = morgoth_10minfea_matrix(df_train_ICANS)
# X_val_data_ICANS, Ygrouped_val_ICANS, Yraw_val_ICANS, Yraw_transformed_val_ICANS, val_dataset_names_ICANS, val_filenames = morgoth_10minfea_matrix(df_val_ICANS)
X_tst_data_ICANS, Ygrouped_tst_ICANS, Yraw_tst_ICANS, Yraw_transformed_tst_ICANS, tst_dataset_names_ICANS, tst_filenames = morgoth_10minfea_matrix(df_test_ICANS)

# X_train_data_ICANS_trch = torch.tensor(X_train_data_ICANS, dtype=torch.float32)
# X_val_data_ICANS_trch   = torch.tensor(X_val_data_ICANS, dtype=torch.float32)
X_tst_data_ICANS_trch   = torch.tensor(X_tst_data_ICANS, dtype=torch.float32)

# Ygrouped_train_ICANS_trch = torch.tensor(Ygrouped_train_ICANS, dtype=torch.long)
# Ygrouped_val_ICANS_trch   = torch.tensor(Ygrouped_val_ICANS, dtype=torch.long)
Ygrouped_tst_ICANS_trch   = torch.tensor(Ygrouped_tst_ICANS, dtype=torch.long)

# Yraw_train_ICANS_trch = torch.tensor(Yraw_train_ICANS, dtype=torch.long)
# Yraw_val_ICANS_trch   = torch.tensor(Yraw_val_ICANS, dtype=torch.long)
Yraw_tst_ICANS_trch   = torch.tensor(Yraw_tst_ICANS, dtype=torch.long)

# train_dataset_names_ICANS = np.array(train_dataset_names_ICANS)
# val_dataset_names_ICANS   = np.array(val_dataset_names_ICANS)
tst_dataset_names_ICANS   = np.array(tst_dataset_names_ICANS)

model_ICANS = MORGOTH_ResNet1D_onlyGAP_CORAL(num_features=17, 
                                               num_classes=3)
        
model_path_ICANS = NESI_ROOT / "BespokeModelCheckpoints" / "ICANS_bespoke.pth"
model_ICANS.load_state_dict(torch.load(model_path_ICANS, map_location=device, weights_only=True))
model_ICANS.eval()

# train_logits_ICANS = get_corn_logits(model_ICANS, X_train_data_ICANS_trch, device)
# val_logits_ICANS   = get_corn_logits(model_ICANS, X_val_data_ICANS_trch, device)
tst_logits_ICANS  = get_corn_logits(model_ICANS, X_tst_data_ICANS_trch, device)

# print('ICANS CORN model Train-set logit size:', train_logits_ICANS.shape)
# print('ICANS CORN model Val-set logit size:', val_logits_ICANS.shape)
print('ICANS CORN model Test-set logit size:', tst_logits_ICANS.shape)

# train_badness_ICANS = icans_badness_from_logits(train_logits_ICANS)
# val_badness_ICANS   = icans_badness_from_logits(val_logits_ICANS)
tst_badness_ICANS  = icans_badness_from_logits(tst_logits_ICANS)

# print("Train ICANS EEG Badness score shape:", train_badness_ICANS.shape)
# print("Val ICANS EEG Badness score shape:", val_badness_ICANS.shape)
print("Test ICANS EEG Badness score shape:", tst_badness_ICANS.shape)



# ---------- NESI vs RASS --------------
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------- mask ----------------
mask_RASS = tst_dataset_names_global == 'RASS'

Yraw_transformed_tst_global_RASS = Yraw_transformed_tst_global[mask_RASS]
tst_scores_global_RASS = tst_scores_global[mask_RASS]

# ---------------- dataframe ----------------
df_gcs = pd.DataFrame({
    "TrueRawRASS": Yraw_tst_RASS,
    "TrueTransformedRASS": Yraw_transformed_tst_global_RASS,
    "GlobalModelScore": tst_scores_global_RASS,
    "BespokeModelScore": tst_badness_RASS
})

# ---------------- extract ----------------
y_true = df_gcs["TrueTransformedRASS"].values
y_global = df_gcs["GlobalModelScore"].values
y_bespoke = df_gcs["BespokeModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)
bespoke_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y_global[idx])
    bespoke_corrs[i], _ = spearmanr(y_true[idx], y_bespoke[idx])

# ---------------- stats ----------------
global_mean_RASS = np.mean(global_corrs)
bespoke_mean_RASS = np.mean(bespoke_corrs)

global_ci_RASS = np.percentile(global_corrs, [2.5, 97.5])
bespoke_ci_RASS = np.percentile(bespoke_corrs, [2.5, 97.5])

# ---------------- WILCOXON TEST ----------------
stat, p_value_RASS = wilcoxon(global_corrs, bespoke_corrs, alternative="greater")

# ---------------- decide winner ----------------
if p_value_RASS < 0.05:
    winner = "GLOBAL MODEL is significantly better than BESPOKE MODEL"
else:
    winner = "No significant difference (or BESPOKE not worse)"

# ---------------- output ----------------
print("GLOBAL MODEL")
print(f"Mean Spearman: {global_mean_RASS:.4f}")
print(f"95% CI: [{global_ci_RASS[0]:.4f}, {global_ci_RASS[1]:.4f}]")

print("\nBESPOKE MODEL")
print(f"Mean Spearman: {bespoke_mean_RASS:.4f}")
print(f"95% CI: [{bespoke_ci_RASS[0]:.4f}, {bespoke_ci_RASS[1]:.4f}]")

print("\nSTATISTICAL TEST (Wilcoxon Signed-Rank, paired)")
print(f"p-value: {p_value_RASS:.3e}")
print(f"Result: {winner}")


# --------- NESI vs GCS -------------
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------- mask ----------------
mask_GCS = tst_dataset_names_global == 'GCS'

Yraw_transformed_tst_global_GCS = Yraw_transformed_tst_global[mask_GCS]
tst_scores_global_GCS = tst_scores_global[mask_GCS]

# ---------------- dataframe ----------------
df_gcs = pd.DataFrame({
    "TrueRawGCS": Yraw_tst_GCS,
    "TrueTransformedGCS": Yraw_transformed_tst_global_GCS,
    "GlobalModelScore": tst_scores_global_GCS,
    "BespokeModelScore": tst_badness_GCS
})

# ---------------- extract ----------------
y_true = df_gcs["TrueTransformedGCS"].values
y_global = df_gcs["GlobalModelScore"].values
y_bespoke = df_gcs["BespokeModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)
bespoke_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y_global[idx])
    bespoke_corrs[i], _ = spearmanr(y_true[idx], y_bespoke[idx])

# ---------------- stats ----------------
global_mean_GCS = np.mean(global_corrs)
bespoke_mean_GCS = np.mean(bespoke_corrs)

global_ci_GCS = np.percentile(global_corrs, [2.5, 97.5])
bespoke_ci_GCS = np.percentile(bespoke_corrs, [2.5, 97.5])

# =========================================================
# Paired permutation test
# =========================================================

observed_diff = np.mean(global_corrs - bespoke_corrs)

n_perm = 10000
perm_diffs = np.zeros(n_perm)

combined = np.vstack([global_corrs, bespoke_corrs]).T

rng = np.random.default_rng(123)

for i in range(n_perm):
    signs = rng.choice([1, -1], size=n_boot)
    perm_diffs[i] = np.mean((global_corrs - bespoke_corrs) * signs)

# one-sided p-value (global > bespoke)
p_value_GCS = np.mean(perm_diffs >= observed_diff)

# ---------------- decide winner ----------------
if p_value_GCS < 0.05:
    winner = "GLOBAL MODEL is significantly better than BESPOKE MODEL"
else:
    winner = "No significant difference (or BESPOKE not worse)"

# ---------------- output ----------------
print("GLOBAL MODEL")
print(f"Mean Spearman: {global_mean_GCS:.4f}")
print(f"95% CI: [{global_ci_GCS[0]:.4f}, {global_ci_GCS[1]:.4f}]")

print("\nBESPOKE MODEL")
print(f"Mean Spearman: {bespoke_mean_GCS:.4f}")
print(f"95% CI: [{bespoke_ci_GCS[0]:.4f}, {bespoke_ci_GCS[1]:.4f}]")

print("\nSTATISTICAL TEST (Paired Permutation Test)")
print(f"p-value: {p_value_GCS:.3e}")
print(f"Result: {winner}")


# --------- NESI vs CAMS -------------
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------- mask ----------------
mask_CAMS = tst_dataset_names_global == 'CAMS'

Yraw_transformed_tst_global_CAMS = Yraw_transformed_tst_global[mask_CAMS]
tst_scores_global_CAMS = tst_scores_global[mask_CAMS]

# ---------------- dataframe ----------------
df_gcs = pd.DataFrame({
    "TrueRawCAMS": Yraw_tst_CAMS,
    "TrueTransformedCAMS": Yraw_transformed_tst_global_CAMS,
    "GlobalModelScore": tst_scores_global_CAMS,
    "BespokeModelScore": tst_badness_CAMS
})

# ---------------- extract ----------------
y_true = df_gcs["TrueTransformedCAMS"].values
y_global = df_gcs["GlobalModelScore"].values
y_bespoke = df_gcs["BespokeModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)
bespoke_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y_global[idx])
    bespoke_corrs[i], _ = spearmanr(y_true[idx], y_bespoke[idx])

# ---------------- stats ----------------
global_mean_CAMS = np.mean(global_corrs)
bespoke_mean_CAMS = np.mean(bespoke_corrs)

global_ci_CAMS = np.percentile(global_corrs, [2.5, 97.5])
bespoke_ci_CAMS = np.percentile(bespoke_corrs, [2.5, 97.5])

# =========================================================
# Paired permutation test
# =========================================================

observed_diff = np.mean(global_corrs - bespoke_corrs)

n_perm = 10000
perm_diffs = np.zeros(n_perm)

combined = np.vstack([global_corrs, bespoke_corrs]).T

rng = np.random.default_rng(123)

for i in range(n_perm):
    signs = rng.choice([1, -1], size=n_boot)
    perm_diffs[i] = np.mean((global_corrs - bespoke_corrs) * signs)

# one-sided p-value (global > bespoke)
p_value_CAMS = np.mean(perm_diffs >= observed_diff)

# ---------------- decide winner ----------------
if p_value_CAMS < 0.05:
    winner = "GLOBAL MODEL is significantly better than BESPOKE MODEL"
else:
    winner = "No significant difference (or BESPOKE not worse)"

# ---------------- output ----------------
print("GLOBAL MODEL")
print(f"Mean Spearman: {global_mean_CAMS:.4f}")
print(f"95% CI: [{global_ci_CAMS[0]:.4f}, {global_ci_CAMS[1]:.4f}]")

print("\nBESPOKE MODEL")
print(f"Mean Spearman: {bespoke_mean_CAMS:.4f}")
print(f"95% CI: [{bespoke_ci_CAMS[0]:.4f}, {bespoke_ci_CAMS[1]:.4f}]")

print("\nSTATISTICAL TEST (Paired Permutation Test)")
print(f"p-value: {p_value_CAMS:.3e}")
print(f"Result: {winner}")

# ------------- NESI vs ICANS -----------------
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------- mask ----------------
mask_ICANS = tst_dataset_names_global == 'ICANS'

Yraw_transformed_tst_global_ICANS = Yraw_transformed_tst_global[mask_ICANS]
tst_scores_global_ICANS = tst_scores_global[mask_ICANS]

# ---------------- dataframe ----------------
df_gcs = pd.DataFrame({
    "TrueRawICANS": Yraw_tst_ICANS,
    "TrueTransformedICANS": Yraw_transformed_tst_global_ICANS,
    "GlobalModelScore": tst_scores_global_ICANS,
    "BespokeModelScore": tst_badness_ICANS
})

# ---------------- extract ----------------
y_true = df_gcs["TrueTransformedICANS"].values
y_global = df_gcs["GlobalModelScore"].values
y_bespoke = df_gcs["BespokeModelScore"].values

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)
bespoke_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y_global[idx])
    bespoke_corrs[i], _ = spearmanr(y_true[idx], y_bespoke[idx])

# ---------------- stats ----------------
global_mean_ICANS = np.mean(global_corrs)
bespoke_mean_ICANS = np.mean(bespoke_corrs)

global_ci_ICANS = np.percentile(global_corrs, [2.5, 97.5])
bespoke_ci_ICANS = np.percentile(bespoke_corrs, [2.5, 97.5])

# =========================================================
# Paired permutation test
# =========================================================

observed_diff = np.mean(global_corrs - bespoke_corrs)

n_perm = 10000
perm_diffs = np.zeros(n_perm)

combined = np.vstack([global_corrs, bespoke_corrs]).T

rng = np.random.default_rng(123)

for i in range(n_perm):
    signs = rng.choice([1, -1], size=n_boot)
    perm_diffs[i] = np.mean((global_corrs - bespoke_corrs) * signs)

# one-sided p-value (global > bespoke)
p_value_ICANS = np.mean(perm_diffs >= observed_diff)

# ---------------- decide winner ----------------
if p_value_ICANS < 0.05:
    winner = "GLOBAL MODEL is significantly better than BESPOKE MODEL"
else:
    winner = "No significant difference (or BESPOKE not worse)"

# ---------------- output ----------------
print("GLOBAL MODEL")
print(f"Mean Spearman: {global_mean_ICANS:.4f}")
print(f"95% CI: [{global_ci_ICANS[0]:.4f}, {global_ci_ICANS[1]:.4f}]")

print("\nBESPOKE MODEL")
print(f"Mean Spearman: {bespoke_mean_ICANS:.4f}")
print(f"95% CI: [{bespoke_ci_ICANS[0]:.4f}, {bespoke_ci_ICANS[1]:.4f}]")

print("\nSTATISTICAL TEST (Paired Permutation Test)")
print(f"p-value: {p_value_ICANS:.3e}")
print(f"Result: {winner}")

import matplotlib.pyplot as plt
import numpy as np

def plot_individual_vs_global_correct(
    rass_bad, rass_y, gcs_bad, gcs_y, cams_bad, cams_y, icans_bad, icans_y,
    global_scores, global_yraw, global_dataset_names,
    spearman_bespoke_RASS, spearman_bespoke_GCS, spearman_bespoke_CAMS, spearman_bespoke_ICANS,
    spearman_global_RASS, spearman_global_GCS, spearman_global_CAMS, spearman_global_ICANS,
    title="Bespoke vs Universal Model Comparison"
):
    plt.rcParams.update({'font.size': 8, 'figure.dpi': 120})

    def to_numpy(x):
        if hasattr(x, "detach"): return x.detach().cpu().numpy()
        return np.array(x)

    def group_data(badness, yraw, levels):
        grouped = []
        for l in levels:
            vals = badness[yraw == l]
            grouped.append(vals if len(vals) > 0 else np.array([np.nan]))
        return grouped

    # --- Convert ---
    rass_bad = to_numpy(rass_bad); rass_y = to_numpy(rass_y)
    gcs_bad  = to_numpy(gcs_bad);  gcs_y  = to_numpy(gcs_y)
    cams_bad = to_numpy(cams_bad); cams_y = to_numpy(cams_y)
    icans_bad = to_numpy(icans_bad); icans_y = to_numpy(icans_y)
    global_scores = to_numpy(global_scores)
    global_yraw = np.array(global_yraw)
    global_dataset_names = np.array(global_dataset_names)

    # --- Levels ---
    rass_levels = [-5, -4, -3, -2, -1, 0]
    gcs_levels  = list(range(3, 16))
    cams_levels = sorted(np.unique(cams_y))
    icans_levels = sorted(np.unique(icans_y))

    colors = ["#AEC6CF", "#FFB7B2", "#B2E2F2", "#CFCFC4", "#FDFD96", "#B39EB5", "#FFD1DC"]
    fig, axes = plt.subplots(2, 4, figsize=(10,5), dpi=150)

    box_style = dict(
        patch_artist=True, showfliers=True,
        flierprops={'marker': 'o', 'markersize': 1.5, 'markerfacecolor': 'black', 'alpha': 0.2},
        widths=0.6, boxprops=dict(edgecolor='black', linewidth=1),
        whiskerprops=dict(color='black', linewidth=1),
        capprops=dict(color='black', linewidth=1),
        medianprops=dict(color='black', linewidth=1)
    )

    datasets_info = [
        ("RASS", rass_bad, rass_y, rass_levels, spearman_bespoke_RASS, spearman_global_RASS),
        ("GCS", gcs_bad, gcs_y, gcs_levels, spearman_bespoke_GCS, spearman_global_GCS),
        ("CAMS", cams_bad, cams_y, cams_levels, spearman_bespoke_CAMS, spearman_global_CAMS),
        ("ICANS", icans_bad, icans_y, icans_levels, spearman_bespoke_ICANS, spearman_global_ICANS),
    ]

    for i, (name, bad, y, levels, rho_ind, rho_global) in enumerate(datasets_info):
        ax_top, ax_bot = axes[0, i], axes[1, i]

        # --- Plotting ---
        bp1 = ax_bot.boxplot(group_data(bad, y, levels), **box_style)
        for j, box in enumerate(bp1['boxes']): box.set_facecolor(colors[j % len(colors)])

        mask = global_dataset_names == name
        bp2 = ax_top.boxplot(group_data(global_scores[mask], global_yraw[mask], levels), **box_style)
        for j, box in enumerate(bp2['boxes']): box.set_facecolor(colors[j % len(colors)])

        # --- Formatting (Spines, Ticks, Labels) ---
        for ax in [ax_top, ax_bot]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.2)
            
            ax.set_xticks(range(1, len(levels) + 1))
            ax.set_xticklabels(levels)
            if name == "GCS": plt.setp(ax.get_xticklabels(), rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Set specific X-labels for the bottom row
        ax_bot.set_xlabel(f"True {name} Score", fontweight='bold', fontsize=10)

        # Stacking titles to prevent horizontal overlap
        ax_bot.set_title(f"ρ={rho_ind}", fontweight='bold', pad=10)
        ax_top.set_title(f"{name}\nρ={rho_global}", fontweight='bold', pad=10)

    axes[1, 0].set_ylabel("Bespoke Model's NESI", fontweight='bold', fontsize=10)
    axes[0, 0].set_ylabel("Universal Model's NESI", fontweight='bold', fontsize=10)

    #fig.suptitle(title, fontsize=10, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(wspace=0.35, hspace=0.5) 
    plt.show()

# ------------- Universal Model ---------------
tst_spearman_conf_global_RASS = (
    f"{global_mean_RASS:.4f} "
    f"([{global_ci_RASS[0]:.4f}, {global_ci_RASS[1]:.4f}])"
)
tst_spearman_conf_global_GCS = (
    f"{global_mean_GCS:.4f} "
    f"([{global_ci_GCS[0]:.4f}, {global_ci_GCS[1]:.4f}])"
)
tst_spearman_conf_global_CAMS = (
    f"{global_mean_CAMS:.4f} "
    f"([{global_ci_CAMS[0]:.4f}, {global_ci_CAMS[1]:.4f}])"
)
tst_spearman_conf_global_ICANS = (
    f"{global_mean_ICANS:.4f} "
    f"([{global_ci_ICANS[0]:.4f}, {global_ci_ICANS[1]:.4f}])"
)

# -------------Bespoke models ---------------
tst_spearman_conf_bespoke_ICANS = (
    f"{bespoke_mean_ICANS:.4f} "
    f"([{bespoke_ci_ICANS[0]:.4f}, {bespoke_ci_ICANS[1]:.4f}])"
)
tst_spearman_conf_bespoke_CAMS = (
    f"{bespoke_mean_CAMS:.4f} "
    f"([{bespoke_ci_CAMS[0]:.4f}, {bespoke_ci_CAMS[1]:.4f}])"
)
tst_spearman_conf_bespoke_RASS = (
    f"{bespoke_mean_RASS:.4f} "
    f"([{bespoke_ci_RASS[0]:.4f}, {bespoke_ci_RASS[1]:.4f}])"
)
tst_spearman_conf_bespoke_GCS = (
    f"{bespoke_mean_GCS:.4f} "
    f"([{bespoke_ci_GCS[0]:.4f}, {bespoke_ci_GCS[1]:.4f}])"
)


plot_individual_vs_global_correct(
    tst_badness_RASS, Yraw_tst_RASS,
    tst_badness_GCS,  Yraw_tst_GCS,
    tst_badness_CAMS, Yraw_tst_CAMS,
    tst_badness_ICANS, Yraw_tst_ICANS,

    tst_scores_global,
    Yraw_tst_global,
    tst_dataset_names_global,

    tst_spearman_conf_bespoke_RASS,
    tst_spearman_conf_bespoke_GCS,
    tst_spearman_conf_bespoke_CAMS,
    tst_spearman_conf_bespoke_ICANS,

    tst_spearman_conf_global_RASS,
    tst_spearman_conf_global_GCS,
    tst_spearman_conf_global_CAMS,
    tst_spearman_conf_global_ICANS,
    
    title="Individual vs Global Model Comparison (Same Hold out Testing set)"
)