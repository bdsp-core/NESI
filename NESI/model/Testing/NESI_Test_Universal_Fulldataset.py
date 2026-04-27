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

X_train_data_global, Ygrouped_train_global, Yraw_train_global, Yraw_transformed_train_global, train_dataset_names_global,trn_filenames_global = morgoth_10minfea_matrix(df_train)
X_val_data_global, Ygrouped_val_global, Yraw_val_global, Yraw_transformed_val_global, val_dataset_names_global, val_filenames_global = morgoth_10minfea_matrix(df_val)
X_tst_data_global, Ygrouped_tst_global, Yraw_tst_global, Yraw_transformed_tst_global, tst_dataset_names_global, tst_filenames_global = morgoth_10minfea_matrix(df_test)

#-------- Save the 591*17 dim input data for train validation and test cases ----------
results_dir = NESI_ROOT / "Results"
results_dir.mkdir(parents=True, exist_ok=True)
import pickle

with open(results_dir / "NESITripletIP_Morgoth_train_data.pkl", "wb") as f:
    pickle.dump(X_train_data_global, f)
print('Input Train data saved!')

with open(results_dir / "NESITripletIP_Morgoth_val_data.pkl", "wb") as f:
    pickle.dump(X_val_data_global, f)
print('Input Validation data saved!')

with open(results_dir / "NESITripletIP_Morgoth_test_data.pkl", "wb") as f:
    pickle.dump(X_tst_data_global, f)
print('Input Test data saved!')

#--------------------------------------------------------------
# YOU CAN LOAD THEM DIRECTLY FROM HERE AND RUN FROM HERE
#--------------------------------------------------------------

X_train_data_global_trch = torch.tensor(X_train_data_global, dtype=torch.float32)
X_val_data_global_trch   = torch.tensor(X_val_data_global, dtype=torch.float32)
X_tst_data_global_trch   = torch.tensor(X_tst_data_global, dtype=torch.float32)

Ygrouped_train_global_trch = torch.tensor(Ygrouped_train_global, dtype=torch.long)
Ygrouped_val_global_trch   = torch.tensor(Ygrouped_val_global, dtype=torch.long)
Ygrouped_tst_global_trch   = torch.tensor(Ygrouped_tst_global, dtype=torch.long)

Yraw_train_global_trch = torch.tensor(Yraw_train_global, dtype=torch.long)
Yraw_val_global_trch   = torch.tensor(Yraw_val_global, dtype=torch.long)
Yraw_tst_global_trch   = torch.tensor(Yraw_tst_global, dtype=torch.long)

train_dataset_names_global = np.array(train_dataset_names_global)
val_dataset_names_global   = np.array(val_dataset_names_global)
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
train_ds_all_global = TensorDataset(X_train_data_global_trch)
val_ds_all_global   = TensorDataset(X_val_data_global_trch)
tst_ds_all_global   = TensorDataset(X_tst_data_global_trch)

train_dl_all_global = DataLoader(train_ds_all_global, batch_size=128, shuffle=False)
val_dl_all_global   = DataLoader(val_ds_all_global, batch_size=128, shuffle=False)
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
train_embeddings_global = get_embeddings(model_trained, train_dl_all_global)
val_embeddings_global = get_embeddings(model_trained, val_dl_all_global)
tst_embeddings_global = get_embeddings(model_trained, tst_dl_all_global)

print("Train embeddings shape:", train_embeddings_global.shape)
print("Val embeddings shape:", val_embeddings_global.shape)
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

train_scores_global = get_scores(model_score, train_embeddings_global, device=device)
val_scores_global   = get_scores(model_score, val_embeddings_global, device=device)
tst_scores_global  = get_scores(model_score, tst_embeddings_global, device=device)

print("Train EEG NESI shape:", train_scores_global.shape)
print("Val EEG NESI shape:", val_scores_global.shape)
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

plot_score_distributions_by_raw(
    predicted_scores=train_scores_global,
    y_raw=Yraw_train_global,
    dataset_names=train_dataset_names_global,
    title="Train Set NESI Distribution"
)
plot_score_distributions_by_raw(
    predicted_scores=val_scores_global,
    y_raw=Yraw_val_global,
    dataset_names=val_dataset_names_global,
    title="Validation Set NESI Distribution"
)
plot_score_distributions_by_raw(
    predicted_scores=tst_scores_global,
    y_raw=Yraw_tst_global,
    dataset_names=tst_dataset_names_global,
    title="Test Set NESI Distribution"
)

## **Save the NESI and TRiplet embeddings in your suitable location**

import pandas as pd

df_train_results = pd.DataFrame({
    "Filename": trn_filenames_global,
    "Dataset": train_dataset_names_global,
    "TrueScore": Yraw_train_global,
    "TransformedTrueScore": Yraw_transformed_train_global,
    "PredictedNESI": train_scores_global
})
df_train_results["WhichSet"] = "Train"


df_val_results = pd.DataFrame({
    "Filename": val_filenames_global,
    "Dataset": val_dataset_names_global,
    "TrueScore": Yraw_val_global,
    "TransformedTrueScore": Yraw_transformed_val_global,
    "PredictedNESI": val_scores_global
})
df_val_results["WhichSet"] = "Validation"


df_tst_results = pd.DataFrame({
    "Filename": tst_filenames_global,
    "Dataset": tst_dataset_names_global,
    "TrueScore": Yraw_tst_global,
    "TransformedTrueScore": Yraw_transformed_tst_global,
    "PredictedNESI": tst_scores_global
})
df_tst_results["WhichSet"] = "Test"

results_dir = NESI_ROOT / "Results"
df_train_results.to_csv(results_dir / "NESI_train_results.csv", index=False)
df_val_results.to_csv(results_dir / "NESI_val_results.csv", index=False)
df_tst_results.to_csv(results_dir / "NESI_test_results.csv", index=False)

# Save Embeddings 
import pickle

with open(results_dir / "NESI_train_embeddings.pkl", "wb") as f:
    pickle.dump(train_embeddings_global, f)

with open(results_dir / "NESI_val_embeddings.pkl", "wb") as f:
    pickle.dump(val_embeddings_global, f)

with open(results_dir / "NESI_test_embeddings.pkl", "wb") as f:
    pickle.dump(tst_embeddings_global, f)