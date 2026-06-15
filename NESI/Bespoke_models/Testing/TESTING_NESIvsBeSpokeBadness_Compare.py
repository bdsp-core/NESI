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


# # **Load Full Dataset**

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


# # **Split subject independently**

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


# # **Feature Engineering**

# In[4]:


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


# # **NESI/Bespoke Model Helper functions**

# In[5]:


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


# In[6]:


def get_scores(model, embeddings, device, batch_size=128):
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

def get_triplet_embeddings_NESI(feature_input_data, triplet_model_path, badnessmodel_path, device):
    # --------- LOAD MODEL ----------
    model_triplet = MORGOTH_ResNet1D_onlyGAP(num_features=17)
    model_triplet.load_state_dict(
        torch.load(triplet_model_path,
                   map_location=device,
                   weights_only=True)
    )
    model_triplet = model_triplet.to(device)
    model_triplet.eval()

    # --------- CREATE LOADERS ----------
    print('Feature input data shape', feature_input_data.shape)
    ds_all_global = TensorDataset(feature_input_data)    
    dl_all_global = DataLoader(ds_all_global, batch_size=128, shuffle=False)

    # --------- FUNCTION TO EXTRACT EMBEDDINGS ----------
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
    embeddings_global = get_embeddings(model_triplet, dl_all_global)
    
    print("embeddings shape:", embeddings_global.shape)

    checkpoint_badnessmodel = torch.load(
    badnessmodel_path,
    map_location=device,
    weights_only=True
    )
    
    badnessmodel_score = EEGScoringModel()
    badnessmodel_score.load_state_dict(checkpoint_badnessmodel["model_state_dict"])
    badnessmodel_score.to(device)
    
    badnessmodel_score.eval()
    
    scores_global = get_scores(badnessmodel_score, embeddings_global, device=device)
    
    print("EEG Badness score shape:", scores_global.shape)
    return embeddings_global, scores_global


# # **For NESI model execute the score**

# In[7]:


X_data_forNESI, Ygrouped_forNESI, Yraw_forNESI, dataset_names_forNESI, filenames_forNESI = morgoth_10minfea_matrix(df_all_metadata)

X_data_forNESI_trch = torch.tensor(X_data_forNESI, dtype=torch.float32)
Ygrouped_forNESI_trch = torch.tensor(Ygrouped_forNESI, dtype=torch.long)
Yraw_forNESI_trch = torch.tensor(Yraw_forNESI, dtype=torch.long)
filenames_forNESI = np.array(filenames_forNESI)


# In[8]:


triplet_model_path = NESI_ROOT / "model" / "ModelCheckpoints" / "ResNetGAP_BestModel.pth"
badnessmodel_path = NESI_ROOT / "model" / "ModelCheckpoints" / "NESI_best_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu" 

NESI_triplet_embeddings, NESI =get_triplet_embeddings_NESI(X_data_forNESI_trch, triplet_model_path, badnessmodel_path, device)


# In[9]:


import numpy as np

# Convert torch tensor to numpy if needed
if hasattr(NESI_triplet_embeddings, "cpu"):
    embeddings_np = NESI_triplet_embeddings.cpu().numpy()
else:
    embeddings_np = np.asarray(NESI_triplet_embeddings)

df_NESI_results = df_all_metadata.copy()
df_NESI_results["NESI"] = NESI

# Store each 40-D embedding as a numpy array in one dataframe cell
df_NESI_results["NESITripletEmbedding"] = list(embeddings_np)

print(len(df_NESI_results))
print(df_NESI_results["NESITripletEmbedding"].iloc[0].shape)
print(df_NESI_results["NESI"].shape)


# In[10]:


df_NESI_results


# # **Bespoke model results**

# ## **RASS Bespoke**

# In[32]:


df_RASS = df_all_metadata[(df_all_metadata['Dataset'] == 'RASS')].reset_index(drop=True)
X_data_forRASSbadness, Ygrouped_forRASSbadness, Yraw_forRASSbadness, dataset_names_forRASSbadness, filenames_forRASSbadness = morgoth_10minfea_matrix(df_RASS)

X_data_forRASSbadness_trch = torch.tensor(X_data_forRASSbadness, dtype=torch.float32)
Ygrouped_forRASSbadness_trch = torch.tensor(Ygrouped_forRASSbadness, dtype=torch.long)
Yraw_forRASSbadness_trch = torch.tensor(Yraw_forRASSbadness, dtype=torch.long)
filenames_forRASSbadness = np.array(filenames_forRASSbadness)


# In[33]:


triplet_model_path = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints" / "TripletCheckpoint" / "cohort_models" / "RASS" / "ResNetGAP_RASS_BestModel.pth"
badnessmodel_path = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints" / "BespokeBadnessCheckpoint" / "cohort_models" / "RASS" / "RASS_Bespoke_Badness_Bet_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu" 

RASSBespoke_triplet_embeddings, RASSBespoke_badness = get_triplet_embeddings_NESI(
    X_data_forRASSbadness_trch, triplet_model_path, badnessmodel_path, device)


# In[34]:


import numpy as np

# Convert torch tensor to numpy if needed
if hasattr(RASSBespoke_triplet_embeddings, "cpu"):
    embeddings_np = RASSBespoke_triplet_embeddings.cpu().numpy()
else:
    embeddings_np = np.asarray(RASSBespoke_triplet_embeddings)

df_RASS = df_RASS.copy()
df_RASS["BespokeBadnessScore"] = RASSBespoke_badness

# Store each 40-D embedding as a numpy array in one dataframe cell
df_RASS["BespokeTripletEmbedding"] = list(embeddings_np)

print(len(df_RASS))
print(df_RASS["BespokeTripletEmbedding"].iloc[0].shape)
print(df_RASS["BespokeBadnessScore"].shape)


# In[36]:

# Save to suitable location
#df_RASS.to_csv('/home/ayush/Desktop/Bespoke_models_new/Results/RASS_full_dataset_results.csv', index= False)


# In[ ]:





# ## **GCS Bespoke**

# In[29]:


df_GCS = df_all_metadata[(df_all_metadata['Dataset'] == 'GCS')].reset_index(drop=True)
X_data_forGCSbadness, Ygrouped_forGCSbadness, Yraw_forGCSbadness, dataset_names_forGCSbadness, filenames_forGCSbadness = morgoth_10minfea_matrix(df_GCS)

X_data_forGCSbadness_trch = torch.tensor(X_data_forGCSbadness, dtype=torch.float32)
Ygrouped_forGCSbadness_trch = torch.tensor(Ygrouped_forGCSbadness, dtype=torch.long)
Yraw_forGCSbadness_trch = torch.tensor(Yraw_forGCSbadness, dtype=torch.long)
filenames_forGCSbadness = np.array(filenames_forGCSbadness)


# In[30]:

triplet_model_path = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints" / "TripletCheckpoint" / "cohort_models" / "GCS" / "ResNetGAP_GCS_BestModel.pth"
badnessmodel_path = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints" / "BespokeBadnessCheckpoint" / "cohort_models" / "GCS" / "GCS_Bespoke_Badness_Bet_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu" 

GCSBespoke_triplet_embeddings, GCSBespoke_badness = get_triplet_embeddings_NESI(
    X_data_forGCSbadness_trch, triplet_model_path, badnessmodel_path, device)


# In[31]:


import numpy as np

# Convert torch tensor to numpy if needed
if hasattr(GCSBespoke_triplet_embeddings, "cpu"):
    embeddings_np = GCSBespoke_triplet_embeddings.cpu().numpy()
else:
    embeddings_np = np.asarray(GCSBespoke_triplet_embeddings)

df_GCS = df_GCS.copy()
df_GCS["BespokeBadnessScore"] = GCSBespoke_badness

# Store each 40-D embedding as a numpy array in one dataframe cell
df_GCS["BespokeTripletEmbedding"] = list(embeddings_np)

print(len(df_GCS))
print(df_GCS["BespokeTripletEmbedding"].iloc[0].shape)
print(df_GCS["BespokeBadnessScore"].shape)


# In[35]:

# save at suitable location
#df_GCS.to_csv('/home/ayush/Desktop/Bespoke_models_new/Results/GCS_full_dataset_results.csv', index= False)


# ## **CAMS Bespoke**

# In[18]:


df_CAMS = df_all_metadata[(df_all_metadata['Dataset'] == 'CAMS')].reset_index(drop=True)
X_data_forCAMSbadness, Ygrouped_forCAMSbadness, Yraw_forCAMSbadness, dataset_names_forCAMSbadness, filenames_forCAMSbadness = morgoth_10minfea_matrix(df_CAMS)

X_data_forCAMSbadness_trch = torch.tensor(X_data_forCAMSbadness, dtype=torch.float32)
Ygrouped_forCAMSbadness_trch = torch.tensor(Ygrouped_forCAMSbadness, dtype=torch.long)
Yraw_forCAMSbadness_trch = torch.tensor(Yraw_forCAMSbadness, dtype=torch.long)
filenames_forCAMSbadness = np.array(filenames_forCAMSbadness)


# In[19]:
triplet_model_path = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints" / "TripletCheckpoint" / "cohort_models" / "CAMS" / "ResNetGAP_CAMS_BestModel.pth"
badnessmodel_path = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints" / "BespokeBadnessCheckpoint" / "cohort_models" / "CAMS" / "CAMS_Bespoke_Badness_Bet_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu" 

CAMSBespoke_triplet_embeddings, CAMSBespoke_badness = get_triplet_embeddings_NESI(
    X_data_forCAMSbadness_trch, triplet_model_path, badnessmodel_path, device)


# In[20]:


import numpy as np

# Convert torch tensor to numpy if needed
if hasattr(CAMSBespoke_triplet_embeddings, "cpu"):
    embeddings_np = CAMSBespoke_triplet_embeddings.cpu().numpy()
else:
    embeddings_np = np.asarray(CAMSBespoke_triplet_embeddings)

df_CAMS = df_CAMS.copy()
df_CAMS["BespokeBadnessScore"] = CAMSBespoke_badness

# Store each 40-D embedding as a numpy array in one dataframe cell
df_CAMS["BespokeTripletEmbedding"] = list(embeddings_np)

print(len(df_CAMS))
print(df_CAMS["BespokeTripletEmbedding"].iloc[0].shape)
print(df_CAMS["BespokeBadnessScore"].shape)


# In[27]:

# Save at suitable location
#df_CAMS.to_csv('/home/ayush/Desktop/Bespoke_models_new/Results/CAMS_full_dataset_results.csv', index= False)


# ## **ICANS Bespoke**

# In[23]:


df_ICANS = df_all_metadata[(df_all_metadata['Dataset'] == 'ICANS')].reset_index(drop=True)
X_data_forICANSbadness, Ygrouped_forICANSbadness, Yraw_forICANSbadness, dataset_names_forICANSbadness, filenames_forICANSbadness = morgoth_10minfea_matrix(df_ICANS)

X_data_forICANSbadness_trch = torch.tensor(X_data_forICANSbadness, dtype=torch.float32)
Ygrouped_forICANSbadness_trch = torch.tensor(Ygrouped_forICANSbadness, dtype=torch.long)
Yraw_forICANSbadness_trch = torch.tensor(Yraw_forICANSbadness, dtype=torch.long)
filenames_forICANSbadness = np.array(filenames_forICANSbadness)


# In[25]:
triplet_model_path = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints" / "TripletCheckpoint" / "cohort_models" / "ICANS" / "ResNetGAP_ICANS_BestModel.pth"
badnessmodel_path = NSEI_ROOT / "Bespoke_models"/ "ModelCheckpoints" / "BespokeBadnessCheckpoint" / "cohort_models" / "ICANS" / "ICANS_Bespoke_Badness_Bet_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu" 

ICANSBespoke_triplet_embeddings, ICANSBespoke_badness = get_triplet_embeddings_NESI(
    X_data_forICANSbadness_trch, triplet_model_path, badnessmodel_path, device)


# In[26]:


import numpy as np

# Convert torch tensor to numpy if needed
if hasattr(ICANSBespoke_triplet_embeddings, "cpu"):
    embeddings_np = ICANSBespoke_triplet_embeddings.cpu().numpy()
else:
    embeddings_np = np.asarray(ICANSBespoke_triplet_embeddings)

df_ICANS = df_ICANS.copy()
df_ICANS["BespokeBadnessScore"] = ICANSBespoke_badness

# Store each 40-D embedding as a numpy array in one dataframe cell
df_ICANS["BespokeTripletEmbedding"] = list(embeddings_np)

print(len(df_ICANS))
print(df_ICANS["BespokeTripletEmbedding"].iloc[0].shape)
print(df_ICANS["BespokeBadnessScore"].shape)


# In[28]:

# Save at suitable location
# df_ICANS.to_csv('/home/ayush/Desktop/Bespoke_models_new/Results/ICANS_full_dataset_results.csv', index= False)


# In[ ]:




