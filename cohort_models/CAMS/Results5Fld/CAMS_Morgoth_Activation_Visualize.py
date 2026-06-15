#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

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
from sklearn.svm import SVC
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


# In[ ]:


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


# # **Morgoth Activation Visualization Code**

# In[ ]:


def morgoth_10minfea_matrix_stat_for_visualization(data_frame, statistic):
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
        
        if statistic == 'mean':
            stat_10min_fea = np.mean(sub_morgoth_fea, axis=0) 
        elif statistic == 'median':
            stat_10min_fea = np.median(sub_morgoth_fea, axis=0)

        all_subject_features.append(stat_10min_fea)
        all_file_names.append(file_name)

    all_subject_features_matrix = np.stack(all_subject_features, axis=0)
    train_labels=data_frame['CAMS_grouped'].values .flatten()  
    raw_labels=data_frame['CAMS_SF'].values .flatten()  
    
    print('Label size==> ',train_labels.shape)
    print('RAW Label size==> ',raw_labels.shape)
    print("Stacked feature matrix shape:", all_subject_features_matrix.shape)
    
    return all_subject_features_matrix, train_labels, raw_labels

# ------------- Morgoth feature visualization ----------------------------
def plot_cams_imagesc(
    X,
    Y,
    cams_value,
    cams_value_type,   
    feature_names,
    max_samples=None,
    figsize=(14, 6)
):
    """
    imagesc-style visualization for one RASS group.

    X : np.ndarray (N, 17)
        Probability features
    Y : np.ndarray (N,)
        RASS labels
    cams_value : int
        RASS group to visualize
    feature_names : list of str
        Names of 17 features
    max_samples : int or None
        If set, randomly subsample columns for visualization
    """

    X_iacns = X[Y == cams_value]

    if X_iacns.shape[0] == 0:
        print(f"No samples for RASS {cams_value}")
        return

    # Optional subsampling (for readability)
    if max_samples is not None and X_iacns.shape[0] > max_samples:
        np.random.seed(42)
        idx = np.random.choice(X_iacns.shape[0], max_samples, replace=False)
        X_iacns = X_iacns[idx]

    # Transpose → (features × samples)
    img = X_iacns.T

    plt.figure(figsize=figsize)
    plt.imshow(
        img,
        aspect='auto',
        cmap='Blues',
        vmin=0,
        vmax=1
    )

    plt.colorbar(label='Probability Activation')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Observations')
    plt.ylabel('EEG Features')
    plt.title(
        f'Morgoth Activation Map – CAMS {cams_value_type}',
        fontsize=13,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()

feature_names = [
    'Awake','N1','N2', # Sleep head output of morgoth
    'Normal/Abnormal', # NM head output
    'Burst/No Burst', # BS head output
    'No Spike','Focal Spike','Generalized Spike', # Spike localize head
    'No Slowing','Focal Slowing','Generalized Slowing', # Slowing head
    'Other','Seizure','LPD','GPD','LRDA','GRDA' # IIIC head
]
feature_stat1='median'
X, Y, Y_raw = morgoth_10minfea_matrix_stat_for_visualization(df_CAMS_metadata, feature_stat1)


# # **3-class (Mild/No: 0-1, Moderate: 2-5, Mild: 6-7)**

# In[ ]:


plot_cams_imagesc(
    X=X,
    Y=Y,
    cams_value=0,
    cams_value_type ='0-1 (Mild)',
    feature_names=feature_names
)
plot_cams_imagesc(
    X=X,
    Y=Y,
    cams_value=1,
    cams_value_type ='2-5 (Moderate)',
    feature_names=feature_names
)
plot_cams_imagesc(
    X=X,
    Y=Y,
    cams_value=2,
    cams_value_type ='6-7 (Severe)',
    feature_names=feature_names
)


# # **Indivividual classes ctivation visulaization**

# In[ ]:


plot_cams_imagesc(
    X=X,
    Y=Y_raw,
    cams_value=0,
    cams_value_type ='0',
    feature_names=feature_names
)
plot_cams_imagesc(
    X=X,
    Y=Y_raw,
    cams_value=1,
    cams_value_type ='1',
    feature_names=feature_names
)
plot_cams_imagesc(
    X=X,
    Y=Y_raw,
    cams_value=2,
    cams_value_type ='2',
    feature_names=feature_names
)
plot_cams_imagesc(
    X=X,
    Y=Y_raw,
    cams_value=0,
    cams_value_type ='3',
    feature_names=feature_names
)
plot_cams_imagesc(
    X=X,
    Y=Y_raw,
    cams_value=1,
    cams_value_type ='4',
    feature_names=feature_names
)
plot_cams_imagesc(
    X=X,
    Y=Y_raw,
    cams_value=2,
    cams_value_type ='5',
    feature_names=feature_names
)
plot_cams_imagesc(
    X=X,
    Y=Y_raw,
    cams_value=0,
    cams_value_type ='6',
    feature_names=feature_names
)
plot_cams_imagesc(
    X=X,
    Y=Y_raw,
    cams_value=1,
    cams_value_type ='7',
    feature_names=feature_names
)

