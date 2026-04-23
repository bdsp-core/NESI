#!/usr/bin/env python
# coding: utf-8

# # **Libraries**

# In[1]:


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


# # **Data curation for model**

# In[2]:
current = Path(__file__).resolve()

ICANS_ROOT = None
for parent in current.parents:
    if parent.name == "ICANS":
        ICANS_ROOT = parent
        break

if ICANS_ROOT is None:
    raise RuntimeError("ICANS folder not found")

# build path relative to repo
metadata_path = ICANS_ROOT / "model" / "Training" / "ICANSTraining_Final_Metadata.csv"
df_ICANS_metadata = pd.read_csv(metadata_path)


# # **Morgoth feature activation**

# In[5]:


# ------------------------------- EEG feature statistics collection -------------------------
BASE = ICANS_ROOT / "MorgothActivations"

slowing_folder_loc = BASE / "SLOWING"
focgen_folder_loc  = BASE / "FOCGEN"
iiic_folder_loc    = BASE / "IIIC"
nm_folder_loc      = BASE / "NM"
bs_folder_loc      = BASE / "BS"
sleep_folder_loc   = BASE / "SLEEP"

def morgoth_10minfea_matrix_stat_for(data_frame, statistic):
    filenames = data_frame['MorgothOutputFilename'].tolist()
    all_subject_features = []
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

    all_subject_features_matrix = np.stack(all_subject_features, axis=0)
    train_labels=data_frame['ICANS_grouped'].values .flatten()  
    print('Label size==> ',train_labels.shape)
    print("Stacked feature matrix shape:", all_subject_features_matrix.shape)
    return all_subject_features_matrix, train_labels


feature_names = [
    'Awake','N1','N2', # Sleep head output of morgoth
    'Normal/Abnormal', # NM head output
    'Burst/No Burst', # BS head output
    'No Spike','Focal Spike','Generalized Spike', # Spike localize head
    'No Slowing','Focal Slowing','Generalized Slowing', # Slowing head
    'Other','Seizure','LPD','GPD','LRDA','GRDA' # IIIC head
]
feature_stat1='median'
X, Y = morgoth_10minfea_matrix_stat_for(df_ICANS_metadata, feature_stat1)


# In[9]:


def plot_icans_imagesc(
    X,
    Y,
    icans_value,
    icans_value_type,   
    feature_names,
    max_samples=None,
    figsize=(14, 6)
):
    """
    imagesc-style visualization for one ICANS group.

    X : np.ndarray (N, 17)
        Probability features
    Y : np.ndarray (N,)
        ICANS labels
    icans_value : int
        ICANS group to visualize
    feature_names : list of str
        Names of 17 features
    max_samples : int or None
        If set, randomly subsample columns for visualization
    """

    X_iacns = X[Y == icans_value]

    if X_iacns.shape[0] == 0:
        print(f"No samples for ICANS {icans_value}")
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
        f'Raw Feature Activation Map (imagesc) – ICANS {icans_value_type}',
        fontsize=13,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


# ## **ICANS-0: morgoth activation**

# In[10]:


plot_icans_imagesc(
    X=X,
    Y=Y,
    icans_value=0,
    icans_value_type ='0 (Mild)',
    feature_names=feature_names
)
plot_icans_imagesc(
    X=X,
    Y=Y,
    icans_value=1,
    icans_value_type ='1-2 (Moderate)',
    feature_names=feature_names
)
plot_icans_imagesc(
    X=X,
    Y=Y,
    icans_value=2,
    icans_value_type ='3-4 (Severe)',
    feature_names=feature_names
)