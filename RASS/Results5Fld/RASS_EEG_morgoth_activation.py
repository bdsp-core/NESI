#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
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
from pathlib import Path
import seaborn as sns


# In[2]:

current = Path(__file__).resolve()
RASS_ROOT = None
for parent in current.parents:
    if parent.name == "RASS":
        RASS_ROOT = parent
        break

if RASS_ROOT is None:
    raise RuntimeError("RASS folder not found")

metadata_path_RASS = RASS_ROOT / "model" / "Training" / "RASSTraining_Final_Metadata.csv"
df_rass_subs=pd.read_csv(metadata_path_RASS)

def extract_pid(filename):
    part = filename.split('_')[0]          
    pid_full = part.split('-')[1]          
    pid = pid_full[5:]                     
    return pid

# Apply to all rows and create new column for BDSPPatientID
df_rass_subs['BDSPPatientID'] = df_rass_subs['Filename'].apply(extract_pid)
df_rass_subs = df_rass_subs[['BDSPPatientID', 'Filename', 'RASS_value']]

# Get a subject independent split: The random seed is 42 (Use this to reproduce result) 
subs_rass=df_rass_subs['BDSPPatientID'].unique()
train_subs, test_subs = train_test_split(subs_rass, test_size=0.15, random_state=42)
df_rass_subs['Split'] = df_rass_subs['BDSPPatientID'].apply(lambda x: 'Train' if x in train_subs else 'Test')

# Train dataframe
df_rass_train = df_rass_subs[df_rass_subs['Split'] == 'Train'].reset_index(drop=True)
# Test dataframe
df_rass_test = df_rass_subs[df_rass_subs['Split'] == 'Test'].reset_index(drop=True)

print("Train dataframe shape:", df_rass_train.shape)
print("Test dataframe shape:", df_rass_test.shape)
print('\n')

# In[3]:



def morgoth_fea_matrix_generate(data_frame, type_split):
    filenames = data_frame['Filename'].tolist()
    
    BASE = RASS_ROOT / "MorgothActivations"

    slowing_folder_loc = BASE / "SLOWING"
    focgen_folder_loc  = BASE / "FOCGEN"
    iiic_folder_loc    = BASE / "IIIC"
    nm_folder_loc      = BASE / "NM"
    bs_folder_loc      = BASE / "BS"
    sleep_folder_loc   = BASE / "SLEEP"


    all_subject_features = []; all_file_names=[];
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
        mean_features = np.median(sub_morgoth_fea, axis=0) 
        all_subject_features.append(mean_features)
        all_file_names.append(file_name)
        
    all_subject_features_matrix = np.stack(all_subject_features, axis=0)
    all_file_names = np.array(all_file_names)
    train_labels=data_frame['RASS_value'].values .flatten()  
    print('Label size==> ',train_labels.shape)
    print("Stacked feature matrix shape:", all_subject_features_matrix.shape)
    return all_subject_features_matrix, train_labels, all_file_names


def grouped_label_6class(feature_data, label_data, rass_groups, filename_data=None):
    keep_indices = np.isin(label_data, rass_groups)
    X_filtered = feature_data[keep_indices]
    Y_filtered = label_data[keep_indices]
    filename_filtered = filename_data[keep_indices] if filename_data is not None else None

    return X_filtered, Y_filtered, filename_filtered


# In[4]:


X_train, Y_train,  train_filenames = morgoth_fea_matrix_generate(df_rass_train, type_split='Train')
X_test, Y_test, test_filenames =morgoth_fea_matrix_generate(df_rass_test, type_split='Test')


# In[5]:


keep_indices_rass = [-5, -4, -3, -2, -1, 0]
X_train, Y_train, train_file_name = grouped_label_6class(X_train, Y_train, keep_indices_rass, train_filenames)
X_test, Y_test, test_file_name = grouped_label_6class(X_test, Y_test, keep_indices_rass, test_filenames)

print('\n')
print('Train Feature Shape: '+str(X_train.shape))
print('Train Label Shape: '+str(Y_train.shape))

print('Test Feature Shape: '+str(X_test.shape))
print('Test Label Shape: '+str(Y_test.shape))
print('\n')

# In[6]:


# col 1-3: Awake , N1 , N2 
# col 4: Normal-Abnormal (binary)
# col 5: Burst-no burst (binary)
# col 6-8: No spike ,  Focal Spike, Generalized Spike
# col 9-11: No slowing, Focal Slowing , Generalized Slowing
# col 12-17: Other , Seizure , LPD , GPD , LRDA , GRDA

feature_names = [
    'Awake','N1','N2', # Sleep head output of morgoth
    'Normal/Abnormal', # NM head output
    'Burst/No Burst', # BS head output
    'No Spike','Focal Spike','Generalized Spike', # Spike localize head
    'No Slowing','Focal Slowing','Generalized Slowing', # Slowing head
    'Other','Seizure','LPD','GPD','LRDA','GRDA' # IIIC head
]


# # **Visualize the morgoth feature heatmaps**

# In[8]:


import numpy as np
import matplotlib.pyplot as plt

def plot_rass_imagesc(
    X,
    Y,
    rass_value,
    feature_names,
    max_samples=None,
    figsize=(10, 6)
):

    X_rass = X[Y == rass_value]

    if X_rass.shape[0] == 0:
        print(f"No samples for RASS {rass_value}")
        return

    if max_samples is not None and X_rass.shape[0] > max_samples:
        np.random.seed(42)
        idx = np.random.choice(X_rass.shape[0], max_samples, replace=False)
        X_rass = X_rass[idx]

    img = X_rass.T  # (features × samples)

    plt.figure(figsize=figsize, dpi=200)
    plt.imshow(
        img,
        aspect='auto',
        cmap='Blues',
        vmin=0,
        vmax=1
    )

    plt.colorbar(label='Probability Activation')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel(f"Observations / No. of 10 min EEG in RASS {rass_value} Class")
    plt.ylabel('EEG Features')
    plt.title(
        f'MORGOTH Feature Activation Map – RASS {rass_value}',
        fontsize=6,
        fontweight='bold'
    )

    # ---- Draw red dotted separators between heads ----
    head_boundaries = [2, 3, 4, 7, 10]  # row indices

    for b in head_boundaries:
        plt.axhline(
            y=b + 0.5,
            color='red',
            linestyle=':',
            linewidth=1.5,
            alpha=0.9
        )

    plt.tight_layout()
    plt.show()


# ## **RASS: -5**

# In[9]:

import numpy as np

X_full = np.concatenate([X_train, X_test], axis=0)
Y_full = np.concatenate([Y_train, Y_test], axis=0)

print('Full Feature Shape:', X_full.shape)
print('Full Label Shape:', Y_full.shape) 

plot_rass_imagesc(
    X=X_full,
    Y=Y_full,
    rass_value=-5,
    feature_names=feature_names
)


# ## **RASS: -4**

# In[10]:


plot_rass_imagesc(
    X=X_full,
    Y=Y_full,
    rass_value=-4,
    feature_names=feature_names
)


# ## **RASS: -3**

# In[11]:


plot_rass_imagesc(
    X=X_full,
    Y=Y_full,
    rass_value=-3,
    feature_names=feature_names
)


# ## **RASS: -2**

# In[12]:


plot_rass_imagesc(
    X=X_full,
    Y=Y_full,
    rass_value=-2,
    feature_names=feature_names
)


# ## **RASS: -1**

# In[13]:


plot_rass_imagesc(
    X=X_full,
    Y=Y_full,
    rass_value=-1,
    feature_names=feature_names
)


# ## **RASS: 0**

# ## SS0

# In[14]:


plot_rass_imagesc(
    X=X_full,
    Y=Y_full,
    rass_value=0,
    feature_names=feature_names
)
