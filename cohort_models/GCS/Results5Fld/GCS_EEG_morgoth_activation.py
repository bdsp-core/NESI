#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
import h5py
import hdf5storage
from pathlib import Path
import seaborn as sns


# In[2]:

current = Path(__file__).resolve()
GCS_ROOT = None
for parent in current.parents:
    if parent.name == "GCS":
        GCS_ROOT = parent
        break

if GCS_ROOT is None:
    raise RuntimeError("GCS folder not found")

metadata_path_GCS = GCS_ROOT / "model" / "Training" / "GCSTraining_Final_Metadata.csv"
df_gcs_subs=pd.read_csv(metadata_path_GCS)

def extract_pid(filename):
    part = filename.split('_')[0]          
    pid_full = part.split('-')[1]          
    pid = pid_full[5:]                     
    return pid

# Apply to all rows and create new column for BDSPPatientID
df_gcs_subs['BDSPPatientID'] = df_gcs_subs['Filename'].apply(extract_pid)
df_gcs_subs = df_gcs_subs[['BDSPPatientID', 'Filename', 'GCS_value']]

# In[3]:

def morgoth_fea_matrix_generate(data_frame):
    filenames = data_frame['Filename'].tolist()
    
    BASE = GCS_ROOT / "MorgothActivations"

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
    train_labels=data_frame['GCS_value'].values .flatten()  
    print('Label size==> ',train_labels.shape)
    print("Stacked feature matrix shape:", all_subject_features_matrix.shape)
    return all_subject_features_matrix, train_labels, all_file_names


# In[4]:


X_full, Y_full, full_filenames = morgoth_fea_matrix_generate(df_gcs_subs)


# In[5]:
print('\n')
print('Full Feature Shape: '+str(X_full.shape))
print('Full Label Shape: '+str(Y_full.shape))
print('Unique classes in full Label Shape: '+str(np.unique(Y_full)))
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

def plot_gcs_imagesc(
    X,
    Y,
    gcs_value,
    feature_names,
    max_samples=None,
    figsize=(10, 6)
):

    X_gcs = X[Y == gcs_value]

    if X_gcs.shape[0] == 0:
        print(f"No samples for GCS {gcs_value}")
        return

    if max_samples is not None and X_gcs.shape[0] > max_samples:
        np.random.seed(42)
        idx = np.random.choice(X_gcs.shape[0], max_samples, replace=False)
        X_gcs = X_gcs[idx]

    img = X_gcs.T  # (features × samples)

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
    plt.xlabel(f"Observations / No. of 10 min EEG in GCS {gcs_value} Class")
    plt.ylabel('EEG Features')
    plt.title(
        f'MORGOTH Feature Activation Map – GCS {gcs_value}',
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


# ## **GCS: 3**

# In[9]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=3,
    feature_names=feature_names
)


# ## **GCS: 4**

# In[10]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=4,
    feature_names=feature_names
)


# ## **GCS: 5**

# In[11]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=5,
    feature_names=feature_names
)


# ## **GCS: 6**

# In[12]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=6,
    feature_names=feature_names
)


# ## **GCS: 7**

# In[13]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=7,
    feature_names=feature_names
)


# ## **GCS: 8**

# In[14]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=8,
    feature_names=feature_names
)


# ## **GCS: 9**

# In[14]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=9,
    feature_names=feature_names
)

# ## **GCS: 10**

# In[14]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=10,
    feature_names=feature_names
)

# ## **GCS: 11**

# In[14]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=11,
    feature_names=feature_names
)

# ## **GCS: 12**

# In[14]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=12,
    feature_names=feature_names
)

# ## **GCS: 13**

# In[14]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=13,
    feature_names=feature_names
)

# ## **GCS: 14**

# In[14]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=14,
    feature_names=feature_names
)

# ## **GCS: 15**

# In[14]:


plot_gcs_imagesc(
    X=X_full,
    Y=Y_full,
    gcs_value=15,
    feature_names=feature_names
)