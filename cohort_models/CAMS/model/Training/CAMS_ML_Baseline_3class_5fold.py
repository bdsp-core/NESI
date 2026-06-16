#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


# # **Helper function**

# ## **Feature Engineering**

# In[2]:

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
def morgoth_10minfea_matrix_stat_for(data_frame, statistic):
    """
    Collect features for all subjects using mean/median or flattened covariance upper-triangle.

    Args:
        data_frame: pd.DataFrame with column 'MorgothOutputFilename' and 'CAMS_SF'
        statistic: 'mean', 'median', or 'cov_upper' for flattened upper-triangle covariance

    Returns:
        all_subject_features_matrix: np.ndarray, shape = (num_subjects, num_features)
        train_labels: np.ndarray, shape = (num_subjects,)
    """
    
    BASE = CAMS_ROOT / "MorgothActivations"

    slowing_folder_loc = BASE / "SLOWING"
    focgen_folder_loc  = BASE / "FOCGEN"
    iiic_folder_loc    = BASE / "IIIC"
    nm_folder_loc      = BASE / "NM"
    bs_folder_loc      = BASE / "BS"
    sleep_folder_loc   = BASE / "SLEEP"

    filenames = data_frame['MorgothOutputFilename'].tolist()
    all_subject_features = []

    for file_name in tqdm(filenames, desc="Processing subjects"):
        
        # Load all feature types
        def load_features(folder_loc):
            df = pd.read_csv(os.path.join(folder_loc, file_name))
            if 'pred_class' in df.columns:
                df = df.drop(columns='pred_class')
            return df.values
        
        sleep_fea = load_features(sleep_folder_loc)
        nm_fea    = load_features(nm_folder_loc)
        bs_fea    = load_features(bs_folder_loc)
        focgen_fea= load_features(focgen_folder_loc)
        slowing_fea = load_features(slowing_folder_loc)
        iiic_fea  = load_features(iiic_folder_loc)
        
        # Concatenate along feature axis (columns)
        sub_morgoth_fea = np.concatenate([sleep_fea, nm_fea, bs_fea, 
                                          focgen_fea, slowing_fea, iiic_fea], axis=1)
        
        # Compute the requested statistic
        if statistic == 'mean':
            stat_10min_fea = np.mean(sub_morgoth_fea, axis=0)
        elif statistic == 'median':
            stat_10min_fea = np.median(sub_morgoth_fea, axis=0)
        elif statistic == 'cov_upper':
            # Covariance across features (columns)
            cov_matrix = np.cov(sub_morgoth_fea, rowvar=False)  # shape: (num_features, num_features)
            # Take upper triangle without diagonal
            triu_indices = np.triu_indices(cov_matrix.shape[0], k=1)
            stat_10min_fea = cov_matrix[triu_indices]           # flattened upper triangle
        else:
            raise ValueError("statistic must be 'mean', 'median', or 'cov_upper'")
        
        all_subject_features.append(stat_10min_fea)

    all_subject_features_matrix = np.stack(all_subject_features, axis=0)
    train_labels = data_frame['CAMS_grouped'].values.flatten()

    print("Stacked feature matrix shape:", all_subject_features_matrix.shape)
    return all_subject_features_matrix, train_labels


# # **K fold cross vlidation**

# In[3]:


def LR_kfold_stratified_group(df_CAMS_metadata, feature_stat, base_model_dir, n_splits=5):
    """
    Stratified group k-fold cross-validation (~15% test per fold) 
    ensuring no subject (BDSPPatientID) appears in both train and test.
    
    Args:
        df_CAMS_metadata: DataFrame with at least ['BDSPPatientID', 'CAMS_grouped']
        n_splits: number of folds (default 7 → ~15% test per fold)
        bs: batch size
    Returns:
        fold_results: list of dicts containing predictions, logits, metrics per fold
    """
    
    base_model_dir = os.path.join(base_model_dir, "LR")
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    # Unique subjects and labels
    df_subjects = df_CAMS_metadata[['BDSPPatientID','CAMS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['CAMS_grouped'].values
    groups = df_subjects['BDSPPatientID'].values  # important to prevent overlap
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
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
        
        # Feature extraction
        X_train_fea, Y_train = morgoth_10minfea_matrix_stat_for(df_CAMS_train, feature_stat)
        X_test_fea, Y_test   = morgoth_10minfea_matrix_stat_for(df_CAMS_test, feature_stat)

        # Model
        model = LogisticRegression(
            penalty='l2',
            C=5.0,
            solver='lbfgs',
            max_iter=5000,
            class_weight='balanced',
            tol=1e-5,
            random_state=42,
            multi_class='multinomial'
        )
        model.fit(X_train_fea, Y_train)

        # Predictions
        Y_pred = model.predict(X_test_fea)
        Y_prob = model.predict_proba(X_test_fea)

        # ---- AUC computation ----
        classes = np.unique(Y_train)
        Y_test_binarized = label_binarize(Y_test, classes=classes)
        auc_per_class = {}
        for i, cls in enumerate(classes):
            auc_per_class[cls] = roc_auc_score(Y_test_binarized[:, i], Y_prob[:, i])

        # Macro AUC
        auc_macro = np.mean(list(auc_per_class.values()))
        
        # Metrics
        acc  = accuracy_score(Y_test, Y_pred)
        pre_macro = precision_score(Y_test, Y_pred, average='macro')
        rec_macro = recall_score(Y_test, Y_pred, average='macro')
        f1_macro  = f1_score(Y_test, Y_pred, average='macro')

        pre_micro = precision_score(Y_test, Y_pred, average='micro')
        rec_micro = recall_score(Y_test, Y_pred, average='micro')
        f1_micro  = f1_score(Y_test, Y_pred, average='micro')

        # Store results
        fold_results.append({
            'fold': fold+1,
            'accuracy': acc,
            'precision_macro': pre_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_micro': pre_micro,
            'recall_micro': rec_micro,
            'f1_micro': f1_micro,
            'auc_macro': auc_macro,
            'auc_per_class': auc_per_class,
            'Y_true':Y_test,
            'Y_pred':Y_pred
        })

        # Save model
        fold_model_dir = Path(base_model_dir) / feature_stat / f"Fold{fold+1}"
        print(fold_model_dir)
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        model_path = fold_model_dir / "LR_model.joblib"
        joblib.dump(model, model_path)

        # Confusion Matrix
        cm = confusion_matrix(Y_test, Y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        # Print metrics
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro -> Precision: {pre_macro:.4f}, Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}")
        print(f"Micro -> Precision: {pre_micro:.4f}, Recall: {rec_micro:.4f}, F1: {f1_micro:.4f}")

        print("*" * 100)

    return fold_results

#--------------------- SVM ----------------
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------- Stratified Group K-Fold SVM -------------------------
def SVM_kfold_stratified_group(df_CAMS_metadata, feature_stat, base_model_dir, n_splits=5):
    base_model_dir = os.path.join(base_model_dir, "SVM")
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    df_subjects = df_CAMS_metadata[['BDSPPatientID','CAMS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['CAMS_grouped'].values
    groups = df_subjects['BDSPPatientID'].values
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(unique_studies, labels, groups=groups)):
        print(f"{'='*30} Fold {fold+1} {'='*30}")
        train_subs = unique_studies[train_idx]
        test_subs = unique_studies[test_idx]
        overlap = set(train_subs).intersection(set(test_subs))
        print("✅ No overlap" if len(overlap)==0 else f"❌ Overlap: {overlap}")
        
        df_CAMS_metadata['Split'] = df_CAMS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        df_train = df_CAMS_metadata[df_CAMS_metadata['Split']=='Train'].reset_index(drop=True)
        df_test  = df_CAMS_metadata[df_CAMS_metadata['Split']=='Test'].reset_index(drop=True)
        
        X_train, Y_train = morgoth_10minfea_matrix_stat_for(df_train, feature_stat)
        X_test, Y_test   = morgoth_10minfea_matrix_stat_for(df_test, feature_stat)
        
        model = SVC(
            kernel='poly',       # change to 'linear' if desired
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        model.fit(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        Y_prob = model.predict_proba(X_test)
        
        classes = np.unique(Y_train)
        Y_test_bin = label_binarize(Y_test, classes=classes)
        auc_per_class = {cls: roc_auc_score(Y_test_bin[:,i], Y_prob[:,i]) for i, cls in enumerate(classes)}
        auc_macro = np.mean(list(auc_per_class.values()))
        
        acc  = accuracy_score(Y_test, Y_pred)
        pre_macro = precision_score(Y_test, Y_pred, average='macro')
        rec_macro = recall_score(Y_test, Y_pred, average='macro')
        f1_macro  = f1_score(Y_test, Y_pred, average='macro')
        pre_micro = precision_score(Y_test, Y_pred, average='micro')
        rec_micro = recall_score(Y_test, Y_pred, average='micro')
        f1_micro  = f1_score(Y_test, Y_pred, average='micro')
        
        fold_results.append({
            'fold': fold+1,
            'accuracy': acc,
            'precision_macro': pre_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_micro': pre_micro,
            'recall_micro': rec_micro,
            'f1_micro': f1_micro,
            'auc_macro': auc_macro,
            'auc_per_class': auc_per_class,
            'Y_true':Y_test,
            'Y_pred':Y_pred
        })
        
        fold_model_dir = Path(base_model_dir) / feature_stat / f"Fold{fold+1}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, fold_model_dir / "SVM_model.joblib")
        
        cm = confusion_matrix(Y_test, Y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        print(f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Micro F1: {f1_micro:.4f}")
        print("*"*100)
    
    return fold_results

# ------------------------------- Stratified Group K-Fold KNN -------------------------
def KNN_kfold_stratified_group(df_CAMS_metadata, feature_stat, base_model_dir, n_splits=5, n_neighbors=5):
    base_model_dir = os.path.join(base_model_dir, "KNN")
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    df_subjects = df_CAMS_metadata[['BDSPPatientID','CAMS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['CAMS_grouped'].values
    groups = df_subjects['BDSPPatientID'].values
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(unique_studies, labels, groups=groups)):
        print(f"{'='*30} Fold {fold+1} {'='*30}")
        train_subs = unique_studies[train_idx]
        test_subs = unique_studies[test_idx]
        overlap = set(train_subs).intersection(set(test_subs))
        print("✅ No overlap" if len(overlap)==0 else f"❌ Overlap: {overlap}")
        
        df_CAMS_metadata['Split'] = df_CAMS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        df_train = df_CAMS_metadata[df_CAMS_metadata['Split']=='Train'].reset_index(drop=True)
        df_test  = df_CAMS_metadata[df_CAMS_metadata['Split']=='Test'].reset_index(drop=True)
        
        X_train, Y_train = morgoth_10minfea_matrix_stat_for(df_train, feature_stat)
        X_test, Y_test   = morgoth_10minfea_matrix_stat_for(df_test, feature_stat)
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        Y_prob = model.predict_proba(X_test)
        
        classes = np.unique(Y_train)
        Y_test_bin = label_binarize(Y_test, classes=classes)
        auc_per_class = {cls: roc_auc_score(Y_test_bin[:,i], Y_prob[:,i]) for i, cls in enumerate(classes)}
        auc_macro = np.mean(list(auc_per_class.values()))
        
        acc  = accuracy_score(Y_test, Y_pred)
        pre_macro = precision_score(Y_test, Y_pred, average='macro')
        rec_macro = recall_score(Y_test, Y_pred, average='macro')
        f1_macro  = f1_score(Y_test, Y_pred, average='macro')
        pre_micro = precision_score(Y_test, Y_pred, average='micro')
        rec_micro = recall_score(Y_test, Y_pred, average='micro')
        f1_micro  = f1_score(Y_test, Y_pred, average='micro')
        
        fold_results.append({
            'fold': fold+1,
            'accuracy': acc,
            'precision_macro': pre_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_micro': pre_micro,
            'recall_micro': rec_micro,
            'f1_micro': f1_micro,
            'auc_macro': auc_macro,
            'auc_per_class': auc_per_class,
            'Y_true':Y_test,
            'Y_pred':Y_pred
        })
        
        fold_model_dir = Path(base_model_dir) / feature_stat / f"Fold{fold+1}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, fold_model_dir / "KNN_model.joblib")
        
        cm = confusion_matrix(Y_test, Y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        print(f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Micro F1: {f1_micro:.4f}")
        print("*"*100)
    
    return fold_results


# # **Load CAMS Metadata**

# In[4]:

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
axes[1].set_title('Grouped CAMS Distribution')
axes[1].set_xticks(range(0, 3))

plt.tight_layout()
plt.show()


# # **LR-based models**

# ## **LR-mean**

# In[5]:

current = Path(__file__).resolve()

CAMS_ROOT = None
for parent in current.parents:
    if parent.name == "CAMS":
        CAMS_ROOT = parent
        break

if CAMS_ROOT is None:
    raise RuntimeError("CAMS folder not found")

feature_stat1 = 'mean'
base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "LR_CAMS"

LR_results_5fld_mean = LR_kfold_stratified_group(df_CAMS_metadata, feature_stat1, base_model_dir, n_splits=5)


# ## **LR-median** 

# In[6]:


feature_stat2 = 'median'
base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "LR_CAMS"

LR_results_5fld_median = LR_kfold_stratified_group(df_CAMS_metadata, feature_stat2, base_model_dir, n_splits=5)


# ## **LR-covarupper**

# In[7]:


feature_stat3 = 'cov_upper'
base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "LR_CAMS"

LR_results_5fld_covar = LR_kfold_stratified_group(df_CAMS_metadata, feature_stat3, base_model_dir, n_splits=5)


# # **SVM-based models**

# ## **SVM-mean** 

# In[8]:

base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "SVM_CAMS"

SVM_results_5fld_mean = SVM_kfold_stratified_group(df_CAMS_metadata, feature_stat1, base_model_dir, n_splits=5)


# ## **SVM-median** 

# In[9]:


base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "SVM_CAMS"

SVM_results_5fld_median = SVM_kfold_stratified_group(df_CAMS_metadata, feature_stat2, base_model_dir, n_splits=5)


# ## **SVM-covarupper**

# In[10]:


base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "SVM_CAMS"

SVM_results_5fld_covar = SVM_kfold_stratified_group(df_CAMS_metadata, feature_stat3, base_model_dir, n_splits=5)


# # **KNN-based models**

# ## **KNN-mean** 

# In[11]:

base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "KNN_CAMS"

KNN_results_5fld_mean = KNN_kfold_stratified_group(df_CAMS_metadata, feature_stat1, base_model_dir, n_splits=5)


# ## **KNN-median** 

# In[12]:


base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "KNN_CAMS"

KNN_results_5fld_median = KNN_kfold_stratified_group(df_CAMS_metadata, feature_stat2, base_model_dir, n_splits=5)


# ## **KNN-covarupper**

# In[13]:


base_model_dir = CAMS_ROOT / "model" / "ModelCheckpointsNew" / "3class" / "KNN_CAMS"

KNN_results_5fld_covar = KNN_kfold_stratified_group(df_CAMS_metadata, feature_stat3, base_model_dir, n_splits=5)


# # **Save Results to your suitable path**

# In[14]:


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


# In[15]:


# save_to_pickle(
#     KNN_results_5fld_mean,
#     '/home/ayush/Desktop/CAM-S_dataset/RESULTS/KNN_5fld_mean_Results_3class.pkl'
# )
# save_to_pickle(
#     KNN_results_5fld_median,
#     '/home/ayush/Desktop/CAM-S_dataset/RESULTS/KNN_5fld_median_Results_3class.pkl'
# )
# save_to_pickle(
#     KNN_results_5fld_covar,
#     '/home/ayush/Desktop/CAM-S_dataset/RESULTS/KNN_5fld_covar_Results_3class.pkl'
# )


# # In[16]:


# save_to_pickle(
#     LR_results_5fld_mean,
#     '/home/ayush/Desktop/CAM-S_dataset/RESULTS/LR_5fld_mean_Results_3class.pkl'
# )
# save_to_pickle(
#     LR_results_5fld_median,
#     '/home/ayush/Desktop/CAM-S_dataset/RESULTS/LR_5fld_median_Results_3class.pkl'
# )
# save_to_pickle(
#     LR_results_5fld_covar,
#     '/home/ayush/Desktop/CAM-S_dataset/RESULTS/LR_5fld_covar_Results_3class.pkl'
# )


# # In[17]:


# save_to_pickle(
#     SVM_results_5fld_mean,
#     '/home/ayush/Desktop/CAM-S_dataset/RESULTS/SVM_5fld_mean_Results_3class.pkl'
# )
# save_to_pickle(
#     SVM_results_5fld_median,
#     '/home/ayush/Desktop/CAM-S_dataset/RESULTS/SVM_5fld_median_Results_3class.pkl'
# )
# save_to_pickle(
#     SVM_results_5fld_covar,
#     '/home/ayush/Desktop/CAM-S_dataset/RESULTS/SVM_5fld_covar_Results_3class.pkl'
# )

