#!/usr/bin/env python
# coding: utf-8

# # **Libraries**

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
RASS_ROOT = None
for parent in current.parents:
    if parent.name == "RASS":
        RASS_ROOT = parent
        break

if RASS_ROOT is None:
    raise RuntimeError("RASS folder not found")

# ------------------------------- EEG feature statistics collection -------------------------
def morgoth_10minfea_matrix_stat_for(data_frame, statistic):
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
        all_file_names.append(file_name)
    
    all_subject_features_matrix = np.stack(all_subject_features, axis=0)
    train_labels = data_frame['RASS_grouped'].values .flatten() 
    all_file_names = np.array(all_file_names)
    print('Label size==> ',train_labels.shape)
    print("Stacked feature matrix shape:", all_subject_features_matrix.shape)
    print("Filename shape ==> ", all_file_names.shape)
    return all_subject_features_matrix, train_labels, all_file_names


# ## **K-fold Cross validation function**

# In[3]:


def LR_kfold_stratified_group(df_RASS_metadata, feature_stat, base_model_dir, n_splits=5):
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
    
    base_model_dir = os.path.join(base_model_dir, "LR")
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    # Unique subjects and labels
    df_subjects = df_RASS_metadata[['BDSPPatientID','RASS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['RASS_grouped'].values
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
        df_RASS_metadata['Split'] = df_RASS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        
        df_RASS_train = df_RASS_metadata[df_RASS_metadata['Split']=='Train'].reset_index(drop=True)
        df_RASS_test  = df_RASS_metadata[df_RASS_metadata['Split']=='Test'].reset_index(drop=True)
        
        # Feature extraction
        X_train_fea, Y_train, train_filenames = morgoth_10minfea_matrix_stat_for(df_RASS_train, feature_stat)
        X_test_fea, Y_test, test_filenames = morgoth_10minfea_matrix_stat_for(df_RASS_test, feature_stat)

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
        one_off_acc = np.mean(np.abs(Y_test - Y_pred) <= 1)
        pre_micro = precision_score(Y_test, Y_pred, average='micro')
        rec_micro = recall_score(Y_test, Y_pred, average='micro')
        f1_micro  = f1_score(Y_test, Y_pred, average='micro')

        # Store results
        fold_results.append({
            'fold': fold+1,
            'accuracy': acc,
            '1-level difference accuracy': one_off_acc,
            'precision_macro': pre_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_micro': pre_micro,
            'recall_micro': rec_micro,
            'f1_micro': f1_micro,
            'auc_macro': auc_macro,
            'auc_per_class': auc_per_class,
            'Y_true':Y_test,
            'Y_pred':Y_pred,
            'Test_filenames': test_filenames
        })

        # Save model
        fold_model_dir = Path(base_model_dir) / feature_stat 
        print(fold_model_dir)
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        model_path = fold_model_dir / f"Fold{fold+1}_LR_model.joblib"
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


# In[4]:


#--------------------- SVM ----------------
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------- Stratified Group K-Fold SVM -------------------------
def SVM_kfold_stratified_group(df_RASS_metadata, feature_stat, base_model_dir, n_splits=5):
    base_model_dir = os.path.join(base_model_dir, "SVM")
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    df_subjects = df_RASS_metadata[['BDSPPatientID','RASS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['RASS_grouped'].values
    groups = df_subjects['BDSPPatientID'].values
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(unique_studies, labels, groups=groups)):
        print(f"{'='*30} Fold {fold+1} {'='*30}")
        train_subs = unique_studies[train_idx]
        test_subs = unique_studies[test_idx]
        overlap = set(train_subs).intersection(set(test_subs))
        print("✅ No overlap" if len(overlap)==0 else f"❌ Overlap: {overlap}")
        
        df_RASS_metadata['Split'] = df_RASS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        df_train = df_RASS_metadata[df_RASS_metadata['Split']=='Train'].reset_index(drop=True)
        df_test  = df_RASS_metadata[df_RASS_metadata['Split']=='Test'].reset_index(drop=True)
        
        X_train, Y_train, train_filenames = morgoth_10minfea_matrix_stat_for(df_train, feature_stat)
        X_test, Y_test, test_filenames = morgoth_10minfea_matrix_stat_for(df_test, feature_stat)
        
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
        one_off_acc = np.mean(np.abs(Y_test - Y_pred) <= 1)
        pre_macro = precision_score(Y_test, Y_pred, average='macro')
        rec_macro = recall_score(Y_test, Y_pred, average='macro')
        f1_macro  = f1_score(Y_test, Y_pred, average='macro')
        pre_micro = precision_score(Y_test, Y_pred, average='micro')
        rec_micro = recall_score(Y_test, Y_pred, average='micro')
        f1_micro  = f1_score(Y_test, Y_pred, average='micro')
        
        fold_results.append({
            'fold': fold+1,
            'accuracy': acc,
            '1-level difference accuracy': one_off_acc,
            'precision_macro': pre_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_micro': pre_micro,
            'recall_micro': rec_micro,
            'f1_micro': f1_micro,
            'auc_macro': auc_macro,
            'auc_per_class': auc_per_class,
            'Y_true':Y_test,
            'Y_pred':Y_pred,
            'Test_filenames': test_filenames
        })
        
        fold_model_dir = Path(base_model_dir) / feature_stat 
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, fold_model_dir / f"Fold{fold+1}_SVM_model.joblib")
        
        cm = confusion_matrix(Y_test, Y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        print(f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Micro F1: {f1_micro:.4f}")
        print("*"*100)
    
    return fold_results

# ------------------------------- Stratified Group K-Fold KNN -------------------------
def KNN_kfold_stratified_group(df_RASS_metadata, feature_stat, base_model_dir, n_splits=5, n_neighbors=5):
    base_model_dir = os.path.join(base_model_dir, "KNN")
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    df_subjects = df_RASS_metadata[['BDSPPatientID','RASS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['RASS_grouped'].values
    groups = df_subjects['BDSPPatientID'].values
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(unique_studies, labels, groups=groups)):
        print(f"{'='*30} Fold {fold+1} {'='*30}")
        train_subs = unique_studies[train_idx]
        test_subs = unique_studies[test_idx]
        overlap = set(train_subs).intersection(set(test_subs))
        print("✅ No overlap" if len(overlap)==0 else f"❌ Overlap: {overlap}")
        
        df_RASS_metadata['Split'] = df_RASS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        df_train = df_RASS_metadata[df_RASS_metadata['Split']=='Train'].reset_index(drop=True)
        df_test  = df_RASS_metadata[df_RASS_metadata['Split']=='Test'].reset_index(drop=True)
        
        X_train, Y_train, train_filenames = morgoth_10minfea_matrix_stat_for(df_train, feature_stat)
        X_test, Y_test, test_filenames = morgoth_10minfea_matrix_stat_for(df_test, feature_stat)
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        Y_prob = model.predict_proba(X_test)
        
        classes = np.unique(Y_train)
        Y_test_bin = label_binarize(Y_test, classes=classes)
        auc_per_class = {cls: roc_auc_score(Y_test_bin[:,i], Y_prob[:,i]) for i, cls in enumerate(classes)}
        auc_macro = np.mean(list(auc_per_class.values()))
        
        acc  = accuracy_score(Y_test, Y_pred)
        one_off_acc = np.mean(np.abs(Y_test - Y_pred) <= 1)
        pre_macro = precision_score(Y_test, Y_pred, average='macro')
        rec_macro = recall_score(Y_test, Y_pred, average='macro')
        f1_macro  = f1_score(Y_test, Y_pred, average='macro')
        pre_micro = precision_score(Y_test, Y_pred, average='micro')
        rec_micro = recall_score(Y_test, Y_pred, average='micro')
        f1_micro  = f1_score(Y_test, Y_pred, average='micro')
        
        fold_results.append({
            'fold': fold+1,
            'accuracy': acc,
            '1-level difference accuracy': one_off_acc,
            'precision_macro': pre_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'precision_micro': pre_micro,
            'recall_micro': rec_micro,
            'f1_micro': f1_micro,
            'auc_macro': auc_macro,
            'auc_per_class': auc_per_class,
            'Y_true':Y_test,
            'Y_pred':Y_pred,
            'Test_filenames': test_filenames
        })
        
        fold_model_dir = Path(base_model_dir) / feature_stat 
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, fold_model_dir / f"Fold{fold+1}_KNN_model.joblib")
        
        cm = confusion_matrix(Y_test, Y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        print(f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Micro F1: {f1_micro:.4f}")
        print("*"*100)
    
    return fold_results


# # **Load RASS Metadata**

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


# # **LR-based model**

# ## **LR-median** 

# In[6]:


feature_stat1 = 'median'
base_model_dir = RASS_ROOT / "model" / "ModelCheckpointsNew" / "LR_RASS"

LR_results_5fld_median = LR_kfold_stratified_group(df_RASS_metadata, feature_stat1, base_model_dir, n_splits=5)


# ## **LR-mean** 

# In[7]:


feature_stat2 = 'mean'
base_model_dir = RASS_ROOT / "model" / "ModelCheckpointsNew" / "LR_RASS"

LR_results_5fld_mean = LR_kfold_stratified_group(df_RASS_metadata, feature_stat2, base_model_dir, n_splits=5)


# ## **LR-Covar_upper** 

# In[8]:


feature_stat3 = 'cov_upper'
base_model_dir = RASS_ROOT / "model" / "ModelCheckpointsNew" / "LR_RASS"

LR_results_5fld_covar = LR_kfold_stratified_group(df_RASS_metadata, feature_stat3, base_model_dir, n_splits=5)


# # **SVM-based model**

# ## **SVM-median**  

# In[9]:


feature_stat1 = 'median'
base_model_dir = RASS_ROOT / "model" / "ModelCheckpointsNew" / "SVM_RASS"

SVM_results_5fld_median = SVM_kfold_stratified_group(df_RASS_metadata, feature_stat1, base_model_dir, n_splits=5)


# ## **SVM-mean** 

# In[10]:


feature_stat2 = 'mean'
base_model_dir = RASS_ROOT / "model" / "ModelCheckpointsNew" / "SVM_RASS"

SVM_results_5fld_mean = SVM_kfold_stratified_group(df_RASS_metadata, feature_stat2, base_model_dir, n_splits=5)


# ## **SVM-Covar_upper**

# In[11]:


feature_stat3 = 'cov_upper'
base_model_dir = RASS_ROOT / "model" / "ModelCheckpointsNew" / "SVM_RASS"

SVM_results_5fld_covar = SVM_kfold_stratified_group(df_RASS_metadata, feature_stat3, base_model_dir, n_splits=5)


# # **KNN-based model**

# ## **KNN-median** 

# In[12]:


feature_stat1 = 'median'
base_model_dir = RASS_ROOT / "model" / "ModelCheckpointsNew" / "KNN_RASS"

KNN_results_5fld_median = KNN_kfold_stratified_group(df_RASS_metadata, feature_stat1, base_model_dir, n_splits=5)


# ## **KNN-mean** 

# In[13]:


feature_stat2 = 'mean'
base_model_dir = RASS_ROOT / "model" / "ModelCheckpointsNew" / "KNN_RASS"

KNN_results_5fld_mean = KNN_kfold_stratified_group(df_RASS_metadata, feature_stat2, base_model_dir, n_splits=5)


# ## **KNN-Covar_upper** 

# In[14]:


feature_stat3 = 'cov_upper'
base_model_dir = RASS_ROOT / "model" / "ModelCheckpointsNew" / "KNN_RASS"

KNN_results_5fld_covar = KNN_kfold_stratified_group(df_RASS_metadata, feature_stat3, base_model_dir, n_splits=5)


# # **Save model results**

# In[15]:


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


# In[16]:

# Save Results at your suitable location

# save_to_pickle(
#     KNN_results_5fld_mean,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/ML/KNN_5fld_mean_Results_6classRASS.pkl'
# )
# save_to_pickle(
#     KNN_results_5fld_median,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/ML/KNN_5fld_median_Results_6classRASS.pkl'
# )
# save_to_pickle(
#     KNN_results_5fld_covar,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/ML/KNN_5fld_covar_Results_6classRASS.pkl'
# )


# # In[17]:


# save_to_pickle(
#     LR_results_5fld_mean,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/ML/LR_5fld_mean_Results_6classRASS.pkl'
# )
# save_to_pickle(
#     LR_results_5fld_median,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/ML/LR_5fld_median_Results_6classRASS.pkl'
# )
# save_to_pickle(
#     LR_results_5fld_covar,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/ML/LR_5fld_covar_Results_6classRASS.pkl'
# )


# # In[18]:


# save_to_pickle(
#     SVM_results_5fld_mean,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/ML/SVM_5fld_mean_Results_6classRASS.pkl'
# )
# save_to_pickle(
#     SVM_results_5fld_median,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/ML/SVM_5fld_median_Results_6classRASS.pkl'
# )
# save_to_pickle(
#     SVM_results_5fld_covar,
#     '/home/ayush/Desktop/MGB_RASS_EEGs/RESULTS_5FOLD_new/ML/SVM_5fld_covar_Results_6classRASS.pkl'
# )

