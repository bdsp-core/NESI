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
GCS_ROOT = None
for parent in current.parents:
    if parent.name == "GCS":
        GCS_ROOT = parent
        break

if GCS_ROOT is None:
    raise RuntimeError("GCS folder not found")

# ------------------------------- EEG feature statistics collection -------------------------
def morgoth_10minfea_matrix_stat_for(data_frame, statistic):
    BASE = GCS_ROOT / "MorgothActivations"

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
    train_labels=data_frame['GCS_grouped'].values .flatten() 
    all_file_names = np.array(all_file_names)
    print('Label size==> ',train_labels.shape)
    print("Stacked feature matrix shape:", all_subject_features_matrix.shape)
    print("Filename shape ==> ", all_file_names.shape)
    return all_subject_features_matrix, train_labels, all_file_names


# ## **K-fold Cross validation function**

# In[3]:


def LR_kfold_stratified_group(df_GCS_metadata, feature_stat, base_model_dir, n_splits=5):
    """
    Stratified group k-fold cross-validation (~15% test per fold) 
    ensuring no subject (BDSPPatientID) appears in both train and test.
    
    Args:
        df_GCS_metadata: DataFrame with at least ['BDSPPatientID', 'GCS_grouped']
        n_splits: number of folds (default 7 → ~15% test per fold)
        bs: batch size
    Returns:
        fold_results: list of dicts containing predictions, logits, metrics per fold
    """
    
    base_model_dir = os.path.join(base_model_dir, "LR")
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    # Unique subjects and labels
    df_subjects = df_GCS_metadata[['BDSPPatientID','GCS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['GCS_grouped'].values
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
        df_GCS_metadata['Split'] = df_GCS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        
        df_GCS_train = df_GCS_metadata[df_GCS_metadata['Split']=='Train'].reset_index(drop=True)
        df_GCS_test  = df_GCS_metadata[df_GCS_metadata['Split']=='Test'].reset_index(drop=True)
        
        # Feature extraction
        X_train_fea, Y_train, train_filenames = morgoth_10minfea_matrix_stat_for(df_GCS_train, feature_stat)
        X_test_fea, Y_test, test_filenames = morgoth_10minfea_matrix_stat_for(df_GCS_test, feature_stat)

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
def SVM_kfold_stratified_group(df_GCS_metadata, feature_stat, base_model_dir, n_splits=5):
    base_model_dir = os.path.join(base_model_dir, "SVM")
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    df_subjects = df_GCS_metadata[['BDSPPatientID','GCS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['GCS_grouped'].values
    groups = df_subjects['BDSPPatientID'].values
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(unique_studies, labels, groups=groups)):
        print(f"{'='*30} Fold {fold+1} {'='*30}")
        train_subs = unique_studies[train_idx]
        test_subs = unique_studies[test_idx]
        overlap = set(train_subs).intersection(set(test_subs))
        print("✅ No overlap" if len(overlap)==0 else f"❌ Overlap: {overlap}")
        
        df_GCS_metadata['Split'] = df_GCS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        df_train = df_GCS_metadata[df_GCS_metadata['Split']=='Train'].reset_index(drop=True)
        df_test  = df_GCS_metadata[df_GCS_metadata['Split']=='Test'].reset_index(drop=True)
        
        X_train, Y_train, train_filenames = morgoth_10minfea_matrix_stat_for(df_train, feature_stat)
        X_test, Y_test, test_filenames   = morgoth_10minfea_matrix_stat_for(df_test, feature_stat)
        
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
def KNN_kfold_stratified_group(df_GCS_metadata, feature_stat, base_model_dir, n_splits=5, n_neighbors=5):
    base_model_dir = os.path.join(base_model_dir, "KNN")
    os.makedirs(base_model_dir, exist_ok=True)
    
    fold_results = []
    
    df_subjects = df_GCS_metadata[['BDSPPatientID','GCS_grouped']].drop_duplicates()
    unique_studies = df_subjects['BDSPPatientID'].values
    labels = df_subjects['GCS_grouped'].values
    groups = df_subjects['BDSPPatientID'].values
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(unique_studies, labels, groups=groups)):
        print(f"{'='*30} Fold {fold+1} {'='*30}")
        train_subs = unique_studies[train_idx]
        test_subs = unique_studies[test_idx]
        overlap = set(train_subs).intersection(set(test_subs))
        print("✅ No overlap" if len(overlap)==0 else f"❌ Overlap: {overlap}")
        
        df_GCS_metadata['Split'] = df_GCS_metadata['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )
        df_train = df_GCS_metadata[df_GCS_metadata['Split']=='Train'].reset_index(drop=True)
        df_test  = df_GCS_metadata[df_GCS_metadata['Split']=='Test'].reset_index(drop=True)
        
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


# # **GCS metadata load**

# In[5]:
metadata_path_GCS = GCS_ROOT / "model" / "Training" / "GCSTraining_Final_Metadata.csv"
df_GCS_metadata=pd.read_csv(metadata_path_GCS)

def group_gcs(y):
    if y in [3,4,5,6,7,8]:
        return 0
    elif y in [9,10,11,12]:
        return 1
    elif y in [13,14,15]:
        return 2
    else:
        return None  # in case of unexpected values

df_GCS_metadata['GCS_grouped'] = df_GCS_metadata['GCS_value'].apply(group_gcs)
def extract_pid(filename):
    part = filename.split('_')[0]          
    pid_full = part.split('-')[1]          
    pid = pid_full[5:]                     
    return pid

# Apply to all rows and create new column for BDSPPatientID
df_GCS_metadata['BDSPPatientID'] = df_GCS_metadata['Filename'].apply(extract_pid)
df_GCS_metadata = df_GCS_metadata[['BDSPPatientID', 'Filename', 'GCS_value', 'GCS_grouped']]


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Clean minimal style
sns.set_style("white", {'axes.grid': False})
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Labels
group_labels = {0: 'Severe (3-8)', 1: 'Moderate (9-12)', 2: 'Mild (13-15)'}

# ----------- Plot 1: GCS_value -----------
sns.histplot(
    data=df_GCS_metadata,
    x='GCS_value',
    bins=range(3, 17),
    color='lightgray',        # neutral color
    edgecolor='black',        # black edges
    alpha=1,
    ax=axes[0],
    discrete=True,
    shrink=0.8
)

axes[0].set_title('GCS Value Distribution', fontsize=15, fontweight='bold', pad=15)
axes[0].set_xlabel('GCS Value (Score)', fontsize=12)
axes[0].set_ylabel('Number of 10 min EEG Segments', fontsize=12)
axes[0].set_xticks(range(3, 16))
sns.despine(ax=axes[0])

# Annotate
for p in axes[0].patches:
    if p.get_height() > 0:
        axes[0].annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            fontsize=10,
            xytext=(0, 6),
            textcoords='offset points'
        )

# ----------- Plot 2: GCS_grouped -----------
grp_counts = df_GCS_metadata['GCS_grouped'].value_counts().sort_index().reset_index()
grp_counts.columns = ['Group', 'Count']
grp_counts['Label'] = grp_counts['Group'].map(group_labels)

sns.barplot(
    data=grp_counts,
    x='Label',
    y='Count',
    color='lightgray',       # neutral color
    edgecolor='black',       # black edges
    ax=axes[1]
)

axes[1].set_title('GCS Severity Group Distribution', fontsize=15, fontweight='bold', pad=15)
axes[1].set_xlabel('Severity Category', fontsize=12)
axes[1].set_ylabel('Number of 10 min EEG Segments', fontsize=12)
sns.despine(ax=axes[1])

# Annotate
for p in axes[1].patches:
    axes[1].annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center',
        fontsize=11,
        fontweight='bold',
        xytext=(0, 6),
        textcoords='offset points'
    )

plt.tight_layout()
plt.show()


# # **LR based model**

# ## **LR-median** 

# In[6]:


feature_stat1 = 'median'
base_model_dir = GCS_ROOT / "model" / "ModelCheckpointsNew" / "LR_GCS"

LR_results_5fld_median = LR_kfold_stratified_group(df_GCS_metadata, feature_stat1, base_model_dir, n_splits=5)


# ## **LR-mean** 

# In[7]:


feature_stat2 = 'mean'
base_model_dir = GCS_ROOT / "model" / "ModelCheckpointsNew" / "LR_GCS"

LR_results_5fld_mean = LR_kfold_stratified_group(df_GCS_metadata, feature_stat2, base_model_dir, n_splits=5)


# ## **LR-covar-upper** 

# In[8]:


feature_stat3 = 'cov_upper'
base_model_dir = GCS_ROOT / "model" / "ModelCheckpointsNew" / "LR_GCS"

LR_results_5fld_covar = LR_kfold_stratified_group(df_GCS_metadata, feature_stat3, base_model_dir, n_splits=5)


# # **SVM based model**

# ## **SVM-median** 

# In[9]:


feature_stat1 = 'median'
base_model_dir = GCS_ROOT / "model" / "ModelCheckpointsNew" / "SVM_GCS"

SVM_results_5fld_median = SVM_kfold_stratified_group(df_GCS_metadata, feature_stat1, base_model_dir, n_splits=5)


# ## **SVM-mean** 

# In[10]:


feature_stat2 = 'mean'
base_model_dir = GCS_ROOT / "model" / "ModelCheckpointsNew" / "SVM_GCS"

SVM_results_5fld_mean = SVM_kfold_stratified_group(df_GCS_metadata, feature_stat2, base_model_dir, n_splits=5)


# ## **SVM-covar-upper** 

# In[11]:


feature_stat3 = 'cov_upper'
base_model_dir = GCS_ROOT / "model" / "ModelCheckpointsNew" / "SVM_GCS"

SVM_results_5fld_covar = SVM_kfold_stratified_group(df_GCS_metadata, feature_stat3, base_model_dir, n_splits=5)


# # **KNN based model**

# ## **KNN-median** 

# In[12]:


feature_stat1 = 'median'
base_model_dir = GCS_ROOT / "model" / "ModelCheckpointsNew" / "KNN_GCS"

KNN_results_5fld_median = KNN_kfold_stratified_group(df_GCS_metadata, feature_stat1, base_model_dir, n_splits=5)


# ## **KNN-mean** 

# In[13]:


feature_stat2 = 'mean'
base_model_dir = GCS_ROOT / "model" / "ModelCheckpointsNew" / "KNN_GCS"

KNN_results_5fld_mean = KNN_kfold_stratified_group(df_GCS_metadata, feature_stat2, base_model_dir, n_splits=5)


# ## **KNN-covar-upper** 

# In[14]:


feature_stat3 = 'cov_upper'
base_model_dir = GCS_ROOT / "model" / "ModelCheckpointsNew" / "KNN_GCS"

KNN_results_5fld_covar = KNN_kfold_stratified_group(df_GCS_metadata, feature_stat3, base_model_dir, n_splits=5)


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
#     '/home/ayush/Desktop/MGB_GCS_EEGs/RESULTS_5FOLD_new/ML/KNN_5fld_mean_Results_3classGCS.pkl'
# )
# save_to_pickle(
#     KNN_results_5fld_median,
#     '/home/ayush/Desktop/MGB_GCS_EEGs/RESULTS_5FOLD_new/ML/KNN_5fld_median_Results_3classGCS.pkl'
# )
# save_to_pickle(
#     KNN_results_5fld_covar,
#     '/home/ayush/Desktop/MGB_GCS_EEGs/RESULTS_5FOLD_new/ML/KNN_5fld_covar_Results_3classGCS.pkl'
# )


# # In[17]:


# save_to_pickle(
#     LR_results_5fld_mean,
#     '/home/ayush/Desktop/MGB_GCS_EEGs/RESULTS_5FOLD_new/ML/LR_5fld_mean_Results_3classGCS.pkl'
# )
# save_to_pickle(
#     LR_results_5fld_median,
#     '/home/ayush/Desktop/MGB_GCS_EEGs/RESULTS_5FOLD_new/ML/LR_5fld_median_Results_3classGCS.pkl'
# )
# save_to_pickle(
#     LR_results_5fld_covar,
#     '/home/ayush/Desktop/MGB_GCS_EEGs/RESULTS_5FOLD_new/ML/LR_5fld_covar_Results_3classGCS.pkl'
# )


# # In[18]:


# save_to_pickle(
#     SVM_results_5fld_mean,
#     '/home/ayush/Desktop/MGB_GCS_EEGs/RESULTS_5FOLD_new/ML/SVM_5fld_mean_Results_3classGCS.pkl'
# )
# save_to_pickle(
#     SVM_results_5fld_median,
#     '/home/ayush/Desktop/MGB_GCS_EEGs/RESULTS_5FOLD_new/ML/SVM_5fld_median_Results_3classGCS.pkl'
# )
# save_to_pickle(
#     SVM_results_5fld_covar,
#     '/home/ayush/Desktop/MGB_GCS_EEGs/RESULTS_5FOLD_new/ML/SVM_5fld_covar_Results_3classGCS.pkl'
# )

