#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
from pathlib import Path
from datetime import datetime

import h5py
import hdf5storage
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr

from tqdm import tqdm



from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

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
        Python object stored in the pickle file.
    """
    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    return obj


# In[3]:


from pathlib import Path

current = Path(__file__).resolve()

RASS_ROOT = None
for parent in current.parents:
    if parent.name == "RASS":
        RASS_ROOT = parent
        break

if RASS_ROOT is None:
    raise RuntimeError("RASS folder not found")

RESULTS_DIR = RASS_ROOT / "Results5Fld"
RESNET_ordinal_onlyGAP  = load_pickle(RESULTS_DIR / "ResNetGAP_5fldResults_6classRASS.pkl")


# In[14]:





def subjectwise_evaluation_plots(
    results,
    figsize=(10, 3), # Default updated to a wider layout
    dpi=150,
    save_path=None
):

    all_y_true = []
    all_y_pred = []
    all_filenames = []

    for fold in results:
        all_y_true.extend(fold['Y_true'])
        all_y_pred.extend(fold['Y_pred'])
        all_filenames.extend(fold['Test_filenames'])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_filenames = np.array(all_filenames)

    def get_subject_id(fname):
        return fname.split('_')[0].replace('sub-S', '')

    subj_true = defaultdict(list)
    subj_pred = defaultdict(list)

    for y_t, y_p, fname in zip(all_y_true, all_y_pred, all_filenames):
        sid = get_subject_id(fname)
        subj_true[sid].append(y_t)
        subj_pred[sid].append(y_p)

    # Metric 1: MAE per subject
    mae_list = []
    for sid in subj_true:
        errs = np.abs(np.array(subj_true[sid]) - np.array(subj_pred[sid]))
        mae_list.append(np.mean(errs))
    mae_list = np.array(mae_list)
    
    # Metric 2: ±1 agreement per subject
    agree_list = []
    for sid in subj_true:
        vals = np.abs(np.array(subj_true[sid]) - np.array(subj_pred[sid]))
        agree_list.append(100 * np.mean(vals <= 1))
    agree_list = np.array(agree_list)
    
    # Metric 3: Spearman per subject
    spearman_list = []
    for sid in subj_true:
        y_t = np.array(subj_true[sid])
        y_p = np.array(subj_pred[sid])
        if len(y_t) < 2:
            continue
        rho, _ = spearmanr(y_t, y_p)
        if not np.isnan(rho):
            spearman_list.append(rho)
    spearman_list = np.array(spearman_list)

    # Plotting
    pastel = "gray"

    # THE FIX: using the figsize variable instead of hardcoding (7,6)
    fig, ax = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    ax[0].hist(mae_list, bins=20, color=pastel, edgecolor='black')
    ax[0].set_xlabel("Mean absolute error", fontsize=12)
    ax[0].set_ylabel("Count of subjects", fontsize=12)
    
    ax[1].hist(agree_list, bins=20, color=pastel, edgecolor='black')
    ax[1].set_xlabel("Percent(|y-y'|<=1) %", fontsize=12)
    
    ax[2].hist(spearman_list, bins=20, color=pastel, edgecolor='black')
    ax[2].set_xlabel("Spearman correlation (ρ)", fontsize=12)
    
    for i in range(3):
        ax[i].tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# In[15]:


subjectwise_evaluation_plots(
    RESNET_ordinal_onlyGAP,
    figsize=(10, 3),
    dpi=150,
    save_path=RASS_ROOT / "Figures" / "RASS_pred_MAE_1levelACC_Corr.png" 
)




