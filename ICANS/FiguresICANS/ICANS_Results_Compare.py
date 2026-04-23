#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
from torchsummary import summary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.layers import CoralLayer
from datetime import datetime
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc)
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


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def plot_model_performance_bar(models_results, model_names=None, metrics=None, figsize=(12, 7), colors=None):
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models_results))]
        
    if metrics is None:
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                   'precision_micro', 'recall_micro', 'f1_micro']
        
    if colors is None:
        cmap = plt.colormaps.get_cmap('Pastel1')
        colors = [cmap(i) for i in np.linspace(0, 1, len(models_results))]
    
    n_metrics = len(metrics)
    n_models = len(models_results)
    
    # Spacing logic
    group_gap = 0.2              
    total_bar_width = 1 - group_gap
    bar_width = total_bar_width / n_models
    
    x = np.arange(n_metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add a light dashed grid behind the bars
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    for i, model in enumerate(models_results):
        means = []
        stds = []
        for m in metrics:
            vals = [fold[m] for fold in model]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        
        offset = (i * bar_width) - (total_bar_width / 2) + (bar_width / 2)
        
        ax.bar(x + offset, means, yerr=stds, width=bar_width * 0.9, 
               color=colors[i], edgecolor='black', linewidth=0.8, 
               capsize=4, label=model_names[i], zorder=3)
    
    # --- PERCENTAGE FORMATTING ---
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Labels and Title (x-axis labels in ALL CAPS)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], rotation=30, ha='right', fontsize=10)
    ax.set_ylabel("Performance Score (%)", fontsize=12, fontweight='bold')
    ax.set_title("Model Performance Comparison Across Folds", fontsize=14, pad=60)
    
    # Adjusted limit for percentage (0 to 105% to leave room for legend/stds)
    ax.set_ylim(0.3, 0.8)
    
    # Legend: Single row above the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=4, frameon=False, fontsize=10, borderaxespad=0.000000001)
    
    plt.tight_layout()
    plt.show()


from pathlib import Path

current = Path(__file__).resolve()

ICANS_ROOT = None
for parent in current.parents:
    if parent.name == "ICANS":
        ICANS_ROOT = parent
        break

if ICANS_ROOT is None:
    raise RuntimeError("ICANS folder not found")

RESULTS_DIR = ICANS_ROOT / "Results5Fld"

# ---------------- SVM ----------------
SVM_fold_results_median   = load_pickle(RESULTS_DIR / "SVM_5fld_median_Results.pkl")
SVM_fold_results_mean     = load_pickle(RESULTS_DIR / "SVM_5fld_mean_Results.pkl")
SVM_fold_results_covupper = load_pickle(RESULTS_DIR / "SVM_5fld_covar_Results.pkl")

# ---------------- LR ----------------
LR_fold_results_median   = load_pickle(RESULTS_DIR / "LR_5fld_median_Results.pkl")
LR_fold_results_mean     = load_pickle(RESULTS_DIR / "LR_5fld_mean_Results.pkl")
LR_fold_results_covupper = load_pickle(RESULTS_DIR / "LR_5fld_covar_Results.pkl")

# ---------------- KNN ----------------
KNN_fold_results_median   = load_pickle(RESULTS_DIR / "KNN_5fld_median_Results.pkl")
KNN_fold_results_mean     = load_pickle(RESULTS_DIR / "KNN_5fld_mean_Results.pkl")
KNN_fold_results_covupper = load_pickle(RESULTS_DIR / "KNN_5fld_covar_Results.pkl")

# ---------------- RESNET ----------------
RESNET_noshuffle   = load_pickle(RESULTS_DIR / "RESNET_5foldNOSHUFFLE_Results.pkl")
RESNET_withshuffle  = load_pickle(RESULTS_DIR / "RESNET_5foldTIMESHUFFLE_Results.pkl")
RESNET_onlyGAP     = load_pickle(RESULTS_DIR / "RESNET_5foldOnlyGap_Results.pkl")


all_models_results = [LR_fold_results_median, LR_fold_results_mean, LR_fold_results_covupper,
                     SVM_fold_results_median, SVM_fold_results_mean, SVM_fold_results_covupper,
                      KNN_fold_results_median, KNN_fold_results_mean, KNN_fold_results_covupper,
                     RESNET_onlyGAP, RESNET_noshuffle, RESNET_withshuffle] 
model_names = ['LR-Median','LR-Mean', 'LR-Covar_upper', 
               'SVM-Median','SVM-Mean','SVM-Covar_upper',
               'KNN-Median','KNN-Mean','KNN-Covar_upper',
               'ResNet-only GAP Ordinal','ResNet-BiLSTM-no shuffle Ordinal','ResNet-BiLSTM-with shuffle Ordinal']

plot_model_performance_bar(all_models_results, model_names)


# # **Conf mat visualization**

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_fold_confmats_grid(
    fold_results,
    class_labels=None,
    mode="both",   # "raw", "normalized", "both"
    font_size=10,
    figsize=(20, 8)
):

    n_folds = len(fold_results)

    # Decide rows
    n_rows = 2 if mode == "both" else 1

    fig, axes = plt.subplots(n_rows, n_folds, figsize=figsize)

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_folds == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, fold in enumerate(fold_results):
        Y_true = fold['Y_true']
        Y_pred = fold['Y_pred']

        cm = confusion_matrix(Y_true, Y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_percent = cm_norm * 100

        # -------- RAW --------
        if mode in ["raw", "both"]:
            ax = axes[0, i]

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        cbar=False, square=True,
                        linewidths=0.5, linecolor='gray',
                        xticklabels=class_labels,
                        yticklabels=class_labels,
                        annot_kws={"size": font_size},
                        ax=ax)

            ax.set_title(f"Fold {i+1}", fontsize=font_size+2)
            ax.set_xlabel("Predicted", fontsize=font_size)
            ax.set_ylabel("True", fontsize=font_size)

        # -------- NORMALIZED --------
        if mode in ["normalized", "both"]:
            row_idx = 1 if mode == "both" else 0
            ax = axes[row_idx, i]

            sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                        cbar=False, square=True,
                        linewidths=0.5, linecolor='gray',
                        xticklabels=class_labels,
                        yticklabels=class_labels,
                        annot_kws={"size": font_size},
                        ax=ax)

            ax.set_title(f"Fold {i+1} (%)", fontsize=font_size+2)
            ax.set_xlabel("Predicted", fontsize=font_size)
            ax.set_ylabel("True", fontsize=font_size)

    plt.tight_layout()
    plt.show()

class_labels = ["0 - Mild", "1-2 Moderate", "3-4 Severe"]

# Only RAW confusion matrices (single row)
plot_fold_confmats_grid(
    RESNET_onlyGAP,
    class_labels=class_labels,
    mode="both",
    font_size=12,
    figsize=(20, 10)
)


# In[ ]:




