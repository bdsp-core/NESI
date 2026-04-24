#!/usr/bin/env python
# coding: utf-8

# # **Libraries**

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
from scipy.stats import spearmanr
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
    'font.size': 5,
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


# # **Load Result Objects**

# In[2]:
current = Path(__file__).resolve()
RASS_ROOT = None
for parent in current.parents:
    if parent.name == "RASS":
        RASS_ROOT = parent
        break

if RASS_ROOT is None:
    raise RuntimeError("RASS folder not found")

RESULTS_DIR = RASS_ROOT / "Results5Fld"

SVM_fold_results_median = load_pickle(RESULTS_DIR / "SVM_5fld_median_Results_6classRASS.pkl")
SVM_fold_results_mean =  load_pickle(RESULTS_DIR / "SVM_5fld_mean_Results_6classRASS.pkl")
SVM_fold_results_covupper = load_pickle(RESULTS_DIR / "SVM_5fld_covar_Results_6classRASS.pkl")

LR_fold_results_median = load_pickle(RESULTS_DIR / "LR_5fld_median_Results_6classRASS.pkl")
LR_fold_results_mean =  load_pickle(RESULTS_DIR / "LR_5fld_mean_Results_6classRASS.pkl")
LR_fold_results_covupper= load_pickle(RESULTS_DIR / "LR_5fld_covar_Results_6classRASS.pkl")

KNN_fold_results_median = load_pickle(RESULTS_DIR / "KNN_5fld_median_Results_6classRASS.pkl")
KNN_fold_results_mean =  load_pickle(RESULTS_DIR / "KNN_5fld_mean_Results_6classRASS.pkl")
KNN_fold_results_covupper = load_pickle(RESULTS_DIR / "KNN_5fld_covar_Results_6classRASS.pkl")

RESNET_ordinal_noshuffle = load_pickle(RESULTS_DIR / "ResNetBiLSTM_NoShuffle_5fldResults_6classRASS.pkl")
RESNET_ordinal_withshuffle  = load_pickle(RESULTS_DIR / "ReSNetBiLSTM_WithShuffle_5fldResults_6classRASS.pkl")
RESNET_ordinal_onlyGAP  = load_pickle(RESULTS_DIR / "ResNetGAP_5fldResults_6classRASS.pkl")


# # **Plot results**

# ## **Fold-wise confmat**

# In[3]:

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_foldwise_confmat(result_obj, model_name, rass_labels=['-5','-4','-3','-2','-1','0']):
    """
    Plots fold-wise confusion matrices with improved scaling and readability.
    """
    n_folds = len(result_obj)
    
    # Increase figsize for better aspect ratio and readability
    # 15x8 provides enough room for 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for i in range(n_folds):
        ax = axes[i]

        y_true = np.array(result_obj[i]['Y_true'])
        y_pred = np.array(result_obj[i]['Y_pred'])

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize='true') * 100

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=rass_labels)
        
        # Increased font sizes for labels and removed colorbar for cleaner look
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='.1f')

        ax.set_title(f'Fold {i+1}', fontsize=12, fontweight='bold', fontfamily='serif', pad=10)
        
        # Styling the text inside the heatmaps
        for text in disp.text_.ravel():
            text.set_fontsize(9)
            text.set_fontweight('bold')
            text.set_fontfamily('serif')

        # Clean up the grid and ticks
        ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        
        # Increase axis label sizes
        ax.set_xlabel('Predicted label', fontsize=10)
        ax.set_ylabel('True label', fontsize=10)

    # Turn off unused subplot
    for j in range(n_folds, len(axes)):
        axes[j].axis('off')

    # Global Title
    plt.suptitle(
        f'Fold-wise Confusion Matrices: {model_name}',
        fontsize=16,
        fontweight='bold',
        fontfamily='serif',
        y=0.95
    )

    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()


# ### **Confmat: LR-mean**

# In[4]:


plot_foldwise_confmat(LR_fold_results_mean , 'LR-mean')


# ### **Confmat: ResNet-BiLSTM without time shuffle**

# In[5]:


plot_foldwise_confmat(RESNET_ordinal_noshuffle, 'ResNet-BiLSTM without time-shuffle')


# ### **Confmat: ResNet with GAP**

# In[6]:


plot_foldwise_confmat(RESNET_ordinal_onlyGAP, 'ResNet-GAP')


# ## **Fold-wise result summary from the save result objects directly**

# In[7]:


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def plot_model_performance_bar(models_results, model_names=None, metrics=None, figsize=(12, 7), colors=None):
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models_results))]
        
    if metrics is None:
        metrics = ['accuracy', '1-level difference accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
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
    ax.set_title("Model Performance Comparison Across Folds\n RASS -5, RASS -4, RASS -3, RASS -2, RASS -1, RASS 0", 
                 fontsize=14, pad=70)
    
    # Adjusted limit for percentage (0 to 105% to leave room for legend/stds)
    ax.set_ylim(0.0, 0.89)
    
    # Legend: Single row above the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
               ncol=4, frameon=False, fontsize=10, borderaxespad=0.000000001)
    
    plt.tight_layout()
    plt.show()


# In[8]:


all_models_results = [LR_fold_results_median, LR_fold_results_mean, LR_fold_results_covupper,
                      SVM_fold_results_median, SVM_fold_results_mean, SVM_fold_results_covupper,
                      KNN_fold_results_median, KNN_fold_results_mean, KNN_fold_results_covupper,
                      RESNET_ordinal_onlyGAP, RESNET_ordinal_noshuffle, RESNET_ordinal_withshuffle] 
model_names = ['LR-Median','LR-Mean', 'LR-Covar_upper', 
               'SVM-Median','SVM-Mean','SVM-Covar_upper',
               'KNN-Median','KNN-Mean','KNN-Covar_upper',
               'ResNet-only GAP Ordinal','ResNet-BiLSTM-no shuffle Ordinal','ResNet-BiLSTM-with shuffle Ordinal']

plot_model_performance_bar(all_models_results, model_names)


# ## **Subjectwise results**

# In[57]:


import pandas as pd

def extract_pid(filename):
    part = filename.split('_')[0]
    pid_full = part.split('-')[1]
    pid = pid_full[5:]
    return pid
# ------------- Gather the fold wise results -------------------------
## ------------Dataframe with BDSPPatientID | Filename | True Grouped RASS | Predicted Grouped RAS

import pandas as pd

def process_model_results_with_diagnosis(
    all_models_results,
    model_names,
    diagnosis_csv_path
):
    """
    Creates model-wise dataframes with:
    BDSPPatientID | Filename | True | Pred | Broad Diagnosis Category
    """

    if len(all_models_results) != len(model_names):
        raise ValueError("Mismatch between number of models and model names")

    # -------- Load diagnosis file --------
    df_diag = pd.read_csv(diagnosis_csv_path)

    df_diag = df_diag[['BDSPPatientID', 'DiagnosisCategory']].rename(
        columns={'DiagnosisCategory': 'Broad Diagnosis Category'}
    )

    # -------- Ensure same dtype for merge --------
    df_diag['BDSPPatientID'] = df_diag['BDSPPatientID'].astype(str)

    output_dfs = {}

    for result, name in zip(all_models_results, model_names):

        y_true_all = []
        y_pred_all = []
        filenames_all = []

        # -------- Collect all folds --------
        for fold_dict in result:
            y_true_all.extend(fold_dict["Y_true"])
            y_pred_all.extend(fold_dict["Y_pred"])
            filenames_all.extend(fold_dict["Test_filenames"])

        # -------- Create dataframe --------
        df = pd.DataFrame({
            "Filename": filenames_all,
            "True Grouped GCS": y_true_all,
            "Predicted Grouped GCS": y_pred_all
        })

        # -------- Extract patient ID --------
        df["BDSPPatientID"] = df["Filename"].apply(extract_pid).astype(str)

        # -------- Merge diagnosis --------
        df = df.merge(df_diag, on="BDSPPatientID", how="left")

        # -------- Reorder columns --------
        df = df[
            [
                "BDSPPatientID",
                "Filename",
                "True Grouped GCS",
                "Predicted Grouped GCS",
                "Broad Diagnosis Category"
            ]
        ]

        # -------- Sort --------
        df = df.sort_values(by="BDSPPatientID").reset_index(drop=True)

        output_dfs[name] = df

    return output_dfs


import numpy as np
import pandas as pd

def compute_all_models_subjectwise(dicts_RASS_SubjectResult, model_names):
    """
    Computes subject-wise metrics for ALL models,
    including Broad Diagnosis Category.

    Returns:
    --------
    dict:
        key = model_name
        value = subject-wise dataframe
    """

    all_results = {}

    for model_name in model_names:

        df = dicts_RASS_SubjectResult[model_name]

        subject_results = []

        for subject_id, g in df.groupby('BDSPPatientID'):

            y_true = g['True Grouped GCS'].values
            y_pred = g['Predicted Grouped GCS'].values

            # ---- segment-level metrics ----
            acc = (y_true == y_pred).astype(int)
            one_level = (np.abs(y_true - y_pred) <= 1).astype(int)
            ae = np.abs(y_true - y_pred)

            # ---- diagnosis (same for all rows of this subject) ----
            diagnosis = g['Broad Diagnosis Category'].iloc[0]

            # ---- subject-level aggregation ----
            subject_results.append({
                "BDSPPatientID": subject_id,
                "Broad Diagnosis Category": diagnosis,
                "Accuracy": np.mean(acc),
                "1-level Accuracy": np.mean(one_level),
                "MAE": np.mean(ae),
                "n_segments": len(g)
            })

        all_results[model_name] = pd.DataFrame(subject_results)

    return all_results


# In[58]:


# ------------- Gather the fold wise results -------------------------
## ------------Dataframe with BDSPPatientID | Filename | True Grouped RASS | Predicted Grouped RASS
dicts_RASS_Subjectwise_predictions_diag = process_model_results_with_diagnosis(
    all_models_results,
    model_names,
    '/home/ayush/Desktop/MGB_RASS_EEGs/combined_rass_unique_bdsp_with_diagnoses.csv'
)

# --------- compute subject wise metrics for all models ----------------
# ----------Each model: BDSPPatientID| Subjectwise ACC | Subjectwise 1-level-ACC | Subjectwise MAE ---------------
subjectwise_metrics_all_models = compute_all_models_subjectwise(
    dicts_RASS_Subjectwise_predictions_diag,
    model_names
)


# ### **MAE plot results**

# In[63]:


import numpy as np
import matplotlib.pyplot as plt

def plot_subjectwise_mae_boxplot(models_results, model_names, colors=None):
    """
    Boxplot of subject-wise MAE across multiple models.

    Parameters:
    -----------
    models_results : dict
        key = model_name
        value = subjectwise dataframe (must contain 'MAE')
    model_names : list
        ordered list of model names
    colors : list or None
        optional custom colors
    """

    # ---------------- Collect MAE ----------------
    mae_data = []
    for m in model_names:
        mae_data.append(models_results[m]['MAE'].values)

    # ---------------- Colors ----------------
    if colors is None:
        cmap = plt.colormaps.get_cmap('Pastel1')
        colors = [cmap(i) for i in np.linspace(0, 1, len(model_names))]

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    bp = ax.boxplot(
        mae_data,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color='black', linewidth=1)
    )

    # ---------------- Style boxes ----------------
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.9)

    # ---------------- Whiskers ----------------
    for w in bp['whiskers']:
        w.set_color('black')

    # ---------------- Caps ----------------
    for c in bp['caps']:
        c.set_color('black')

    # ---------------- Medians ----------------
    for m in bp['medians']:
        m.set_color('black')
        m.set_linewidth(2)

    # ---------------- Outliers ----------------
    for f in bp['fliers']:
        f.set_markerfacecolor('gray')
        f.set_markeredgecolor('black')
        f.set_alpha(0.7)

    # ---------------- Labels ----------------
    ax.set_xticks(np.arange(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontfamily='serif')

    ax.set_ylabel('MAE (Subject-wise)', fontfamily='serif')
    ax.set_title('Subject-wise MAE Comparison Across Models',
                 fontweight='bold', fontfamily='serif')

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


# In[64]:


plot_subjectwise_mae_boxplot(subjectwise_metrics_all_models , model_names)


# ### **Subjectwise ACC and 1-level difference ACC plot**

# In[65]:


import numpy as np
import matplotlib.pyplot as plt

def plot_acc_1level_boxplots(subjectwise_metrics_all_models, model_names):
    # ---------------- Colors ----------------
    cmap = plt.colormaps.get_cmap('Pastel1')
    colors = [cmap(i) for i in np.linspace(0, 1, len(model_names))]

    # ---------------- Collect data ----------------
    acc_data = [subjectwise_metrics_all_models[m]['Accuracy'].values for m in model_names]
    one_data = [subjectwise_metrics_all_models[m]['1-level Accuracy'].values for m in model_names]

    # ---------------- Figure ----------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    n_models = len(model_names)
    
    # ADJUST THESE FOR "LOOKS"
    width = 0.6  # Wider boxes
    gap_between_groups = 4  # Space between Accuracy and 1-level Accuracy
    
    # ---------------- X positions ----------------
    # We place boxes at 1, 2, 3... then jump for the next group
    x_acc = np.arange(n_models)
    x_one = np.arange(n_models) + n_models + gap_between_groups

    # ---------------- Plot ----------------
    legend_handles = []

    def draw_group(data, positions):
        handles = []
        for i, (d, color) in enumerate(zip(data, colors)):
            bp = ax.boxplot(
                d,
                positions=[positions[i]],
                widths=width,
                patch_artist=True,
                showfliers=True,
                # Thinner median and box lines look more modern
                medianprops=dict(color='black', linewidth=1.2),
                boxprops=dict(linewidth=1),
                whiskerprops=dict(linewidth=1),
                capprops=dict(linewidth=1),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=2, alpha=0.5)
            )
            
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_edgecolor('#333333')
            handles.append(bp['boxes'][0])
        return handles

    # Draw both groups
    h1 = draw_group(acc_data, x_acc)
    draw_group(one_data, x_one)
    legend_handles = h1

    # ---------------- X-axis labels ----------------
    ax.set_xticks([np.mean(x_acc), np.mean(x_one)])
    ax.set_xticklabels(['Accuracy', '1-level Accuracy'], 
                       fontfamily='serif', fontweight='bold', fontsize=5)

    # ---------------- Legend ----------------
    # ncol=3 or 4 depending on how many models you have
    ax.legend(
        legend_handles,
        model_names,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        frameon=False,
        fontsize=5
    )

    # ---------------- Styling ----------------
    ax.set_ylabel('Score', fontfamily='serif', fontsize=5)
    ax.set_title('Subject-wise Performance Across Models',
                 fontweight='bold', fontfamily='serif', pad=50, fontsize=7)

    # Clean up the spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Ensure y-axis starts at 0 and ends at 1
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()


# In[66]:


plot_acc_1level_boxplots(
    subjectwise_metrics_all_models,
    model_names
)


# ### **Disease stratified results**

# #### **Acc and 1-level diffrene ACC**

# In[91]:


import numpy as np
import matplotlib.pyplot as plt

def plot_model_diagnosis_boxplots_acc_1levelacc(subjectwise_metrics_all_models, model_name):
    df = subjectwise_metrics_all_models[model_name]

    # ---------------- Data Preparation ----------------
    diagnoses = sorted(df['Broad Diagnosis Category'].dropna().unique())
    
    # Use a cleaner color palette
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(diagnoses)))

    acc_data = [df[df['Broad Diagnosis Category'] == d]['Accuracy'].values * 100 for d in diagnoses]
    one_data = [df[df['Broad Diagnosis Category'] == d]['1-level Accuracy'].values * 100 for d in diagnoses]

    # ---------------- Aesthetics & Sizing ----------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    
    n_diag = len(diagnoses)
    width = 0.6  # Much wider boxes for better visibility
    gap_between_groups = 3 # Space between the two main categories
    
    # Define positions so they are clustered tightly
    x_acc = np.arange(n_diag)
    x_one = np.arange(n_diag) + n_diag + gap_between_groups

    # ---------------- Plotting Function ----------------
    def draw_boxes(data, positions):
        boxes = []
        for i, (d_slice, color) in enumerate(zip(data, colors)):
            bp = ax.boxplot(
                d_slice,
                positions=[positions[i]],
                widths=width,
                patch_artist=True,
                showfliers=True,
                # Modern, thinner lines
                medianprops=dict(color='black', linewidth=1.2),
                boxprops=dict(linewidth=0.8, color='#333333'),
                whiskerprops=dict(linewidth=0.8, color='#333333'),
                capprops=dict(linewidth=0.8, color='#333333'),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.4, markeredgecolor='none')
            )
            bp['boxes'][0].set_facecolor(color)
            boxes.append(bp['boxes'][0])
        return boxes

    # Execute plotting
    legend_handles = draw_boxes(acc_data, x_acc)
    draw_boxes(one_data, x_one)

    # ---------------- Labels & Styling ----------------
    # Center the group labels
    ax.set_xticks([np.mean(x_acc), np.mean(x_one)])
    ax.set_xticklabels(['Accuracy', '1-level Accuracy'], 
                       fontfamily='serif', fontweight='bold', fontsize=5)

    ax.set_ylabel('Score (%)', fontfamily='serif', fontsize=5)
    ax.set_ylim(-2, 102) # Slight padding so 0 and 100 aren't cut off

    # Clean legend placement
    ax.legend(
        legend_handles,
        diagnoses,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(diagnoses),
        frameon=False,
        fontsize=10
    )

    # Clean up the chart (Remove top/right spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    
    ax.set_title(f'{model_name}: Performance Across Diagnosis Categories',
                 fontweight='bold', fontfamily='serif', fontsize=7, pad=55)

    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


# In[92]:


plot_model_diagnosis_boxplots_acc_1levelacc(
    subjectwise_metrics_all_models,
    model_name='ResNet-only GAP Ordinal'
)


# In[94]:


# plot_model_diagnosis_boxplots_acc_1levelacc(
#     subjectwise_metrics_all_models,
#     model_name='ResNet-BiLSTM-no shuffle Ordinal'
# )


# #### **MAE**

# In[95]:


import numpy as np
import matplotlib.pyplot as plt

def plot_model_diagnosis_boxplots_mae(subjectwise_metrics_all_models, model_name):
    df = subjectwise_metrics_all_models[model_name]

    # ---------------- Data Preparation ----------------
    diagnoses = sorted(df['Broad Diagnosis Category'].dropna().unique())
    
    # Pastel colors
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(diagnoses)))

    mae_data = [
        df[df['Broad Diagnosis Category'] == d]['MAE'].values
        for d in diagnoses
    ]

    # ---------------- Figure ----------------
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    width = 0.6
    x_pos = np.arange(len(diagnoses))

    legend_handles = []

    # ---------------- Plotting ----------------
    for i, (d_slice, color) in enumerate(zip(mae_data, colors)):

        bp = ax.boxplot(
            d_slice,
            positions=[x_pos[i]],
            widths=width,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color='black', linewidth=1.2),
            boxprops=dict(linewidth=0.8, color='#333333'),
            whiskerprops=dict(linewidth=0.8, color='#333333'),
            capprops=dict(linewidth=0.8, color='#333333'),
            flierprops=dict(
                marker='o',
                markerfacecolor='gray',
                markersize=2,
                alpha=0.4,
                markeredgecolor='none'
            )
        )

        bp['boxes'][0].set_facecolor(color)
        legend_handles.append(bp['boxes'][0])

    # ---------------- Labels ----------------
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        diagnoses,
        rotation=30,
        ha='right',
        fontfamily='serif',
        fontsize=5
    )

    ax.set_ylabel('MAE', fontweight='bold', fontfamily='serif', fontsize=5)

    # ---------------- Legend ----------------
    ax.legend(
        legend_handles,
        diagnoses,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(diagnoses),
        frameon=False,
        fontsize=5
    )

    # ---------------- Styling ----------------
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')

    ax.set_title(
        f'{model_name}: MAE Across Diagnosis Categories',
        fontweight='bold',
        fontfamily='serif',
        fontsize=7,
        pad=40
    )

    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


# In[97]:


plot_model_diagnosis_boxplots_mae(
    subjectwise_metrics_all_models,
    model_name='ResNet-only GAP Ordinal'
)