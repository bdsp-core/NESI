#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
from datetime import datetime
from pathlib import Path
import pacmap

import h5py
import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from tqdm import tqdm

from coral_pytorch.dataset import (
    corn_label_from_logits
)
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import corn_loss


# # **Helper function**

# In[2]:


# ---------------- Custom Function ----------------
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


def save_pickle(obj, save_path):
    """
    Save any Python object as a pickle file.

    Parameters
    ----------
    obj : any
        Object to save.
    save_path : str
        Output .pkl file path.
    """

    dir_name = os.path.dirname(save_path)

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {save_path}")

def extract_pid(filename):
    part = filename.split('_')[0]          
    pid_full = part.split('-')[1]          
    pid = pid_full[5:]                     
    return pid

def group_gcs(y):
    if y in [3,4,5,6,7,8]:
        return 2 # Bad Neurological state
    elif y in [9,10,11,12]:
        return 1 # Moderate Neurological state
    elif y in [13,14,15]:
        return 0 # Good Neurological state
    else:
        return None 

def group_rass(y):
    if y in [-5, -4]:
        return 2 # Bad Neurological state
    elif y in [-3, -2]:
        return 1 # Moderate Neurological state
    elif y in [-1, 0]:
        return 0 # Good Neurological state
    else:
        return None 
        
def group_icans(y):
    if y in [0]:
        return 0 # Good Neurological state
    elif y in [1, 2]:
        return 1 # Moderate Neurological state
    elif y in [3, 4]:
        return 2 # Bad Neurological state
    else:
        return None 

def group_cams(x):
    if 0 <= x <= 1:
        return 0 # Good Neurological state
    elif 2 <= x <= 5:
        return 1 # Moderate Neurological state
    else:
        return 2 # Bad Neurological state


# In[3]:



#------ RASS Root Path--------
if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

RASS_ROOT = None
for parent in current.parents:
    if (parent / "NESI").exists():
        RASS_ROOT = parent
        break

if RASS_ROOT is None:
    raise RuntimeError("RASS folder not found")

#------ GCS Root Path--------
if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

GCS_ROOT = None
for parent in current.parents:
    if (parent / "NESI").exists():
        GCS_ROOT = parent
        break

if GCS_ROOT is None:
    raise RuntimeError("GCS folder not found")


#------ CAMS Root Path--------
from pathlib import Path

if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

CAMS_ROOT = None
for parent in current.parents:
    if (parent / "NESI").exists():
        CAMS_ROOT = parent
        break

if CAMS_ROOT is None:
    raise RuntimeError("CAMS folder not found")



#------ ICANS Root Path--------
if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

ICANS_ROOT = None
for parent in current.parents:
    if (parent / "NESI").exists():
        ICANS_ROOT = parent
        break

if ICANS_ROOT is None:
    raise RuntimeError("ICANS folder not found")

Supplementray_fig_Root = ICANS_ROOT


# # **RASS PacMAPs**

# In[4]:


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_rass_pacmap_fixed_colors(emb, y_raw, title="PaCMAP Embedding", save_path=None, figsz=(11, 7),dpi=300):
    emb = np.array(emb)
    y_raw = np.array(y_raw)
    
    # Structural bug fixed here for -5 (removed the trailing duplicate string element)
    group_info = {
         0: ('limegreen', 'darkgreen', 'No/Mild'), 
        -1: ('mediumseagreen', 'darkgreen', 'No/Mild'),
        -2: ('gold', 'darkgoldenrod', 'Moderate'),
        -3: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        -4: ('salmon', 'darkred', 'Severe'), 
        -5: ('red', 'darkred', 'Severe')
    }
    
    fig, ax = plt.subplots(figsize=figsz, dpi=dpi)
    legend_elements = []
    
    rass_classes_ordered = [-5, -4, -3, -2, -1, 0]
    
    for raw_val in rass_classes_ordered:
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0: 
            continue
            
        face_c, edge_c, group_name = group_info[raw_val]
        
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=0.5,
            zorder=2
        )
        
        # Increased markersize slightly to 8 so the edge outlines fit nicely in the legend layout
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"RASS {raw_val} ({group_name})"
            )
        )

    # ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel("Dimension 1", labelpad=8, fontweight='bold')
    ax.set_ylabel("Dimension 2", labelpad=8, fontweight='bold')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    ax.spines['left'].set_edgecolor('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['bottom'].set_linewidth(1.0)
        
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="RASS scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=3
    )
    
    plt.tight_layout()
    
    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        
    plt.show()



# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

def plot_rass_and_feature_pacmaps(
    emb,
    y_raw,
    X_features,
    feature_names,
    Y_tst_NESI_RASS=None,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=300
):

    emb = np.asarray(emb)
    y_raw = np.asarray(y_raw)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[10, 6, 6, 6],
        hspace=0.25,
        wspace=0.10
    )

    #####################################################################
    # TOP PANEL (UNCHANGED)
    #####################################################################
    ax_main = fig.add_subplot(gs[0, 1:5])

    group_info = {
         0: ('limegreen', 'darkgreen', 'No/Mild'),
        -1: ('mediumseagreen', 'darkgreen', 'No/Mild'),
        -2: ('gold', 'darkgoldenrod', 'Moderate'),
        -3: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        -4: ('salmon', 'darkred', 'Severe'),
        -5: ('red', 'darkred', 'Severe')
    }

    legend_elements = []
    rass_classes_ordered = [-5, -4, -3, -2, -1, 0]

    for raw_val in rass_classes_ordered:

        mask = (y_raw == raw_val)

        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax_main.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=0.5,
            zorder=2,
            rasterized=True
        )

        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"RASS {raw_val} ({group_name})"
            )
        )

    ax_main.set_xlabel("Dimension 1", fontweight='bold')
    ax_main.set_ylabel("Dimension 2", fontweight='bold')

    for spine in ['top', 'right']:
        ax_main.spines[spine].set_visible(False)

    ax_main.grid(True, linestyle='--', alpha=0.3)

    ax_main.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="RASS scale detailed key",
        frameon=True,
        fontsize=12,
        title_fontsize=12,
        ncol=3
    )

    #####################################################################
    # FEATURE PANELS (3 × 6) WITH INDIVIDUAL COLORBARS
    #####################################################################

    cmap_all = 'Blues'

    for i in range(18):

        row = 1 + (i // 6)
        col = i % 6

        ax = fig.add_subplot(gs[row, col])

        # ---------------------------
        # NESI PANEL
        # ---------------------------
        if i == 17:

            if plot_nesi and Y_tst_NESI_RASS is not None:

                sc = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=Y_tst_NESI_RASS,
                    cmap=cmap_all,
                    s=4,
                    alpha=1.0,
                    rasterized=True
                )

                ax.set_title("NESI", fontsize=10, fontweight='bold')

                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=9)
            else:
                ax.axis('off')

        elif i < X_features.shape[1]:

            vals = X_features[:, i]

            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=vals,
                cmap=cmap_all,
                s=4,
                alpha=1.0,
                rasterized=True
            )

            ax.set_title(feature_names[i], fontsize=10, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=9)

        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# In[6]:


median_feature_RASS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "X_tst_data_global_RASS_median.pkl"
X_tst_data_global_RASS_median = load_pickle(median_feature_RASS_savepath)

Y_raw_RASS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Yraw_tst_global_RASS.pkl"
Yraw_tst_global_RASS = load_pickle(Y_raw_RASS_savepath)

pacmap_RASS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "RASS_pacmap_data.pkl"
RASS_pacmap_data = load_pickle(pacmap_RASS_savepath)

NESI_RASS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Y_tst_NESI_RASS.pkl"
Y_tst_NESI_RASS = load_pickle(NESI_RASS_savepath)


# In[7]:


feature_names = [
    'Awake','N1','N2',
    'Normal/Abnormal',
    'Burst/No Burst',
    'No Spike','Focal Spike','Generalized Spike',
    'No Slowing','Focal Slowing','Generalized Slowing',
    'Other','Seizure','LPD','GPD','LRDA','GRDA'
]

fig_RASS = plot_rass_and_feature_pacmaps(
    emb=RASS_pacmap_data,
    y_raw=Yraw_tst_global_RASS,
    X_features=X_tst_data_global_RASS_median,
    feature_names=feature_names,
    Y_tst_NESI_RASS=Y_tst_NESI_RASS,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=300
)

savefig_path_RASS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "RASS_new_pacmap.png"
fig_RASS.savefig(savefig_path_RASS, dpi=600, bbox_inches="tight")
plt.show()


# # **GCS PacMAPs**

# In[8]:


def plot_gcs_pacmap_fixed_colors(emb, y_raw, title="PaCMAP Embedding", save_path=None, figsz=(11,7),dpi=300):
    emb = np.array(emb)
    y_raw = np.array(y_raw)
    
    group_info = {
        3: ('red', 'darkred', 'Severe'), 
        4: ('salmon', 'darkred', 'Severe'),
        5: ('tomato', 'darkred', 'Severe'), 
        6: ('darksalmon', 'darkred', 'Severe'),
        7: ('indianred', 'darkred', 'Severe'), 
        8: ('lightcoral', 'darkred', 'Severe'),
        
        9: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        10: ('darkkhaki', 'darkgoldenrod', 'Moderate'),
        11: ('palegoldenrod', 'darkgoldenrod', 'Moderate'),
        12: ('khaki', 'darkgoldenrod', 'Moderate'),

        13: ('mediumseagreen', 'darkgreen', 'No/Mild'), 
        14: ('limegreen', 'darkgreen', 'No/Mild'),
        15: ('lime', 'darkgreen', 'No/Mild'),
    }
    
    fig, ax = plt.subplots(figsize=figsz, dpi=dpi)
    legend_elements = []
    
    for raw_val in range(3, 16):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0: 
            continue
            
        face_c, edge_c, group_name = group_info[raw_val]
        
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=0.5,
            zorder=2
        )
        
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"GCS {raw_val} ({group_name})"
            )
        )

    ax.set_xlabel("Dimension 1", labelpad=8, fontweight='normal')
    ax.set_ylabel("Dimension 2", labelpad=8, fontweight='normal')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    ax.spines['left'].set_edgecolor('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['bottom'].set_linewidth(1.0)
        
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    # ---------------------------------------------------
    # 🔥 LEGEND FIX (STRICT COLUMN CONTROL)
    # ---------------------------------------------------

    def H(label):
        for h in legend_elements:
            if h.get_label() == label:
                return h
        return Line2D([], [], alpha=0)

    ordered_labels = [
        # Column 1 (3–6 Severe)
        "GCS 3 (Severe)", "GCS 4 (Severe)", "GCS 5 (Severe)", "GCS 6 (Severe)",
        
        # Column 2 (7–8)
        "GCS 7 (Severe)", "GCS 8 (Severe)", None, None,
        
        # Column 3 (9–12 Moderate)
        "GCS 9 (Moderate)", "GCS 10 (Moderate)", "GCS 11 (Moderate)", "GCS 12 (Moderate)",
        
        # Column 4 (13–15 No/Mild)
        "GCS 13 (No/Mild)", "GCS 14 (No/Mild)", "GCS 15 (No/Mild)", None
    ]

    ordered_handles = [
        H(lbl) if lbl is not None else Line2D([], [], alpha=0)
        for lbl in ordered_labels
    ]

    ax.legend(
        handles=ordered_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="GCS scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=4
    )

    plt.tight_layout()

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

def plot_gcs_and_feature_pacmaps(
    emb,
    y_raw,
    X_features,
    feature_names,
    Y_tst_NESI_GCS=None,
    plot_nesi=False,
    figsize=(24, 20),
    dpi=300
):

    emb = np.asarray(emb)
    y_raw = np.asarray(y_raw)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[10, 6, 6, 6],
        hspace=0.25,
        wspace=0.10
    )

    #####################################################################
    # TOP: GCS MAIN PLOT (same styling as your function)
    #####################################################################
    ax_main = fig.add_subplot(gs[0, 1:5])

    group_info = {
        3: ('red', 'darkred', 'Severe'),
        4: ('salmon', 'darkred', 'Severe'),
        5: ('tomato', 'darkred', 'Severe'),
        6: ('darksalmon', 'darkred', 'Severe'),
        7: ('indianred', 'darkred', 'Severe'),
        8: ('lightcoral', 'darkred', 'Severe'),

        9: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        10: ('darkkhaki', 'darkgoldenrod', 'Moderate'),
        11: ('palegoldenrod', 'darkgoldenrod', 'Moderate'),
        12: ('khaki', 'darkgoldenrod', 'Moderate'),

        13: ('mediumseagreen', 'darkgreen', 'No/Mild'),
        14: ('limegreen', 'darkgreen', 'No/Mild'),
        15: ('lime', 'darkgreen', 'No/Mild'),
    }

    legend_elements = []

    for raw_val in range(3, 16):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax_main.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=0.5,
            zorder=2
        )

        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"GCS {raw_val} ({group_name})"
            )
        )

    ax_main.set_xlabel("Dimension 1", fontweight='bold')
    ax_main.set_ylabel("Dimension 2", fontweight='bold')
    ax_main.grid(True, linestyle='--', alpha=0.3)

    def H(label):
        for h in legend_elements:
            if h.get_label() == label:
                return h
        return Line2D([], [], alpha=0)

    ordered_labels = [
        # Column 1 (3–6 Severe)
        "GCS 3 (Severe)", "GCS 4 (Severe)", "GCS 5 (Severe)", "GCS 6 (Severe)",
        
        # Column 2 (7–8)
        "GCS 7 (Severe)", "GCS 8 (Severe)", None, None,
        
        # Column 3 (9–12 Moderate)
        "GCS 9 (Moderate)", "GCS 10 (Moderate)", "GCS 11 (Moderate)", "GCS 12 (Moderate)",
        
        # Column 4 (13–15 No/Mild)
        "GCS 13 (No/Mild)", "GCS 14 (No/Mild)", "GCS 15 (No/Mild)", None
    ]

    ordered_handles = [
        H(lbl) if lbl is not None else Line2D([], [], alpha=0)
        for lbl in ordered_labels
    ]
    ax_main.legend(
        handles=ordered_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="GCS scale detailed key",
        frameon=True,
        fontsize=12,
        title_fontsize=12,
        ncol=4
    )
    for spine in ['top', 'right', ]:
        ax_main.spines[spine].set_visible(False)
    
    ax_main.spines['left'].set_visible(True)
    ax_main.spines['left'].set_edgecolor('black')
    ax_main.spines['left'].set_linewidth(1.2)

    ax_main.spines['bottom'].set_visible(True)
    ax_main.spines['bottom'].set_edgecolor('black')
    ax_main.spines['bottom'].set_linewidth(1.2)
    #####################################################################
    # FEATURE GRID (3 × 6 = 18 SLOTS)
    #####################################################################

    cmap_all = 'Blues'

    for i in range(18):

        row = 1 + (i // 6)
        col = i % 6

        ax = fig.add_subplot(gs[row, col])

        # -----------------------------
        # LAST SLOT = NESI (optional)
        # -----------------------------
        if i == 17:

            if plot_nesi and Y_tst_NESI_GCS is not None:

                sc = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=Y_tst_NESI_GCS,
                    cmap=cmap_all,
                    s=4,
                    alpha=1.0,
                    rasterized=True
                )

                ax.set_title("NESI", fontsize=10, fontweight='bold')

                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=9)

            else:
                ax.axis('off')

        # -----------------------------
        # FEATURE PANELS
        # -----------------------------
        elif i < X_features.shape[1]:

            vals = X_features[:, i]

            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=vals,
                cmap=cmap_all,
                s=4,
                alpha=1.0,
                rasterized=True
            )

            ax.set_title(feature_names[i], fontsize=10, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=9)

        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# In[10]:


median_feature_GCS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "X_tst_data_global_GCS_median.pkl"
X_tst_data_global_GCS_median = load_pickle(median_feature_GCS_savepath)

Y_raw_GCS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Yraw_tst_global_GCS.pkl"
Yraw_tst_global_GCS = load_pickle(Y_raw_GCS_savepath)

pacmap_GCS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "GCS_pacmap_data.pkl"
GCS_pacmap_data = load_pickle(pacmap_GCS_savepath)

NESI_GCS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Y_tst_NESI_GCS.pkl"
Y_tst_NESI_GCS = load_pickle(NESI_GCS_savepath)


feature_names = [
    'Awake','N1','N2',
    'Normal/Abnormal',
    'Burst/No Burst',
    'No Spike','Focal Spike','Generalized Spike',
    'No Slowing','Focal Slowing','Generalized Slowing',
    'Other','Seizure','LPD','GPD','LRDA','GRDA'
]

fig_GCS = plot_gcs_and_feature_pacmaps(
    emb=GCS_pacmap_data,
    y_raw=Yraw_tst_global_GCS,
    X_features=X_tst_data_global_GCS_median,
    feature_names=feature_names,
    Y_tst_NESI_GCS=Y_tst_NESI_GCS,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=300
)

savefig_path_GCS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "GCS_new_pacmap.png"
fig_GCS.savefig(savefig_path_GCS, dpi=600, bbox_inches="tight")
plt.show()


# # **CAMS PacMAPs**

# In[11]:


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_cams_pacmap_fixed_colors(emb, y_raw, title="PaCMAP Embedding", save_path=None, figsz=(11,7), dpi=300):
    emb = np.array(emb)
    y_raw = np.array(y_raw)
    
    group_info = {
        0: ('limegreen', 'darkgreen', 'No/Mild'), 
        1: ('mediumseagreen', 'darkgreen', 'No/Mild'),
        2: ('khaki', 'darkgoldenrod', 'Moderate'),
        3: ('palegoldenrod', 'darkgoldenrod', 'Moderate'),
        4: ('darkkhaki', 'darkgoldenrod', 'Moderate'),
        5: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        6: ('salmon', 'darkred', 'Severe'), 
        7: ('red', 'darkred', 'Severe')
    }
    
    fig, ax = plt.subplots(figsize=figsz, dpi=dpi)
    legend_elements = []
    
    for raw_val in range(8):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0: 
            continue
            
        face_c, edge_c, group_name = group_info[raw_val]
        
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=1.0,
            zorder=2
        )
        
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"CAMS {raw_val} ({group_name})"
            )
        )

    # ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel("Dimension 1", labelpad=8, fontweight='normal')
    ax.set_ylabel("Dimension 2", labelpad=8, fontweight='normal')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    ax.spines['left'].set_edgecolor('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['bottom'].set_linewidth(1.0)
        
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="CAM-S scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=4
    )
    
    plt.tight_layout()
    
    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        
    plt.show()




# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import os

def plot_cams_and_feature_pacmaps(
    emb,
    y_raw,
    X_features,
    feature_names,
    Y_tst_NESI_CAMS=None,
    plot_nesi=False,
    figsize=(24, 20),
    dpi=300
):

    emb = np.asarray(emb)
    y_raw = np.asarray(y_raw)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[10, 6, 6, 6],
        hspace=0.25,
        wspace=0.10
    )

    #####################################################################
    # TOP: CAM-S MAIN PLOT (UNCHANGED STYLE)
    #####################################################################
    ax_main = fig.add_subplot(gs[0, 1:5])

    group_info = {
        0: ('limegreen', 'darkgreen', 'No/Mild'),
        1: ('mediumseagreen', 'darkgreen', 'No/Mild'),
        2: ('khaki', 'darkgoldenrod', 'Moderate'),
        3: ('palegoldenrod', 'darkgoldenrod', 'Moderate'),
        4: ('darkkhaki', 'darkgoldenrod', 'Moderate'),
        5: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        6: ('salmon', 'darkred', 'Severe'),
        7: ('red', 'darkred', 'Severe')
    }

    legend_elements = []

    for raw_val in range(8):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax_main.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=1.0,
            zorder=2
        )

        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"CAMS {raw_val} ({group_name})"
            )
        )

    ax_main.set_xlabel("Dimension 1", fontweight='normal')
    ax_main.set_ylabel("Dimension 2", fontweight='normal')

    for spine in ['top', 'right']:
        ax_main.spines[spine].set_visible(False)

    ax_main.spines['left'].set_edgecolor('black')
    ax_main.spines['left'].set_linewidth(1.0)
    ax_main.spines['bottom'].set_edgecolor('black')
    ax_main.spines['bottom'].set_linewidth(1.0)

    ax_main.grid(True, linestyle='--', alpha=0.3)

    ax_main.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        title="CAM-S scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=4
    )

    #####################################################################
    # FEATURE GRID (3 × 6 = 18 SLOTS)
    #####################################################################

    cmap_all = 'Blues'

    for i in range(18):

        row = 1 + (i // 6)
        col = i % 6

        ax = fig.add_subplot(gs[row, col])

        # -----------------------------
        # NESI SLOT
        # -----------------------------
        if i == 17:

            if plot_nesi and Y_tst_NESI_CAMS is not None:

                sc = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=Y_tst_NESI_CAMS,
                    cmap=cmap_all,
                    s=4,
                    alpha=1.0,
                    rasterized=True
                )

                ax.set_title("NESI", fontsize=10, fontweight='bold')

                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=9)

            else:
                ax.axis('off')

        # -----------------------------
        # FEATURE PANELS
        # -----------------------------
        elif i < X_features.shape[1]:

            vals = X_features[:, i]

            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=vals,
                cmap=cmap_all,
                s=4,
                alpha=1.0,
                rasterized=True
            )

            ax.set_title(feature_names[i], fontsize=10, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=9)

        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# In[13]:


median_feature_CAMS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "X_tst_data_global_CAMS_median.pkl"
X_tst_data_global_CAMS_median = load_pickle(median_feature_CAMS_savepath)

Y_raw_CAMS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Yraw_tst_global_CAMS.pkl"
Yraw_tst_global_CAMS = load_pickle(Y_raw_CAMS_savepath)

pacmap_CAMS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "CAMS_pacmap_data.pkl"
CAMS_pacmap_data = load_pickle(pacmap_CAMS_savepath)

NESI_CAMS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Y_tst_NESI_CAMS.pkl"
Y_tst_NESI_CAMS = load_pickle(NESI_CAMS_savepath)


feature_names = [
    'Awake','N1','N2',
    'Normal/Abnormal',
    'Burst/No Burst',
    'No Spike','Focal Spike','Generalized Spike',
    'No Slowing','Focal Slowing','Generalized Slowing',
    'Other','Seizure','LPD','GPD','LRDA','GRDA'
]

fig_CAMS = plot_cams_and_feature_pacmaps(
    emb=CAMS_pacmap_data,
    y_raw=Yraw_tst_global_CAMS,
    X_features=X_tst_data_global_CAMS_median,
    feature_names=feature_names,
    Y_tst_NESI_CAMS=Y_tst_NESI_CAMS,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=600
)

savefig_path_CAMS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "CAMS_new_pacmap.png"
fig_CAMS.savefig(savefig_path_CAMS, dpi=600, bbox_inches="tight")
plt.show()


# # **ICANS PacMAPs**

# In[14]:


def plot_ICANS_pacmap_fixed_colors(emb, y_raw, title="PaCMAP Embedding", save_path=None, figsz=(11, 7), dpi=300):
    emb = np.array(emb)
    y_raw = np.array(y_raw)
    
    group_info = {
        0: ('limegreen', 'darkgreen', 'No/Mild'), 
        1: ('gold', 'darkgoldenrod', 'Moderate'),
        2: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        3: ('salmon', 'darkred', 'Severe'), 
        4: ('red', 'darkred', 'Severe')
    }
    
    fig, ax = plt.subplots(figsize=figsz, dpi=dpi)

    legend_elements = []

    for raw_val in range(8):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=1.0,
            zorder=2
        )

        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"ICANS {raw_val} ({group_name})"
            )
        )

    # ---------------------------------------------------
    # 🔥 LEGEND FIX (FORCED COLUMN GROUPING)
    # ---------------------------------------------------

    def get_handle(label):
        for h in legend_elements:
            if h.get_label() == label:
                return h
        return None

    ordered_labels = [
        "ICANS 0 (No/Mild)",   None,                None,
        "ICANS 1 (Moderate)",  "ICANS 2 (Moderate)", None,
        "ICANS 3 (Severe)",    "ICANS 4 (Severe)",   None
    ]

    ordered_handles = [
        get_handle(l) if l is not None else Line2D([], [], alpha=0)
        for l in ordered_labels
    ]

    ax.legend(
        handles=ordered_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        title="ICANS scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=3,
        handletextpad=0.6,
        columnspacing=1.5
    )

    # ---------------------------------------------------
    # keep EVERYTHING else unchanged
    # ---------------------------------------------------

    ax.set_xlabel("Dimension 1", labelpad=8, fontweight='bold')
    ax.set_ylabel("Dimension 2", labelpad=8, fontweight='bold')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    ax.spines['left'].set_edgecolor('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['bottom'].set_linewidth(1.0)
        
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    plt.tight_layout()

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import os

def plot_ICANS_and_feature_pacmaps(
    emb,
    y_raw,
    X_features,
    feature_names,
    Y_tst_NESI_ICANS=None,
    plot_nesi=False,
    figsize=(24, 20),
    dpi=300
):

    emb = np.asarray(emb)
    y_raw = np.asarray(y_raw)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[10, 6, 6, 6],
        hspace=0.25,
        wspace=0.10
    )

    #####################################################################
    # TOP: ICANS MAIN PLOT (UNCHANGED STYLE)
    #####################################################################
    ax_main = fig.add_subplot(gs[0, 1:5])

    group_info = {
        0: ('limegreen', 'darkgreen', 'No/Mild'),
        1: ('gold', 'darkgoldenrod', 'Moderate'),
        2: ('goldenrod', 'darkgoldenrod', 'Moderate'),
        3: ('salmon', 'darkred', 'Severe'),
        4: ('red', 'darkred', 'Severe')
    }

    legend_elements = []

    for raw_val in range(5):
        mask = (y_raw == raw_val)
        if np.sum(mask) == 0:
            continue

        face_c, edge_c, group_name = group_info[raw_val]

        ax_main.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=1.0,
            facecolor=face_c,
            edgecolor=edge_c,
            linewidth=1.0,
            zorder=2
        )

        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=face_c,
                markeredgecolor=edge_c,
                markersize=6,
                markeredgewidth=1,
                label=f"ICANS {raw_val} ({group_name})"
            )
        )

    # -------------------------
    # EXACT ICANS LEGEND STYLE
    # -------------------------
    def get_handle(label):
        for h in legend_elements:
            if h.get_label() == label:
                return h
        return None

    ordered_labels = [
        "ICANS 0 (No/Mild)", None, None,
        "ICANS 1 (Moderate)", "ICANS 2 (Moderate)", None,
        "ICANS 3 (Severe)", "ICANS 4 (Severe)", None
    ]

    ordered_handles = [
        get_handle(l) if l is not None else Line2D([], [], alpha=0)
        for l in ordered_labels
    ]

    ax_main.legend(
        handles=ordered_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        title="ICANS scale detailed key",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=3,
        handletextpad=0.6,
        columnspacing=1.5
    )

    ax_main.set_xlabel("Dimension 1", fontweight='bold')
    ax_main.set_ylabel("Dimension 2", fontweight='bold')

    for spine in ['top', 'right']:
        ax_main.spines[spine].set_visible(False)

    ax_main.spines['left'].set_edgecolor('black')
    ax_main.spines['left'].set_linewidth(1.0)
    ax_main.spines['bottom'].set_edgecolor('black')
    ax_main.spines['bottom'].set_linewidth(1.0)

    ax_main.grid(True, linestyle='--', alpha=0.3)

    #####################################################################
    # FEATURE GRID (3 × 6 = 18 SLOTS)
    #####################################################################

    cmap_all = 'Blues'

    for i in range(18):

        row = 1 + (i // 6)
        col = i % 6

        ax = fig.add_subplot(gs[row, col])

        # -------------------------
        # NESI SLOT
        # -------------------------
        if i == 17:

            if plot_nesi and Y_tst_NESI_ICANS is not None:

                sc = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=Y_tst_NESI_ICANS,
                    cmap=cmap_all,
                    s=4,
                    alpha=1.0,
                    rasterized=True
                )

                ax.set_title("NESI", fontsize=10, fontweight='bold')

                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=9)

            else:
                ax.axis('off')

        # -------------------------
        # FEATURES
        # -------------------------
        elif i < X_features.shape[1]:

            vals = X_features[:, i]

            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=vals,
                cmap=cmap_all,
                s=4,
                alpha=1.0,
                rasterized=True
            )

            ax.set_title(feature_names[i], fontsize=10, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=9)

        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# In[16]:


median_feature_ICANS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "X_tst_data_global_ICANS_median.pkl"
X_tst_data_global_ICANS_median = load_pickle(median_feature_ICANS_savepath)

Y_raw_ICANS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Yraw_tst_global_ICANS.pkl"
Yraw_tst_global_ICANS = load_pickle(Y_raw_ICANS_savepath)

pacmap_ICANS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "ICANS_pacmap_data.pkl"
ICANS_pacmap_data = load_pickle(pacmap_ICANS_savepath)

NESI_ICANS_savepath = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "Y_tst_NESI_ICANS.pkl"
Y_tst_NESI_ICANS = load_pickle(NESI_ICANS_savepath)




feature_names = [
    'Awake','N1','N2',
    'Normal/Abnormal',
    'Burst/No Burst',
    'No Spike','Focal Spike','Generalized Spike',
    'No Slowing','Focal Slowing','Generalized Slowing',
    'Other','Seizure','LPD','GPD','LRDA','GRDA'
]

fig_ICANS = plot_ICANS_and_feature_pacmaps(
    emb=ICANS_pacmap_data,
    y_raw=Yraw_tst_global_ICANS,
    X_features=X_tst_data_global_ICANS_median,
    feature_names=feature_names,
    Y_tst_NESI_ICANS=Y_tst_NESI_ICANS,
    plot_nesi=True,
    figsize=(24, 20),
    dpi=600
)

savefig_path_ICANS = Supplementray_fig_Root /  "SupplementaryScorePredictionResults" / "PacMAPS_colored_byMorgoth_NESI_heads" / "ICANS_new_pacmap.png"
fig_ICANS.savefig(savefig_path_ICANS, dpi=600, bbox_inches="tight")
plt.show()

