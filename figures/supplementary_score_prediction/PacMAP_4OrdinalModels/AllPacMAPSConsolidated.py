
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
from matplotlib.lines import Line2D
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


from pathlib import Path

if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

PACMAP_ROOT = None

for parent in current.parents:
    if (parent / "PacMAP_4OrdinalModels").exists():
        PACMAP_ROOT = parent
        break

if PACMAP_ROOT is None:
    raise RuntimeError("PacMAP_4OrdinalModels folder not found")

RASS_pacmap_data = load_pickle(PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "RASS_pacmap_data.pkl")
Yraw_tst_global_RASS = load_pickle(PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "Yraw_tst_global_RASS.pkl")

CAMS_pacmap_data = load_pickle(PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "CAMS_pacmap_data.pkl")
Yraw_tst_global_CAMS = load_pickle(PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "Yraw_tst_global_CAMS.pkl")

GCS_pacmap_data = load_pickle(PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "GCS_pacmap_data.pkl")
Yraw_tst_global_GCS = load_pickle(PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "Yraw_tst_global_GCS.pkl")

ICANS_pacmap_data = load_pickle(PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "ICANS_pacmap_data.pkl")
Yraw_tst_global_ICANS = load_pickle(PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "Yraw_tst_global_ICANS.pkl")


# # **Whole dataset Pacmap for 4 models**

def plot_all_pacmaps_grid(data_dict, figsize=(8, 8), dpi=150, save_path=None):
    # Consolidated dataset configuration definitions for all 4 scales
    dataset_configs = {
        'RASS': {
            'ordered_classes': [-5, -4, -3, -2, -1, 0],
            'title': 'RASS PaCMAP Embedding',
            'legend_title': 'RASS scale detailed key',
            'label_prefix': 'RASS',
            'legend_cols': 3,
            'ordered_labels': [
                "RASS 0 (No/Mild)",   "RASS -1 (No/Mild)",  "RASS -2 (Moderate)",
                "RASS -3 (Moderate)", "RASS -4 (Severe)",   "RASS -5 (Severe)"
            ],
            'group_info': {
                 0: ('limegreen', 'darkgreen', 'No/Mild'), 
                -1: ('mediumseagreen', 'darkgreen', 'No/Mild'),
                -2: ('gold', 'darkgoldenrod', 'Moderate'),
                -3: ('goldenrod', 'darkgoldenrod', 'Moderate'),
                -4: ('salmon', 'darkred', 'Severe'), 
                -5: ('red', 'darkred', 'Severe')
            }
        },
        'CAMS': {
            'ordered_classes': [0, 1, 2, 3, 4, 5, 6, 7],
            'title': 'CAMS PaCMAP Embedding',
            'legend_title': 'CAMS scale detailed key',
            'label_prefix': 'CAMS',
            'legend_cols': 4,
            'ordered_labels': [
                "CAMS 0 (No/Mild)",  "CAMS 1 (No/Mild)",  "CAMS 2 (Moderate)", "CAMS 3 (Moderate)",
                "CAMS 4 (Moderate)", "CAMS 5 (Moderate)", "CAMS 6 (Severe)",   "CAMS 7 (Severe)"
            ],
            'group_info': {
                0: ('limegreen', 'darkgreen', 'No/Mild'), 
                1: ('mediumseagreen', 'darkgreen', 'No/Mild'),
                2: ('khaki', 'darkgoldenrod', 'Moderate'),
                3: ('palegoldenrod', 'darkgoldenrod', 'Moderate'),
                4: ('darkkhaki', 'darkgoldenrod', 'Moderate'),
                5: ('goldenrod', 'darkgoldenrod', 'Moderate'),
                6: ('salmon', 'darkred', 'Severe'), 
                7: ('red', 'darkred', 'Severe')
            }
        },
        'ICANS': {
            'ordered_classes': [0, 1, 2, 3, 4],
            'title': 'ICANS PaCMAP Embedding',
            'legend_title': 'ICANS scale detailed key',
            'label_prefix': 'ICANS',
            'legend_cols': 3,
            'ordered_labels': [
                "ICANS 0 (No/Mild)", None, None,
                "ICANS 1 (Moderate)", "ICANS 2 (Moderate)",  None,
                "ICANS 3 (Severe)",   "ICANS 4 (Severe)",   None
            ],
            'group_info': {
                0: ('limegreen', 'darkgreen', 'No/Mild'), 
                1: ('gold', 'darkgoldenrod', 'Moderate'),
                2: ('goldenrod', 'darkgoldenrod', 'Moderate'),
                3: ('salmon', 'darkred', 'Severe'), 
                4: ('red', 'darkred', 'Severe')
            }
        },
        'GCS': {
            'ordered_classes': list(range(3, 16)),
            'title': 'GCS PaCMAP Embedding',
            'legend_title': 'GCS scale detailed key',
            'label_prefix': 'GCS',
            'legend_cols': 4, # Changed to 3 columns to prevent clipping issues
            'ordered_labels': [
                "GCS 3 (Severe)",   "GCS 4 (Severe)", "GCS 5 (Severe)", "GCS 6 (Severe)",
                "GCS 7 (Severe)", "GCS 8 (Severe)", None, None,
                "GCS 9 (Moderate)", "GCS 10 (Moderate)", "GCS 11 (Moderate)", "GCS 12 (Moderate)",
                "GCS 13 (No/Mild)", "GCS 14 (No/Mild)", "GCS 15 (No/Mild)", None
            ],
            'group_info': {
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
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    
    # Fully populate the 2x2 grid layout matrix
    subplot_mapping = {
        'RASS': axes[0, 0],
        'GCS': axes[0, 1],
        'CAMS': axes[1, 0],
        'ICANS': axes[1, 1]
    }

    for scale_name, ax in subplot_mapping.items():
        emb = np.array(data_dict[scale_name]['logits'])
        y_raw = np.array(data_dict[scale_name]['y_true'])
        
        cfg = dataset_configs[scale_name]
        raw_legend_elements = []
        
        for raw_val in cfg['ordered_classes']:
            mask = (y_raw == raw_val)
            if np.sum(mask) == 0:
                continue
                
            face_c, edge_c, group_name = cfg['group_info'][raw_val]
            
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=3, # Bumped slightly for better visibility
                alpha=0.8,
                facecolor=face_c,
                edgecolor=edge_c,
                linewidth=0.3,
                zorder=2
            )
            
            raw_legend_elements.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=face_c,
                    markeredgecolor=edge_c,
                    markersize=5, # Slightly bigger so legends are legible
                    markeredgewidth=0.8,
                    label=f"{cfg['label_prefix']} {raw_val} ({group_name})"
                )
            )

        def get_handle(label_str):
            for h in raw_legend_elements:
                if h.get_label() == label_str:
                    return h
            return None

        # Build column groupings dynamically while padding missing slots safely
        ordered_handles = [
            get_handle(lbl) if lbl is not None else Line2D([], [], alpha=0, label='')
            for lbl in cfg['ordered_labels']
        ]
        ordered_labels_cleaned = [lbl if lbl is not None else '' for lbl in cfg['ordered_labels']]

        # Axis Panel Styling Configurations
        ax.set_xlabel("Dimension 1", labelpad=6, fontweight='normal', fontsize=9)
        ax.set_ylabel("Dimension 2", labelpad=6, fontweight='normal', fontsize=9)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        ax.spines['left'].set_edgecolor('black')
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['bottom'].set_linewidth(1.0)
            
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        
        # Position legends directly UNDER each subplot instead of floating high above it
        ax.legend(
            handles=ordered_handles,
            labels=ordered_labels_cleaned,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15), 
            ncol=cfg['legend_cols'], # Using 2 columns makes the legend block cleaner under the plots
            title=cfg['legend_title'],
            frameon=True,
            fontsize=5,
            title_fontsize=6
        )

    # Balanced padding that won't squash main scatter plots
    plt.tight_layout()

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        
    plt.show()


# In[138]:


embedding_payload = {
    'RASS': {
        'logits': RASS_pacmap_data,
        'y_true': Yraw_tst_global_RASS
    },
    'CAMS': {
        'logits': CAMS_pacmap_data,
        'y_true': Yraw_tst_global_CAMS
    },
    'ICANS': {
        'logits': ICANS_pacmap_data,
        'y_true': Yraw_tst_global_ICANS
    },
    'GCS': {
        'logits': GCS_pacmap_data,        
        'y_true': Yraw_tst_global_GCS     
    }
}

# Run execution rendering
plot_all_pacmaps_grid(
    data_dict=embedding_payload,
    save_path=PACMAP_ROOT / "PacMAP_4OrdinalModels"/ "FourPacMAPsTogether.png"
)