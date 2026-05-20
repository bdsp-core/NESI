# -------------------------
# Standard Library
# -------------------------
import os
import pickle
from pathlib import Path

# -------------------------
# Core Scientific Stack
# -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# -------------------------
# Visualization
# -------------------------
import seaborn as sns
# -------------------------
# I/O / Models
# -------------------------
import joblib
import h5py
import hdf5storage
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
from pathlib import Path

def find_project_root(marker="RASS"):
    if "__file__" in globals():
        current = Path(__file__).resolve()
    else:
        current = Path.cwd().resolve()

    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent

    raise RuntimeError(f"Project root not found (missing '{marker}' folder)")

PROJECT_ROOT = find_project_root("RASS")

#-------------------------- RASS Results -----------------------------------------------

RESULTS_DIR_RASS = PROJECT_ROOT / "RASS"/ "Results5Fld"

SVM_median_RASS = load_pickle(RESULTS_DIR_RASS / "SVM_5fld_median_Results_6classRASS.pkl")
SVM_mean_RASS =  load_pickle(RESULTS_DIR_RASS / "SVM_5fld_mean_Results_6classRASS.pkl")
SVM_covupper_RASS = load_pickle(RESULTS_DIR_RASS / "SVM_5fld_covar_Results_6classRASS.pkl")

LR_median_RASS = load_pickle(RESULTS_DIR_RASS / "LR_5fld_median_Results_6classRASS.pkl")
LR_mean_RASS =  load_pickle(RESULTS_DIR_RASS / "LR_5fld_mean_Results_6classRASS.pkl")
LR_covupper_RASS= load_pickle(RESULTS_DIR_RASS / "LR_5fld_covar_Results_6classRASS.pkl")

KNN_median_RASS = load_pickle(RESULTS_DIR_RASS / "KNN_5fld_median_Results_6classRASS.pkl")
KNN_mean_RASS =  load_pickle(RESULTS_DIR_RASS / "KNN_5fld_mean_Results_6classRASS.pkl")
KNN_covupper_RASS = load_pickle(RESULTS_DIR_RASS / "KNN_5fld_covar_Results_6classRASS.pkl")

RESNET_noshuffle_RASS = load_pickle(RESULTS_DIR_RASS / "ResNetBiLSTM_NoShuffle_5fldResults_6classRASS.pkl")
RESNET_withshuffle_RASS  = load_pickle(RESULTS_DIR_RASS / "ReSNetBiLSTM_WithShuffle_5fldResults_6classRASS.pkl")
RESNET_onlyGAP_RASS  = load_pickle(RESULTS_DIR_RASS / "ResNetGAP_5fldResults_6classRASS.pkl")

#-------------------------- GCS Results -----------------------------------------------

RESULTS_DIR_GCS = PROJECT_ROOT / "GCS"/ "Results5Fld"

SVM_median_GCS = load_pickle(RESULTS_DIR_GCS / "SVM_5fld_median_Results_3classGCS.pkl")
SVM_mean_GCS =  load_pickle(RESULTS_DIR_GCS / "SVM_5fld_mean_Results_3classGCS.pkl")
SVM_covupper_GCS = load_pickle(RESULTS_DIR_GCS / "SVM_5fld_covar_Results_3classGCS.pkl")

LR_median_GCS = load_pickle(RESULTS_DIR_GCS / "LR_5fld_median_Results_3classGCS.pkl")
LR_mean_GCS =  load_pickle(RESULTS_DIR_GCS / "LR_5fld_mean_Results_3classGCS.pkl")
LR_covupper_GCS = load_pickle(RESULTS_DIR_GCS / "LR_5fld_covar_Results_3classGCS.pkl")

KNN_median_GCS = load_pickle(RESULTS_DIR_GCS / "KNN_5fld_median_Results_3classGCS.pkl")
KNN_mean_GCS =  load_pickle(RESULTS_DIR_GCS / "KNN_5fld_mean_Results_3classGCS.pkl")
KNN_covupper_GCS = load_pickle(RESULTS_DIR_GCS / "KNN_5fld_covar_Results_3classGCS.pkl")

RESNET_noshuffle_GCS = load_pickle(RESULTS_DIR_GCS / "ResNetBiLSTM_NoShuffle_5fldResults_3classGCS.pkl")
RESNET_withshuffle_GCS  = load_pickle(RESULTS_DIR_GCS / "ReSNetBiLSTM_WithShuffle_5fldResults_3classGCS.pkl")
RESNET_onlyGAP_GCS  = load_pickle(RESULTS_DIR_GCS / "ResNetGAP_5fldResults_3classGCS.pkl")


#-------------------------- CAMS Results -----------------------------------------------

RESULTS_DIR_CAMS = PROJECT_ROOT / "CAMS"/ "Results5Fld"

SVM_median_CAMS = load_pickle(RESULTS_DIR_CAMS / "SVM_5fld_median_Results_3class.pkl")
SVM_mean_CAMS =  load_pickle(RESULTS_DIR_CAMS / "SVM_5fld_mean_Results_3class.pkl")
SVM_covupper_CAMS = load_pickle(RESULTS_DIR_CAMS / "SVM_5fld_covar_Results_3class.pkl")

LR_median_CAMS = load_pickle(RESULTS_DIR_CAMS / "LR_5fld_median_Results_3class.pkl")
LR_mean_CAMS =  load_pickle(RESULTS_DIR_CAMS / "LR_5fld_mean_Results_3class.pkl")
LR_covupper_CAMS = load_pickle(RESULTS_DIR_CAMS / "LR_5fld_covar_Results_3class.pkl")

KNN_median_CAMS = load_pickle(RESULTS_DIR_CAMS / "KNN_5fld_median_Results_3class.pkl")
KNN_mean_CAMS =  load_pickle(RESULTS_DIR_CAMS / "KNN_5fld_mean_Results_3class.pkl")
KNN_covupper_CAMS = load_pickle(RESULTS_DIR_CAMS / "KNN_5fld_covar_Results_3class.pkl")

RESNET_noshuffle_CAMS = load_pickle(RESULTS_DIR_CAMS / "RESNET_5foldNOSHUFFLE_Results_3class.pkl")
RESNET_withshuffle_CAMS  = load_pickle(RESULTS_DIR_CAMS / "RESNET_5foldTIMESHUFFLE_Results_3class.pkl")
RESNET_onlyGAP_CAMS  = load_pickle(RESULTS_DIR_CAMS / "RESNET_5foldOnlyGap_Results_3class.pkl")

#-------------------------- ICANS Results -----------------------------------------------

RESULTS_DIR_ICANS = PROJECT_ROOT / "ICANS"/ "Results5Fld"

SVM_median_ICANS = load_pickle(RESULTS_DIR_ICANS / "SVM_5fld_median_Results.pkl")
SVM_mean_ICANS =  load_pickle(RESULTS_DIR_ICANS / "SVM_5fld_mean_Results.pkl")
SVM_covupper_ICANS = load_pickle(RESULTS_DIR_ICANS / "SVM_5fld_covar_Results.pkl")

LR_median_ICANS = load_pickle(RESULTS_DIR_ICANS / "LR_5fld_median_Results.pkl")
LR_mean_ICANS =  load_pickle(RESULTS_DIR_ICANS / "LR_5fld_mean_Results.pkl")
LR_covupper_ICANS = load_pickle(RESULTS_DIR_ICANS / "LR_5fld_covar_Results.pkl")

KNN_median_ICANS = load_pickle(RESULTS_DIR_ICANS / "KNN_5fld_median_Results.pkl")
KNN_mean_ICANS =  load_pickle(RESULTS_DIR_ICANS / "KNN_5fld_mean_Results.pkl")
KNN_covupper_ICANS = load_pickle(RESULTS_DIR_ICANS / "KNN_5fld_covar_Results.pkl")

RESNET_noshuffle_ICANS = load_pickle(RESULTS_DIR_ICANS / "RESNET_5foldNOSHUFFLE_Results.pkl")
RESNET_withshuffle_ICANS  = load_pickle(RESULTS_DIR_ICANS / "RESNET_5foldTIMESHUFFLE_Results.pkl")
RESNET_onlyGAP_ICANS  = load_pickle(RESULTS_DIR_ICANS / "RESNET_5foldOnlyGap_Results.pkl")

#-------------------------- GCS Results -----------------------------------------------

RESULTS_DIR_GCS = PROJECT_ROOT / "GCS"/ "Results5Fld"

SVM_median_GCS = load_pickle(RESULTS_DIR_GCS / "SVM_5fld_median_Results_3classGCS.pkl")
SVM_mean_GCS =  load_pickle(RESULTS_DIR_GCS / "SVM_5fld_mean_Results_3classGCS.pkl")
SVM_covupper_GCS = load_pickle(RESULTS_DIR_GCS / "SVM_5fld_covar_Results_3classGCS.pkl")

LR_median_GCS = load_pickle(RESULTS_DIR_GCS / "LR_5fld_median_Results_3classGCS.pkl")
LR_mean_GCS =  load_pickle(RESULTS_DIR_GCS / "LR_5fld_mean_Results_3classGCS.pkl")
LR_covupper_GCS = load_pickle(RESULTS_DIR_GCS / "LR_5fld_covar_Results_3classGCS.pkl")

KNN_median_GCS = load_pickle(RESULTS_DIR_GCS / "KNN_5fld_median_Results_3classGCS.pkl")
KNN_mean_GCS =  load_pickle(RESULTS_DIR_GCS / "KNN_5fld_mean_Results_3classGCS.pkl")
KNN_covupper_GCS = load_pickle(RESULTS_DIR_GCS / "KNN_5fld_covar_Results_3classGCS.pkl")

RESNET_noshuffle_GCS = load_pickle(RESULTS_DIR_GCS / "ResNetBiLSTM_NoShuffle_5fldResults_3classGCS.pkl")
RESNET_withshuffle_GCS  = load_pickle(RESULTS_DIR_GCS / "ReSNetBiLSTM_WithShuffle_5fldResults_3classGCS.pkl")
RESNET_onlyGAP_GCS  = load_pickle(RESULTS_DIR_GCS / "ResNetGAP_5fldResults_3classGCS.pkl")


def plot_multi_task_performance(tasks_dict,
                                 model_names,
                                 metrics_common,
                                 task_specific_metrics=None,
                                 figsize=(10,7),
                                 dpi=210,
                                 colors=None):

    import matplotlib.ticker as mtick
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------------------------
    # TASK-SPECIFIC SETTINGS
    # ---------------------------
    task_titles = {
        "RASS": "RASS\nRASS -5, -4, -3, -2, -1, 0",

        "GCS": "GCS\nMild (GCS 13–15), Moderate (GCS 9–12), Severe (GCS 3–8)",

        "CAMS": "CAM-S\nMild (CAM-S 0–1), Moderate (CAM-S 2–5), Severe (CAM-S 6–7)",

        "ICANS": "ICANS\nMild (ICANS 0), Moderate (ICANS 1–2), Severe (ICANS 3–4)"
    }

    task_ylims = {
        "RASS": (0.0, 0.89),
        "GCS": (0.30, 0.80),
        "CAMS": (0.30, 0.80),
        "ICANS": (0.30, 0.80)
    }

    # ---------------------------
    # FIGURE
    # ---------------------------
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        dpi=dpi,
        sharey=False
    )

    axes = axes.flatten()

    # ---------------------------
    # COLORS
    # ---------------------------
    if colors is None:
        cmap = plt.colormaps.get_cmap('Pastel1')
        colors = [
            cmap(i)
            for i in np.linspace(0, 1, len(model_names))
        ]

    # ---------------------------
    # PLOTTING LOOP
    # ---------------------------
    for ax, (task_name, models_results) in zip(axes, tasks_dict.items()):

        metrics = list(metrics_common)

        if (
            task_specific_metrics
            and task_name in task_specific_metrics
        ):
            metrics += task_specific_metrics[task_name]

        n_metrics = len(metrics)
        n_models = len(models_results)

        # spacing
        group_gap = 0.2
        total_bar_width = 1 - group_gap
        bar_width = total_bar_width / n_models

        x = np.arange(n_metrics)

        # grid
        ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.6,
            zorder=0
        )

        # ---------------------------
        # BARS
        # ---------------------------
        for i, model in enumerate(models_results):

            means = []
            stds = []

            for m in metrics:
                vals = [fold[m] for fold in model]

                means.append(np.mean(vals))
                stds.append(np.std(vals))

            offset = (
                (i * bar_width)
                - (total_bar_width / 2)
                + (bar_width / 2)
            )

            ax.bar(
                x + offset,
                means,
                yerr=stds,
                width=bar_width * 1.2,
                color=colors[i],
                edgecolor='black',
                linewidth=0.7,
                error_kw={"elinewidth": 0.7}, 
                capsize=1,
                label=model_names[i],
                zorder=3
            )

        # ---------------------------
        # AXIS FORMATTING
        # ---------------------------
        ax.yaxis.set_major_formatter(
            mtick.PercentFormatter(1.0)
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            [m.upper() for m in metrics],
            rotation=20,
            ha='right',
            fontsize=5
        )

        ax.tick_params(axis='y', labelsize=6)

        ax.set_title(
            task_titles[task_name],
            fontsize=6,
            fontweight='normal',
            pad=4
        )

        ax.set_ylim(task_ylims[task_name])

    # ---------------------------
    # Y LABELS
    # ---------------------------
    axes[0].set_ylabel(
        "Performance (%)",
        fontsize=6,
        fontweight='normal'
    )

    axes[2].set_ylabel(
        "Performance (%)",
        fontsize=6,
        fontweight='normal'
    )

    # ---------------------------
    # GLOBAL LEGEND
    # ---------------------------
    fig.legend(
        model_names,
        loc='upper center',
        bbox_to_anchor=(0.5, 1),
        ncol=6,
        frameon=False,
        fontsize=5
    )

    # ---------------------------
    # LAYOUT
    # ---------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


all_rass_models = [
    LR_median_RASS, LR_mean_RASS, LR_covupper_RASS,
    SVM_median_RASS, SVM_mean_RASS, SVM_covupper_RASS,
    KNN_median_RASS, KNN_mean_RASS, KNN_covupper_RASS,
    RESNET_onlyGAP_RASS, RESNET_noshuffle_RASS, RESNET_withshuffle_RASS
]

all_gcs_models = [
    LR_median_GCS, LR_mean_GCS, LR_covupper_GCS,
    SVM_median_GCS, SVM_mean_GCS, SVM_covupper_GCS,
    KNN_median_GCS, KNN_mean_GCS, KNN_covupper_GCS,
    RESNET_onlyGAP_GCS, RESNET_noshuffle_GCS, RESNET_withshuffle_GCS
]

all_cams_models = [
    LR_median_CAMS, LR_mean_CAMS, LR_covupper_CAMS,
    SVM_median_CAMS, SVM_mean_CAMS, SVM_covupper_CAMS,
    KNN_median_CAMS, KNN_mean_CAMS, KNN_covupper_CAMS,
    RESNET_onlyGAP_CAMS, RESNET_noshuffle_CAMS, RESNET_withshuffle_CAMS
]

all_icans_models = [
    LR_median_ICANS, LR_mean_ICANS, LR_covupper_ICANS,
    SVM_median_ICANS, SVM_mean_ICANS, SVM_covupper_ICANS,
    KNN_median_ICANS, KNN_mean_ICANS, KNN_covupper_ICANS,
    RESNET_onlyGAP_ICANS, RESNET_noshuffle_ICANS, RESNET_withshuffle_ICANS
]

tasks = {
    "RASS": all_rass_models,
    "GCS": all_gcs_models,
    "CAMS": all_cams_models,
    "ICANS": all_icans_models
}

metrics_common = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro"
]

task_specific_metrics = {
    "RASS": ["1-level difference accuracy"]
}

model_names = [
    'LR-Median',
    'LR-Mean',
    'LR-Covar_upper',

    'SVM-Median',
    'SVM-Mean',
    'SVM-Covar_upper',

    'KNN-Median',
    'KNN-Mean',
    'KNN-Covar_upper',

    'ResNet-only GAP',
    'ResNet-BiLSTM-no shuffle',
    'ResNet-BiLSTM-with shuffle'
]

pastel_colors = [
    "#AFCBFF",  # pastel blue
    "#A0E7E5",  # pastel cyan
    "#B4F8C8",  # pastel mint green
    "#FBE7C6",  # pastel peach
    "#FFB7B2",  # pastel pink
    "#E2C2FF",  # pastel lavender
    "#FFD6A5",  # pastel orange
    "#FDFFB6",  # pastel yellow
    "#CAFFBF",  # soft lime green
    "#9BF6FF",  # sky cyan
    "#BDB2FF",  # soft violet-blue
    "#FFC6FF"   # pastel magenta
]

plot_multi_task_performance(
    tasks,
    model_names=model_names,
    metrics_common=metrics_common,
    task_specific_metrics=task_specific_metrics,
    colors=pastel_colors
)