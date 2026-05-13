#!/usr/bin/env python
# coding: utf-8

# # **Libraries**

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from pathlib import Path

# # **Load all cohort metadata**

current = Path(__file__).resolve()
NESI_ROOT = None
for parent in current.parents:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI folder not found")

# # **Helper functions**

# In[4]:


#----------------------- Raw Scorue distribution plot -----------------------------------

def plot_true_rawscore_histograms(df, title):
    sites = ["RASS", "GCS", "CAMS", "ICANS"]

    fig, axes = plt.subplots(1, 4, figsize=(7,4), dpi=210)  

    for i, site in enumerate(sites):
        ax = axes[i]

        subset = df[df["Dataset"] == site]

        sns.histplot(
            subset["TrueRawScores"].dropna(),
            bins=30,
            kde=True,
            ax=ax
        )

        ax.set_title(site, fontsize=7)
        ax.set_xlabel("True Raw Score", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(axis='both', labelsize=7)

        # show sample size (VERY useful for imbalance datasets)
        ax.text(
            0.95, 0.95,
            f"n={len(subset)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0, edgecolor="none")
        )

    fig.suptitle(title, fontsize=9, y=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    return fig


# # **Load the predicted NESI-sheet for whole dataset (RASS-GCS-ICANS-CAMS)**

# In[5]:


# Load data
NESI_result_sheet = NESI_ROOT / "MORGOTHActivationViz_GroupedbyNESI" / "UniversalBadnessModelResult_Full.csv"
df = pd.read_csv(NESI_result_sheet)

# -------------------------
# Define NESI bins
# -------------------------
bins = [-3, -1, 1, 3]
labels = ['[-3, -1)', '[-1, 1)', '[1, 3]']

df['NESI_Bin'] = pd.cut(
    df['NESI'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# -------------------------
# Split into 3 datasets
# -------------------------
df_bin1 = df[df['NESI_Bin'] == '[-3, -1)'].copy().reset_index(drop=True)  # Mild
df_bin2 = df[df['NESI_Bin'] == '[-1, 1)'].copy().reset_index(drop=True)   # Moderate
df_bin3 = df[df['NESI_Bin'] == '[1, 3]'].copy().reset_index(drop=True)    # Severe

# -------------------------
# Print summary
# -------------------------
print(f"Bin 1 [-3, -1): Mild            -> {len(df_bin1)} samples")
print(f"Bin 2 [-1,  1): Moderate        -> {len(df_bin2)} samples")
print(f"Bin 3 [ 1,  3]: Severe          -> {len(df_bin3)} samples")

# -------------------------
# Sanity check (optional but recommended)
# -------------------------
print("\nDistribution check:")
print(df['NESI_Bin'].value_counts(dropna=False))

# fig_bin1 = plot_true_rawscore_histograms(df_bin1, "NESI ∈ [-3, -1)")
# plt.show()
# fig_bin2 = plot_true_rawscore_histograms(df_bin2, "NESI ∈ [-1, 1)")
# plt.show()
# fig_bin3 = plot_true_rawscore_histograms(df_bin3, "NESI ∈ [1, 3]")
# plt.show()


# ## **Proportional-stratified sampling-based balanced dataset creation**

# In[6]:


def create_balanced_bin_df(df_bin, 
                            target_total_large=500,   # total samples to draw from GCS/RASS
                            min_per_class=10,          # floor so rare classes aren't zeroed out
                            random_seed=42):
    """
    Proportional-stratified sampling for GCS and RASS:
      - Compute the empirical class distribution from the original bin
      - Allocate samples proportionally to target_total_large
      - Apply a min_per_class floor so rare classes survive
      - CAMS and ICANS: keep ALL samples
    
    This preserves the shape of the original histogram while drastically
    reducing GCS/RASS dominance.
    """
    parts = []

    for dataset in ['GCS', 'RASS', 'CAMS', 'ICANS']:
        sub = df_bin[df_bin['Dataset'] == dataset].copy()
        if len(sub) == 0:
            continue

        if dataset in ['GCS', 'RASS']:

            # ── 1. Empirical class counts & proportions ──
            class_counts = sub['TrueRawScores'].value_counts().sort_index()
            class_props  = class_counts / class_counts.sum()

            # ── 2. Proportional allocation ──
            raw_alloc = (class_props * target_total_large).round().astype(int)

            # ── 3. Apply floor: ensure min_per_class for every class ──
            alloc = raw_alloc.clip(lower=min_per_class)

            # ── 4. Cap at actual class size (can't sample more than available) ──
            alloc = alloc.combine(class_counts, min)

            # ── 5. Sample each class ──
            sampled_parts = []
            for score, n in alloc.items():
                cls_df = sub[sub['TrueRawScores'] == score]
                sampled_parts.append(
                    cls_df.sample(n=int(n), random_state=random_seed)
                )

            sampled = pd.concat(sampled_parts).reset_index(drop=True)
            parts.append(sampled)

            # ── Summary ──
            print(f"\n  {dataset}: {len(sub)} → {len(sampled)} total")
            print(f"  {'Score':>8} | {'Original':>10} | {'Proportion':>12} | {'Sampled':>10}")
            print(f"  {'-'*46}")
            for score in class_counts.index:
                print(f"  {score:>8} | {class_counts[score]:>10} | "
                      f"{class_props[score]:>11.1%} | {alloc[score]:>10}")

        else:  # CAMS, ICANS — keep all
            parts.append(sub)
            print(f"\n  {dataset}: {len(sub)} → {len(sub)} (all kept)")

    balanced_df = pd.concat(parts).reset_index(drop=True)
    return balanced_df


# ── Apply to each bin ──
print("=" * 50)
print("── Bin 1  NESI [-3, -2) · Mild ──")
df_bin1_bal = create_balanced_bin_df(df_bin1, target_total_large=500, min_per_class=10)
print(f"\n  ✓ Bin1 total: {len(df_bin1_bal)}")

print("\n" + "=" * 50)
print("── Bin 2  NESI [-2, 0) · Moderate-Mild ──")
df_bin2_bal = create_balanced_bin_df(df_bin2, target_total_large=500, min_per_class=10)
print(f"\n  ✓ Bin2 total: {len(df_bin2_bal)}")

print("\n" + "=" * 50)
print("── Bin 3  NESI [0, 1) · Moderate-Severe ──")
df_bin3_bal = create_balanced_bin_df(df_bin3, target_total_large=500, min_per_class=10)
print(f"\n  ✓ Bin3 total: {len(df_bin3_bal)}")


# In[8]:


# fig_bin1_bal = plot_true_rawscore_histograms(df_bin1_bal, "NESI ∈ [-3, -1)")
# plt.show()
# fig_bin2_bal = plot_true_rawscore_histograms(df_bin2_bal, "NESI ∈ [-1, 1)")
# plt.show()
# fig_bin3_bal = plot_true_rawscore_histograms(df_bin3_bal, "NESI ∈ [1, 3]")
# plt.show()


# # **Feature Engineering-MORGOTH functions**

# In[12]:

#------ RASS Root Path--------
if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

RASS_ROOT = None
for parent in current.parents:
    if (parent / "RASS").exists():
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
    if (parent / "GCS").exists():
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
    if (parent / "CAMS").exists():
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
    if (parent / "ICANS").exists():
        ICANS_ROOT = parent
        break

if ICANS_ROOT is None:
    raise RuntimeError("ICANS folder not found")

print(ICANS_ROOT)

def morgoth_output_file_location(data_group):
    base_paths = {
        "RASS": RASS_ROOT / "RASS" / "MorgothActivations",
        "GCS": GCS_ROOT / "GCS" / "MorgothActivations",
        "CAMS": CAMS_ROOT / "CAMS" / "MorgothActivations",
        "ICANS": ICANS_ROOT / "ICANS" / "MorgothActivations"
    }

    root = base_paths[data_group]

    return {
        "SLOWING": os.path.join(root, "SLOWING"),
        "FOCGEN": os.path.join(root, "FOCGEN"),
        "IIIC": os.path.join(root, "IIIC"),
        "NM": os.path.join(root, "NM"),
        "BS": os.path.join(root, "BS"),
        "SLEEP": os.path.join(root, "SLEEP")
    }


def load_feature(path, filename):
    df = pd.read_csv(os.path.join(path, filename))
    if "pred_class" in df.columns:
        df = df.drop(columns=["pred_class"])
    return df.values


def morgoth_10minfea_matrix(data_frame):
    features = []
    file_names = []

    for row in tqdm(data_frame.itertuples(index=False), total=len(data_frame)):
        fname = row.MorgothOutputFilename
        dataset = row.Dataset

        paths = morgoth_output_file_location(dataset)

        sleep = load_feature(paths["SLEEP"], fname)
        nm = load_feature(paths["NM"], fname)
        bs = load_feature(paths["BS"], fname)

        focgen = load_feature(paths["FOCGEN"], fname)
        slowing = load_feature(paths["SLOWING"], fname)
        iiic = load_feature(paths["IIIC"], fname)

        subject_feature = np.concatenate(
            [sleep, nm, bs, focgen, slowing, iiic],
            axis=1
        )

        median_features = np.median(subject_feature, axis=0)
        features.append(median_features)

    
    X = np.stack(features, axis=0)    
    Y_raw = data_frame["TrueRawScores"].to_numpy()
    dataset_names = data_frame["Dataset"].to_numpy()
    file_names = np.array(file_names)

    print("Feature matrix shape:", X.shape)
    print("Raw Label shape:", Y_raw.shape)
    
    return X, Y_raw, dataset_names, file_names


# In[13]:


X_bin1, Y_raw_bin1, dataset_names_bin1, _ = morgoth_10minfea_matrix(df_bin1_bal)
X_bin2, Y_raw_bin2, dataset_names_bin2, _ = morgoth_10minfea_matrix(df_bin2_bal)
X_bin3, Y_raw_bin3, dataset_names_bin3, _ = morgoth_10minfea_matrix(df_bin3_bal)


feature_names = [
    'Awake','N1','N2', # Sleep head output of morgoth
    'Normal/Abnormal', # NM head output
    'Burst/No Burst', # BS head output
    'No Spike','Focal Spike','Gen. Spike', # Spike localize head
    'No Slowing','Focal Slowing','Gen. Slowing', # Slowing head
    'Other','Seizure','LPD','GPD','LRDA','GRDA' # IIIC head
]

# # **MORGOTH activation Visualization plot:**

# In[54]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb, LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase

DATASET_COLORS = {
    'GCS':   'b',
    'RASS':  'g',
    'CAMS':  'r',
    'ICANS': 'm',
}

def activation_to_rgb(activation_col, base_color_hex):
    base_rgb = np.array(to_rgb(base_color_hex))
    white    = np.ones(3)
    rgb = (1 - activation_col[:, None]) * white + activation_col[:, None] * base_rgb
    return rgb

def plot_nesi_imagesc(
    X,
    Y_raw,
    dataset_names,
    nesi_bin_label,
    feature_names,
    max_samples=None,
    figsize=(7,6),
    head_boundaries=None
):
    if head_boundaries is None:
        head_boundaries = [2, 3, 4, 7, 10]

    dataset_names = np.array(dataset_names)
    Y_raw         = np.array(Y_raw)
    scale_order   = ['GCS', 'RASS', 'CAMS', 'ICANS']

    # ── 1. Sort ──
    sorted_idx       = []
    scale_boundaries = []
    scale_groups     = []
    cursor = 0

    for scale in scale_order:
        mask = dataset_names == scale
        if mask.sum() == 0:
            continue
        group_idx = np.where(mask)[0]
        group_idx = group_idx[np.argsort(Y_raw[group_idx])]
        if max_samples is not None and len(group_idx) > max_samples:
            np.random.seed(42)
            group_idx = np.random.choice(group_idx, max_samples, replace=False)
        sorted_idx.extend(group_idx.tolist())
        scale_groups.append((scale, cursor, cursor + len(group_idx)))
        cursor += len(group_idx)
        scale_boundaries.append(cursor)

    sorted_idx   = np.array(sorted_idx)
    X_sorted     = X[sorted_idx]
    names_sorted = dataset_names[sorted_idx]
    n_features   = X_sorted.shape[1]
    n_samples    = X_sorted.shape[0]

    # ── 2. Build RGB image ──
    rgb_img = np.ones((n_features, n_samples, 3))
    for col_idx in range(n_samples):
        scale     = names_sorted[col_idx]
        color_hex = DATASET_COLORS.get(scale, '#333333')
        rgb_img[:, col_idx, :] = activation_to_rgb(X_sorted[col_idx], color_hex)

    # ── 3. Figure layout: legend | colorbars | heatmap ──
    fig = plt.figure(figsize=figsize, dpi=210)
    gs  = gridspec.GridSpec(
        3, 1,
        height_ratios=[0.04, 0.10, 1],
        hspace=0.18,
    )

    ax_legend = fig.add_subplot(gs[0])
    ax_cbars  = fig.add_subplot(gs[1])
    ax_main   = fig.add_subplot(gs[2])

    # ── 4. Legend ──
    ax_legend.set_axis_off()
    ax_legend.patch.set_visible(False)
    patches = [
        mpatches.Patch(color=DATASET_COLORS[s], label=s, alpha=0.8)
        for s in scale_order if s in dataset_names
    ]
    ax_legend.legend(
        handles=patches,
        loc='center',
        ncols=4,
        fontsize=8,
        title='True clinical scale',
        title_fontsize=8,
        framealpha=0.7,
        bbox_to_anchor=(0.5, 1.7)
    )

    # ── 5. Four horizontal colorbars proportional to dataset size ──
    ax_cbars.set_axis_off()
    ax_cbars.patch.set_visible(False)

    total_samples = n_samples
    gap = 0.01

    for (scale, start, end) in scale_groups:
        color   = DATASET_COLORS[scale]
        x_left  = start / total_samples
        width_f = (end - start) / total_samples

        inset = ax_cbars.inset_axes(
            [x_left + gap, 0.05, width_f - 2 * gap, 0.55],
            transform=ax_cbars.transAxes
        )
        inset.patch.set_visible(False)

        cmap_scale = LinearSegmentedColormap.from_list(
            f'cmap_{scale}',
            [to_rgb('white'), to_rgb(color)]
        )
        sm = plt.cm.ScalarMappable(cmap=cmap_scale, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cb = plt.colorbar(sm, cax=inset, orientation='horizontal')

        # Adaptive ticks based on group size
        if (end - start) < 200:
            cb.set_ticks([0, 0.5, 1.0])
            cb.set_ticklabels(['0', '0.5', '1'], fontsize=5)
        else:
            cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cb.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=5)

        cb.ax.tick_params(labelsize=5, length=2, pad=1)
        cb.ax.set_title(
            f'{scale}\n (n={end - start})',
            fontsize=6.5, fontweight='bold',
            color=to_rgb(color), pad=3
        )

    # ── 6. Main heatmap ──
    ax_main.imshow(
        rgb_img,
        aspect='auto',
        extent=[0, n_samples, n_features - 0.5, -0.5]
    )
    ax_main.set_xlim(0, n_samples)

    # ── 7. Vertical black dashed dividers ──
    for boundary in scale_boundaries[:-1]:
        ax_main.axvline(
            x=boundary, color='k', linestyle='--', linewidth=2.0, alpha=1.0
        )

    # ── 8. Horizontal red dotted separators ──
    for b in head_boundaries:
        ax_main.axhline(
            y=b + 0.5, color='red', linestyle=':', linewidth=1.75, alpha=0.9
        )

    # ── 9. Axes labels ──
    ax_main.set_yticks(range(len(feature_names)))
    ax_main.set_yticklabels(feature_names, fontsize=7)
    ax_main.set_xlabel('No. of observations / No. of 10 min EEG snippets', fontsize=8)
    ax_main.set_ylabel('MORGOTH event level EEG features', fontsize=8)

    # ── 10. Title ──
    fig.suptitle(
        f'MORGOTH Feature Activation Map for NESI ∈ {nesi_bin_label}',
        fontsize=9,
        fontweight='bold',
        y=0.989
    )

    plt.show()
    return fig


# In[55]:


fig_nesi_mild = plot_nesi_imagesc(X_bin1, Y_raw_bin1, dataset_names_bin1, nesi_bin_label='[-3,  -1)',   
                  feature_names=feature_names)

mild_nesi_savepath = NESI_ROOT / "FigureNESI" / "MildNESIMorgothActivation.png"

mild_nesi_savepath.parent.mkdir(parents=True, exist_ok=True)

fig_nesi_mild.savefig(
    mild_nesi_savepath,
    dpi=300,
    bbox_inches="tight"
)

print("Saved at:", mild_nesi_savepath.resolve())

# In[ ]:

fig_nesi_moderate = plot_nesi_imagesc(X_bin2, Y_raw_bin2, dataset_names_bin2, nesi_bin_label='[-1,  1)',   
                  feature_names=feature_names)

moderate_nesi_savepath = NESI_ROOT / "FigureNESI" / "ModerateNESIMorgothActivation(-1,1).png"
moderate_nesi_savepath.parent.mkdir(parents=True, exist_ok=True)
fig_nesi_moderate.savefig(moderate_nesi_savepath, dpi=300, bbox_inches="tight")

# In[ ]:

fig_nesi_worst = plot_nesi_imagesc(X_bin3, Y_raw_bin3, dataset_names_bin3, nesi_bin_label='[1,  3]',   
                  feature_names=feature_names)

worst_nesi_savepath = NESI_ROOT / "FigureNESI" / "WorstNESIMorgothActivation(1, 3).png"
worst_nesi_savepath.parent.mkdir(parents=True, exist_ok=True)
fig_nesi_worst.savefig(worst_nesi_savepath, dpi=300, bbox_inches="tight")