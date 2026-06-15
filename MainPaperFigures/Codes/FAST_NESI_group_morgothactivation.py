"""
Single combined MORGOTH-activation figure across the full NESI range.

Differences from Arka's original (which produced 3 separate figures, one per
NESI bin):
  - Single figure, four panels (GCS | RASS | CAMS | ICANS), using *all* cases
    from all three NESI bins.
  - Left → right within each panel = progressively WORSE clinical state.
    Sort direction is per-scale:
        GCS    descending (15  →  3)
        RASS   descending ( 0  → -5)
        CAMS   ascending  ( 0  →  7)
        ICANS  ascending  ( 0  →  4)
  - Each raw-score level subsampled to TARGET_PER_LEVEL columns so progression
    is visually balanced (random drop of excess; rare levels kept as-is).
  - Hierarchical sort within each panel:
        primary   = raw clinical score (in worse direction)
        secondary = NESI (ascending → worse)
  - Extra feature row at the top: NESI itself, rescaled to [0,1] for the
    standard white→panel-color encoding (higher NESI = worse → more saturated).

Per-case NESI is recovered by reproducing Arka's balanced-sample DataFrame
using the same random_state=42, which is byte-identical to the pickled arrays
(verified by exact match on TrueRawScores and Dataset).
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from nesi_fig_style import apply_style, save_fig, FS_LEGEND

# Data and outputs live alongside this script.
from pathlib import Path

if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

NESI_ROOT = None

for parent in current.parents:
    if (parent / "NESI").exists():
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI folder not found")

SCRIPT_DIR = Path(__file__).resolve().parent
YAMA_DIR = NESI_ROOT / "NESI" / "MORGOTHActivationViz_GroupedbyNESI"
OUT_DIR = NESI_ROOT / "MainPaperFigures"

TARGET_PER_LEVEL = 40
EQUALIZE_SEED = 42

DATASET_COLORS = {'GCS': 'b', 'RASS': 'g', 'CAMS': 'r', 'ICANS': 'm'}
DATASET_COLORS_BW = {'GCS': 'k', 'RASS': 'k', 'CAMS': 'k', 'ICANS': 'k'}
SCALE_ORDER = ['GCS', 'RASS', 'CAMS', 'ICANS']
# True → sort descending (high score first), so that left=best, right=worst
SCALE_REVERSE = {'GCS': True, 'RASS': True, 'CAMS': False, 'ICANS': False}

# Boundaries between MORGOTH heads, in the ORIGINAL (no-NESI-row) feature
# index space. With NESI prepended as row 0, these are shifted by +1.
HEAD_BOUNDARIES_BASE = [2, 3, 4, 7, 10]
FEATURE_NAMES_BASE = [
    'Awake', 'N1', 'N2',
    'Normal/Abnormal',
    'Burst/No Burst',
    'No Spike', 'Focal Spike', 'Gen. Spike',
    'No Slowing', 'Focal Slowing', 'Gen. Slowing',
    'Other', 'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA',
]

# NESI bin definitions, must match Arka's
NESI_BINS = [-3, -1, 1, 3]
NESI_BIN_LABELS = ['[-3, -1)', '[-1, 1)', '[1, 3]']


# ── Reproduce Arka's balanced DataFrame to recover NESI per case ──────────
def create_balanced_bin_df(df_bin, target_total_large=500, min_per_class=10,
                            random_seed=42):
    parts = []
    for dataset in ['GCS', 'RASS', 'CAMS', 'ICANS']:
        sub = df_bin[df_bin['Dataset'] == dataset].copy()
        if len(sub) == 0:
            continue
        if dataset in ['GCS', 'RASS']:
            class_counts = sub['TrueRawScores'].value_counts().sort_index()
            class_props = class_counts / class_counts.sum()
            raw_alloc = (class_props * target_total_large).round().astype(int)
            alloc = raw_alloc.clip(lower=min_per_class)
            alloc = alloc.combine(class_counts, min)
            sp = []
            for score, n in alloc.items():
                cls_df = sub[sub['TrueRawScores'] == score]
                sp.append(cls_df.sample(n=int(n), random_state=random_seed))
            parts.append(pd.concat(sp).reset_index(drop=True))
        else:
            parts.append(sub)
    return pd.concat(parts).reset_index(drop=True)


def load_all_cases():
    """Returns X (n_cases, 17), Y (raw score), names (scale), nesi (continuous)."""
    df = pd.read_csv(YAMA_DIR / "UniversalBadnessModelResult_Full.csv")
    df['NESI_Bin'] = pd.cut(df['NESI'], bins=NESI_BINS, labels=NESI_BIN_LABELS,
                            include_lowest=True)

    X_list, Y_list, names_list, nesi_list = [], [], [], []
    for i, label in enumerate(NESI_BIN_LABELS, start=1):
        df_bin = df[df.NESI_Bin == label].copy().reset_index(drop=True)
        bal = create_balanced_bin_df(df_bin)

        with open(YAMA_DIR / f"NESIbin{i}_data.pkl", "rb") as f:
            d = pickle.load(f)
        X = d[f"X_bin{i}"]
        Y = d[f"Y_raw_bin{i}"]
        names = d[f"dataset_names_bin{i}"]

        assert np.array_equal(bal['TrueRawScores'].to_numpy(), Y), \
            f"bin{i}: TrueRawScores mismatch — random_state reproduction broke"
        assert np.array_equal(bal['Dataset'].to_numpy(), names), \
            f"bin{i}: Dataset mismatch"

        X_list.append(X)
        Y_list.append(Y)
        names_list.append(names)
        nesi_list.append(bal['NESI'].to_numpy())

    return (np.vstack(X_list),
            np.concatenate(Y_list),
            np.concatenate(names_list),
            np.concatenate(nesi_list))


# ── Plotting helpers ────────────────────────────────────────────────────────
def activation_to_rgb(activation_col, base_color_hex):
    base_rgb = np.array(to_rgb(base_color_hex))
    white = np.ones(3)
    return (1 - activation_col[:, None]) * white + activation_col[:, None] * base_rgb


def equalize_and_order(X, Y_raw, names, nesi,
                       target_per_level=TARGET_PER_LEVEL,
                       seed=EQUALIZE_SEED):
    """Per-panel: cap each raw-score level at target_per_level, sort by
    raw score (in the panel's worse direction), secondary by NESI."""
    rng = np.random.default_rng(seed)

    sorted_idx, scale_boundaries, scale_groups = [], [], []
    level_boundaries, level_blocks = [], []
    cursor = 0

    for scale in SCALE_ORDER:
        mask = names == scale
        if mask.sum() == 0:
            continue
        scale_idx = np.where(mask)[0]
        scale_start = cursor

        unique_scores = np.unique(Y_raw[scale_idx])
        if SCALE_REVERSE[scale]:
            unique_scores = unique_scores[::-1]

        for k, score in enumerate(unique_scores):
            cls = scale_idx[Y_raw[scale_idx] == score]
            if len(cls) > target_per_level:
                cls = rng.choice(cls, target_per_level, replace=False)
            cls = cls[np.argsort(nesi[cls])]  # within-level: NESI ascending → worse
            block_start = cursor
            sorted_idx.extend(cls.tolist())
            cursor += len(cls)
            level_blocks.append((scale, score, block_start, cursor))
            if k < len(unique_scores) - 1:
                level_boundaries.append(cursor)

        scale_groups.append((scale, scale_start, cursor))
        scale_boundaries.append(cursor)

    return (np.array(sorted_idx), scale_boundaries, scale_groups,
            level_boundaries, level_blocks)


def build_rgb_image(X_sorted, nesi_sorted_scaled, names_sorted, color_map):
    """Returns (n_features+1, n_samples, 3) with NESI as row 0."""
    n_samples, n_morgoth = X_sorted.shape
    n_features = n_morgoth + 1
    rgb_img = np.ones((n_features, n_samples, 3))
    for j in range(n_samples):
        color_hex = color_map.get(names_sorted[j], '#333333')
        rgb_img[0, j, :] = activation_to_rgb(
            np.array([nesi_sorted_scaled[j]]), color_hex
        )[0]
        rgb_img[1:, j, :] = activation_to_rgb(X_sorted[j], color_hex)
    return rgb_img


def plot_combined(X, Y_raw, names, nesi, out_path=None, figsize=(11, 6.5),
                  bw=False, basename=None):
    apply_style()
    names = np.array(names)
    Y_raw = np.array(Y_raw)
    nesi = np.array(nesi, dtype=float)

    color_map = DATASET_COLORS_BW if bw else DATASET_COLORS
    divider_color = 'k'  # black dividers in both modes

    (sorted_idx, scale_boundaries, scale_groups,
     level_boundaries, level_blocks) = equalize_and_order(X, Y_raw, names, nesi)

    X_sorted = X[sorted_idx]
    names_sorted = names[sorted_idx]
    nesi_sorted = nesi[sorted_idx]

    nesi_scaled = np.clip((nesi_sorted + 3.0) / 6.0, 0, 1)

    rgb_img = build_rgb_image(X_sorted, nesi_scaled, names_sorted, color_map)
    n_features, n_samples, _ = rgb_img.shape

    feature_names = ['NESI'] + FEATURE_NAMES_BASE
    head_boundaries = [b + 1 for b in HEAD_BOUNDARIES_BASE]  # shifted by NESI row

    # Stdout summary
    print(f"\n  Combined figure — total columns = {n_samples}")
    for scale, _start, _end in scale_groups:
        blocks = [b for b in level_blocks if b[0] == scale]
        per = ", ".join(
            f"{int(s) if float(s).is_integer() else s}:{e - st}"
            for (_sc, s, st, e) in blocks
        )
        print(f"    {scale} ({_end - _start} total)  per-level → {per}")

    fig = plt.figure(figsize=figsize, dpi=210)
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.06, 1], hspace=0.05)
    ax_labels = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    # Per-panel labels only (no colorbar gradient)
    ax_labels.set_axis_off()
    for scale, start, end in scale_groups:
        color = color_map[scale]
        mid_f = 0.5 * (start + end) / n_samples
        ax_labels.text(
            mid_f, 0.4, f'{scale}\n(n={end - start})',
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color=to_rgb(color), transform=ax_labels.transAxes,
        )

    ax_main.imshow(
        rgb_img, aspect='auto',
        extent=[0, n_samples, n_features - 0.5, -0.5],
    )
    ax_main.set_xlim(0, n_samples)

    # Bold dashed dividers between clinical scales (full-height)
    for boundary in scale_boundaries[:-1]:
        ax_main.axvline(x=boundary, color=divider_color, linestyle='--',
                        linewidth=2.0)
    # Thin horizontal separators between every feature row
    for r in range(1, n_features):
        ax_main.axhline(y=r - 0.5, color=divider_color,
                        linewidth=0.6, alpha=0.95)
    # Bolder horizontal lines between MORGOTH heads
    for b in head_boundaries:
        ax_main.axhline(y=b + 0.5, color=divider_color,
                        linewidth=1.5, alpha=1.0)
    # Bolder line separating NESI row from MORGOTH rows
    ax_main.axhline(y=0.5, color=divider_color, linewidth=1.75, alpha=1.0)

    # ── Bottom x-axis: score labels at block centers, ticks at boundaries ──
    block_centers = [0.5 * (b[2] + b[3]) for b in level_blocks]
    block_labels = [
        f"{int(b[1])}" if float(b[1]).is_integer() else f"{b[1]:g}"
        for b in level_blocks
    ]
    ax_main.set_xticks(block_centers)
    ax_main.set_xticklabels(block_labels, fontsize=7)
    ax_main.tick_params(axis='x', which='major', length=0, pad=2)
    # Minor ticks at level boundaries (visible as short tick marks below axis)
    ax_main.set_xticks(level_boundaries, minor=True)
    ax_main.tick_params(axis='x', which='minor', length=4,
                        color=divider_color, width=0.7)

    ax_main.set_yticks(range(len(feature_names)))
    ax_main.set_yticklabels(feature_names, fontsize=7)
    ax_main.get_yticklabels()[0].set_fontweight('bold')

    ax_main.set_xlabel('True clinical score   (within each panel: left = best, '
                       'right = worst)',
                       fontsize=8)
    ax_main.set_ylabel('NESI  +  MORGOTH event-level EEG features', fontsize=8)

    # Intensity colorbar (white=0 -> black=1). Descriptive title lives in caption.
    gray_cmap = LinearSegmentedColormap.from_list('wb', ['white', 'black'])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=gray_cmap)
    cb = fig.colorbar(sm, ax=ax_main, fraction=0.015, pad=0.012)
    cb.set_label('Feature activation probability (dark = higher)\n'
                 'NESI row: white→black = low→high NESI',
                 fontsize=FS_LEGEND + 1)
    cb.ax.tick_params(labelsize=FS_LEGEND)

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved {out_path}")
    if basename is not None:
        save_fig(fig, basename)
    plt.close(fig)


def main():
    X, Y, names, nesi = load_all_cases()
    print(f"Loaded {len(Y)} cases total "
          f"(GCS:{(names=='GCS').sum()}, RASS:{(names=='RASS').sum()}, "
          f"CAMS:{(names=='CAMS').sum()}, ICANS:{(names=='ICANS').sum()})")
    # Official main-paper figure is the black-and-white version (Figure4).
    plot_combined(X, Y, names, nesi, bw=True, basename='Figure4')


if __name__ == '__main__':
    main()
