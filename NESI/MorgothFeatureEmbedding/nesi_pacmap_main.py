"""
Canonical PaCMAP pipeline + all paper figures.

Run order:
  1. Stratify per-patient median NESI into 12 bins, sample 400 patients/bin.
  2. Compute Slowing = 1 - NoSlowing, then logit + per-column z-score.
  3. PaCMAP with NESI weighted 2x as the 14th input feature (canonical
     embedding); a no-NESI PaCMAP is also run for comparison figures.
  4. Render canonical figures (1x2 overview, clinical atlas), supporting
     figures (no-NESI vs with-NESI side-by-side, weight sweep, smoothed
     IIIC map), and dump XY+rgb+labels to NESI_pacmap_iiic_data.npz for
     the HTML smoothness explorer.

Coloring rule for the IIIC categorical map:
    if Burst_vs_NoBurst >= BURST_THRESHOLD:
        color = black                             # burst suppression
    else:
        color = palette[argmax over
                        (Seizure, LPD, GPD, LRDA, GRDA, Other)]
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, to_rgb, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import pacmap

SCRIPT_DIR = Path(__file__).resolve().parent
IN_CSV = SCRIPT_DIR / "NESI_window_features.csv"
OUT_IIIC = SCRIPT_DIR / "NESI_pacmap_iiic_color.png"
OUT_IIIC_SMOOTH = SCRIPT_DIR / "NESI_pacmap_iiic_smooth.png"
OUT_IIIC_WITH_NESI = SCRIPT_DIR / "NESI_pacmap_iiic_with_nesi.png"
OUT_WEIGHT_SWEEP = SCRIPT_DIR / "NESI_pacmap_weight_sweep.png"
OUT_FIG1_OVERVIEW = SCRIPT_DIR / "NESI_pacmap_fig1_overview.png"
OUT_FIG2_ATLAS = SCRIPT_DIR / "NESI_pacmap_fig2_atlas.png"
OUT_NPZ = SCRIPT_DIR / "NESI_pacmap_iiic_data.npz"

# Scatter rendering: jitter copies + larger dots to fill in the map visually.
JITTER_COPIES = 4            # 1 original + 4 jittered = 5x points
JITTER_SIGMA_FRAC = 0.006    # jitter std as fraction of embedding std
DOT_SIZE = 4.0               # marker size for the scatter figures

# NESI-included PaCMAP: how much weight to give NESI relative to the
# other 13 (already-z-scored) features. weight=1 means NESI contributes
# the same variance as each other feature; weight=2 means 4x variance.
NESI_FEATURE_WEIGHT = 2.0

# Weight sweep for the composite figure (0 = no NESI in feature set).
NESI_WEIGHT_SWEEP = [0.0, 1.0, 2.0, 3.0]

# Shared sequential colormap for all probability/loading panels.
# Hex anchors per the agreed convention: light-gray -> pale-blue -> mid-blue
# -> dark-blue, so low values fade into the background and high values
# stand out.
PROB_CMAP_HEX = ['#eeeeee', '#bcdff1', '#2b8cbe', '#08306b']
PROB_CMAP = LinearSegmentedColormap.from_list('prob_seq', PROB_CMAP_HEX)

# Display titles (clean, human-readable) for each input feature.
TITLE_MAP = {
    'Awake': 'Awake',
    'N1': 'N1',
    'N2': 'N2',
    'Normal_vs_Abnormal': 'Normal vs Abnormal',
    'Burst_vs_NoBurst': 'Burst vs No Burst',
    'Slowing': 'Slowing',
    'FocalSlowing': 'Focal slowing',
    'GenSlowing': 'Generalized slowing',
    'IIIC_Seizure': 'Seizure',
    'IIIC_LPD': 'LPD',
    'IIIC_GPD': 'GPD',
    'IIIC_LRDA': 'LRDA',
    'IIIC_GRDA': 'GRDA',
}

# Direction labels for the two binary-class panels.
BINARY_SUBTITLE = {
    'Normal_vs_Abnormal': "P(Normal) — 0 = Abnormal, 1 = Normal",
    'Burst_vs_NoBurst':   "P(Burst) — 0 = NoBurst, 1 = Burst",
}

# Clinical groupings for the atlas figure.
FEATURE_GROUPS = [
    ('State',                       ['Awake', 'N1', 'N2']),
    ('Global abnormality',          ['Normal_vs_Abnormal',
                                     'Burst_vs_NoBurst', 'Slowing']),
    ('Focal abnormalities',         ['FocalSlowing', 'IIIC_LRDA',
                                     'IIIC_LPD']),
    ('Generalized abnormalities',   ['GenSlowing', 'IIIC_GRDA',
                                     'IIIC_GPD']),
    ('Seizure / ictal',             ['IIIC_Seizure']),
]

# ─── Clinical 4-category colors for Figure 2's panel borders ──────────────
SECTION_BORDER = {
    'physiologic':        '#7FB97F',   # green
    'slowing':            '#D9B872',   # tan / yellow
    'rhythmic_periodic':  '#E89B9B',   # pink
    'ictal_suppression':  '#D45959',   # red
}

# Each panel in the Figure 2 redesign -> which border color group.
PANEL_SECTION = {
    'Awake':              'physiologic',
    'Normal_vs_Abnormal': 'physiologic',
    'N1':                 'physiologic',
    'N2':                 'physiologic',
    'FocalSlowing':       'slowing',
    'GenSlowing':         'slowing',
    'IIIC_LRDA':          'rhythmic_periodic',
    'IIIC_GRDA':          'rhythmic_periodic',
    'IIIC_LPD':           'rhythmic_periodic',
    'IIIC_GPD':           'rhythmic_periodic',
    'IIIC_Seizure':       'ictal_suppression',
    'Burst_vs_NoBurst':   'ictal_suppression',
}

# Short panel titles for the redesigned Figure 2.
SHORT_TITLE = {
    'Awake':              'Awake',
    'Normal_vs_Abnormal': 'Normal',
    'N1':                 'N1',
    'N2':                 'N2',
    'FocalSlowing':       'Focal slowing',
    'GenSlowing':         'Gen. slowing',
    'IIIC_LRDA':          'LRDA',
    'IIIC_GRDA':          'GRDA',
    'IIIC_LPD':           'LPD',
    'IIIC_GPD':           'GPD',
    'IIIC_Seizure':       'Seizure',
    'Burst_vs_NoBurst':   'Suppression',
}

# Figure 2 ordering (top = benign → bottom = severe).
PHYSIOLOGIC_PANELS = ['Awake', 'Normal_vs_Abnormal', 'N1', 'N2']
# Each tuple is (focal_panel, generalized_panel).
ABNORMAL_ROWS = [
    ('FocalSlowing', 'GenSlowing'),
    ('IIIC_LRDA',    'IIIC_GRDA'),
    ('IIIC_LPD',     'IIIC_GPD'),
    ('IIIC_Seizure', 'Burst_vs_NoBurst'),
]
COORDS_CSV = SCRIPT_DIR / "NESI_pacmap_coords.csv"

TARGET_PATIENTS_PER_NESI_BIN = 400
NESI_BIN_EDGES = np.arange(-3.0, 3.01, 0.5)
SEED = 42
BURST_THRESHOLD = 0.5    # Burst_vs_NoBurst probability above this -> "burst"

# IIIC color palette matching the standard Westover-lab convention.
# These are the "max intensity" anchors for each category's gradient.
IIIC_PALETTE = {
    'Seizure': '#E13238',   # red
    'LPD':     '#F08C2A',   # orange
    'GPD':     '#F2D549',   # yellow
    'LRDA':    '#AABF45',   # green
    'GRDA':    '#7AC8E3',   # cyan / light blue
    'Other':   '#5C3A87',   # purple anchor for "Other" gradient (dark = sleep)
    'Burst':   '#000000',   # black
}
IIIC_LEGEND_ORDER = ['Burst', 'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

# Per-category probability column used to drive within-category gradient.
# For "Other", coloring is driven by p_Awake (light = awake, dark = sleep).
CAT_PROB_COL = {
    'Burst':   'Burst_vs_NoBurst',
    'Seizure': 'IIIC_Seizure',
    'LPD':     'IIIC_LPD',
    'GPD':     'IIIC_GPD',
    'LRDA':    'IIIC_LRDA',
    'GRDA':    'IIIC_GRDA',
    'Other':   'Awake',
}


def _light_tint(hex_color, mix=0.85):
    """Pale tint of a base color: mix*white + (1-mix)*base."""
    rgb = np.array(to_rgb(hex_color))
    return tuple(mix * 1.0 + (1.0 - mix) * rgb)


def make_category_cmap(cat):
    """LinearSegmentedColormap with a [light, dark/saturated] anchor pair
    appropriate for the category.

    For most categories: light tint -> full base color (so cmap(p) gives
    pale at p=0, full at p=1).
    For Burst (black base): light gray -> black.
    For Other (purple base): light lavender -> deep purple, BUT the
    underlying probability is p_Awake where high p_Awake = awake (light),
    so we *invert* by mapping p=0 (sleep) to dark and p=1 (awake) to light.
    """
    base = IIIC_PALETTE[cat]
    if cat == 'Other':
        return LinearSegmentedColormap.from_list(
            f'cat_{cat}',
            [base, _light_tint(base, mix=0.85)],
        )
    light = _light_tint(base, mix=0.80)
    return LinearSegmentedColormap.from_list(f'cat_{cat}', [light, base])


CAT_CMAPS = {cat: make_category_cmap(cat) for cat in IIIC_PALETTE}

# 13 input features after computing Slowing = 1 - NoSlowing
INPUT_FEATURES = [
    'Awake', 'N1', 'N2',
    'Normal_vs_Abnormal',
    'Burst_vs_NoBurst',
    'Slowing', 'FocalSlowing', 'GenSlowing',
    'IIIC_Seizure', 'IIIC_LPD', 'IIIC_GPD', 'IIIC_LRDA', 'IIIC_GRDA',
]


def stratify_patients(df, edges, target, seed):
    rng = np.random.default_rng(seed)
    pat_nesi = (df.groupby('PatientID')['NESI'].median()
                  .reset_index().rename(columns={'NESI': 'PatientMedianNESI'}))
    pat_nesi['NESI_bin'] = pd.cut(pat_nesi.PatientMedianNESI,
                                  bins=edges, include_lowest=True)
    keep_pids = []
    for b, grp in pat_nesi.groupby('NESI_bin', observed=True):
        n = min(target, len(grp))
        keep_pids.extend(grp.sample(n=n, random_state=rng.integers(1e9)).PatientID.tolist())
    return df[df.PatientID.isin(keep_pids)].copy()


def logit_z(X, eps=0.1):
    X = np.clip(X, 0, 1)
    L = np.log((X + eps) / (1.0 - X + eps))
    mu = L.mean(axis=0, keepdims=True)
    sd = L.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (L - mu) / sd


def jitter_xy_rgb(XY, rgb, n_copies=JITTER_COPIES,
                   sigma_frac=JITTER_SIGMA_FRAC, seed=0):
    """Augment by adding n_copies jittered duplicates of every point.

    Jitter scale = sigma_frac * embedding_std (per axis). Returns
    (XY_aug, rgb_aug) with shape (n*(n_copies+1), 2) and (...,3).
    """
    rng = np.random.default_rng(seed)
    sx = float(XY[:, 0].std()) * sigma_frac
    sy = float(XY[:, 1].std()) * sigma_frac
    parts_xy = [XY]
    parts_rgb = [rgb]
    for _ in range(n_copies):
        jit = np.column_stack([
            rng.normal(0, sx, size=len(XY)),
            rng.normal(0, sy, size=len(XY)),
        ])
        parts_xy.append(XY + jit)
        parts_rgb.append(rgb)
    return np.vstack(parts_xy), np.vstack(parts_rgb)


def assign_iiic_label(sub):
    """Vectorized: return array of category labels per row."""
    iiic_cols = ['IIIC_Seizure', 'IIIC_LPD', 'IIIC_GPD',
                 'IIIC_LRDA', 'IIIC_GRDA', 'IIIC_Other']
    iiic_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    M = sub[iiic_cols].to_numpy()
    argmax = np.argmax(M, axis=1)
    labels = np.array(iiic_names)[argmax]
    burst = sub['Burst_vs_NoBurst'].to_numpy() >= BURST_THRESHOLD
    labels[burst] = 'Burst'
    return labels


def build_iiic_rgb(sub, labels):
    """Per-point RGB driven by category + per-category probability gradient."""
    rgb = np.zeros((len(sub), 3))
    for cat in IIIC_PALETTE:
        m = labels == cat
        if not m.any():
            continue
        probs = sub.loc[m, CAT_PROB_COL[cat]].to_numpy()
        if len(probs) > 20:
            lo, hi = np.quantile(probs, [0.05, 0.95])
        else:
            lo, hi = float(probs.min()), float(probs.max())
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        norm_p = np.clip((probs - lo) / (hi - lo), 0, 1)
        rgb[m] = CAT_CMAPS[cat](norm_p)[:, :3]
    return rgb


def _draw_iiic_legend_on_ax(legend_ax, counts):
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    legend_ax.set_axis_off()
    n_cat = len(IIIC_LEGEND_ORDER)
    row_h = 1.0 / n_cat
    for i, cat in enumerate(IIIC_LEGEND_ORDER):
        n = int(counts.get(cat, 0))
        cmap = CAT_CMAPS[cat]
        gradient = np.linspace(1, 0, 64).reshape(-1, 1)
        y_lo = 1.0 - (i + 1) * row_h + 0.04 * row_h
        y_hi = 1.0 - i * row_h - 0.04 * row_h
        legend_ax.imshow(
            gradient, cmap=cmap, aspect='auto',
            extent=[0.0, 0.40, y_lo, y_hi],
            transform=legend_ax.transAxes, zorder=2,
        )
        suffix = "← sleep | wake →" if cat == 'Other' else "← low | high →"
        legend_ax.text(
            0.46, (y_lo + y_hi) / 2,
            f"{cat}  (n={n})\n  intensity: {suffix}",
            ha='left', va='center', fontsize=8,
            transform=legend_ax.transAxes,
        )
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)


def _render_two_panel(XY, rgb, labels, nesi_values, counts, *,
                      n_patients, n_windows, out_path, subtitle):
    """2-panel figure: (A) IIIC categorical colouring with within-category
    gradients; (B) same coordinates coloured by NESI."""
    XY_aug, rgb_aug = jitter_xy_rgb(XY, rgb)
    labels_aug = np.tile(labels, JITTER_COPIES + 1)
    nesi_aug = np.tile(np.asarray(nesi_values, dtype=float),
                       JITTER_COPIES + 1)
    n_aug = len(XY_aug)
    print(f"  scatter: {n_windows} -> {n_aug} points after jitter "
          f"(x{JITTER_COPIES + 1}), dot size {DOT_SIZE}")

    fig = plt.figure(figsize=(18, 8.5), dpi=180)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[7.2, 1.4, 7.2],
        wspace=0.08,
        left=0.03, right=0.97, top=0.86, bottom=0.05,
    )
    axA = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    axB = fig.add_subplot(gs[2])

    # ── A. IIIC categorical scatter ──
    plot_order = ['Other', 'GRDA', 'LRDA', 'GPD', 'LPD', 'Seizure', 'Burst']
    for cat in plot_order:
        m = labels_aug == cat
        if not m.any():
            continue
        axA.scatter(XY_aug[m, 0], XY_aug[m, 1],
                    c=rgb_aug[m], s=DOT_SIZE, marker='o',
                    edgecolors='none', alpha=0.55)
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_title("A. Coloured by dominant IIIC pattern",
                  fontsize=11, fontweight='bold')

    _draw_iiic_legend_on_ax(legend_ax, counts)

    # ── B. NESI-coloured scatter (same coordinates) ──
    sc = axB.scatter(XY_aug[:, 0], XY_aug[:, 1],
                     c=nesi_aug, cmap='coolwarm', vmin=-3, vmax=3,
                     s=DOT_SIZE, marker='o', edgecolors='none', alpha=0.55)
    axB.set_xticks([]); axB.set_yticks([])
    axB.set_title("B. Coloured by NESI", fontsize=11, fontweight='bold')
    cbar = plt.colorbar(sc, ax=axB, fraction=0.04, pad=0.02)
    cbar.set_label("NESI", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{subtitle}\n"
        f"n={n_windows} windows from {n_patients} patients "
        f"(jittered x{JITTER_COPIES + 1})",
        fontsize=12, fontweight='bold', y=0.97,
    )
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_smoothed_iiic_map(XY, rgb, labels, counts, n_patients,
                            n_pix=800, sigma_px=7, density_floor_quantile=0.5):
    """Render a smooth, continuous version of the IIIC color map.

    Approach: rasterize each point as a Gaussian blob whose weight is its RGB
    color, accumulate a separate density image (count of points), Gaussian-
    blur all four, then take per-pixel mean color = sum(rgb*K) / sum(K).
    Pixels whose smoothed density is below a threshold are rendered white
    (background) so the map fades out where there is no data.

    Density only governs visibility (which pixels are masked); the colour
    itself is the unweighted mean of the local point colours, so the
    intensity gradients you set within each category are preserved.
    """
    xmin, xmax = XY[:, 0].min(), XY[:, 0].max()
    ymin, ymax = XY[:, 1].min(), XY[:, 1].max()
    pad = 0.04 * max(xmax - xmin, ymax - ymin)
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

    # Map each point to a pixel.
    xpix = ((XY[:, 0] - xmin) / (xmax - xmin) * (n_pix - 1)).astype(int)
    ypix = ((XY[:, 1] - ymin) / (ymax - ymin) * (n_pix - 1)).astype(int)
    valid = (xpix >= 0) & (xpix < n_pix) & (ypix >= 0) & (ypix < n_pix)

    density_img = np.zeros((n_pix, n_pix), dtype=np.float64)
    color_imgs = np.zeros((n_pix, n_pix, 3), dtype=np.float64)
    # np.add.at handles repeated (yi, xi) correctly.
    np.add.at(density_img, (ypix[valid], xpix[valid]), 1.0)
    for c in range(3):
        np.add.at(color_imgs[..., c], (ypix[valid], xpix[valid]),
                  rgb[valid, c])

    density_smooth = gaussian_filter(density_img, sigma=sigma_px)
    color_smooth = np.stack(
        [gaussian_filter(color_imgs[..., c], sigma=sigma_px) for c in range(3)],
        axis=-1,
    )

    nonzero = density_smooth > 0
    mean_color = np.ones_like(color_smooth)
    for c in range(3):
        mean_color[..., c] = np.where(
            nonzero,
            color_smooth[..., c] / np.maximum(density_smooth, 1e-12),
            1.0,
        )

    # Mask pixels with very low local density -> background (white).
    nz_vals = density_smooth[nonzero]
    floor = float(np.quantile(nz_vals, density_floor_quantile)) * 0.02
    mask_empty = density_smooth < floor
    mean_color[mask_empty] = 1.0

    fig, ax = plt.subplots(figsize=(9.5, 8.5), dpi=180)
    ax.imshow(mean_color, origin='lower',
              extent=[xmin, xmax, ymin, ymax], interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        "PaCMAP — smoothed map (Gaussian-splatted mean colour)\n"
        "Category by hue, intensity by within-category probability\n"
        f"n={int(valid.sum())} windows from {n_patients} patients",
        fontsize=10.5, fontweight='bold',
    )

    # Same legend as Figure 2.
    legend_ax = fig.add_axes([0.79, 0.30, 0.06, 0.55])
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    legend_ax.set_axis_off()
    n_cat = len(IIIC_LEGEND_ORDER)
    row_h = 1.0 / n_cat
    for i, cat in enumerate(IIIC_LEGEND_ORDER):
        n = int(counts.get(cat, 0))
        cmap = CAT_CMAPS[cat]
        gradient = np.linspace(1, 0, 64).reshape(-1, 1)
        y_lo = 1.0 - (i + 1) * row_h + 0.04 * row_h
        y_hi = 1.0 - i * row_h - 0.04 * row_h
        legend_ax.imshow(
            gradient, cmap=cmap, aspect='auto',
            extent=[0.0, 0.20, y_lo, y_hi],
            transform=legend_ax.transAxes, zorder=2,
        )
        label_text = f"{cat}  (n={n})"
        if cat == 'Other':
            label_text += "\n  intensity: ← sleep | wake →"
        else:
            label_text += "\n  intensity: ← low | high →"
        legend_ax.text(
            0.24, (y_lo + y_hi) / 2, label_text,
            ha='left', va='center', fontsize=8,
            transform=legend_ax.transAxes,
        )
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)

    fig.subplots_adjust(left=0.04, right=0.78, top=0.90, bottom=0.04)
    fig.savefig(OUT_IIIC_SMOOTH, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT_IIIC_SMOOTH}")


def _scatter_nesi_on_ax(ax, XY_aug, nesi_aug):
    """Diverging blue-white-red NESI, extremes plotted on top."""
    order = np.argsort(np.abs(nesi_aug))
    sc = ax.scatter(
        XY_aug[order, 0], XY_aug[order, 1],
        c=nesi_aug[order], cmap='RdBu_r', vmin=-3, vmax=3,
        s=DOT_SIZE, marker='o', edgecolors='none', alpha=0.55,
    )
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def _scatter_iiic_rich_on_ax(ax, XY_aug, rgb_aug, labels_aug, counts):
    """Categorical IIIC coloring (per-point RGB already built upstream)."""
    plot_order = ['Other', 'GRDA', 'LRDA', 'GPD', 'LPD', 'Seizure', 'Burst']
    for cat in plot_order:
        m = labels_aug == cat
        if not m.any():
            continue
        ax.scatter(XY_aug[m, 0], XY_aug[m, 1],
                   c=rgb_aug[m], s=DOT_SIZE, marker='o',
                   edgecolors='none', alpha=0.55)
    ax.set_xticks([]); ax.set_yticks([])
    handles = [
        mpatches.Patch(color=IIIC_PALETTE[cat],
                       label=f"{cat} (n={int(counts.get(cat, 0))})")
        for cat in IIIC_LEGEND_ORDER
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              framealpha=0.9, title="IIIC category", title_fontsize=8)


def _scatter_feature_on_ax(ax, XY_aug, vals_aug):
    """Shared 0-1 sequential feature scatter; low values plotted first."""
    order = np.argsort(vals_aug)
    sc = ax.scatter(
        XY_aug[order, 0], XY_aug[order, 1],
        c=vals_aug[order], cmap=PROB_CMAP, vmin=0, vmax=1,
        s=DOT_SIZE, marker='o', edgecolors='none', alpha=0.55,
    )
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def _feature_title(fname):
    title = TITLE_MAP.get(fname, fname)
    if fname in BINARY_SUBTITLE:
        title = f"{title}\n{BINARY_SUBTITLE[fname]}"
    return title


def render_fig1_overview(XY, rgb, labels, sub, counts, *,
                          n_patients, n_windows, out_path):
    """Figure 1: 1x2 overview. A = NESI, B = Rich IIIC categorical."""
    XY_aug, rgb_aug = jitter_xy_rgb(XY, rgb)
    labels_aug = np.tile(labels, JITTER_COPIES + 1)
    nesi_aug = np.tile(sub['NESI'].to_numpy(), JITTER_COPIES + 1)

    fig = plt.figure(figsize=(14, 6.5), dpi=140)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 0.04, 1.0],
        wspace=0.10,
        left=0.03, right=0.97, top=0.90, bottom=0.04,
    )
    axA = fig.add_subplot(gs[0, 0])
    ax_nesi_cb = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[0, 2])

    sc_n = _scatter_nesi_on_ax(axA, XY_aug, nesi_aug)
    axA.set_title("A. NESI (continuous)", fontsize=12, fontweight='bold')
    cb = fig.colorbar(sc_n, cax=ax_nesi_cb)
    cb.set_label("NESI", fontsize=10); cb.ax.tick_params(labelsize=8)

    _scatter_iiic_rich_on_ax(axB, XY_aug, rgb_aug, labels_aug, counts)
    axB.set_title("B. Dominant IIIC category", fontsize=12, fontweight='bold')

    fig.suptitle(
        f"PaCMAP overview  (NESI weight = 2)  —  "
        f"n={n_windows} windows from {n_patients} patients",
        fontsize=12, fontweight='bold', y=0.97,
    )
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def _style_panel_border(ax, color, lw=2.5):
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(lw)


def render_fig2_atlas(XY, sub, *, n_patients, n_windows, out_path):
    """Figure 2 redesign: top row = 4 physiologic panels; below that, 4 rows
    of 2 panels each split into focal | generalized columns. Top-to-bottom
    progression = benign -> severe. Each panel has a category-coloured
    border. Shared 0-1 colorbar on the right; legend at the bottom.
    """
    XY_aug, _ = jitter_xy_rgb(XY, np.zeros((len(XY), 3)))
    feat_aug = {f: np.tile(sub[f].to_numpy(), JITTER_COPIES + 1)
                for f in INPUT_FEATURES}

    # Layout:
    #   cols = [arrow, p0, p1, p2, p3, colorbar]      6 columns
    #   rows = [physio, gap, focal-gen-header, ab1, ab2, ab3, ab4, legend]
    fig = plt.figure(figsize=(14.5, 22.0), dpi=130)
    gs = fig.add_gridspec(
        8, 6,
        width_ratios=[0.10, 1.0, 1.0, 1.0, 1.0, 0.05],
        height_ratios=[1.0, 0.05, 0.20, 1.0, 1.0, 1.0, 1.0, 0.40],
        hspace=0.22, wspace=0.10,
        left=0.04, right=0.95, top=0.965, bottom=0.02,
    )

    # ── Benign → severe arrow column on the left (spans all panel rows) ──
    ax_arrow = fig.add_subplot(gs[0:7, 0])
    ax_arrow.set_axis_off()
    ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
    ax_arrow.annotate(
        '', xy=(0.6, 0.04), xytext=(0.6, 0.96),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->', lw=1.8, color='0.4'),
    )
    ax_arrow.text(0.6, 1.00, 'benign', ha='center', va='bottom',
                  fontsize=11, color='0.3', fontweight='bold',
                  transform=ax_arrow.transAxes)
    ax_arrow.text(0.6, 0.00, 'severe', ha='center', va='top',
                  fontsize=11, color='0.3', fontweight='bold',
                  transform=ax_arrow.transAxes)

    # ── Row 0: 4 physiologic panels (Awake | Normal | N1 | N2) ──
    for j, fname in enumerate(PHYSIOLOGIC_PANELS):
        ax = fig.add_subplot(gs[0, 1 + j])
        _scatter_feature_on_ax(ax, XY_aug, feat_aug[fname])
        ax.set_title(SHORT_TITLE[fname], fontsize=11, fontweight='bold')
        _style_panel_border(ax, SECTION_BORDER[PANEL_SECTION[fname]])

    # ── Row 2: focal/generalized column headers ──
    ax_h_focal = fig.add_subplot(gs[2, 1:3])
    ax_h_focal.set_axis_off()
    ax_h_focal.text(0.5, 0.5, "Focal · lateralized",
                    fontsize=12, fontweight='bold', color='0.15',
                    ha='center', va='center', transform=ax_h_focal.transAxes)
    ax_h_gen = fig.add_subplot(gs[2, 3:5])
    ax_h_gen.set_axis_off()
    ax_h_gen.text(0.5, 0.5, "Generalized · diffuse",
                  fontsize=12, fontweight='bold', color='0.15',
                  ha='center', va='center', transform=ax_h_gen.transAxes)

    # ── Rows 3-6: focal vs generalized pairs, top->bottom = benign->severe ──
    for i, (focal, gen) in enumerate(ABNORMAL_ROWS):
        grid_row = 3 + i

        ax_f = fig.add_subplot(gs[grid_row, 1:3])
        _scatter_feature_on_ax(ax_f, XY_aug, feat_aug[focal])
        ax_f.set_title(SHORT_TITLE[focal], fontsize=11, fontweight='bold')
        _style_panel_border(ax_f, SECTION_BORDER[PANEL_SECTION[focal]])

        ax_g = fig.add_subplot(gs[grid_row, 3:5])
        _scatter_feature_on_ax(ax_g, XY_aug, feat_aug[gen])
        ax_g.set_title(SHORT_TITLE[gen], fontsize=11, fontweight='bold')
        _style_panel_border(ax_g, SECTION_BORDER[PANEL_SECTION[gen]])

    # ── Shared probability colorbar on the right (panel rows only) ──
    ax_cb = fig.add_subplot(gs[0:7, 5])
    sm = plt.cm.ScalarMappable(cmap=PROB_CMAP, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("Feature probability / loading  "
                  "(0 = absent / low, 1 = high)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    # ── Bottom legend row: category swatches + severity note ──
    ax_legend = fig.add_subplot(gs[7, 1:5])
    ax_legend.set_axis_off()
    legend_handles = [
        mpatches.Patch(facecolor='white', edgecolor=SECTION_BORDER['physiologic'],
                       linewidth=2.5, label='physiologic'),
        mpatches.Patch(facecolor='white', edgecolor=SECTION_BORDER['slowing'],
                       linewidth=2.5, label='slowing'),
        mpatches.Patch(facecolor='white',
                       edgecolor=SECTION_BORDER['rhythmic_periodic'],
                       linewidth=2.5, label='rhythmic / periodic'),
        mpatches.Patch(facecolor='white',
                       edgecolor=SECTION_BORDER['ictal_suppression'],
                       linewidth=2.5, label='ictal / suppression'),
    ]
    ax_legend.legend(
        handles=legend_handles, loc='upper center', ncol=4, fontsize=10,
        bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.6,
    )
    ax_legend.text(
        0.5, 0.10,
        "Top → bottom = increasing NESI severity",
        ha='center', va='center', fontsize=10, color='0.4',
        transform=ax_legend.transAxes, style='italic',
    )

    fig.suptitle(
        f"PaCMAP feature-loading atlas  (NESI weight = 2)  —  "
        f"n={n_windows} windows from {n_patients} patients",
        fontsize=12, fontweight='bold', y=0.985,
    )
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def run_pacmap_with_nesi_weight(Xn, nesi_z, weight, seed):
    """PaCMAP on either Xn (weight=0) or Xn concatenated with nesi_z*weight."""
    if weight == 0:
        X_in = Xn
    else:
        X_in = np.hstack([Xn, (nesi_z * weight).reshape(-1, 1)])
    return pacmap.PaCMAP(
        n_components=2, n_neighbors=10,
        MN_ratio=0.5, FP_ratio=2.0,
        random_state=seed,
    ).fit_transform(X_in, init='random')


def render_weight_sweep(weights, embeddings, rgb, labels, nesi_values,
                         counts, *, n_patients, n_windows, n_input_features,
                         out_path):
    """Composite figure: one row per weight, columns A (IIIC) and B (NESI).
    Shared IIIC legend + NESI colorbar on the right edge.
    """
    nesi_arr = np.asarray(nesi_values, dtype=float)
    n_rows = len(weights)

    fig = plt.figure(figsize=(14.5, 6.0 * n_rows), dpi=130)
    gs = fig.add_gridspec(
        n_rows, 4,
        width_ratios=[6.0, 6.0, 1.6, 0.25],
        hspace=0.10, wspace=0.05,
        left=0.06, right=0.965, top=0.955, bottom=0.025,
    )

    last_sc = None
    plot_order = ['Other', 'GRDA', 'LRDA', 'GPD', 'LPD', 'Seizure', 'Burst']

    for i, (weight, XY) in enumerate(zip(weights, embeddings)):
        XY_aug, rgb_aug = jitter_xy_rgb(XY, rgb, seed=int(weight * 100))
        labels_aug = np.tile(labels, JITTER_COPIES + 1)
        nesi_aug = np.tile(nesi_arr, JITTER_COPIES + 1)

        axA = fig.add_subplot(gs[i, 0])
        axB = fig.add_subplot(gs[i, 1])

        for cat in plot_order:
            m = labels_aug == cat
            if not m.any():
                continue
            axA.scatter(XY_aug[m, 0], XY_aug[m, 1],
                        c=rgb_aug[m], s=DOT_SIZE, marker='o',
                        edgecolors='none', alpha=0.55)
        axA.set_xticks([]); axA.set_yticks([])

        last_sc = axB.scatter(
            XY_aug[:, 0], XY_aug[:, 1],
            c=nesi_aug, cmap='coolwarm', vmin=-3, vmax=3,
            s=DOT_SIZE, marker='o', edgecolors='none', alpha=0.55,
        )
        axB.set_xticks([]); axB.set_yticks([])

        # Row label and per-row titles
        share = weight ** 2 / (n_input_features + weight ** 2)
        row_label = (f"NESI weight = {weight:g}\n"
                     f"({share:.0%} of feature variance)")
        if weight == 0:
            row_label = "NESI weight = 0\n(not in feature set)"
        axA.set_ylabel(row_label, fontsize=11, fontweight='bold',
                       labelpad=10)
        if i == 0:
            axA.set_title("A. Coloured by dominant IIIC pattern",
                          fontsize=12, fontweight='bold')
            axB.set_title("B. Coloured by NESI",
                          fontsize=12, fontweight='bold')

    # Shared IIIC legend (spans all rows)
    legend_ax = fig.add_subplot(gs[:, 2])
    _draw_iiic_legend_on_ax(legend_ax, counts)

    # Shared NESI colorbar (spans all rows)
    cbar_ax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(last_sc, cax=cbar_ax)
    cbar.set_label("NESI", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"PaCMAP layout vs NESI weighting in the feature set\n"
        f"Each row: same PaCMAP, two colorings (A = IIIC, B = NESI). "
        f"n={n_windows} windows from {n_patients} patients "
        f"(jittered x{JITTER_COPIES + 1})",
        fontsize=12, fontweight='bold', y=0.985,
    )
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} window-vectors, {df.PatientID.nunique()} patients")

    # Stratify
    sub = stratify_patients(df, NESI_BIN_EDGES,
                            TARGET_PATIENTS_PER_NESI_BIN, SEED)
    print(f"After stratify: {len(sub)} windows from {sub.PatientID.nunique()} patients")
    print("Per scale (vectors):", sub.Dataset.value_counts().to_dict())

    # Compute Slowing = 1 - NoSlowing (BEFORE logit+z)
    sub = sub.copy()
    sub['Slowing'] = 1.0 - sub['NoSlowing']

    X = sub[INPUT_FEATURES].to_numpy()
    Xn = logit_z(X)
    print(f"Normalized matrix: {Xn.shape}; per-col std={np.round(Xn.std(0),3)}")

    print("Running PaCMAP...")
    XY = pacmap.PaCMAP(n_components=2, n_neighbors=10,
                       MN_ratio=0.5, FP_ratio=2.0,
                       random_state=SEED).fit_transform(Xn, init='random')

    # Save coords
    coords = sub[['PatientID', 'Dataset', 'MorgothOutputFilename',
                  'TrueRawScores', 'NESI']].copy()
    coords['pacmap_x'] = XY[:, 0]
    coords['pacmap_y'] = XY[:, 1]
    coords.to_csv(COORDS_CSV, index=False)
    print(f"Wrote coords -> {COORDS_CSV}")

    # ── Figure 2: IIIC categorical coloring with within-category gradient ──
    labels = assign_iiic_label(sub)
    counts = pd.Series(labels).value_counts()
    print("\nIIIC color category counts:")
    print(counts.to_string())

    rgb = build_iiic_rgb(sub, labels)
    _render_two_panel(
        XY, rgb, labels, sub.NESI.to_numpy(), counts,
        n_patients=sub.PatientID.nunique(),
        n_windows=len(sub),
        out_path=OUT_IIIC,
        subtitle=("PaCMAP from 13 features (logit + z-scored); "
                  "NESI not included as feature"),
    )

    # Save numpy bundle for the HTML explorer and other downstream tools.
    np.savez_compressed(
        OUT_NPZ,
        XY=XY, rgb=rgb, labels=labels,
        category_order=np.array(IIIC_LEGEND_ORDER),
    )
    print(f"Saved {OUT_NPZ}")

    # ── Figure 3: Gaussian-splat smoothed version ──
    plot_smoothed_iiic_map(XY, rgb, labels, counts, sub.PatientID.nunique())

    # ── Figure 4: NESI included as an extra input feature (weight 2x) ──
    print(f"\nRunning PaCMAP with NESI as 14th feature "
          f"(scale={NESI_FEATURE_WEIGHT:g}x)...")
    nesi_vals = sub['NESI'].to_numpy().astype(float)
    nesi_z = (nesi_vals - nesi_vals.mean()) / (nesi_vals.std() or 1.0)
    Xn_with_nesi = np.hstack([Xn, (nesi_z * NESI_FEATURE_WEIGHT).reshape(-1, 1)])
    XY_with_nesi = pacmap.PaCMAP(
        n_components=2, n_neighbors=10,
        MN_ratio=0.5, FP_ratio=2.0,
        random_state=SEED,
    ).fit_transform(Xn_with_nesi, init='random')
    print(f"  shape={Xn_with_nesi.shape}; "
          f"NESI variance contribution ~ "
          f"{NESI_FEATURE_WEIGHT**2 / (Xn.shape[1] + NESI_FEATURE_WEIGHT**2):.1%}")

    _render_two_panel(
        XY_with_nesi, rgb, labels, sub.NESI.to_numpy(), counts,
        n_patients=sub.PatientID.nunique(),
        n_windows=len(sub),
        out_path=OUT_IIIC_WITH_NESI,
        subtitle=(f"PaCMAP from 13 features + NESI "
                  f"(NESI weight = {NESI_FEATURE_WEIGHT:g}x)"),
    )

    # ── Figure 5: composite weight sweep ──
    print(f"\nWeight sweep: {NESI_WEIGHT_SWEEP}")
    cached = {0.0: XY, NESI_FEATURE_WEIGHT: XY_with_nesi}
    embeddings = []
    for w in NESI_WEIGHT_SWEEP:
        if w in cached:
            embeddings.append(cached[w])
            print(f"  weight={w:g}: reusing cached embedding")
            continue
        print(f"  weight={w:g}: running PaCMAP...")
        XY_w = run_pacmap_with_nesi_weight(Xn, nesi_z, w, SEED)
        embeddings.append(XY_w)

    render_weight_sweep(
        NESI_WEIGHT_SWEEP, embeddings, rgb, labels,
        sub.NESI.to_numpy(), counts,
        n_patients=sub.PatientID.nunique(),
        n_windows=len(sub),
        n_input_features=Xn.shape[1],
        out_path=OUT_WEIGHT_SWEEP,
    )

    # ── Figure 6 (overview) and Figure 7 (atlas) on the weight-2 layout ──
    render_fig1_overview(
        XY_with_nesi, rgb, labels, sub, counts,
        n_patients=sub.PatientID.nunique(),
        n_windows=len(sub),
        out_path=OUT_FIG1_OVERVIEW,
    )
    render_fig2_atlas(
        XY_with_nesi, sub,
        n_patients=sub.PatientID.nunique(),
        n_windows=len(sub),
        out_path=OUT_FIG2_ATLAS,
    )


if __name__ == '__main__':
    main()
