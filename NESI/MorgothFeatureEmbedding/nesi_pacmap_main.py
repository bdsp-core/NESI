"""
Canonical PaCMAP pipeline for the NESI paper.

Run order:
  1. Stratify per-patient median NESI into 12 bins, sample 400 patients/bin.
  2. Logit + per-column z-score the 12 input features.
  3. No-NESI PaCMAP -> coords CSV + NPZ bundle (for the HTML explorer).
  4. PaCMAP with NESI weighted 2x as the 13th input feature -> canonical
     embedding for Figure 1 (overview) and Figure 2 (atlas).

Coloring rule for the IIIC categorical map used in Figure 1 panel B:
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
from matplotlib.colors import to_rgb, LinearSegmentedColormap
import pacmap

SCRIPT_DIR = Path(__file__).resolve().parent
IN_CSV = SCRIPT_DIR / "NESI_window_features.csv"
OUT_FIG1_OVERVIEW = SCRIPT_DIR / "NESI_pacmap_fig1_overview.png"
OUT_FIG2_ATLAS = SCRIPT_DIR / "NESI_pacmap_fig2_atlas.png"
OUT_NPZ = SCRIPT_DIR / "NESI_pacmap_iiic_data.npz"

# Scatter rendering: jitter copies + larger dots to fill in the map visually.
JITTER_COPIES = 4            # 1 original + 4 jittered = 5x points
JITTER_SIGMA_FRAC = 0.006    # jitter std as fraction of embedding std
DOT_SIZE = 4.0               # marker size for the scatter figures

# NESI-included PaCMAP: how much weight to give NESI relative to the
# other 12 (already-z-scored) features. weight=2 means NESI contributes
# 4x the variance of each individual feature.
NESI_FEATURE_WEIGHT = 2.0

# Shared sequential colormap for all probability/loading panels.
# Light-gray -> black, so low values fade into the background and high
# values stand out.
PROB_CMAP_HEX = ['#eeeeee', '#b0b0b0', '#505050', '#000000']
PROB_CMAP = LinearSegmentedColormap.from_list('prob_seq', PROB_CMAP_HEX)

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

# Figure 2 panel layout: 3 rows of 4 feature panels below the big NESI map.
FIG2_ROWS = [
    ['Normal_vs_Abnormal', 'Awake',      'N1',           'N2'],
    ['IIIC_Seizure',       'IIIC_LPD',   'IIIC_GPD',     'IIIC_LRDA'],
    ['IIIC_GRDA',          'GenSlowing', 'FocalSlowing', 'Burst_vs_NoBurst'],
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

# 12 input features (global "Slowing" intentionally omitted; FocalSlowing
# and GenSlowing already cover the spatial decomposition).
INPUT_FEATURES = [
    'Awake', 'N1', 'N2',
    'Normal_vs_Abnormal',
    'Burst_vs_NoBurst',
    'FocalSlowing', 'GenSlowing',
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


def _lock_panel_aspect(ax):
    # Keep every PaCMAP panel in fig1/fig2 at the same data aspect,
    # regardless of how wide its grid slot is.
    ax.set_aspect('equal', adjustable='box')


def _scatter_nesi_on_ax(ax, XY_aug, nesi_aug):
    """Diverging blue-white-red NESI, extremes plotted on top."""
    order = np.argsort(np.abs(nesi_aug))
    sc = ax.scatter(
        XY_aug[order, 0], XY_aug[order, 1],
        c=nesi_aug[order], cmap='RdBu_r', vmin=-3, vmax=3,
        s=DOT_SIZE, marker='o', edgecolors='none', alpha=0.55,
    )
    ax.set_xticks([]); ax.set_yticks([])
    _lock_panel_aspect(ax)
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
    _lock_panel_aspect(ax)
    handles = [
        mpatches.Patch(color=IIIC_PALETTE[cat],
                       label=f"{cat} (n={int(counts.get(cat, 0))})")
        for cat in IIIC_LEGEND_ORDER
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              frameon=False, title="IIIC category", title_fontsize=8)


def _scatter_feature_on_ax(ax, XY_aug, vals_aug):
    """Shared 0-1 sequential feature scatter; low values plotted first."""
    order = np.argsort(vals_aug)
    sc = ax.scatter(
        XY_aug[order, 0], XY_aug[order, 1],
        c=vals_aug[order], cmap=PROB_CMAP, vmin=0, vmax=1,
        s=DOT_SIZE, marker='o', edgecolors='none', alpha=0.55,
    )
    ax.set_xticks([]); ax.set_yticks([])
    _lock_panel_aspect(ax)
    return sc


def render_fig1_overview(XY, rgb, labels, sub, counts, *,
                          n_patients, n_windows, out_path):
    """Figure 1: 1x2 overview. A = NESI, B = Rich IIIC categorical."""
    XY_aug, rgb_aug = jitter_xy_rgb(XY, rgb)
    labels_aug = np.tile(labels, JITTER_COPIES + 1)
    nesi_aug = np.tile(sub['NESI'].to_numpy(), JITTER_COPIES + 1)

    fig = plt.figure(figsize=(14, 6.5), dpi=140)
    gs = fig.add_gridspec(
        1, 2,
        wspace=0.06,
        left=0.03, right=0.97, top=0.94, bottom=0.04,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    _scatter_nesi_on_ax(axA, XY_aug, nesi_aug)
    axA.set_title("A. NESI (continuous)", fontsize=12, fontweight='bold')
    for spine in axA.spines.values():
        spine.set_visible(False)

    _scatter_iiic_rich_on_ax(axB, XY_aug, rgb_aug, labels_aug, counts)
    axB.set_title("B. Dominant IIIC category", fontsize=12, fontweight='bold')
    for spine in axB.spines.values():
        spine.set_visible(False)

    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def render_fig2_atlas(XY, sub, *, n_patients, n_windows, out_path):
    """Figure 2: large PaCMAP coloured by NESI on top, then 3 rows of 4
    feature panels (all using the shared light-gray -> dark-blue PROB_CMAP).
    """
    XY_aug, _ = jitter_xy_rgb(XY, np.zeros((len(XY), 3)))
    feat_aug = {f: np.tile(sub[f].to_numpy(), JITTER_COPIES + 1)
                for f in INPUT_FEATURES}
    nesi_aug = np.tile(sub['NESI'].to_numpy().astype(float),
                       JITTER_COPIES + 1)

    def _strip_spines(ax):
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Layout:
    #   cols = 4 panel columns (no colorbar column)
    #   rows = [big NESI, gap, row1, row2, row3]
    #   NESI slot height ≈ 4 small-panel rows so its aspect-locked data
    #   box fills (almost) the full width of the 4-column slot.
    fig = plt.figure(figsize=(14.0, 24.0), dpi=130)
    gs = fig.add_gridspec(
        5, 4,
        height_ratios=[4.0, 0.20, 1.0, 1.0, 1.0],
        hspace=0.18, wspace=0.10,
        left=0.02, right=0.98, top=0.98, bottom=0.02,
    )

    # ── Large NESI panel (same gray -> black cmap, vmin=-3, vmax=3) ──
    ax_nesi = fig.add_subplot(gs[0, :])
    order = np.argsort(np.abs(nesi_aug))
    ax_nesi.scatter(
        XY_aug[order, 0], XY_aug[order, 1],
        c=nesi_aug[order], cmap=PROB_CMAP, vmin=-3, vmax=3,
        s=DOT_SIZE, marker='o', edgecolors='none', alpha=0.55,
    )
    ax_nesi.set_xticks([]); ax_nesi.set_yticks([])
    ax_nesi.set_title("NESI", fontsize=13, fontweight='bold')
    _lock_panel_aspect(ax_nesi)
    _strip_spines(ax_nesi)

    # ── 3 feature rows × 4 panels each ──
    # "Normal_vs_Abnormal" stores P(Abnormal); invert for the "Normal" panel.
    for row_idx, row in enumerate(FIG2_ROWS):
        grid_row = 2 + row_idx
        for col_idx, fname in enumerate(row):
            ax = fig.add_subplot(gs[grid_row, col_idx])
            vals = feat_aug[fname]
            if fname == 'Normal_vs_Abnormal':
                vals = 1.0 - vals
            _scatter_feature_on_ax(ax, XY_aug, vals)
            ax.set_title(SHORT_TITLE[fname], fontsize=11, fontweight='bold')
            _strip_spines(ax)

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

    sub = sub.copy()
    X = sub[INPUT_FEATURES].to_numpy()
    Xn = logit_z(X)
    print(f"Normalized matrix: {Xn.shape}; per-col std={np.round(Xn.std(0),3)}")

    print("Running PaCMAP...")
    XY = pacmap.PaCMAP(n_components=2, n_neighbors=10,
                       MN_ratio=0.5, FP_ratio=2.0,
                       random_state=SEED).fit_transform(Xn, init='random')

    # coords.csv also carries per-window feature values + IIIC_Other so the
    # interactive explorer can recolor without re-joining the input CSV.
    explorer_feat_cols = list(INPUT_FEATURES) + ['IIIC_Other']
    coords = sub[['PatientID', 'Dataset', 'MorgothOutputFilename',
                  'TrueRawScores', 'NESI'] + explorer_feat_cols].copy()
    coords['pacmap_x'] = XY[:, 0]
    coords['pacmap_y'] = XY[:, 1]
    # XY_with_nesi columns are added below after the canonical embedding runs.

    # IIIC categorical labels + per-point RGB for fig1 panel B and the
    # NPZ bundle consumed by the HTML smoothness explorer.
    labels = assign_iiic_label(sub)
    counts = pd.Series(labels).value_counts()
    print("\nIIIC color category counts:")
    print(counts.to_string())

    rgb = build_iiic_rgb(sub, labels)

    np.savez_compressed(
        OUT_NPZ,
        XY=XY, rgb=rgb, labels=labels,
        category_order=np.array(IIIC_LEGEND_ORDER),
    )
    print(f"Saved {OUT_NPZ}")

    # Canonical embedding: NESI as 13th feature (z-scored, scaled 2x).
    print(f"\nRunning PaCMAP with NESI as 13th feature "
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

    coords['pacmap_x_nesi'] = XY_with_nesi[:, 0]
    coords['pacmap_y_nesi'] = XY_with_nesi[:, 1]
    coords.to_csv(COORDS_CSV, index=False)
    print(f"Wrote coords (both layouts) -> {COORDS_CSV}")

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
