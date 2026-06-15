"""
Fit monotonic curves mapping NESI -> each discrete clinical scale (GCS, RASS, CAMS, ICANS).

Approach: proportional-odds ordinal logistic regression (cumulative-link model).
For each scale we model P(Y <= k | NESI) = sigmoid(theta_k - beta * NESI).
The expected (interpreted) score is then E[Y | NESI] = sum_k k * P(Y = k | NESI).

We deliberately fit on TransformedRawScores so every scale points the same way
(higher = worse, same direction as NESI). GCS and RASS are already reversed in
that column, CAMS and ICANS are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm
from scipy.optimize import minimize
from scipy.special import expit  # numerically stable sigmoid
from scipy.stats import spearmanr


from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
# Input metadata lives in the sibling MORGOTHActivationViz_GroupedbyNESI folder
CSV_PATH = str(SCRIPT_DIR.parent / "MORGOTHActivationViz_GroupedbyNESI"
                / "UniversalBadnessModelResult_Full.csv")
FIT_ON_SPLIT = "Train"    # fit ordinal-logistic curves on this split only
OUT_DIR = str(SCRIPT_DIR)

# Blue -> red colormap for NESI; red = worse (higher NESI).
NESI_CMAP = plt.cm.RdBu_r
NESI_VMIN, NESI_VMAX = -3.0, 3.0
NESI_NORM = Normalize(vmin=NESI_VMIN, vmax=NESI_VMAX)

# shared main-paper figure style (MainPaperFigures/Codes/nesi_fig_style.py)
import sys as _sys
_sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "MainPaperFigures" / "Codes"))
try:
    from nesi_fig_style import (apply_style as _apply_style, save_fig as _save_fig,
                                NESI_CBAR_LABEL as _NESI_CBAR_LABEL)
except Exception:
    _apply_style = lambda: None
    _save_fig = None
    _NESI_CBAR_LABEL = "NESI (z-scored; red = more impaired)"


# ---------------------------------------------------------------------------
# Ordinal logistic (proportional-odds) fit
# ---------------------------------------------------------------------------

def fit_ordinal_logistic(nesi: np.ndarray, y: np.ndarray):
    """Fit P(Y<=k | x) = sigmoid(theta_k - beta * x) with ordered thresholds.

    y must be integer-coded 0..K-1 (consecutive).
    Returns (thresholds theta of length K-1, slope beta, log-likelihood).
    """
    y = y.astype(int)
    K = int(y.max()) + 1
    n_thresh = K - 1

    # Parameterize thresholds as theta_0, delta_1, ... delta_{K-2} with delta>0
    # so the thresholds stay ordered.
    def unpack(params):
        theta0 = params[0]
        deltas = np.exp(params[1:n_thresh])  # positive gaps
        theta = np.concatenate([[theta0], theta0 + np.cumsum(deltas)])
        beta = params[n_thresh]
        return theta, beta

    def neg_log_lik(params):
        theta, beta = unpack(params)
        eta = theta[None, :] - beta * nesi[:, None]          # (n, K-1)
        cum = expit(eta)                                      # P(Y<=k)
        cum = np.concatenate(
            [np.zeros((len(nesi), 1)), cum, np.ones((len(nesi), 1))], axis=1
        )
        probs = np.diff(cum, axis=1)                          # P(Y=k)
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.log(probs[np.arange(len(y)), y]).sum()

    # Init: evenly spaced thresholds, small negative beta (higher x -> higher y).
    init = np.zeros(n_thresh + 1)
    init[0] = -1.0
    init[1:n_thresh] = np.log(0.5)
    init[n_thresh] = -1.0

    res = minimize(neg_log_lik, init, method="L-BFGS-B")
    theta, beta = unpack(res.x)
    return theta, beta, -res.fun


def predict_expected(nesi_grid: np.ndarray, theta: np.ndarray, beta: float, K: int):
    """Expected score E[Y | NESI] on a grid of NESI values."""
    eta = theta[None, :] - beta * nesi_grid[:, None]
    cum = expit(eta)
    cum = np.concatenate(
        [np.zeros((len(nesi_grid), 1)), cum, np.ones((len(nesi_grid), 1))], axis=1
    )
    probs = np.diff(cum, axis=1)
    ks = np.arange(K)
    return probs @ ks, probs


def predict_mean_var(nesi_grid: np.ndarray, theta, beta, K):
    """Return E[Y|NESI], Var[Y|NESI], and the full P(Y=k|NESI) matrix."""
    E, probs = predict_expected(nesi_grid, theta, beta, K)
    ks = np.arange(K)
    E2 = probs @ (ks ** 2)
    Var = np.maximum(E2 - E ** 2, 0.0)
    return E, Var, probs


def joint_density(nesi_empirical, fx, fy):
    """Model-implied P(X_t=j, Y_t=i) marginalized over empirical NESI.

    Returns an array shape (K_y, K_x) indexed in TRANSFORMED space.
    """
    _, _, px = predict_mean_var(nesi_empirical, fx["theta"], fx["beta"], fx["K"])
    _, _, py = predict_mean_var(nesi_empirical, fy["theta"], fy["beta"], fy["K"])
    return (py[:, :, None] * px[:, None, :]).mean(axis=0)


def expected_nesi_given_xy(nesi_empirical, fx, fy):
    """Posterior E[NESI | X_t=j, Y_t=i] using both ordinal models.

    E[NESI|X,Y] = sum_i NESI_i * P(X|NESI_i) P(Y|NESI_i) / sum_i P(X|NESI_i) P(Y|NESI_i)
    Returns an array shape (K_y, K_x) indexed in TRANSFORMED space.
    """
    _, _, px = predict_mean_var(nesi_empirical, fx["theta"], fx["beta"], fx["K"])
    _, _, py = predict_mean_var(nesi_empirical, fy["theta"], fy["beta"], fy["K"])
    w = py[:, :, None] * px[:, None, :]                       # (n, K_y, K_x)
    numer = (nesi_empirical[:, None, None] * w).sum(axis=0)   # (K_y, K_x)
    denom = w.sum(axis=0)
    return numer / np.where(denom > 0, denom, np.nan)


# ---------------------------------------------------------------------------
# Gaussian-approximation "worm" model:
#   Y | NESI ~ N(E[Y|NESI], max(Var[Y|NESI], sd_floor^2))   truncated at scale.
#   Joint (X, Y) | NESI assumed conditionally independent.
#   Marginalize over a UNIFORM NESI prior on [-3, 3] for the worm envelope.
# ---------------------------------------------------------------------------

SD_FLOOR = 0.35     # minimum SD so the worm never collapses to zero width


def _worm_gaussian_stats(nesi, fit, scale):
    """Return (mean, sd) in RAW score units for each NESI."""
    E_t, V_t, _ = predict_mean_var(nesi, fit["theta"], fit["beta"], fit["K"])
    mean_raw = transformed_to_raw(scale, E_t)
    # linear transform raw = ±t+c so Var unchanged
    sd_raw = np.sqrt(np.maximum(V_t, SD_FLOOR ** 2))
    return mean_raw, sd_raw


def worm_joint_density(fx, fy, sx, sy, x_grid, y_grid,
                       nesi_range=(-3.0, 3.0), n_nesi=200):
    """Marginal p(x, y) and posterior E[NESI | x, y] under uniform NESI prior.

    Returns (density, E_nesi), each of shape (len(y_grid), len(x_grid)).
    """
    nesi = np.linspace(nesi_range[0], nesi_range[1], n_nesi)
    mu_x, sd_x = _worm_gaussian_stats(nesi, fx, sx)
    mu_y, sd_y = _worm_gaussian_stats(nesi, fy, sy)

    # Gaussian pdfs on the grid for each NESI.
    dx = (x_grid[:, None] - mu_x[None, :]) / sd_x[None, :]
    dy = (y_grid[:, None] - mu_y[None, :]) / sd_y[None, :]
    px = np.exp(-0.5 * dx ** 2) / (sd_x[None, :] * np.sqrt(2 * np.pi))  # (nx, n_nesi)
    py = np.exp(-0.5 * dy ** 2) / (sd_y[None, :] * np.sqrt(2 * np.pi))  # (ny, n_nesi)

    density = (py @ px.T) / n_nesi                          # (ny, nx)
    numer = (py * nesi[None, :]) @ px.T                     # (ny, nx)
    E_nesi = numer / np.where(density > 0, density * n_nesi, np.nan)
    return density, E_nesi


def worm_diagonal_density(fit, scale, nesi_grid, y_grid, sd_floor=SD_FLOOR):
    """Conditional p(y | NESI) as a 2D grid (y × NESI), plus raw mean curve."""
    mu_y, sd_y = _worm_gaussian_stats(nesi_grid, fit, scale)
    dy = (y_grid[:, None] - mu_y[None, :]) / sd_y[None, :]
    cond = np.exp(-0.5 * dy ** 2) / (sd_y[None, :] * np.sqrt(2 * np.pi))
    # Treat NESI prior as uniform — use joint p(y, NESI) = cond * 1/n so the
    # centile thresholds are comparable to the 2D off-diagonal case.
    joint = cond / len(nesi_grid)
    return joint, mu_y


def density_levels_for_centiles(density, centiles=(0.5, 0.8, 0.95)):
    """Density thresholds that bracket the highest-density regions containing
    each cumulative fraction of mass."""
    flat = np.sort(density.ravel())[::-1]
    total = flat.sum()
    if total <= 0:
        return sorted(centiles)
    cum = np.cumsum(flat) / total
    thresholds = []
    for c in centiles:
        idx = int(np.searchsorted(cum, c))
        idx = min(idx, len(flat) - 1)
        thresholds.append(float(flat[idx]))
    return sorted(thresholds)


# ---------------------------------------------------------------------------
# Mapping transformed score <-> clinical raw score
# ---------------------------------------------------------------------------
# The CSV's TransformedRawScores column normalizes direction so higher = worse.
# We recover the clinically familiar score from the transformed index.

def transformed_to_raw(scale: str, t: np.ndarray) -> np.ndarray:
    if scale == "GCS":
        return 15 - t        # 0->15 (best), 12->3 (worst)
    if scale == "RASS":
        return -t            # 0->0, 5->-5
    if scale == "CAMS":
        return t             # unchanged
    if scale == "ICANS":
        return t             # unchanged
    raise ValueError(scale)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _apply_style()
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    df_all = pd.read_csv(CSV_PATH)
    df_all = df_all.dropna(subset=["NESI", "TransformedRawScores", "Dataset"])

    print("=" * 70)
    print(f"Total rows: {len(df_all):,}")
    print(f"NESI (all splits): min={df_all.NESI.min():.3f}  "
          f"max={df_all.NESI.max():.3f}  mean={df_all.NESI.mean():.3f}  "
          f"std={df_all.NESI.std():.3f}")
    if "WhichSet" in df_all.columns:
        print("\nRows by Dataset × WhichSet:")
        ct = pd.crosstab(df_all.Dataset, df_all.WhichSet, margins=True)
        print(ct.to_string())
        df = df_all[df_all["WhichSet"] == FIT_ON_SPLIT].copy()
        print(f"\nFitting curves on split = {FIT_ON_SPLIT!r}  (n={len(df):,})")
    else:
        df = df_all
        print("\n(no WhichSet column; fitting on all rows)")
    print("=" * 70)

    scales = ["GCS", "RASS", "CAMS", "ICANS"]
    fits: dict[str, dict] = {}

    for scale in scales:
        sub = df[df.Dataset == scale]
        y = sub.TransformedRawScores.to_numpy().astype(int)
        nesi = sub.NESI.to_numpy()
        K = int(y.max()) + 1

        theta, beta, ll = fit_ordinal_logistic(nesi, y)
        fits[scale] = dict(theta=theta, beta=beta, K=K, n=len(sub), loglik=ll)

        print(f"\n--- {scale}  (n={len(sub)}, K={K} categories) ---")
        print("Raw score distribution:")
        raw_counts = sub.TrueRawScores.value_counts().sort_index()
        for s, c in raw_counts.items():
            print(f"   score {int(s):>3d}: {c:>5d}")

        print(f"Proportional-odds logistic fit:")
        print(f"   slope beta = {beta:+.4f}   log-lik = {ll:.1f}")
        print(f"   thresholds theta = {np.array2string(theta, precision=3, separator=', ')}")

        # Emit the closed-form formula.
        # For category k (0..K-1):
        #   P(Y<=k) = sigmoid(theta_k - beta*NESI),  theta_K-1 := +inf
        #   P(Y=k)  = P(Y<=k) - P(Y<=k-1)
        #   E[Y]    = sum_k k * P(Y=k)
        print(f"   predicted transformed-score  E[Y|NESI] = "
              f"sum_{{k=0..{K-1}}} k * [s(theta_k - {beta:+.4f}*NESI) "
              f"- s(theta_{{k-1}} - {beta:+.4f}*NESI)]")
        print(f"   raw-{scale} score  = "
              + ("15 - E[Y|NESI]"   if scale == "GCS"
                 else "-E[Y|NESI]"  if scale == "RASS"
                 else "E[Y|NESI]"))

    # -----------------------------------------------------------------------
    # MAIN FIGURE: 4x4 matrix that combines the old Plot 1 and Plot 2.
    #   Diagonal cells:  scale vs NESI  (x-axis = NESI)
    #   Off-diagonal:    cross-scale worm (x-axis = scale_x, y-axis = scale_y)
    # Visual simplifications (per reviewer feedback):
    #   * 80 / 95 % centile contours only (drop 50 %)
    #   * contour-fill alpha=0.7 so the red-blue field is quieter than the
    #     spine and contour lines
    #   * NESI waypoint labels only on one exemplar diagonal (GCS vs NESI);
    #     dots still appear on all cells as visual anchors
    # -----------------------------------------------------------------------
    nesi_grid = np.linspace(NESI_VMIN, NESI_VMAX, 400)
    nesi_fine = np.linspace(NESI_VMIN, NESI_VMAX, 200)
    fill_levels = np.linspace(NESI_VMIN, NESI_VMAX, 41)
    centiles = (0.8, 0.95)
    nesi_marks = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    contour_lw = [0.9, 1.3]        # matches centiles outer-in
    fill_alpha = 0.70

    n = len(scales)
    fig, axes = plt.subplots(n, n, figsize=(13, 12))

    for i, sy in enumerate(scales):
        for j, sx in enumerate(scales):
            ax = axes[i, j]

            if i == j:
                # Diagonal: score (sy) as a function of NESI.
                f = fits[sy]
                raw_values_all = transformed_to_raw(sy, np.arange(f["K"]))
                y_lo = raw_values_all.min() - 0.5
                y_hi = raw_values_all.max() + 0.5
                y_grid = np.linspace(y_lo, y_hi, 200)

                density, mu_y = worm_diagonal_density(
                    f, sy, nesi_fine, y_grid)
                levels = density_levels_for_centiles(density, centiles)
                Xg, Yg = np.meshgrid(nesi_fine, y_grid)
                mask = density >= levels[0]
                # On a diagonal, E[NESI | NESI=x, score=y] = x trivially.
                nesi_field = np.where(mask, Xg, np.nan)
                ax.contourf(Xg, Yg, nesi_field, levels=fill_levels,
                            cmap=NESI_CMAP, norm=NESI_NORM,
                            extend="both", alpha=fill_alpha)
                ax.contour(Xg, Yg, density, levels=levels,
                           colors="black", linewidths=contour_lw, alpha=0.6)
                ax.plot(nesi_fine, mu_y, color="black", lw=1.7, zorder=5)

                mu_marks, _ = _worm_gaussian_stats(nesi_marks, f, sy)
                ax.scatter(nesi_marks, mu_marks, c=nesi_marks,
                           cmap=NESI_CMAP, norm=NESI_NORM, s=45,
                           edgecolor="white", linewidth=1.2, zorder=6)
                # Labels only on the bottom-right exemplar (ICANS vs NESI).
                if i == n - 1:
                    for nv, ym in zip(nesi_marks, mu_marks):
                        ax.annotate(f"{nv:+.0f}", (nv, ym),
                                    xytext=(5, 5), textcoords="offset points",
                                    fontsize=7, ha="left", va="bottom",
                                    zorder=7,
                                    bbox=dict(boxstyle="round,pad=0.15",
                                              fc="white", ec="none",
                                              alpha=0.8))

                ax.set_xlim(NESI_VMIN, NESI_VMAX)
                ax.set_ylim(y_lo, y_hi)
                title_text = f"{sy} vs NESI"
            else:
                # Off-diagonal: cross-scale worm.
                fx, fy = fits[sx], fits[sy]
                x_raw_all = transformed_to_raw(sx, np.arange(fx["K"]))
                y_raw_all = transformed_to_raw(sy, np.arange(fy["K"]))
                x_lo, x_hi = x_raw_all.min() - 0.5, x_raw_all.max() + 0.5
                y_lo, y_hi = y_raw_all.min() - 0.5, y_raw_all.max() + 0.5
                x_grid = np.linspace(x_lo, x_hi, 160)
                y_grid = np.linspace(y_lo, y_hi, 160)

                density, E_nesi = worm_joint_density(
                    fx, fy, sx, sy, x_grid, y_grid,
                    nesi_range=(NESI_VMIN, NESI_VMAX))
                levels = density_levels_for_centiles(density, centiles)
                Xg, Yg = np.meshgrid(x_grid, y_grid)

                mask = density >= levels[0]
                nesi_field = np.where(mask, E_nesi, np.nan)
                ax.contourf(Xg, Yg, nesi_field, levels=fill_levels,
                            cmap=NESI_CMAP, norm=NESI_NORM,
                            extend="both", alpha=fill_alpha)
                ax.contour(Xg, Yg, density, levels=levels,
                           colors="black", linewidths=contour_lw, alpha=0.6)

                mu_x_s, _ = _worm_gaussian_stats(nesi_fine, fx, sx)
                mu_y_s, _ = _worm_gaussian_stats(nesi_fine, fy, sy)
                ax.plot(mu_x_s, mu_y_s, color="black", lw=1.7, zorder=5)

                mu_x_m, _ = _worm_gaussian_stats(nesi_marks, fx, sx)
                mu_y_m, _ = _worm_gaussian_stats(nesi_marks, fy, sy)
                ax.scatter(mu_x_m, mu_y_m, c=nesi_marks, cmap=NESI_CMAP,
                           norm=NESI_NORM, s=40, edgecolor="white",
                           linewidth=1.1, zorder=6)

                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
                title_text = f"{sy} vs {sx}"

            # Axis labels and ticks only on the bottom row / left column.
            # Interior diagonals would also want NESI ticks, but they bleed
            # into the cell below under tight hspace — drop them and rely
            # on the bottom-right diagonal + colorbar for NESI readout.
            if j == 0:
                ax.set_ylabel(sy, fontsize=10)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both",
                               left=False, labelleft=False)
            if i == n - 1:
                ax.set_xlabel("NESI" if i == j else sx, fontsize=10)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", which="both",
                               bottom=False, labelbottom=False)

            ax.text(0.03, 0.97, title_text,
                    transform=ax.transAxes,
                    fontsize=8, fontweight="bold", ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="0.7", lw=0.5, alpha=1.0))
            if i == j:
                ax.set_facecolor("#eef2f6")  # tint diagonal (score vs NESI) panels
            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.06, top=0.94,
                        wspace=0.08, hspace=0.08)

    sm = plt.cm.ScalarMappable(cmap=NESI_CMAP, norm=NESI_NORM)
    sm.set_array([])
    cax = fig.add_axes([0.935, 0.15, 0.013, 0.7])
    fig.colorbar(sm, cax=cax, label=_NESI_CBAR_LABEL)

    # Descriptive title intentionally omitted from artwork (lives in the caption).
    fig.savefig(f"{OUT_DIR}/figure_main.png", dpi=150, bbox_inches="tight")
    if _save_fig is not None:
        _save_fig(fig, "Figure3")
    plt.close(fig)
    print(f"\nWrote {OUT_DIR}/figure_main.png")

    # -----------------------------------------------------------------------
    # SUPPLEMENTARY FIGURE: rater-vs-rater reproducibility (1x4 strip).
    # Under the fitted ordinal-logistic noise, two independent raters of
    # the same patient have joint distribution p(X_A, X_B) obtained by the
    # same machinery as the cross-scale joints, with X = Y = same scale.
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, n, figsize=(16, 4.5))
    for ax, scale in zip(axes, scales):
        f = fits[scale]
        raw_values_all = transformed_to_raw(scale, np.arange(f["K"]))
        lo = raw_values_all.min() - 0.5
        hi = raw_values_all.max() + 0.5
        grid = np.linspace(lo, hi, 160)

        density, E_nesi = worm_joint_density(
            f, f, scale, scale, grid, grid,
            nesi_range=(NESI_VMIN, NESI_VMAX))
        levels = density_levels_for_centiles(density, centiles)
        Xg, Yg = np.meshgrid(grid, grid)

        mask = density >= levels[0]
        nesi_field = np.where(mask, E_nesi, np.nan)
        ax.contourf(Xg, Yg, nesi_field, levels=fill_levels,
                    cmap=NESI_CMAP, norm=NESI_NORM,
                    extend="both", alpha=fill_alpha)
        ax.contour(Xg, Yg, density, levels=levels,
                   colors="black", linewidths=contour_lw, alpha=0.6)

        mu_s, _ = _worm_gaussian_stats(nesi_fine, f, scale)
        ax.plot(mu_s, mu_s, color="black", lw=1.7, zorder=5)

        # Reference y = x
        ref = np.linspace(lo, hi, 2)
        ax.plot(ref, ref, color="gray", lw=0.8, linestyle="--",
                alpha=0.6, zorder=4)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{scale} (rater A)", fontsize=10)
        ax.set_ylabel(f"{scale} (rater B)", fontsize=10)
        ax.set_title(scale, fontsize=11)
        ax.grid(alpha=0.15)

    sm = plt.cm.ScalarMappable(cmap=NESI_CMAP, norm=NESI_NORM)
    sm.set_array([])
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.935, 0.2, 0.010, 0.6])
    fig.colorbar(sm, cax=cax, label="NESI (red = worse)")

    fig.suptitle(
        "Supplementary: rater-vs-rater reproducibility implied by the "
        "noise model.  Two independent raters of the same patient; y = x "
        "dashed for reference.  80 / 95 % credible regions.",
        fontsize=10.5, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    fig.savefig(f"{OUT_DIR}/figure_supp_rater_noise.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/figure_supp_rater_noise.png")
    print(f"Wrote {OUT_DIR}/plot2_pairwise.png")


    # Build the predicted-scores curve (used by CSV export below).
    predicted = {"NESI": nesi_grid}
    for scale in scales:
        f = fits[scale]
        exp_t, _ = predict_expected(nesi_grid, f["theta"], f["beta"], f["K"])
        predicted[f"p{scale}"] = transformed_to_raw(scale, exp_t)


    # -----------------------------------------------------------------------
    # Export: dense (NESI, pGCS, pRASS, pCAMS, pICANS) curve for the user
    # and a coarser lookup table for quick reference.
    # -----------------------------------------------------------------------
    curve = pd.DataFrame(predicted)
    curve.to_csv("predicted_scores_vs_NESI.csv", index=False)
    print(f"\nWrote predicted_scores_vs_NESI.csv  ({len(curve)} rows, "
          f"NESI from {curve.NESI.min():.3f} to {curve.NESI.max():.3f})")

    nesi_table = np.linspace(-3.0, 3.0, 13)
    rows = []
    for x in nesi_table:
        row = {"NESI": x}
        for scale in scales:
            f = fits[scale]
            exp_t, _ = predict_expected(np.array([x]), f["theta"], f["beta"], f["K"])
            row[f"p{scale}"] = float(transformed_to_raw(scale, exp_t)[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv("nesi_to_scale_lookup.csv", index=False)
    print("\nPredicted raw clinical score at representative NESI values:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    print("\nWrote nesi_to_scale_lookup.csv")


if __name__ == "__main__":
    main()
