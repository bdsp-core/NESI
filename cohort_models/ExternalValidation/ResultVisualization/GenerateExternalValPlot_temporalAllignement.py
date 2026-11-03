#!/usr/bin/env python
# coding: utf-8

# In[1]:

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:
current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")


df_nesi_nearest_for_RASS= pd.read_csv(NESI_ROOT/"cohort_models"/"ExternalValidation"/"ResultVisualization"/"RASS_external_validation_result.csv")
df_nesi_nearest_for_GCS= pd.read_csv(NESI_ROOT/"cohort_models"/"ExternalValidation"/"ResultVisualization"/"GCS_external_validation_result.csv")


# # **Bootstrap CI Helper function**

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def spearman_bootstrap_ci(x, y, n_boot=1000, ci=95, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan

    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    boot = []
    n = len(x)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]

        if len(np.unique(xb)) > 1 and len(np.unique(yb)) > 1:
            r = spearmanr(xb, yb).statistic
            if np.isfinite(r):
                boot.append(r)

    if not boot:
        return rho, np.nan, np.nan

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot, [alpha, 100 - alpha])

    return rho, lo, hi


# # **COMBINED dataset plot Helpers**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_temporal_subplots(df_rass, df_gcs, n_boot=1000, ci=95, seed=42, dpi=400):
    horizons = [(30, '30 min'), (60, '60 min'), (120, '120 min'), (None, 'All data')]
    cohorts = [
        ('RASS', df_rass, 'TrueRASS', 'TransformedTrueRASS', '#7EB0D5'),
        ('GCS', df_gcs, 'TrueGCS', 'TransformedTrueGCS', '#F4B860')
    ]

    fig, axes = plt.subplots(2, 4, figsize=(23, 12), dpi=dpi)
    results = []

    for row, (scale, df, score_col, corr_col, color) in enumerate(cohorts):
        all_scores = sorted(df[score_col].dropna().unique())
        positions = np.arange(len(all_scores))

        for col, (horizon, label) in enumerate(horizons):
            ax = axes[row, col]
            subset = df.copy() if horizon is None else df[df['Temporal distance (min)'] <= horizon].copy()
            subset = subset.dropna(subset=[score_col, corr_col, 'NESI_nearest'])

            rho, lo, hi = spearman_bootstrap_ci(
                subset['NESI_nearest'], subset[corr_col],
                n_boot=n_boot, ci=ci, seed=seed
            )

            data = [subset.loc[subset[score_col] == s, 'NESI_nearest'].values for s in all_scores]
            valid = [(p, d) for p, d in zip(positions, data) if len(d) > 0]

            if valid:
                bp = ax.boxplot(
                    [d for p, d in valid],
                    positions=[p for p, d in valid],
                    widths=0.65,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=1.7,
                                    markerfacecolor='lightgray',
                                    markeredgecolor='none', alpha=0.6),
                    medianprops=dict(color='black', linewidth=1.4),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1),
                    boxprops=dict(edgecolor='black', linewidth=1)
                )
                for box in bp['boxes']:
                    box.set_facecolor(color)

            n_patients = subset['SubjectID'].nunique()
            n_pairs = len(subset)
            panel = chr(65 + row * 4 + col)

            ax.set_title(
                f'({panel}) {scale} | {"All assessments" if horizon is None else f"Within {horizon} min"}\n'
                f'ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]\n'
                f'Assessments = {n_pairs:,} | Patients = {n_patients:,}',
                fontsize=14, fontweight='bold', pad=12, linespacing=1.4
            )

            ax.set_xticks(positions)
            ax.set_xticklabels([str(int(s)) for s in all_scores], fontsize=9,
                               rotation=0 if scale == 'GCS' else 0)
            ax.set_xlabel(f'{scale} score', fontsize=15, labelpad=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(-0.6, len(all_scores)-0.4)
            ax.grid(False)

            if col == 0:
                ax.set_ylabel('NESI', fontsize=13)

            results.append({
                'Scale': scale,
                'Time window': label,
                'Assessments': n_pairs,
                'Unique patients': n_patients,
                'Spearman rho': rho,
                'CI lower': lo,
                'CI upper': hi
            })

    fig.suptitle(
        'External Validation Performance: Temporal Alignment Sensitivity Analysis',
        fontsize=18, fontweight='bold', y=1.05
    )

    fig.text(
        0.5, 0.975,
        'RASS (A–D) and GCS (E–H): Association between NESI and clinical neurological severity across temporal alignment windows',
        ha='center', fontsize=15
    )

    footnote = (
        'Temporal distance: time between the clinical assessment and the nearest EEG segment boundary; '
        'assessments within the EEG segment have a distance of zero.\n'
        'Boxes: median and interquartile range (IQR); whiskers: 1.5 × IQR; points: outliers. '
        'n denotes the number of assessments at each clinical score.\n'
        'Higher NESI values indicate greater neurological dysfunction. '
        'ρ: Spearman correlation; brackets: 95% bootstrap confidence intervals (1,000 iterations).'
    )

    fig.text(0.5, 0.015, footnote, ha='center', va='bottom',
             fontsize=14, style='italic', linespacing=1.6)

    plt.subplots_adjust(
        left=0.055, right=0.99,
        top=0.85, bottom=0.15,
        wspace=0.20, hspace=0.55
    )
    plt.savefig(NESI_ROOT/"cohort_models"/"ExternalValidation"/"ResultVisualization"/"NESI_RASS_External_Validation.png", dpi=600, bbox_inches="tight")

    plt.show()

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    return results_df

df_temporal_results_combined = plot_combined_temporal_subplots(
    df_nesi_nearest_for_RASS,
    df_nesi_nearest_for_GCS,
    n_boot=1000,
    ci=95,
    seed=42,
    dpi=100
)


# ------------------- version2: Same plot in barplot style with error bars ---------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_temporal_bars(df_rass, df_gcs, n_boot=1000, ci=95, seed=42, dpi=400):
    horizons = [(30, '30 min'), (60, '60 min'), (120, '120 min')]
    cohorts = [
        ('RASS', df_rass, 'TransformedTrueRASS', '#7EB0D5'),
        ('GCS', df_gcs, 'TransformedTrueGCS', '#F4B860')
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=dpi)
    results = []

    for i, (scale, df, corr_col, color) in enumerate(cohorts):
        ax = axes[i]
        rhos, lower, upper, counts = [], [], [], []

        for horizon, label in horizons:
            subset = df[df['Temporal distance (min)'] <= horizon].copy()
            subset = subset.dropna(subset=[corr_col, 'NESI_nearest'])

            rho, lo, hi = spearman_bootstrap_ci(
                subset['NESI_nearest'], subset[corr_col],
                n_boot=n_boot, ci=ci, seed=seed
            )

            n_pairs = len(subset)
            n_patients = subset['SubjectID'].nunique()

            rhos.append(rho)
            lower.append(rho - lo)
            upper.append(hi - rho)
            counts.append((n_pairs, n_patients))

            results.append({
                'Scale': scale,
                'Time window': label,
                'Assessments': n_pairs,
                'Unique patients': n_patients,
                'Spearman rho': rho,
                'CI lower': lo,
                'CI upper': hi
            })

        x = np.arange(len(horizons))
        bars = ax.bar(
            x, rhos, width=0.48, color=color,
            edgecolor='black', linewidth=0.8,
            yerr=np.array([lower, upper]),
            capsize=5, error_kw=dict(ecolor='black', lw=1.5, capthick=1.5)
        )

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in horizons], fontsize=12)
        ax.set_xlabel('Temporal alignment window', fontsize=13, labelpad=10)
        ax.set_ylabel('Spearman correlation (ρ)', fontsize=13)
        ax.set_title(f'({chr(65+i)}) {scale}', fontsize=16, fontweight='bold', pad=15)

        ymin = min(0, min(np.array(rhos) - np.array(lower)) - 0.05)
        ymax = max(np.array(rhos) + np.array(upper)) + 0.16
        ax.set_ylim(ymin, ymax)

        for j, (rho, err, (n_pairs, n_patients)) in enumerate(zip(rhos, upper, counts)):
            ax.text(
                j, rho + err + 0.025,
                f'ρ = {rho:.2f} [{rho - lower[j]:.2f}, {rho + upper[j]:.2f}]\n'
                f'Assessments = {n_pairs:,}\nPatients = {n_patients:,}',
                ha='center', va='bottom', fontsize=10, linespacing=1.4
            )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(False)
        ax.set_axisbelow(True)

    fig.suptitle(
        'External Validation Performance: Temporal Alignment Sensitivity Analysis',
        fontsize=18, fontweight='bold', y=1.08
    )

    fig.text(
        0.5, 1.005,
        'RASS (A) and GCS (B): Association between NESI and clinical neurological severity across temporal alignment windows',
        ha='center', fontsize=12
    )

    footnote = (
        'Temporal distance: time between the clinical assessment and the nearest EEG segment boundary; '
        'assessments within the EEG segment have a distance of zero.\n'
        'Bars: Spearman correlation (ρ); error bars: 95% bootstrap confidence intervals (1,000 iterations). '
        'Assessments and unique patients are shown above each bar.\n'
        'Higher NESI values indicate greater neurological dysfunction.'
    )

    fig.text(
        0.5, 0.05, footnote,
        ha='center', va='top', fontsize=11,
        style='italic', linespacing=1.5
    )

    plt.subplots_adjust(
        left=0.08, right=0.98,
        top=0.82, bottom=0.18,
        wspace=0.25
    )

    plt.show()

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    return results_df

# df_temporal_results_combined = plot_combined_temporal_bars(
#     df_nesi_nearest_for_RASS,
#     df_nesi_nearest_for_GCS,
#     n_boot=1000,
#     ci=95,
#     seed=42,
#     dpi=400
# )


# In[ ]:





# In[ ]:





# In[ ]:




