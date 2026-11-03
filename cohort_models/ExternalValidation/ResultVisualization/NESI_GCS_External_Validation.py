#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")




df_kimchi =pd.read_excel(NESI_ROOT/"cohort_models"/ "ExternalValidation" / "KIMCHI_FINAL_METADATA.xlsx")
df_kimchi['EEG_segment_StartdateTime'] = pd.to_datetime(df_kimchi['EEG_segment_StartdateTime'],format='mixed',errors='coerce')

df_kimchi['EEG_segment_EnddateTime'] = (
    df_kimchi['EEG_segment_StartdateTime'] +
    pd.to_timedelta(df_kimchi['Segment duration_s'], unit='s')
)
df_kimchi['NESIPredictionFilename'] = (
    df_kimchi['Filename'].astype(str) + '_ErikaSegment_' +
    df_kimchi['seg_start_rel_s'].astype(str) + '_NESI_predictions.csv'
)
df_kimchi = df_kimchi[['Filename', 'EEG_StartDateTime', 'EEG_segment_StartdateTime', 'EEG_segment_EnddateTime',
       'seg_start_rel_s', 'seg_end_rel_s', 'Segment duration_s', 'NESIPredictionFilename',
       'GCS_closest_DateTime', 'GCS_closest', 'RASS_closest_Datetime',
       'RASS_raw_closest', 'RASS_numeric_closest', 'AnyCAM_closest_Datetime',
       'AnyCAM_closestScore', 'segment_name',
       'Segmentduration_sec_AfterAutoreject', 'EEGrun',
       ]]


# In[2]:

df_kimchi['GCS_closest_DateTime'] = pd.to_datetime(df_kimchi['GCS_closest_DateTime'], errors='coerce')

start = df_kimchi['EEG_segment_StartdateTime']
end = df_kimchi['EEG_segment_EnddateTime']
gcs = df_kimchi['GCS_closest_DateTime']

inside = (gcs >= start) & (gcs <= end)
after = gcs > end
before = gcs < start
missing = gcs.isna()

print(f"Total rows                  : {len(df_kimchi)}")
print(f"GCS inside EEG segment      : {inside.sum()}")
print(f"GCS after EEG segment end   : {after.sum()}")
print(f"GCS before EEG segment start: {before.sum()}")
print(f"Missing GCS datetime        : {missing.sum()}")
print(f"Total classified            : {inside.sum() + after.sum() + before.sum() + missing.sum()}")


# In[3]:

df_kimchi['RASS_closest_Datetime'] = pd.to_datetime(df_kimchi['RASS_closest_Datetime'], errors='coerce')

start = df_kimchi['EEG_segment_StartdateTime']
end = df_kimchi['EEG_segment_EnddateTime']
gcs = df_kimchi['RASS_closest_Datetime']

inside = (gcs >= start) & (gcs <= end)
after = gcs > end
before = gcs < start
missing = gcs.isna()

print(f"Total rows                  : {len(df_kimchi)}")
print(f"RASS inside EEG segment      : {inside.sum()}")
print(f"RASS after EEG segment end   : {after.sum()}")
print(f"RASS before EEG segment start: {before.sum()}")
print(f"Missing RASS datetime        : {missing.sum()}")
print(f"Total classified            : {inside.sum() + after.sum() + before.sum() + missing.sum()}")


# # **For GCS extract the nearest NESI**

# In[4]:

def compute_nearest_nesi_for_GCS(df_kimchi, folder_path, window_size=10, statistic='median', nesi_column='NESI'):
    folder_path = Path(folder_path)
    df = df_kimchi.copy()

    if window_size != 'all' and (not isinstance(window_size, (int, np.integer)) or window_size < 1):
        raise ValueError("window_size must be a positive integer or 'all'")
    if statistic not in ['mean', 'median']:
        raise ValueError("statistic must be 'mean' or 'median'")

    files = {p.name: p for p in folder_path.glob('*.csv')}
    df['NESIPredictionFilename'] = df['NESIPredictionFilename'].astype('string').str.strip()

    for col in ['EEG_segment_StartdateTime', 'EEG_segment_EnddateTime', 'GCS_closest_DateTime']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Computing nearest NESI'):
        filename = row['NESIPredictionFilename']
        nesi_value = np.nan

        if pd.notna(filename) and filename in files:
            pred = pd.read_csv(files[filename])

            if nesi_column not in pred.columns:
                raise KeyError(f"Column '{nesi_column}' not found in {filename}. Available: {pred.columns.tolist()}")

            values = pd.to_numeric(pred[nesi_column], errors='coerce').to_numpy(dtype=float)

            start = row['EEG_segment_StartdateTime']
            end = row['EEG_segment_EnddateTime']
            rass_time = row['GCS_closest_DateTime']

            if len(values) > 0:
                if window_size == 'all':
                    selected = values
                elif pd.notna(start) and pd.notna(end) and pd.notna(rass_time):
                    n = min(window_size, len(values))

                    if rass_time < start:
                        selected = values[:n]
                    elif rass_time > end:
                        selected = values[-n:]
                    else:
                        seconds = (rass_time - start).total_seconds()
                        last_idx = int(np.floor(seconds - 599))
                        last_idx = int(np.clip(last_idx, 0, len(values) - 1))
                        first_idx = max(0, last_idx - n + 1)
                        selected = values[first_idx:last_idx + 1]
                else:
                    selected = np.array([])

                if np.isfinite(selected).any():
                    nesi_value = np.nanmean(selected) if statistic == 'mean' else np.nanmedian(selected)

        results.append({
            'NESIPredictionFilename': filename,
            'EEG_segment_StartdateTime': row['EEG_segment_StartdateTime'],
            'EEG_segment_EnddateTime': row['EEG_segment_EnddateTime'],
            'GCS_closest_DateTime': row['GCS_closest_DateTime'],
            'GCS_closest': row['GCS_closest'],
            'NESI_nearest': nesi_value
        })

    result_df = pd.DataFrame(results)
    matched = df['NESIPredictionFilename'].isin(files)

    print(f"\nWindow size               : {window_size}")
    print(f"Statistic                 : {statistic}")
    print(f"Total metadata rows       : {len(df)}")
    print(f"Total prediction CSV files: {len(files)}")
    print(f"Matched metadata rows     : {matched.sum()}")
    print(f"Unmatched metadata rows   : {(~matched).sum()}")
    print(f"Valid NESI values         : {result_df['NESI_nearest'].notna().sum()}")
    print(f"Missing NESI values       : {result_df['NESI_nearest'].isna().sum()}")

    if (~matched).any():
        print("\nFirst 10 unmatched filenames:")
        print(df.loc[~matched, 'NESIPredictionFilename'].head(10).to_list())

    return result_df


# In[5]:


folder_path = Path(NESI_ROOT/"cohort_models"/"ExternalValidation"/"ProductsV2"/"NESIPredictions")

df_nesi_nearest_for_GCS = compute_nearest_nesi_for_GCS(
    df_kimchi=df_kimchi,
    folder_path=folder_path,
    window_size='all',
    statistic='mean',
    nesi_column='NESI'
)

from pathlib import Path
existing_files = {f.name for f in folder_path.glob('*.csv')}

nan_mask = df_nesi_nearest_for_GCS['NESI_nearest'].isna()
missing_file_mask = ~df_nesi_nearest_for_GCS['NESIPredictionFilename'].isin(existing_files)

print(f"Total NaN NESI cases              : {nan_mask.sum()}")
print(f"NaN cases with missing CSV files : {(nan_mask & missing_file_mask).sum()}")
print(f"NaN cases with existing CSV files: {(nan_mask & ~missing_file_mask).sum()}")

print("\nNaN cases and CSV availability:")
print(df_nesi_nearest_for_GCS.loc[nan_mask, ['NESIPredictionFilename', 'NESI_nearest']].assign(
    CSV_Exists=lambda x: x['NESIPredictionFilename'].isin(existing_files)
).to_string(index=False))

df_nesi_nearest_for_GCS = df_nesi_nearest_for_GCS.loc[~(nan_mask & missing_file_mask)].reset_index(drop=True)
print(f"\nRemaining rows      : {len(df_nesi_nearest_for_GCS)}")
print(f"Remaining NaN cases : {df_nesi_nearest_for_GCS['NESI_nearest'].isna().sum()}")


# In[6]:


df_nesi_nearest_for_GCS['SubjectID'] = df_nesi_nearest_for_GCS['NESIPredictionFilename'].str.split('_').str[0]
df_nesi_nearest_for_GCS = df_nesi_nearest_for_GCS.rename(columns={
    "GCS_closest": "TrueGCS",
})
def group_gcs(gcs):
    if pd.isna(gcs):
        return None
    if 3 <= gcs <= 8:
        return "Severe"
    elif 9 <= gcs <= 12:
        return "Moderate"
    elif 13 <= gcs <= 15:
        return "Mild"
    return None

df_nesi_nearest_for_GCS["TrueGCSGrouped"] = df_nesi_nearest_for_GCS["TrueGCS"].apply(group_gcs)
df_nesi_nearest_for_GCS["TransformedTrueGCS"] = 15 - pd.to_numeric(df_nesi_nearest_for_GCS["TrueGCS"], errors="coerce")


# In[7]:


df_nesi_nearest_for_GCS = df_nesi_nearest_for_GCS.copy()

start = pd.to_datetime(df_nesi_nearest_for_GCS['EEG_segment_StartdateTime'], errors='coerce')
end = pd.to_datetime(df_nesi_nearest_for_GCS['EEG_segment_EnddateTime'], errors='coerce')
rass = pd.to_datetime(df_nesi_nearest_for_GCS['GCS_closest_DateTime'], errors='coerce')

df_nesi_nearest_for_GCS['WhenGCSColected'] = np.select(
    [rass < start, rass > end, (rass >= start) & (rass <= end)],
    ['Before EEG segment', 'After EEG segment', 'Inside EEG segment'],
    default='Missing'
)

df_nesi_nearest_for_GCS['Temporal distance (min)'] = np.select(
    [rass < start, rass > end, (rass >= start) & (rass <= end)],
    [(start - rass).dt.total_seconds() / 60,
     (rass - end).dt.total_seconds() / 60,
     0],
    default=np.nan
)

df_nesi_nearest_for_GCS = df_nesi_nearest_for_GCS[['SubjectID','NESIPredictionFilename', 'EEG_segment_StartdateTime',
                                                     'EEG_segment_EnddateTime', 'GCS_closest_DateTime', 
                                                     'TrueGCS', 'TransformedTrueGCS',
                                                     'NESI_nearest', 'WhenGCSColected', 'Temporal distance (min)']]


# In[8]:


import pandas as pd
import numpy as np

dist = pd.to_numeric(df_nesi_nearest_for_GCS['Temporal distance (min)'], errors='coerce').dropna()

print("GCS Assessment-to-EEG Temporal Distance Distribution")
print(f"Total assessments : {len(dist)}")
print(f"Mean ± SD         : {dist.mean():.2f} ± {dist.std():.2f} min")
print(f"Median (IQR)      : {dist.median():.2f} ({dist.quantile(0.25):.2f}–{dist.quantile(0.75):.2f}) min")
print(f"Minimum           : {dist.min():.2f} min")
print(f"Maximum           : {dist.max():.2f} min")
print(f"5th percentile    : {dist.quantile(0.05):.2f} min")
print(f"10th percentile   : {dist.quantile(0.10):.2f} min")
print(f"25th percentile   : {dist.quantile(0.25):.2f} min")
print(f"50th percentile   : {dist.quantile(0.50):.2f} min")
print(f"75th percentile   : {dist.quantile(0.75):.2f} min")
print(f"90th percentile   : {dist.quantile(0.90):.2f} min")
print(f"95th percentile   : {dist.quantile(0.95):.2f} min")

print("\nCases retained at different temporal windows:")
for t in [0, 15, 30, 45, 60, 90, 120, 180, 240]:
    n = (dist <= t).sum()
    print(f"Within {t:>3} min : {n:>4} / {len(dist)} ({100*n/len(dist):.1f}%)")


# In[ ]:





# In[9]:


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



# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_gcs_temporal_subplots(df, n_boot=1000, ci=95, seed=42, dpi=400):
    horizons = [(30, '30 min'), (60, '60 min'), (120, '120 min'), (None, 'All data')]
    score_col = 'TrueGCS'
    corr_col = 'TransformedTrueGCS'
    nesi_col = 'NESI_nearest'
    time_col = 'Temporal distance (min)'
    patient_col = 'SubjectID'

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.8), dpi=dpi, sharey=True)
    all_scores = sorted(df[score_col].dropna().unique())
    results = []

    for i, (horizon, label) in enumerate(horizons):
        ax = axes[i]
        subset = df.copy() if horizon is None else df[df[time_col] <= horizon].copy()
        subset = subset.dropna(subset=[score_col, corr_col, nesi_col])

        rho, lo, hi = spearman_bootstrap_ci(
            subset[nesi_col], subset[corr_col],
            n_boot=n_boot, ci=ci, seed=seed
        )

        data = [subset.loc[subset[score_col] == s, nesi_col].values for s in all_scores]
        positions = np.arange(len(all_scores))
        valid = [(p, d) for p, d in zip(positions, data) if len(d) > 0]

        if valid:
            bp = ax.boxplot(
                [d for p, d in valid],
                positions=[p for p, d in valid],
                widths=0.65,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker='o', markersize=2.8,
                                markerfacecolor='lightgray',
                                markeredgecolor='none', alpha=0.6),
                medianprops=dict(color='black', linewidth=1.4),
                whiskerprops=dict(color='black', linewidth=1),
                capprops=dict(color='black', linewidth=1),
                boxprops=dict(edgecolor='black', linewidth=1)
            )
            for box in bp['boxes']:
                box.set_facecolor('#F4B860')

        n_patients = subset[patient_col].nunique()
        n_pairs = len(subset)

        ax.set_title(
            f'({chr(65+i)}) {label}\n'
            f'ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]\n'
            f'EEG–GCS pairs = {n_pairs:,} | Patients = {n_patients:,}',
            fontsize=10, fontweight='bold', pad=14, linespacing=1.5
        )

        ax.set_xticks(positions)
        ax.set_xticklabels([str(int(s)) for s in all_scores], fontsize=14)
        ax.set_xlabel('True GCS', fontsize=14, labelpad=10)
        ax.tick_params(axis='y', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-0.6, len(all_scores)-0.4)
        ax.grid(False)

        if i == 0:
            ax.set_ylabel('NESI', fontsize=15)

        results.append({
            'Time window': label,
            'EEG-GCS pairs': n_pairs,
            'Unique patients': n_patients,
            'Spearman rho': rho,
            'CI lower': lo,
            'CI upper': hi
        })

    fig.suptitle(
        'External Validation: NESI vs. GCS',
        fontsize=17, fontweight='bold', y=1.08
    )
    fig.text(
        0.5, 1.0,
        'Temporal distance is defined as the time interval between the GCS assessment and the nearest EEG segment boundary (start or end)',
        ha='center', fontsize=16
    )

    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.15, top=0.77, wspace=0.15)
    plt.show()

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    return results_df

df_temporal_results = plot_gcs_temporal_subplots(
    df_nesi_nearest_for_GCS,
    n_boot=1000,
    ci=95,
    seed=42,
    dpi=100
)


# In[15]:


df_nesi_nearest_for_GCS.to_csv(NESI_ROOT/"cohort_models""/ExternalValidation"/"ResultVisualization"/"GCS_external_validation_result.csv", 
                               index=False)


# In[ ]:




