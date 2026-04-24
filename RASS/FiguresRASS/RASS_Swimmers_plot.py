import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="DataFrameGroupBy.apply operated on the grouping columns"
)
# Load the Cohort RASS data

current = Path(__file__).resolve()
RASS_ROOT = None
for parent in current.parents:
    if parent.name == "RASS":
        RASS_ROOT = parent
        break

if RASS_ROOT is None:
    raise RuntimeError("RASS folder not found")


s0001_RASS_datapath = RASS_ROOT / "Cohort" / "RASS_MGH_HarvardEEG_metadata.csv" 
s0002_RASS_datapath = RASS_ROOT / "Cohort" / "RASS_BWH_HarvardEEG_metadata.csv" 

df_s0001=pd.read_csv(s0001_RASS_datapath)
df_s0002=pd.read_csv(s0002_RASS_datapath)

RASS_EEG_all_df_censor_mgb=pd.concat([df_s0002, df_s0001], ignore_index=True)
subs_rass=RASS_EEG_all_df_censor_mgb['BDSPPatientID'].unique()

RASS_EEG_all_df_censor_mgb=RASS_EEG_all_df_censor_mgb.drop(columns=['DateOfBirth',	'DateOfDeath',	'CensorDate',
                                                                    'HospitalAdmitDTS', 'HospitalDischargeDTS',
                                                                    'AgeAtVisit',	'PatientRace', 'EthnicGroupDSC', 'SexDSC'])

# Process the RASS cohort MGB (MGH+BWH)for swimmers plot

df = RASS_EEG_all_df_censor_mgb.copy()
idx = (
    df.groupby(['BDSPPatientID', 'SessionID'])['DurationInSeconds']
      .max()
      .reset_index()
      .sort_values(['BDSPPatientID', 'DurationInSeconds'], ascending=[True, False])
      .drop_duplicates(subset=['BDSPPatientID'])
      [['BDSPPatientID', 'SessionID']]
)

df_filtered = df.merge(idx, on=['BDSPPatientID', 'SessionID'], how='inner')
df = df_filtered.copy()

df['RASS'] = df['R PHS IP RASS']
df['sed_component'] = np.maximum(0, -df['RASS'])
df['weighted_sed'] = df['sed_component'] * df['DurationInSeconds']

patient_order = (
    df.groupby('BDSPPatientID')
      .apply(lambda x: x['weighted_sed'].sum() / x['DurationInSeconds'].sum())
      .rename('SedBurden')
      .sort_values(ascending=False)
      .reset_index()
)

df_sorted = (
    df.merge(patient_order[['BDSPPatientID']], on='BDSPPatientID', how='left')
      .set_index('BDSPPatientID')
      .loc[patient_order['BDSPPatientID']]
      .reset_index()
)

# Group by patient and get number of unique BidsFolder and SessionID combinations
check = df_sorted.groupby('BDSPPatientID').agg({
    'BidsFolder': 'nunique',
    'SessionID': 'nunique'
})

# Show any patients where there is more than one BidsFolder or SessionID
violations = check[(check['BidsFolder'] > 1) | (check['SessionID'] > 1)]
# print("Patients with inconsistent mapping:")
# print(violations)

if violations.empty:
    print("All BDSPPatientID values have exactly one unique BidsFolder and SessionID.")
else:
    print(f"Found {len(violations)} patients with inconsistent mapping.")


# First, for each patient, find the most common BidsFolder (or just the first one)
patient_folder = df_sorted.groupby('BDSPPatientID')['BidsFolder'] \
                          .agg(lambda x: x.mode()[0])  # mode gives the most frequent

# Merge back to keep only rows with that BidsFolder
df_filtered = df_sorted.merge(patient_folder.rename('ChosenBidsFolder'),
                              on='BDSPPatientID')

df_filtered = df_filtered[df_filtered['BidsFolder'] == df_filtered['ChosenBidsFolder']].copy()

# Drop the helper column if you want
df_filtered.drop(columns='ChosenBidsFolder', inplace=True)

# Verify that each patient now has only one unique BidsFolder
check = df_filtered.groupby('BDSPPatientID')['BidsFolder'].nunique()
# print(check[check > 1])  # should be empty

# Ensure datetime columns
df_filtered['RASSRecordedDTS'] = pd.to_datetime(df_filtered['RASSRecordedDTS'])
df_filtered['EEGBeginDTS'] = pd.to_datetime(df_filtered['EEGBeginDTS'])
# EEG duration in minutes
df_filtered['EEGDurationInHour'] = df_filtered['DurationInSeconds'] / 3600
# RASS timing in minutes relative to EEG start
df_filtered['RASSHour'] = (df_filtered['RASSRecordedDTS'] - df_filtered['EEGBeginDTS']).dt.total_seconds() / 3600


# Histogram
# Define custom bins in Hours RASS timing relative to EEG start (Hours)
bins = [0, 0.01, 0.2, 1, 2, 5, 10, 15, 20, 25, 30]

# Plot histogram
# plt.figure(figsize=(26, 4))

# Added rwidth=0.8 to create space between bars
# n, bins_edges, patches = plt.hist(
#     df_filtered['RASSHour'],
#     bins=bins,
#     edgecolor='black',
#     rwidth=0.8  # Adjust this value (0 to 1) to change the gap size
# )
# plt.xlabel('RASS timing relative to EEG start (Hours)')
# plt.ylabel('Number of RASS recordings')
# plt.title('Histogram of RASSHour with Gaps')
# plt.xticks(bins)
# plt.show()

n, bins_edges = np.histogram(
    df_filtered['RASSHour'],
    bins=bins
)
# Print counts per bin
for i in range(len(bins)-1):
    print(f"{bins[i]} - {bins[i+1]} Hour: {int(n[i])} recordings")


# -------------------------------------------------------------------
#			       Linear Time Scale based Swimmers plot 
#  -------------------------------------------------------------------

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 5
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5

# ---------- Data Preparation ----------
df_full = df_filtered.copy().dropna(subset=['EEGDurationInHour', 'RASSHour', 'RASS'])
df_full = df_full[df_full['RASSHour'] <= 30]

# Convert RASSHour to minutes for linear scale
df_full['RASSMinute'] = df_full['RASSHour'] * 60
df_full['EEGDurationMinute'] = df_full['EEGDurationInHour'] * 60

unique_pids = df_full['BDSPPatientID'].unique()
pid_to_y = {pid: i for i, pid in enumerate(unique_pids)}
df_full['y_coord'] = df_full['BDSPPatientID'].map(pid_to_y)

# ---------- Color map for RASS ----------
rass_levels = np.arange(-5, 5)
base_cmap = plt.cm.Blues_r
colors = base_cmap(np.linspace(0.15, 0.95, len(rass_levels)))
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(-5.5, 5.5, 1), cmap.N)

# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
epsilon = 0.01  # minimal shift to avoid 0 issues

# 1. Background Duration Lines
duration_df = df_full.groupby('y_coord')['EEGDurationMinute'].max().reset_index()
ax.hlines(
    y=duration_df['y_coord'],
    xmin=epsilon,
    xmax=duration_df['EEGDurationMinute'].clip(upper=1800),
    color='lightgray',
    linewidth=0.4,
    alpha=0.2,
    zorder=1
)

# 2. RASS Points
x_coords = np.where(df_full['RASSMinute'] < epsilon, epsilon, df_full['RASSMinute'])
ax.scatter(
    x_coords,
    df_full['y_coord'],
    c=df_full['RASS'],
    cmap=cmap,
    norm=norm,
    s=1,
    alpha=0.8,
    edgecolors='none',
    zorder=1,
    rasterized=True
)

# ---------- Linear X-axis in Minutes ----------
ax.set_xscale('linear')
ax.set_xlim(0, 1800)  # 0–1800 min = 0–30 hours

# Ticks every 10 minutes, label every hour
tick_minutes = np.arange(0, 1801, 10)
tick_labels = [f"{int(t//60)}h" if t % 60 == 0 else '' for t in tick_minutes]
ax.set_xticks(tick_minutes)
ax.set_xticklabels(tick_labels, rotation=45)

ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.4)

ax.set_xlabel("Time since EEG start (Minutes)")
ax.set_ylabel(f"No. of unique patients")
ax.set_title("Swimmer Plot (0–30 Hours): RASS Assessments")
ax.set_ylim(len(unique_pids), 0)
ax.invert_yaxis()
ax.margins(y=0)

# ---------- Colorbar ----------
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, label="RASS Level", shrink=0.5, pad=0.02)
cbar.set_ticks(rass_levels)

plt.tight_layout()
plt.show()
 
# -------------------------------------------------------------------
#			       Log Time Scale based Swimmers plot 
#  -------------------------------------------------------------------

# ---------- Data Preparation ----------
df_full = df_filtered.copy().dropna(subset=['EEGDurationInHour', 'RASSHour', 'RASS'])
df_full = df_full[df_full['RASSHour'] <= 30]

unique_pids = df_full['BDSPPatientID'].unique()
pid_to_y = {pid: i for i, pid in enumerate(unique_pids)}
df_full['y_coord'] = df_full['BDSPPatientID'].map(pid_to_y)

# ---------- Color map for RASS ----------
rass_levels = np.arange(-5, 5)
base_cmap = plt.cm.Blues_r
colors = base_cmap(np.linspace(0.15, 0.95, len(rass_levels)))
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(-5.5, 5.5, 1), cmap.N)

# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

# We set epsilon to be just before 1 minute so the 1-min tick shows up nicely
epsilon = 0.005  # ~18 seconds

# 1. Background Duration Lines
duration_df = df_full.groupby('y_coord')['EEGDurationInHour'].max().reset_index()
ax.hlines(
    y=duration_df['y_coord'],
    xmin=epsilon,
    xmax=duration_df['EEGDurationInHour'].clip(upper=30),
    color='lightgray',
    linewidth=0.4,
    alpha=0.2,
    zorder=1
)

# 2. RASS Points
# Shift values < epsilon to epsilon for log display
x_coords = np.where(df_full['RASSHour'] < epsilon, epsilon, df_full['RASSHour'])

ax.scatter(
    x_coords,
    df_full['y_coord'],
    c=df_full['RASS'],
    cmap=cmap,
    norm=norm,
    s=1,         # Tiny points to prevent vertical overlap
    alpha=0.8,
    edgecolors='none',
    zorder=1,
    rasterized=True
)

# ---------- Log Scale & Custom Ticks ----------
ax.set_xscale('log')

# Define ticks by converting minutes to hours
# 1m, 6m, 10m, 1h, 5h, 10h, 20h, 30h
tick_hours = [1/60, 10/60, 30/60,1.0, 5.0, 10.0, 20.0, 30.0]
tick_labels = ['1m', '10m', '30m','1h', '5h', '10h', '20h', '30h']

ax.set_xlim(epsilon, 30)
ax.set_xticks(tick_hours)
ax.set_xticklabels(tick_labels)

# Minor ticks to help guide the eye across log spans
ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.4)

ax.set_xlabel("Time since EEG start (Log Scale: Minutes to Hours)")
ax.set_ylabel(f"No. of Unique Patients")
ax.set_title("Swimmer Plot (0-30 Hours): RASS Assessments")
ax.set_ylim(len(unique_pids), 0)
ax.invert_yaxis()
ax.margins(y=0)
# ---------- Colorbar ----------
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, label="RASS Level", shrink=0.5, pad=0.02)
cbar.set_ticks(rass_levels)
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------
#	 	Zoomed Linear Time Scale based Swimmers plot (0-10 min) 
#  -------------------------------------------------------------------

# ---------- Data Preparation ----------
df_full = df_filtered.copy().dropna(subset=['EEGDurationInHour', 'RASSHour', 'RASS'])
df_full = df_full[df_full['RASSHour'] <= 30]

# Convert RASSHour to minutes for linear scale
df_full['RASSMinute'] = df_full['RASSHour'] * 60
df_full['EEGDurationMinute'] = df_full['EEGDurationInHour'] * 60

unique_pids = df_full['BDSPPatientID'].unique()
pid_to_y = {pid: i for i, pid in enumerate(unique_pids)}
df_full['y_coord'] = df_full['BDSPPatientID'].map(pid_to_y)

# ---------- Color map for RASS ----------
rass_levels = np.arange(-5, 5)
base_cmap = plt.cm.Blues_r
colors = base_cmap(np.linspace(0.15, 0.95, len(rass_levels)))
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(-5.5, 5.5, 1), cmap.N)

# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
epsilon = 0.01  # minimal shift to avoid 0 issues

# 1. Background Duration Lines
duration_df = df_full.groupby('y_coord')['EEGDurationMinute'].max().reset_index()
ax.hlines(
    y=duration_df['y_coord'],
    xmin=epsilon,
    xmax=duration_df['EEGDurationMinute'].clip(upper=10),  # Clip at 10 min
    color='lightgray',
    linewidth=0.4,
    alpha=0.2,
    zorder=1
)

# 2. RASS Points
x_coords = np.where(df_full['RASSMinute'] < epsilon, epsilon, df_full['RASSMinute'])
ax.scatter(
    x_coords,
    df_full['y_coord'],
    c=df_full['RASS'],
    cmap=cmap,
    norm=norm,
    s=2,
    alpha=0.8,
    edgecolors='none',
    zorder=2,
    rasterized=True
)

# ---------- Linear X-axis in Minutes ----------
ax.set_xscale('linear')
ax.set_xlim(0, 10)  # Only show 0–10 minutes

# Ticks every 1 minute for fine resolution
tick_minutes = np.arange(0, 11, 1)
tick_labels = [f"{int(t)}m" for t in tick_minutes]
ax.set_xticks(tick_minutes)
ax.set_xticklabels(tick_labels, rotation=45)

ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.4)

ax.set_xlabel("Time since EEG start (Minutes)")
ax.set_ylabel(f"No. of Unique Patients")
ax.set_title("Swimmer Plot (0–10 Minutes): RASS Assessments")
ax.set_ylim(len(unique_pids), 0)
ax.invert_yaxis()
ax.margins(y=0)

# ---------- Colorbar ----------
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, label="RASS Level", shrink=0.5, pad=0.02)
cbar.set_ticks(rass_levels)

plt.tight_layout()
plt.show()