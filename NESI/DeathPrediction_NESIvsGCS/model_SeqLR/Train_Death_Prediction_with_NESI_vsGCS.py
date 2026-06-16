#!/usr/bin/env python
# coding: utf-8

# # **Library**

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import h5py
import hdf5storage
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)
from tqdm import tqdm
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


# # **Load GCS Cohort**

# In[2]:

current = Path(__file__).resolve()
GCS_ROOT = None
for parent in current.parents:
    if parent.name == "GCS":
        GCS_ROOT = parent
        break

if GCS_ROOT is None:
    raise RuntimeError("GCS folder not found")


current = Path(__file__).resolve()
GCS_ROOT = None
for parent in current.parents:
    if parent.name == "GCS":
        GCS_ROOT = parent
        break

if GCS_ROOT is None:
    raise RuntimeError("GCS folder not found")


metadata_s0001_path = GCS_ROOT / "cohort_models" / "GCS" / "Cohort" / "GCS_MGH_HarvardEEG_metadata.csv"
metadata_s0002_path = GCS_ROOT / "cohort_models" / "GCS" / "Cohort" / "GCS_BWH_HarvardEEG_metadata.csv"
metadata_i0002_path = GCS_ROOT / "cohort_models" / "GCS" / "Cohort" / "GCS_BIDMC_HarvardEEG_metadata.csv"

df_s0001=pd.read_csv(metadata_s0001_path)
df_s0002=pd.read_csv(metadata_s0002_path)

df_GCS_MGB=pd.concat([df_s0002, df_s0001], ignore_index=True)
df_GCS_MGB = df_GCS_MGB.dropna(subset=["R CPN GLASGOW COMA SCALE SCORE"])
subs_gcs_mgb=df_GCS_MGB['BDSPPatientID'].unique()

df_GCS_BIDMC=pd.read_csv(metadata_i0002_path)
subs_gcs_bidmc=df_GCS_BIDMC['BDSPPatientID'].unique()

df_GCS_MGB['SiteID'] = 'S0001'
df_GCS_BIDMC['SiteID'] = 'S0002'

df_GCS_MGB = df_GCS_MGB[['BDSPPatientID', 'SiteID', 'DateOfDeath', 'CensorDate',
                        'EEGBeginDTS', 'EEGExamEndDTS', 'GCSRecordedDTS',
                         'R CPN GLASGOW COMA SCALE SCORE', 'DurationInSeconds',
                         'BidsFolder', 'SessionID']]


df_GCS_BIDMC = df_GCS_BIDMC.rename(columns={
    'censor_date': 'CensorDate',
    'death_date': 'DateOfDeath',
    'RecordedDTS': 'GCSRecordedDTS',
    'StartTime':'EEGBeginDTS', 
    'EndTime':'EEGExamEndDTS', 
    'GCS_total': 'R CPN GLASGOW COMA SCALE SCORE',
    
})

df_GCS_BIDMC = df_GCS_BIDMC[['BDSPPatientID', 'SiteID', 'DateOfDeath', 'CensorDate',
                             'EEGBeginDTS', 'EEGExamEndDTS', 'GCSRecordedDTS',
                            'R CPN GLASGOW COMA SCALE SCORE','DurationInSeconds', 
                             'BidsFolder', 'SessionID']]

df_full_GCS_Dataset = pd.concat([df_GCS_MGB, df_GCS_BIDMC], axis=0)

print('Total unique patients in Full (MGH+BWH+BIDMC) GCS cohort ==> ', df_full_GCS_Dataset['BDSPPatientID'].nunique())
print('Total unique patients in MGH+BWH GCS cohort ==> ', df_GCS_MGB['BDSPPatientID'].nunique())
print('Total unique patients in BIDMC GCS cohort ==> ', df_GCS_BIDMC['BDSPPatientID'].nunique())


# # **Load Single Session Death Cohort**

# In[3]:

current = Path(__file__).resolve()
DEATH_ROOT = None
for parent in current.parents:
    if parent.name == "Death":
        DEATH_ROOT = parent
        break

if DEATH_ROOT is None:
    raise RuntimeError("DEATH folder not found")

death_single_seeion_path = DEATH_ROOT / "mortality_analysis" / "Cohort" / "YAMA_FINAL_DEATH_RANDOM_SINGLE_SESSIONS_COHORT.csv"
df_death_singlesession = pd.read_csv(death_single_seeion_path)

print('Total unique patients death cohort ==> ', df_death_singlesession['BDSPPatientID'].nunique())
print('\n COlumn names in death dataset ==> ', df_death_singlesession.columns)

patient_id_col = "BDSPPatientID"

# =========================
# 1. Convert Yes/No → 1/0
# =========================
df = df_death_singlesession.copy()

df["DiedInHospital"] = df["DiedInHospital"].map({"Yes": 1, "No": 0})

# =========================
# 2. Collapse to patient-level
# =========================
site_patient = (
    df.sort_values([patient_id_col, "SiteID"])
    .groupby(patient_id_col)
    .agg({
        "SiteID": "first",
        "DiedInHospital": "max"
    })
    .reset_index()
)

# =========================
# 3. Create stacked table
# =========================
site_table = pd.crosstab(
    site_patient["SiteID"],
    site_patient["DiedInHospital"]
)

site_table = site_table.reindex(columns=[0, 1], fill_value=0)

print('\n', site_table)

# =========================
# 4. Plot stacked bar
# =========================
ax = site_table.plot(kind="bar", stacked=True, figsize=(10, 6))

plt.title("Unique Patients Outcome by SiteID")
plt.xlabel("SiteID")
plt.ylabel("Number of Patients")

# labels inside bars
for i, (_, row) in enumerate(site_table.iterrows()):
    bottom = 0
    for col in site_table.columns:
        val = row[col]
        if val > 0:
            plt.text(i, bottom + val/2, str(val), ha="center", va="center")
        bottom += val

plt.legend(["Alive (0)", "Died In-hospital (1)"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # **Find common subjcets in GCS cohort as the death cohort**

# In[4]:


# =========================
# UNIQUE SUBJECT COUNTS
# =========================
SubIDs_death = set(df_death_singlesession["BDSPPatientID"].unique())
SubIDs_GCS_MGB = set(df_GCS_MGB["BDSPPatientID"].unique())
SubIDs_GCS_BIDMC = set(df_GCS_BIDMC["BDSPPatientID"].unique())

# =========================
# INTERSECTIONS
# =========================
common_GCS_MGB_death = SubIDs_GCS_MGB.intersection(SubIDs_death)
common_GCS_BIDMC_death = SubIDs_GCS_BIDMC.intersection(SubIDs_death)

# =========================
# PRINT RESULTS
# =========================
print("Total Death subjects:", len(SubIDs_death))
print("Total GCS MGB subjects:", len(SubIDs_GCS_MGB))
print("Total GCS BIDMC subjects:", len(SubIDs_GCS_BIDMC))

print("\nCommon (GCS MGB ∩ Death):", len(common_GCS_MGB_death))
print("Common (GCS BIDMC ∩ Death):", len(common_GCS_BIDMC_death))


# ## **Keeping only those subjects rows which are common in: DEATH ∩ GCS MGB**

# In[5]:


# =========================
# KEEP ONLY COMMON PATIENTS (DEATH ∩ GCS MGB)
# =========================

# merge all GCS subjects
gcs_subjects = set(df_full_GCS_Dataset["BDSPPatientID"].unique())

# death subjects
death_subjects = set(df_death_singlesession["BDSPPatientID"].unique())

# intersection
common_subjects = gcs_subjects.intersection(death_subjects)

# filter death dataframe
df_death_filtered = df_death_singlesession[
    df_death_singlesession["BDSPPatientID"].isin(common_subjects)
].copy()

df_death_filtered["DurationInHours"] = np.floor(
    df_death_filtered["DurationInSeconds"] / 3600
).astype(int)

df_death_filtered = df_death_filtered[['BDSPPatientID', 'SiteID', 'DateOfDeath', 'CensorDate',
                                       'EEGBeginDTS', 'EEGExamEndDTS', 'DurationInSeconds', 'DurationInHours',
                                       'BidsFolder', 'SessionID', 'DiedInHospital'
                                      ]]

print("\nOriginal death rows:", df_death_singlesession.shape)
print("Filtered death rows:", df_death_filtered.shape)
print("Unique patients after filter:", df_death_filtered["BDSPPatientID"].nunique())

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# MAKE SURE BINARY (YES/NO -> 1/0) OR KEEP AS IS
# =========================
df = df_death_filtered.copy()

# if still string
if df["DiedInHospital"].dtype == "object":
    df["DiedInHospital"] = df["DiedInHospital"].map({"Yes": "Yes", "No": "No"})

# =========================
# STACKED TABLE: SITE x OUTCOME
# =========================
site_table = pd.crosstab(df["SiteID"], df["DiedInHospital"])

# ensure both columns exist
site_table = site_table.reindex(columns=["No", "Yes"], fill_value=0)

# =========================
# PLOT
# =========================
ax = site_table.plot(kind="bar", stacked=True, figsize=(10,6))

plt.title("In-Hospital Outcome by Site (Filtered Dataset)")
plt.xlabel("SiteID")
plt.ylabel("Number of Patients")

# annotate bars
for i, (_, row) in enumerate(site_table.iterrows()):
    bottom = 0
    for col in site_table.columns:
        val = row[col]
        if val > 0:
            plt.text(i, bottom + val/2, str(val), ha="center", va="center")
        bottom += val

plt.legend(title="DiedInHospital")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # **Gathering the GCS values and GCS record dates for these patients**

# ## **Finding who has GCS recorded while EEG recording was going on**

# In[6]:


import pandas as pd

# =========================
# COPY DATA
# =========================
df_death_filtered = df_death_filtered.copy()
df_gcs = df_GCS_MGB.copy()

# =========================
# CLEAN COLUMN NAME (optional but safer)
# =========================
df_gcs = df_gcs.rename(columns={
    "R CPN GLASGOW COMA SCALE SCORE": "GCSValue"
})

# =========================
# EXACT EEG SESSION MATCH
# (PatientID + EEG start + EEG end must ALL match)
# =========================
merged = df_gcs.merge(
    df_death_filtered[
        ["BDSPPatientID", "EEGBeginDTS", "EEGExamEndDTS"]
    ],
    on=["BDSPPatientID", "EEGBeginDTS", "EEGExamEndDTS"],
    how="inner"
)

# =========================
# AGGREGATE ALL GCS PER EEG SESSION
# =========================
gcs_agg = merged.groupby(
    ["BDSPPatientID", "EEGBeginDTS", "EEGExamEndDTS"]
).agg(
    GCSRecordDTS=("GCSRecordedDTS", list),
    GCSValues=("GCSValue", list)
).reset_index()

# =========================
# MERGE BACK INTO DEATH DATASET
# =========================
df_death_filtered = df_death_filtered.merge(
    gcs_agg,
    on=["BDSPPatientID", "EEGBeginDTS", "EEGExamEndDTS"],
    how="left"
)

# =========================
# DROP CASES WHERE GCS IS NOT FOUND
# =========================
df_death_with_GCS = df_death_filtered[~df_death_filtered['GCSValues'].isna()]

print('Total unique patients in death cohort with GCS recording available ==> ', df_death_with_GCS['BDSPPatientID'].nunique())


# ## **Visualize the distribution of patient from each site having GCS and death outcome**

# In[7]:


# =========================
# MAKE SURE BINARY (YES/NO -> 1/0) OR KEEP AS IS
# =========================
df = df_death_with_GCS.copy()

# if still string
if df["DiedInHospital"].dtype == "object":
    df["DiedInHospital"] = df["DiedInHospital"].map({"Yes": "Yes", "No": "No"})

# =========================
# STACKED TABLE: SITE x OUTCOME
# =========================
site_table = pd.crosstab(df["SiteID"], df["DiedInHospital"])

# ensure both columns exist
site_table = site_table.reindex(columns=["No", "Yes"], fill_value=0)

# =========================
# PLOT
# =========================
ax = site_table.plot(kind="bar", stacked=True, figsize=(10,6))

plt.title("In-Hospital Death Outcome by Site (Patients with successfully GCS data found)")
plt.xlabel("SiteID")
plt.ylabel("Number of Patients")

# annotate bars
for i, (_, row) in enumerate(site_table.iterrows()):
    bottom = 0
    for col in site_table.columns:
        val = row[col]
        if val > 0:
            plt.text(i, bottom + val/2, str(val), ha="center", va="center")
        bottom += val

plt.legend(title="DiedInHospital")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[8]:


# ----------------- VIsualization for whom GCS were not found --------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Filter rows where GCSValues is NaN
df_missing_gcs = df_death_filtered[df_death_filtered['GCSValues'].isna()]

# Step 2: Group by SiteID and DiedInHospital
site_counts = df_missing_gcs.groupby(['SiteID', 'DiedInHospital']).size().unstack(fill_value=0)

# Ensure both columns exist
if 'No' not in site_counts.columns:
    site_counts['No'] = 0
if 'Yes' not in site_counts.columns:
    site_counts['Yes'] = 0

# Sort by SiteID (optional)
site_counts = site_counts.sort_index()

# Step 3: Plot stacked bar chart
plt.figure(figsize=(10, 6))

plt.bar(site_counts.index, site_counts['No'], label='Alive (No)')
plt.bar(site_counts.index, site_counts['Yes'], bottom=site_counts['No'], label='Died (Yes)')

# Step 4: Add value labels on bars
for i, site in enumerate(site_counts.index):
    alive = site_counts.loc[site, 'No']
    died = site_counts.loc[site, 'Yes']
    
    if alive > 0:
        plt.text(site, alive / 2, str(alive), ha='center', va='center', fontsize=9)
    if died > 0:
        plt.text(site, alive + died / 2, str(died), ha='center', va='center', fontsize=9)

# Step 5: Labels and styling
plt.xlabel("SiteID")
plt.ylabel("Count")
plt.title("Site-wise Outcome Distribution (where GCS is Missing for Those Subjects)")
plt.legend()
plt.xticks(rotation=45)

plt.show()


# ## **Get the sequential GCS for each subject (LOCF)**
# Example:
# 
# - Hour 1 model: [GCS_1_imputed]
# - Hour 2 model: [GCS_1_imputed, GCS_2]        ← real measurement
# - Hour 3 model: [GCS_1_imputed, GCS_2, GCS_2] ← Last Observation Carried Forward
# - Hour 4 model: [GCS_1_imputed, GCS_2, GCS_2, GCS_4] ← real

# In[9]:


import pandas as pd
import numpy as np

df = df_death_with_GCS.copy()

# ---------------------------------------------
# PARSE TIME (FULL PRECISION)
# ---------------------------------------------
df['EEGBeginDTS'] = pd.to_datetime(df['EEGBeginDTS'])
df['EEGExamEndDTS'] = pd.to_datetime(df['EEGExamEndDTS'])

df['GCSRecordDTS'] = df['GCSRecordDTS'].apply(pd.to_datetime)

# ---------------------------------------------
# BUILD FUNCTION
# ---------------------------------------------
def build_seq_gcs_time(row):

    eeg_start = row['EEGBeginDTS']

    gcs_times = list(row['GCSRecordDTS'])
    gcs_vals  = list(row['GCSValues'])

    if len(gcs_times) == 0:
        return [], 0

    # sort by time
    pairs = sorted(zip(gcs_times, gcs_vals), key=lambda x: x[0])
    gcs_times, gcs_vals = zip(*pairs)

    gcs_times = list(gcs_times)
    gcs_vals  = list(gcs_vals)

    last_gcs_time = gcs_times[-1]

    # ---------------------------------------------
    # IMPORTANT: use FULL seconds precision
    # ---------------------------------------------
    delta_seconds = (last_gcs_time - eeg_start).total_seconds()

    if delta_seconds <= 0:
        return [], 0

    # convert to hourly bins (ceil to include partial hour)
    total_hours = int(np.ceil(delta_seconds / 3600))

    seq = []

    current_gcs = gcs_vals[0]
    j = 0

    # ---------------------------------------------
    # 1-BASED HOURLY GRID
    # ---------------------------------------------
    for h in range(1, total_hours + 1):

        current_time = eeg_start + pd.Timedelta(hours=h)

        # LOCF update
        while j < len(gcs_times) and gcs_times[j] <= current_time:
            current_gcs = gcs_vals[j]
            j += 1

        seq.append(current_gcs)

        # STOP RULE (NO EXTRAPOLATION)
        if current_time >= last_gcs_time:
            break

    return seq, len(seq)

# ---------------------------------------------
# APPLY
# ---------------------------------------------
df[['Seq_GCS', 'NoOfSeqGCS']] = df.apply(
    lambda row: pd.Series(build_seq_gcs_time(row)),
    axis=1
)

# ---------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------
df_SeqGCS_curation_final = df[[
    'BDSPPatientID',
    'SiteID',
    'DurationInHours',
    'DiedInHospital',
    'Seq_GCS',
    'NoOfSeqGCS'
]].copy()
mask = df_SeqGCS_curation_final['DurationInHours'] < df_SeqGCS_curation_final['NoOfSeqGCS']

# apply fix only to problematic rows
df_SeqGCS_curation_final.loc[mask, 'Seq_GCS'] = df_SeqGCS_curation_final.loc[mask, 'Seq_GCS'].apply(lambda x: x[1:] if len(x) > 1 else x)

# recompute length
df_SeqGCS_curation_final['NoOfSeqGCS'] = df_SeqGCS_curation_final['Seq_GCS'].apply(len)
# ---------------------------------------------
# SANITY CHECK
# ---------------------------------------------
df_SeqGCS_curation_final


# # **Seq-GCS-based model training metadata creation**

# ## **Filename compilation**

# In[10]:


# ------------ BATCH-1 ---------------------
batch1_data_path = DEATH_ROOT / "mortality_analysis" / "Cohort" / "Subject_inhospDied_label_batch1_MGB_BIDMC.csv"
ddd_all_label_batch1 = pd.read_csv(batch1_data_path)

# ------------ BATCH-2 ---------------------
batch2_data_path = DEATH_ROOT / "mortality_analysis" / "Cohort" / "Subject_inhospDied_label_batch2_MGB_BIDMC.csv"
ddd_all_label_batch2 = pd.read_csv(batch2_data_path)

# ---------- Combined Batch-1 and Batch-2 ---------------
ddd_all_label_combined = pd.concat(
    [ddd_all_label_batch1, ddd_all_label_batch2],
    ignore_index=True
)
# ------------------ Some of them appear in two sites (we just just consider them in one of the sites) -------------
multi_site_mask = (
    ddd_all_label_combined
    .groupby('BDSPPatientID')['SiteID']
    .nunique()
    .gt(1)
)

multi_site_ids = multi_site_mask[multi_site_mask].index

ddd_all_label_combined = ddd_all_label_combined[
    ~ddd_all_label_combined['BDSPPatientID'].isin(multi_site_ids) |
    ((ddd_all_label_combined['BDSPPatientID'].isin(multi_site_ids)) &
     (ddd_all_label_combined['SiteID'] == 'S0001'))
]

ddd_with_gcs = ddd_all_label_combined[ddd_all_label_combined['BDSPPatientID'].isin(df_death_with_GCS['BDSPPatientID'].unique())]
print('Total unique patients from cohort metadata == '+str(ddd_with_gcs['BDSPPatientID'].nunique())) 

# -------------------------------- Death EEG compilation and arranging in order --------------------------
import os
import pandas as pd


######################### CAUTION (YOU NEED TO DOWNLOAD THE MORGOTH ACTIVATIONS DATA FROM AWS FROM DEATH FOLDER) ################
dir_bs = DEATH_ROOT / "mortality_analysis" / "MorgothActivations" / "BS"

files = [f for f in os.listdir(dir_bs) if f.endswith('.csv')]

df_death_eeg_file_compilation = pd.DataFrame({'Filename': files})
df_death_eeg_file_compilation['Filename'] = df_death_eeg_file_compilation['Filename'].str.replace('.csv', '', regex=False)

df_death_eeg_file_compilation['SiteID'] = df_death_eeg_file_compilation['Filename'].str[4:9]

def extract_pid(filename):
    part = filename.split('_')[0]
    pid_full = part.split('-')[1]
    pid = pid_full[5:]
    return pid

df_death_eeg_file_compilation['BDSPPatientID'] = df_death_eeg_file_compilation['Filename'].apply(extract_pid)

ddd_with_gcs = ddd_with_gcs.copy()
ddd_with_gcs['BDSPPatientID'] = ddd_with_gcs['BDSPPatientID'].astype(str)
df_death_eeg_file_compilation['BDSPPatientID'] = df_death_eeg_file_compilation['BDSPPatientID'].astype(str)

# Getting 'DiedInHospital' column from ddd_with_gcs
df_death_eeg_file_compilation['DiedInHospital'] = (
    df_death_eeg_file_compilation['BDSPPatientID']
    .map(ddd_with_gcs.set_index('BDSPPatientID')['DiedInHospital'])
)

# Sorting the rows by BDSPID
df_death_eeg_file_compilation = df_death_eeg_file_compilation.sort_values(
    by='BDSPPatientID'
).reset_index(drop=True)

# Sorting the filenames per person
df_death_eeg_file_compilation['seg_num'] = (
    df_death_eeg_file_compilation['Filename']
    .str.extract(r'_seg(\d+)$')
    .astype(int)
)

df_death_eeg_file_compilation = (
    df_death_eeg_file_compilation
    .sort_values(by=['BDSPPatientID', 'seg_num'])
    .drop(columns='seg_num')
    .reset_index(drop=True)
)

# Getting the total duration of EEG for each patient for that session that have been downloaded
duration = df_death_eeg_file_compilation.groupby('BDSPPatientID')['Filename'].transform('count')
df_death_eeg_file_compilation['DurationEEGInHourperSubject'] = duration

df_death_eeg_file_compilation = df_death_eeg_file_compilation[
    ~df_death_eeg_file_compilation['DiedInHospital'].isna()
]
print('Total unique patients after EEG extrcation == '
      +str(df_death_eeg_file_compilation['BDSPPatientID'].nunique())) 


# ## **Place the GCS values for each segment sequentially next to the filename**

# In[11]:


import pandas as pd
import numpy as np

# -------------------------------------------------
# COPY DATAFRAMES (avoid view issues)
# -------------------------------------------------
df_death_eeg_file_compilation = df_death_eeg_file_compilation.copy()
df_SeqGCS_curation_final = df_SeqGCS_curation_final.copy()

# -------------------------------------------------
# CLEAN PATIENT ID FUNCTION (CRITICAL FIX)
# -------------------------------------------------
def clean_id(x):
    try:
        return str(int(float(x)))  # fixes 111192769.0 -> 111192769
    except:
        return str(x).strip()

# apply to BOTH datasets
df_death_eeg_file_compilation['BDSPPatientID'] = df_death_eeg_file_compilation['BDSPPatientID'].apply(clean_id)
df_SeqGCS_curation_final['BDSPPatientID'] = df_SeqGCS_curation_final['BDSPPatientID'].apply(clean_id)

# -------------------------------------------------
# BUILD PATIENT-LEVEL MAP
# -------------------------------------------------
seq_map = (
    df_SeqGCS_curation_final[['BDSPPatientID', 'NoOfSeqGCS']]
    .drop_duplicates('BDSPPatientID')
)

# -------------------------------------------------
# CHECK OVERLAP (IMPORTANT DEBUG STEP)
# -------------------------------------------------
overlap = set(df_death_eeg_file_compilation['BDSPPatientID']) & set(seq_map['BDSPPatientID'])
print("Overlap count:", len(overlap))

# -------------------------------------------------
# MERGE (BROADCAST NoOfSeqGCS TO ALL ROWS)
# -------------------------------------------------
df_death_eeg_file_compilation = df_death_eeg_file_compilation.merge(
    seq_map,
    on='BDSPPatientID',
    how='left'
)

# -------------------------------------------------
# FINAL SANITY CHECKS
# -------------------------------------------------
print("Missing NoOfSeqGCS:",
      df_death_eeg_file_compilation['NoOfSeqGCS'].isna().sum())

print("Total rows:",
      len(df_death_eeg_file_compilation))

print("Unique patients:",
      df_death_eeg_file_compilation['BDSPPatientID'].nunique())

df = df_death_eeg_file_compilation.copy()

df['BDSPPatientID'] = df['BDSPPatientID'].astype(str)

df_death_eegcompilation_truncated = df.groupby('BDSPPatientID', group_keys=False).apply(
    lambda x: x.head(int(x['NoOfSeqGCS'].iloc[0]))
).reset_index(drop=True)


# In[12]:


df = df_death_eegcompilation_truncated.copy()
seq_df = df_SeqGCS_curation_final.copy()

# standardize key
df['BDSPPatientID'] = df['BDSPPatientID'].astype(str)
seq_df['BDSPPatientID'] = seq_df['BDSPPatientID'].astype(str)

# merge Seq_GCS list into EEG dataframe
df = df.merge(
    seq_df[['BDSPPatientID', 'Seq_GCS']],
    on='BDSPPatientID',
    how='left'
)

# -------------------------------------------------
# expand list into rows (1-to-1 mapping)
# -------------------------------------------------
df['row_rank'] = df.groupby('BDSPPatientID').cumcount()

df['GCSValue'] = df.apply(
    lambda x: x['Seq_GCS'][x['row_rank']],
    axis=1
)

df_death_eegcompilation_truncated_final = df.drop(columns=['row_rank', 'Seq_GCS'])

df_death_eegcompilation_truncated_final = df_death_eegcompilation_truncated_final.rename(columns={
    'NoOfSeqGCS': 'DurationGCSSyncedEEGInHour'})
df_death_eegcompilation_truncated_final = df_death_eegcompilation_truncated_final[
    df_death_eegcompilation_truncated_final['DurationEEGInHourperSubject'] >=
    df_death_eegcompilation_truncated_final['DurationGCSSyncedEEGInHour']
].reset_index(drop=True)

print('Total subjects with whom we will run GCS-Death Association Expt = ', 
      df_death_eegcompilation_truncated_final['BDSPPatientID'].nunique())


# ## **Visualize the EEG duration Histogram with label of death outcome**

# In[13]:


#  ---------------------------------- Patient EEG duration histogram with label of death -----------------------

# Take unique patients with duration and death label
df_duration_label = df_death_eegcompilation_truncated_final[['BDSPPatientID', 'DurationGCSSyncedEEGInHour', 'DiedInHospital']].drop_duplicates()

# Separate by label
died_yes = df_duration_label[df_duration_label['DiedInHospital'] == 'yes']['DurationGCSSyncedEEGInHour']
died_no  = df_duration_label[df_duration_label['DiedInHospital'] == 'no']['DurationGCSSyncedEEGInHour']

# Histogram bins
bins = range(1, df_duration_label['DurationGCSSyncedEEGInHour'].max()+2)

# Compute counts per bin
counts_no, _ = np.histogram(died_no, bins=bins)
counts_yes, _ = np.histogram(died_yes, bins=bins)


# Plot
plt.figure(figsize=(12,6))
plt.hist([died_no, died_yes], bins=bins, stacked=True, edgecolor='black', label=['Alive','Died in hospital'])
plt.xlabel('EEG Duration in sync with availble GCS recording (hours) in a single session')
plt.ylabel('Number of Patients\n(Each patient has only one randomly\nchosen session of out of multiple admission)')
plt.title('Stacked Histogram of EEG Duration per Patient by Outcome')
plt.legend()

# Sparse xticks
max_bin = df_duration_label['DurationGCSSyncedEEGInHour'].max()
plt.xlim(0,30)
plt.show()


# # **Sequential LR**

# In[14]:


import matplotlib.pyplot as plt
import numpy as np

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------- Seqential dataframe creation --------------------------
def build_patient_wide_sequential_df(
    df,
    hour_points,
    fea_name,
):
    patient_rows = []

    grouped = df.groupby('BDSPPatientID')
    total_patients = grouped.ngroups

    for pid, df_pid in tqdm(grouped, total=total_patients, desc='Processing patients'):
        df_pid = df_pid.sort_values('Filename').reset_index(drop=True)

        died_label = df_pid['DiedInHospital'].iloc[0]
        total_hours = int(df_pid['DurationGCSSyncedEEGInHour'].iloc[0])

        hour_features = []

        for _, row in df_pid.iterrows():
            fea_10min = row[fea_name]
            
            hour_features.append(fea_10min)

        hour_features = np.array(hour_features)

        row_dict = {
            'BDSPPatientID': pid,
            'DurationGCSSyncedEEGInHour': total_hours,
            'DiedInHospital': died_label
        }

        for h in hour_points:
            col_name = f'Hour_{h}_features'
            if h <= total_hours:
                row_dict[col_name] = hour_features[:h].reshape(-1)
            else:
                row_dict[col_name] = np.nan

        patient_rows.append(row_dict)

    return pd.DataFrame(patient_rows)


def plot_duration_histogram_horizontal(df_data, split_name, dataset_type):
    # Filter and drop duplicates
    df_sub = df_data[
        df_data['Split'] == split_name
    ][['BDSPPatientID', 'DurationGCSSyncedEEGInHour', 'DiedInHospital']].drop_duplicates()

    died_yes = df_sub[df_sub['DiedInHospital'] == 'yes']['DurationGCSSyncedEEGInHour']
    died_no  = df_sub[df_sub['DiedInHospital'] == 'no']['DurationGCSSyncedEEGInHour']

    # Define bins and centers
    max_val = int(df_sub['DurationGCSSyncedEEGInHour'].max())
    bins = range(1, max_val + 2)
    bin_centers = np.array(bins[:-1])

    counts_no, _ = np.histogram(died_no, bins=bins)
    counts_yes, _ = np.histogram(died_yes, bins=bins)

    plt.figure(figsize=(12, 7))
    
    # FIX: Set bar_height to less than 1.0 to create a gap (e.g., 0.8)
    bar_height = 0.8  
    
    bars_no = plt.barh(bin_centers, counts_no, height=bar_height, edgecolor='black', label='Alive')
    bars_yes = plt.barh(bin_centers, counts_yes, height=bar_height, left=counts_no, edgecolor='black', label='Died')

    # Add text labels inside bars
    for i in range(len(bin_centers)):
        if counts_no[i] > 0:
            plt.text(counts_no[i]/2, bin_centers[i], str(counts_no[i]), 
                     va='center', ha='center', color='white', fontweight='bold')
        if counts_yes[i] > 0:
            plt.text(counts_no[i] + counts_yes[i]/2, bin_centers[i], str(counts_yes[i]), 
                     va='center', ha='center', color='white', fontweight='bold')

    plt.xlabel('Number of Patients', fontsize=12)
    plt.ylabel('EEG Duration in sync with GCS recording(hours)\n(Each patient has only one randomly chosen session)', fontsize=12)
    plt.title(f'Stacked Horizontal Histogram of EEG Duration per Patient by Outcome ({split_name})', fontsize=14)
    plt.legend()

    # Y-axis ticks and limits
    plt.yticks(range(1, max_val + 1, max(1, max_val // 10)))
    
    if dataset_type == 'full dataset':
        plt.ylim(0, 51)
    else:
        plt.ylim(0, 26) # Starting from 0 ensures the first bar isn't cut off

    plt.grid(axis='x', linestyle='--', alpha=0.7) # Optional: adds grid for easier reading
    plt.tight_layout()
    plt.show()


# In[15]:


import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib
from tqdm import tqdm

current = Path(__file__).resolve()
NESI_ROOT = None
for parent in current.parents:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI folder not found")

def Seq_LR_kfold(df_death_data, seeds, kfold, model_type, hour_points, fea_name):
    base_model_dir = NESI_ROOT / "DeathPrediction_NESIvsGCS" / "ModelCheckpoints"
    fold_results = []

    for fld in range(kfold):
        print('---------------------------------- Fold '+str(fld+1)+' ----------------------------------------------')
        
        # Subject-independent split
        df_split_helper = df_death_data[['BDSPPatientID', 'DiedInHospital']].drop_duplicates()
        train_subs, test_subs = train_test_split(
            df_split_helper['BDSPPatientID'].values,
            test_size=0.15,
            random_state=seeds[fld],
            stratify=df_split_helper['DiedInHospital'].values
        )
        df_death_data['Split'] = df_death_data['BDSPPatientID'].apply(
            lambda x: 'Train' if x in train_subs else 'Test'
        )

        df_death_train_lt24 = df_death_data[
            (df_death_data['Split'] == 'Train') &
            (df_death_data['DurationGCSSyncedEEGInHour'] < 24)
        ].reset_index(drop=True)
        
        df_death_test_lt24 = df_death_data[
            (df_death_data['Split'] == 'Test') &
            (df_death_data['DurationGCSSyncedEEGInHour'] < 24)
        ].reset_index(drop=True)

        # Plot histograms
        # plot_duration_histogram_horizontal(df_death_train_lt24, 'Train', '<23hr')
        # plot_duration_histogram_horizontal(df_death_test_lt24, 'Test', '<23hr')

        # Build sequential wide dataframe
        df_seq_wide_train = build_patient_wide_sequential_df(
            df_death_train_lt24,
            hour_points,
            fea_name
        )
        df_seq_wide_test = build_patient_wide_sequential_df(
            df_death_test_lt24,
            hour_points,
            fea_name
        )

        # Fold-specific folder
        fold_model_dir = Path(base_model_dir) / model_type / f"Fold{fld+1}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        # Train LR models and save
        results_lr = {}
        for h in tqdm(hour_points, desc=f'Training LR Fold {fld+1}'):
            col = f'Hour_{h}_features'
            train_sub = df_seq_wide_train[df_seq_wide_train[col].notna()]
            test_sub = df_seq_wide_test[df_seq_wide_test[col].notna()]

            if len(train_sub) == 0 or len(test_sub) == 0:
                continue

            X_train = np.vstack(train_sub[col].values)
            y_train = (train_sub['DiedInHospital'] == 'yes').astype(int).values
            X_test = np.vstack(test_sub[col].values)
            y_test = (test_sub['DiedInHospital'] == 'yes').astype(int).values

            clf = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                class_weight='balanced',
                C=0.5,
                l1_ratio=0.7,
                max_iter=5000,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)

            # Save model for this hour
            model_file = fold_model_dir / f"lr_hour_{h}.joblib"
            joblib.dump(clf, model_file)

            y_prob = clf.predict_proba(X_test)[:,1]
            y_pred = clf.predict(X_test)

            results_lr[h] = {
                'n_train': len(train_sub),
                'n_test': len(test_sub),
                'auroc': roc_auc_score(y_test, y_prob),
                'auprc': average_precision_score(y_test, y_prob),
                'acc': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'coef': clf.coef_,
                'intercept': clf.intercept_
            }

        fold_results.append(results_lr)

    print('All folds complete!')
    return fold_results


# ## **Model training**

# In[25]:


seeds = [10, 12, 42, 5, 30]
kfold = 5
hour_points = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]
model_type = "SeqLR_GCS"
fea_name = 'GCSValue'
results_lr_gcs = Seq_LR_kfold(df_death_eegcompilation_truncated_final, 
                                 seeds, kfold, model_type, hour_points, fea_name)


# # **NESI-based Death Prediction**

# In[20]:

################### CAUTION (YOU NEED TO DOWNLOAD THE MORGOTH ACTIVATIONS FROM AWS) #########################
# ------------------------------- EEG feature statistics collection -------------------------
slowing_folder_loc=DEATH_ROOT / "MorgothActivations" / "SLOWING"
focgen_folder_loc=DEATH_ROOT / "MorgothActivations" / "FOCGEN"
iiic_folder_loc=DEATH_ROOT / "MorgothActivations" / "IIIC"
nm_folder_loc=DEATH_ROOT / "MorgothActivations" / "NM"
bs_folder_loc=DEATH_ROOT / "MorgothActivations" / "BS"
sleep_folder_loc=DEATH_ROOT / "MorgothActivations" / "SLEEP"

def morgoth_10minfea_matrix_stat_for(filename):
    file_name_csv = filename + '.csv'

    slowing_fea_sub = pd.read_csv(os.path.join(slowing_folder_loc, file_name_csv))
    slowing_fea_sub = slowing_fea_sub.drop(columns='pred_class')
    slowing_fea_sub_val = slowing_fea_sub.values

    focgen_fea_sub = pd.read_csv(os.path.join(focgen_folder_loc, file_name_csv))
    focgen_fea_sub = focgen_fea_sub.drop(columns='pred_class')
    focgen_fea_sub_val = focgen_fea_sub.values

    iiic_fea_sub = pd.read_csv(os.path.join(iiic_folder_loc, file_name_csv))
    iiic_fea_sub = iiic_fea_sub.drop(columns='pred_class')
    iiic_fea_sub_val = iiic_fea_sub.values


    bs_fea_sub = pd.read_csv(os.path.join(bs_folder_loc, file_name_csv))
    bs_fea_sub_val = bs_fea_sub.values

    nm_fea_sub = pd.read_csv(os.path.join(nm_folder_loc, file_name_csv))
    nm_fea_sub_val = nm_fea_sub.values

    sleep_fea_sub = pd.read_csv(os.path.join(sleep_folder_loc, file_name_csv))
    sleep_fea_sub = sleep_fea_sub.drop(columns='pred_class')
    sleep_fea_sub_val = sleep_fea_sub.values
    
    sub_morgoth_fea = np.concatenate([sleep_fea_sub_val, nm_fea_sub_val,
                                      bs_fea_sub_val, focgen_fea_sub_val,
                                      slowing_fea_sub_val, iiic_fea_sub_val], axis=1)
    
    
    return sub_morgoth_fea


# ------------------------- NESI functions -------------------------------
#-------------------------- ResNet-GAP only model -------------------------------------
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # If channels differ → use 1x1 conv for skip
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        return F.relu(out)

class MORGOTH_ResNet1D_onlyGAP(nn.Module):
    def __init__(self, num_features, filters=None, use_logit=True):
        super().__init__()
        self.use_logit = use_logit
        

        if filters is None:
            filters = [64, 128, 128, 256, 256]

        # Initial conv
        self.conv0 = nn.Conv1d(num_features, filters[0], kernel_size=7, padding=3)
        self.bn0 = nn.BatchNorm1d(filters[0])
        self.pool0 = nn.MaxPool1d(kernel_size=2)

        # ResNet blocks
        blocks = []
        in_ch = filters[0]
        for out_ch in filters:
            blocks.append(ResidualBlock1D(in_ch, out_ch))
            blocks.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch
        self.resnet_layers = nn.Sequential(*blocks)

        # GAP but DON'T squeeze yet
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 40)
    def forward(self, x):
        if self.use_logit:
            eps = 1e-6
            x = torch.log((x + eps) / (1 - x + eps))

        x = x.permute(0, 2, 1)

        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.resnet_layers(x)

        # GAP: (B,C,1)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# ------------ NESI model ----------------
import torch.nn.functional as F
class EEGScoringModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=30, dropout=0.3):
        super(EEGScoringModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)  # scalar badness score
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# --------- LOAD TRIPLET MODEL ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
Triplet_model_path = NESI_ROOT / "model" / "ModelCheckpoints" / "ResNetGAP_BestModel.pth" 
Triplet_model_trained = MORGOTH_ResNet1D_onlyGAP(num_features=17)
Triplet_model_trained.load_state_dict(
    torch.load(Triplet_model_path,
               map_location=device,
               weights_only=True)
)
Triplet_model_trained = Triplet_model_trained.to(device)

# --------------- LOAD NESI MODEL -------------------
NESI_model_path = NESI_ROOT / "model" / "ModelCheckpoints" / "NESI_best_model.pth"
checkpoint = torch.load(
    NESI_model_path,
    map_location=device,
    weights_only=True
)

NESI_model_score = EEGScoringModel()
NESI_model_score.load_state_dict(checkpoint["model_state_dict"])
NESI_model_score = NESI_model_score.to(device)


# In[21]:


df_death_eegcompilation_truncated_final = df_death_eegcompilation_truncated_final.copy()
nesi_preds = []
Triplet_model_trained.eval()
NESI_model_score.eval()

for _, row in tqdm(df_death_eegcompilation_truncated_final.iterrows(),
                   total=len(df_death_eegcompilation_truncated_final),
                   desc="Predicting NESI"):
    fname = row['Filename']
    fea = morgoth_10minfea_matrix_stat_for(fname)
    x = torch.tensor(fea, dtype=torch.float32).unsqueeze(0).to(device)    
    with torch.no_grad():
        emb = Triplet_model_trained(x)
        score = NESI_model_score(emb)
        score = score.squeeze().item()

    nesi_preds.append(score)

nesi_preds= np.array(nesi_preds)
df_death_eegcompilation_truncated_final['NESI_Predicted'] = nesi_preds


# In[23]:


seeds = [10, 12, 42, 5, 30]
kfold = 5
hour_points = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]
model_type = "SeqLR_NESI"
fea_name = 'NESI_Predicted'
results_lr_NESI = Seq_LR_kfold(df_death_eegcompilation_truncated_final, 
                                 seeds, kfold, model_type, hour_points, fea_name)


# In[27]:


import pickle

# save at your suitable location
# with open('/home/ayush/Desktop/results_NESI_Death.pkl', 'wb') as f:
#     pickle.dump(results_lr_NESI, f)

# with open('/home/ayush/Desktop/results_GCS_Death.pkl', 'wb') as f:
#     pickle.dump(results_lr_gcs, f)

# print('All results saved !!!!!! GO SLEEP...')


# In[1]:


import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_pickle(filepath):
    """
    Load a Python object from a pickle file.
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


# ---------------- LOAD RESULTS ----------------
def load_pickle(filepath):
    """
    Load a Python object from a pickle file.
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj

current = Path(__file__).resolve()
NESI_ROOT = None
for parent in current.parents:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI folder not found")

# ---------------- LOAD RESULTS ----------------
gcs_death_result = NESI_ROOT / "DeathPrediction_NESIvsGCS" / "Results" / "results_GCS_Death.pkl"
nesi_death_result = NESI_ROOT / "DeathPrediction_NESIvsGCS" / "Results" / "results_NESI_Death.pkl"

results_lr_gcs = load_pickle(gcs_death_result)
results_lr_nesi = load_pickle(nesi_death_result)


# ---------------- SETTINGS ----------------
hour_points = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


# ---------------- FUNCTION ----------------
def compute_mean_ci(results_obj, metric='auroc'):
    
    metric_matrix = {h: [] for h in hour_points}

    # results_obj = list of folds
    for fold_result in results_obj:
        for h in hour_points:
            if h in fold_result:
                metric_matrix[h].append(fold_result[h][metric])

    means = []
    ci_lower = []
    ci_upper = []

    for h in hour_points:
        vals = np.array(metric_matrix[h])

        mean = vals.mean()
        std = vals.std(ddof=1)

        ci = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0

        means.append(mean)
        ci_lower.append(mean - ci)
        ci_upper.append(mean + ci)

    return means, ci_lower, ci_upper


# ---------------- COMPUTE ----------------
gcs_mean, gcs_low, gcs_up = compute_mean_ci(results_lr_gcs)
nesi_mean, nesi_low, nesi_up = compute_mean_ci(results_lr_nesi)


# ---------------- PLOT ----------------
plt.figure(figsize=(9,6))

# GCS
plt.plot(hour_points, gcs_mean, marker='o', label='GCS')
plt.fill_between(hour_points, gcs_low, gcs_up, alpha=0.2)

# NESI
plt.plot(hour_points, nesi_mean, marker='s', label='NESI')
plt.fill_between(hour_points, nesi_low, nesi_up, alpha=0.2)

plt.xlabel('Hour')
plt.ylabel('AUROC')
plt.title('AUROC vs Time')
plt.xticks(hour_points)

plt.grid(True)
plt.legend()

plt.show()