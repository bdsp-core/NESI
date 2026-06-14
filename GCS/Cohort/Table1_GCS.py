#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import re
from pathlib import Path


# # **Load the cohort metadata file**

# In[4]:


current = Path(__file__).resolve()
GCS_ROOT = None
for parent in current.parents:
    if parent.name == "GCS":
        GCS_ROOT = parent
        break

if GCS_ROOT is None:
    raise RuntimeError("GCS folder not found")


if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

DIAG_ROOT = None
for parent in current.parents:
    if (parent / "DiagnosisMetadtafiles").exists():
        DIAG_ROOT = parent
        break

if DIAG_ROOT is None:
    raise RuntimeError("DIAG folder not found")  


metadata_s0001_path = GCS_ROOT / "Cohort" / "GCS_MGH_HarvardEEG_metadata.csv"
metadata_s0002_path = GCS_ROOT / "Cohort" / "GCS_BWH_HarvardEEG_metadata.csv"
metadata_i0002_path = GCS_ROOT / "Cohort" / "GCS_BIDMC_HarvardEEG_metadata.csv"

df_s0001=pd.read_csv(metadata_s0001_path)
df_s0002=pd.read_csv(metadata_s0002_path)

GCS_EEG_all_df_censor_mgb=pd.concat([df_s0002, df_s0001], ignore_index=True)

df_GCS_BIDMC=pd.read_csv(metadata_i0002_path)
subs_gcs_bidmc=df_GCS_BIDMC['BDSPPatientID'].unique()

GCS_EEG_all_df_censor_mgb['SiteID'] = 'S0001'
df_GCS_BIDMC['SiteID'] = 'S0002'


# In[5]:


# Mapping from BIDMC column names to standard column names
col_mapping = {
    
    'RecordedDTS': 'GCSRecordedDTS',
    'gcs_e': 'R CPN GLASGOW COMA SCALE EYE OPENING',
    'gcs_v': 'R CPN GLASGOW COMA SCALE BEST VERBAL RESPONSE',
    'gcs_m': 'R CPN GLASGOW COMA SCALE BEST MOTOR RESPONSE',
    'GCS_total': 'R CPN GLASGOW COMA SCALE SCORE',
    't0_admit': 'HospitalAdmitDTS',
    't1_discharge': 'HospitalDischargeDTS',
    'death_date': 'DateOfDeath',
    'censor_date': 'CensorDate',
    'race': 'PatientRace',
    'ethnicity': 'EthnicGroupDSC',
    'age': 'AgeAtVisit',
    'sex': 'SexDSC',
    # Keep other columns same if needed
    'BDSPPatientID': 'BDSPPatientID',
    'SessionID': 'SessionID',
    'BidsFolder': 'BidsFolder',
    'CreationTime': 'CreationTime',
    'StartTime': 'EEGBeginDTS',  # If you want EEG timing columns
    'EndTime': 'EEGExamEndDTS',
    'DurationInSeconds': 'DurationInSeconds',
    'SiteID': 'SiteID'
}

# Apply mapping to your BIDMC DataFrame
df_GCS_BIDMC_renamed = df_GCS_BIDMC.rename(columns=col_mapping)
# Ensure SexDSC column exists (after renaming)
if 'SexDSC' in df_GCS_BIDMC_renamed.columns:
    df_GCS_BIDMC_renamed['SexDSC'] = df_GCS_BIDMC_renamed['SexDSC'].map({
        'F': 'Female',
        'M': 'Male'
    })


# In[6]:


# Keep needed columns
columns_need = [
    'SiteID',
    'BDSPPatientID',
    'HospitalAdmitDTS',
    'DateOfDeath',
    'CensorDate',
    'SexDSC',
    'PatientRace',
    'EthnicGroupDSC',
    'R CPN GLASGOW COMA SCALE SCORE',
    'AgeAtVisit'
]
gcs_mgb = GCS_EEG_all_df_censor_mgb[columns_need].copy()
gcs_bidmc = df_GCS_BIDMC_renamed[columns_need].copy()
gcs_all_sites = pd.concat([gcs_mgb, gcs_bidmc], ignore_index=True)


# # **Total no of unique patients**

# In[7]:


print(
    'Total Unique subjects in GCS cohort ==',
    gcs_all_sites['BDSPPatientID'].nunique()
)


# # **Age**

# In[16]:


print("#######################################################")
print("#                    AGE                              #")
print("#######################################################")
# One age per unique subject
sub_age = (
    gcs_all_sites[['BDSPPatientID', 'AgeAtVisit']]
    .drop_duplicates(subset='BDSPPatientID')
)

ages = sub_age['AgeAtVisit'].dropna()

median_age = ages.median()
q1_age = ages.quantile(0.25)
q3_age = ages.quantile(0.75)

print(f'Age at Visit (median [Q1, Q3]): {median_age:.1f} [{q1_age:.1f}, {q3_age:.1f}]')


# In[ ]:





# In[17]:


print('#######################################################')
print('#                    SEX                              #')
print('#######################################################')
# One row per subject
sub_sex = (
    gcs_all_sites[['BDSPPatientID', 'SexDSC']]
    .drop_duplicates(subset='BDSPPatientID')
)

n_subjects = sub_sex['BDSPPatientID'].nunique()

print(f"Total Unique subjects = {n_subjects}\n")

sex_counts = sub_sex['SexDSC'].value_counts(dropna=False)

for sex, count in sex_counts.items():
    pct = 100 * count / n_subjects
    print(f"{sex}: {count} ({pct:.1f}%)")


# # **PatientRace**

# In[18]:


print('#######################################################')
print('#                    PATIENT RACE                     #')
print('#######################################################')

# ============================================================
# RACE HARMONIZATION
# ============================================================
RACE_TOKENS = {
    "White": ["white"],
    "Black/African American": ["black", "african"],
    "Asian": ["asian"],
}

def map_race(x):
    if pd.isna(x):
        return "Other/Unknown"

    s = str(x).strip().lower()

    if s == "" or s in [
        "nan", "none", "unknown", "declined",
        "not reported", "unavailable"
    ]:
        return "Other/Unknown"

    if any(k in s for k in [
        "more", "multiple", "mixed",
        "biracial", "multiracial", "multi"
    ]):
        return "More than one race"

    parts = re.split(r"[;,/|]+|\band\b|\bor\b", s)

    found = set()
    for p in parts:
        p = p.strip()
        if not p:
            continue

        for race, keys in RACE_TOKENS.items():
            if any(k in p for k in keys):
                found.add(race)

    if len(found) >= 2:
        return "More than one race"
    elif len(found) == 1:
        return next(iter(found))
    else:
        return "Other/Unknown"


# ============================================================
# UNIQUE SUBJECT-LEVEL RACE SUMMARY
# ============================================================

# Take one race entry per unique subject
sub_race = (
    gcs_all_sites[['BDSPPatientID', 'PatientRace']]
    .drop_duplicates(subset='BDSPPatientID')
)

# Harmonize race
race_cat = sub_race['PatientRace'].apply(map_race)

# Count categories
race_counts = race_cat.value_counts()

# Total unique subjects
n_subjects = sub_race['BDSPPatientID'].nunique()

print(f"Race Summary (N = {n_subjects})\n")

for race in [
    "White",
    "Black/African American",
    "Asian",
    "More than one race",
    "Other/Unknown"
]:
    count = race_counts.get(race, 0)
    pct = 100 * count / n_subjects

    print(f"{race}: {count} ({pct:.1f}%)")


# # **Recorded Death Information**

# In[21]:


print('#######################################################')
print('#            Recorded Death Information               #')
print('#######################################################')

# One row per unique subject
sub_df = gcs_all_sites[['BDSPPatientID', 'DateOfDeath']].drop_duplicates(
    subset='BDSPPatientID'
)

total_subjects = sub_df['BDSPPatientID'].nunique()

# Define groups
has_death = sub_df['DateOfDeath'].notna()
censored = sub_df['DateOfDeath'].isna()

n_death = has_death.sum()
n_censored = censored.sum()

print(f"Survival Status Summary (N = {total_subjects})\n")

print(f"Death recorded: {n_death} ({100*n_death/total_subjects:.1f}%)")
print(f"Censored (no death date): {n_censored} ({100*n_censored/total_subjects:.1f}%)")


# # **Diagnosis**

# In[23]:


# -------------------------- Process Diagnosis files ----------------------------
diag_files = [
    DIAG_ROOT / "DiagnosisMetadtafiles"/ "File1.xlsx",
    DIAG_ROOT / "DiagnosisMetadtafiles"/ "File2.xlsx",
    DIAG_ROOT / "DiagnosisMetadtafiles"/ "File3.xlsx",
]
def pct(n, denom):
    if denom == 0:
        return 0.0
    return round(100.0 * n / denom, 1)

def norm_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s

def norm_cols(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
    )
    return df

def as_str_series(x, n_fallback=0):
    if x is None:
        return pd.Series([""] * n_fallback, dtype=str)
    return x.astype(str)

diag_dfs = []
for f in diag_files:
    df = pd.read_excel(f)
    df = norm_cols(df)
    df.columns = df.columns.str.lower()
    diag_dfs.append(df)

diagnosis = pd.concat(diag_dfs, ignore_index=True)

if "BDSPPatientID" not in diagnosis.columns:
    possible_id_cols = [c for c in diagnosis.columns if "patientid" in c]
    if len(possible_id_cols) == 0:
        raise ValueError(f"Could not find BDSPPatientID column. Columns: {list(diagnosis.columns)}")
    diagnosis = diagnosis.rename(columns={possible_id_cols[0]: "BDSPPatientID"})

if "diagnosis category" not in diagnosis.columns:
    possible_cat_cols = [c for c in diagnosis.columns if ("diagnosis" in c and "category" in c)]
    if len(possible_cat_cols) == 0:
        raise ValueError(f"Could not find diagnosis category column. Columns: {list(diagnosis.columns)}")
    diagnosis = diagnosis.rename(columns={possible_cat_cols[0]: "diagnosis category"})

diagnosis["BDSPPatientID"] = diagnosis["BDSPPatientID"].apply(norm_id)
diagnosis["diagnosis category"] = as_str_series(diagnosis["diagnosis category"]).str.strip().str.lower()

def map_dx(cat: str) -> str:
    c = (cat or "").strip().lower()

    if any(k in c for k in [
        "intracerebral", "ich", "hemorrhage", "ischemic", "stroke", "subdural", "sdh",
        "traumatic", "tbi", "brain tumor", "tumor",
        "seizure", "sz",
        "dement", "neuroinfection", "neuroinflamm", "neuro"
    ]):
        return "Neurological"

    if any(k in c for k in ["cardiac arrest", "arrest", "endocarditis", "cardiac", "circulatory"]):
        return "Cardiac/Circulatory"

    if any(k in c for k in [
        "sepsis",
        "toxic", "metabolic", "tme", "encephalopathy",
        "renal failure", "renal",
        "liver failure", "hepatic", "liver"
    ]):
        return "Systemic/Metabolic"

    if any(k in c for k in ["respiratory", "pneumonia"]):
        return "Respiratory"

    if any(k in c for k in ["altered mental status", "ams", "psychiatric", "other", "unknown", "unspecified", "na", "n/a"]):
        return "Unknown/Other"

    return "Unknown/Other"

diagnosis["DxGroup"] = diagnosis["diagnosis category"].apply(map_dx)



# In[28]:


print('#######################################################')
print('#                    DIAGNOSIS                        #')
print('#######################################################')
# ------------------------ Get stats about total patients -------------------------------
sub_death_all= gcs_all_sites['BDSPPatientID'].unique()
sub_death_all = np.char.replace(sub_death_all.astype(str), '.0', '')

print('\nTotal patients in YAMA-Death cohort: ' +str(len(sub_death_all)))
print('Patients Diagnosis found in combined diagnosis file: ' +str(diagnosis['BDSPPatientID'].nunique()))
common_subs = set(map(str, sub_death_all)).intersection(
    diagnosis['BDSPPatientID'].astype(str).unique()
)
print('Common patients:', len(common_subs))


# Clean IDs in death cohort
gcs_all_sites['BDSPPatientID'] = (
     gcs_all_sites['BDSPPatientID']
    .astype(str)
    .str.replace(r'\.0$', '', regex=True)
)

# Clean IDs in diagnosis dataframe
diagnosis['BDSPPatientID'] = (
    diagnosis['BDSPPatientID']
    .astype(str)
    .str.replace(r'\.0$', '', regex=True)
)

# Merge DxGroup onto the death cohort dataframe
gcs_all_sites =  gcs_all_sites.merge(
    diagnosis[['BDSPPatientID', 'DxGroup']],
    on='BDSPPatientID',
    how='left'
)

# Rename column and fill missing values
gcs_all_sites =  gcs_all_sites.rename(
    columns={'DxGroup': 'DiagnosisGroup'}
)

gcs_all_sites['DiagnosisGroup'] = (
     gcs_all_sites['DiagnosisGroup']
    .fillna('Unknown/Other')
)

# Subject-level counts per diagnosis
subject_counts = (
     gcs_all_sites
    .groupby('DiagnosisGroup')['BDSPPatientID']
    .nunique()
    .sort_values(ascending=False)
)

# Total unique subjects
total_subjects =  gcs_all_sites['BDSPPatientID'].nunique()

print("\nDiagnosis Group Summary (N = %d)\n" % total_subjects)

for diag, count in subject_counts.items():
    pct = 100 * count / total_subjects
    print(f"{diag}: {count} ({pct:.1f}%)")

