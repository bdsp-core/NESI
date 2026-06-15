#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from pathlib import Path


# In[2]:


current = Path(__file__).resolve()
CAMS_ROOT = None
for parent in current.parents:
    if parent.name == "CAMS":
        CAMS_ROOT = parent
        break

if CAMS_ROOT is None:
    raise RuntimeError("CAMS folder not found")


cams_metadata_path = CAMS_ROOT / "Cohort" / "CAMS_cohort_metadata.csv"

df_cams=pd.read_csv(cams_metadata_path)


# # **Total no. of Unique patients**

# In[3]:


print(
    'Total Unique subjects in CAMS cohort ==',
    df_cams['BDSPPatientID'].nunique()
)


# # **Age**

# In[4]:


print("#######################################################")
print("#                    AGE                              #")
print("#######################################################")
# One age per unique subject
sub_age = (
    df_cams[['BDSPPatientID', 'Age ']]
    .drop_duplicates(subset='BDSPPatientID')
)

ages = sub_age['Age '].dropna()

median_age = ages.median()
q1_age = ages.quantile(0.25)
q3_age = ages.quantile(0.75)

print(f'Age at Visit (median [Q1, Q3]): {median_age:.1f} [{q1_age:.1f}, {q3_age:.1f}]')


# # **SexDSC**

# In[5]:


print('#                    SEX                              #')
print('#######################################################')

# One row per subject
sub_sex = (
    df_cams[['BDSPPatientID', 'Gender']]
    .drop_duplicates(subset='BDSPPatientID')
)

# Merge NaN into Unknown category
sub_sex['Gender'] = sub_sex['Gender'].fillna('Unknown or not disclosed')

n_subjects = sub_sex['BDSPPatientID'].nunique()

print(f"Total Unique subjects = {n_subjects}\n")

sex_counts = sub_sex['Gender'].value_counts()

for sex, count in sex_counts.items():
    pct = 100 * count / n_subjects
    print(f"{sex}: {count} ({pct:.1f}%)")


# # **PatientRace**

# In[6]:


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
    df_cams[['BDSPPatientID', 'Race']]
    .drop_duplicates(subset='BDSPPatientID')
)

# Harmonize race
race_cat = sub_race['Race'].apply(map_race)

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

# In[7]:


print('#######################################################')
print('#            Recorded Death Information               #')
print('#######################################################')

# One row per unique subject
sub_df = df_cams[['BDSPPatientID', 'DateOfDeath (shifted)']].drop_duplicates(
    subset='BDSPPatientID'
)

total_subjects = sub_df['BDSPPatientID'].nunique()

# Define groups
has_death = sub_df['DateOfDeath (shifted)'].notna()
censored = sub_df['DateOfDeath (shifted)'].isna()

n_death = has_death.sum()
n_censored = censored.sum()

print(f"Survival Status Summary (N = {total_subjects})\n")

print(f"Death recorded: {n_death} ({100*n_death/total_subjects:.1f}%)")
print(f"Censored (no death date): {n_censored} ({100*n_censored/total_subjects:.1f}%)")


# # **Diagnosis**

# In[10]:


print('#######################################################')
print('#                    DIAGNOSIS                        #')
print('#######################################################')

def map_dx(x):
    s = str(x).strip().lower()

    if s == "" or s in ["nan", "none"]:
        return "Unknown/Other"

    if any(k in s for k in [
        "stroke", "seizure", "status epilepticus", "epilepsy", "tbi",
        "brain", "intracerebral", "hemorrhage", "sdh", "neuro"
    ]):
        return "Neurological"

    if any(k in s for k in [
        "cardiac", "arrest", "heart", "stemi", "nstemi", "vf", "vt", "pea", "circulatory"
    ]):
        return "Cardiac/Circulatory"

    if any(k in s for k in [
        "sepsis", "metabolic", "renal", "kidney", "hepatic", "liver",
        "shock", "encephalopathy", "dka"
    ]):
        return "Systemic/Metabolic"

    if any(k in s for k in [
        "respiratory", "pneumonia", "copd", "asthma", "ards", "pulmonary", "hypox"
    ]):
        return "Respiratory"

    return "Unknown/Other"


def first_existing_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of these columns were found: {candidates}")
    return None


dx_col_1 = first_existing_col(df_cams, ["Primary dx"], required=False)
dx_col_2 = first_existing_col(df_cams, ["Principal problem"], required=False)
dx_col_3 = first_existing_col(df_cams, ["Reason for hosp admit"], required=False)

dx_source = pd.Series("", index=df_cams.index)

if dx_col_1 is not None:
    dx_source = dx_source.mask(dx_source.eq(""), df_cams[dx_col_1].fillna("").astype(str))
if dx_col_2 is not None:
    dx_source = dx_source.mask(dx_source.eq(""), df_cams[dx_col_2].fillna("").astype(str))
if dx_col_3 is not None:
    dx_source = dx_source.mask(dx_source.eq(""), df_cams[dx_col_3].fillna("").astype(str))


df_cams["DxGroup"] = dx_source.apply(map_dx)

# =========================
# UNIQUE PATIENT LEVEL
# =========================
patient_col = "BDSPPatientID"   # change if your column name is different

df_pat = df_cams[[patient_col, "DxGroup"]].dropna().drop_duplicates(subset=[patient_col])

dx_counts = df_pat["DxGroup"].value_counts()

total = dx_counts.sum()

print(f"\nTotal unique patients: {total}\n")

for k, v in dx_counts.items():
    pct = (v / total) * 100
    print(f"{k}: {v} ({pct:.1f}%)")

