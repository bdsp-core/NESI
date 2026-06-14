#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import re
from pathlib import Path


# In[25]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lifelines import KaplanMeierFitter
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
def get_n_percent2(df, id_col, target_col, simplify_func=None):
    """
    Returns counts in 'N (%)' format for a target column grouped by unique IDs.

    Parameters:
    - df: pandas DataFrame
    - id_col: column with unique patient IDs
    - target_col: column to summarize (e.g., 'SexDSC', 'PatientRace')
    - simplify_func: optional function to simplify values (like race/ethnicity)

    Returns:
    - pandas Series with index as categories and values as 'N (%)' strings
    """
    # Get first occurrence per patient
    per_patient = df.groupby(id_col)[target_col].first().reset_index()
    
    # Apply simplification if provided
    if simplify_func:
        per_patient[target_col] = per_patient[target_col].apply(simplify_func)
    
    # Count values
    counts = per_patient[target_col].value_counts().sort_index()
    
    # Total patients
    total = counts.sum()
    
    # Format as 'N (%)'
    n_percent = counts.apply(lambda x: f"{x} ({x/total*100:.1f}%)")
    
    return n_percent

def get_n_percent(df, id_col, target_col, simplify_func=None):
    """
    Returns counts in 'N (%)' format for a target column grouped by unique IDs.
    Parameters:
    - df: pandas DataFrame
    - id_col: column with unique patient IDs
    - target_col: column to summarize (e.g.,
                                            Sex
                                            Race  
                                            Ethnicity  
                                            Ethnic Background
                                            Marital Status)
    - simplify_func: optional function to simplify values (like race/ethnicity)

    Returns:
    - pandas Series with index as categories and values as 'N (%)' strings
    """
    # unique patients
    per_patient = df.groupby(id_col)[target_col].first().reset_index()
    
    # Apply simplification if provided
    if simplify_func:
        per_patient[target_col] = per_patient[target_col].apply(simplify_func)
    
    # Count values, total sunjects and percent
    counts = per_patient[target_col].value_counts().sort_index()
    total = counts.sum()
    n_percent = counts.apply(lambda x: f"{x} ({x/total*100:.1f}%)")
    print('Total: '+str(total))
    return n_percent


def simplify_race(race):
    if pd.isna(race) or race.strip() == '' or 'Unavailable' in race or 'Declined' in race or 'not listed' in race:
        return 'Unknown'

    # Split multiple categories by semicolon
    parts = [p.strip() for p in race.split(';')]

    if len(parts) > 1:
        return 'More than one race'

    # Single race cases
    race_clean = parts
    if 'Black or African American' in race_clean:
        return 'Black or African American'
    elif 'Black or African American, Unknown or not disclosed' in race_clean:
        return 'Black or African American'
    
    elif 'Black or African American, White' in race_clean:
        return 'More than one race'
    elif 'White, Unknown or not disclosed' in race_clean:
        return 'More than one race'
    
    elif 'White' in race_clean:
        return 'White'
    elif 'Asian' in race_clean:
        return 'Asian'

    elif 'American Indian or Alaska Native' in race_clean:
        return 'American Indian or Alaska Native'
      
    else:
        return 'Unknown'


# In[12]:


current = Path(__file__).resolve()
ICANS_ROOT = None
for parent in current.parents:
    if parent.name == "ICANS":
        ICANS_ROOT = parent
        break

if ICANS_ROOT is None:
    raise RuntimeError("ICANS folder not found")


icans_metadata_path = ICANS_ROOT/ "Cohort" / "ICANS_cohort_metadata.csv"
df_icans=pd.read_csv(icans_metadata_path)

print(
    'Total Unique subjects in ICANS cohort ==',
    df_icans['StudyID'].nunique()
)


# In[14]:


# # Apply filter
df_icans['Age'] = pd.to_numeric(df_icans['Age'], errors='coerce')
df_icans = df_icans[df_icans['Age'] >= 18].copy()

print(
    'Total Unique subjects in ICANS cohort ==',
    df_icans['StudyID'].nunique()
)


# # **Age**

# In[20]:


print("#######################################################")
print("#                    AGE                              #")
print("#######################################################")
# One age per unique subject
sub_age = (
    df_icans[['StudyID', 'Age']]
    .drop_duplicates(subset='StudyID')
)

ages = sub_age['Age'].dropna()

median_age = ages.median()
q1_age = ages.quantile(0.25)
q3_age = ages.quantile(0.75)

print(f'Age at Visit (median [Q1, Q3]): {median_age:.1f} [{q1_age:.1f}, {q3_age:.1f}]')


# # **Sex**

# In[21]:


print('#                    SEX                              #')
print('#######################################################')

# One row per subject
sub_sex = (
    df_icans[['StudyID', 'Gender']]
    .drop_duplicates(subset='StudyID')
)

# Merge NaN into Unknown category
sub_sex['Gender'] = sub_sex['Gender'].fillna('Unknown or not disclosed')

n_subjects = sub_sex['StudyID'].nunique()

print(f"Total Unique subjects = {n_subjects}\n")

sex_counts = sub_sex['Gender'].value_counts()

for sex, count in sex_counts.items():
    pct = 100 * count / n_subjects
    print(f"{sex}: {count} ({pct:.1f}%)")


# # **Race**

# In[27]:


# -------------------------------RACE----------------------------
print('#######################################################')
print('#                    PATIENT RACE                     #')
print('#######################################################')

df_race_icans = (
    df_icans
    .groupby('StudyID')[['PatientRaceCD']]
    .first()
    .reset_index()
)
race_map = {
    1.0: 'White',
    4.0: 'Asian',
    2.0: 'Black'
}
df_race_icans['Race'] = df_race_icans['PatientRaceCD'].map(race_map)
# Everything not mapped → Other/Unknown
df_race_icans['Race'] = df_race_icans['Race'].fillna('Other/Unknown')
# Summary
race_summary_icans = get_n_percent(df_race_icans, 'StudyID', 'Race')

print(race_summary_icans)
print('\n')


# # **Recorded Death Information**

# In[29]:


print('#######################################################')
print('#            Recorded Death Information               #')
print('#######################################################')

death_per_patient = (
    df_icans
    .groupby('StudyID')['DeathDate']
    .first()
    .reset_index()
)

# Convert to datetime and keep only date
death_per_patient['DeathDate'] = pd.to_datetime(death_per_patient['DeathDate'], errors='coerce').dt.date

# Create a summary column
death_per_patient['DeathStatus'] = death_per_patient['DeathDate'].apply(
    lambda x: 'Died (Death date known)' if pd.notna(x) else 'Unknown date of death'
)

# Count patients and calculate percent
counts = death_per_patient['DeathStatus'].value_counts()
percent = death_per_patient['DeathStatus'].value_counts(normalize=True) * 100

# Combine counts and percent into one string
death_summary = counts.astype(str) + ' (' + percent.round(1).astype(str) + '%)'

print(f"Total patients: {len(death_per_patient)}\n")
print(death_summary)


# # **Diagnosis**

# In[33]:


print('#######################################################')
print('#               DIAGNOSIS-MALIGNANCY                  #')
print('#######################################################')
# Keep one row per patient
malignancy_per_patient = (
    df_icans
    .groupby('StudyID')['Malignancy  (ALL, DLBCL, MM)']
    .first()
    .reset_index()
)

# Clean text
malignancy_per_patient['Malignancy_clean'] = (
    malignancy_per_patient['Malignancy  (ALL, DLBCL, MM)']
    .str.strip()
    .str.lower()
)

# Map to paper categories
diagnosis_map = {
    'dlbcl': 'DLBCL',
    'dl bcl': 'DLBCL',
    'd_lbcl': 'DLBCL',

    'pmbcl': 'PMBCL',
    'primary mediastinal large b cell lymphoma': 'PMBCL',
    'mediastinal large b cell lymphoma': 'PMBCL',

    'mantle cell lymphoma': 'MCL',
    'mcl': 'MCL',

    'follicular lymphoma': 'FL',
    'fl': 'FL',

    'mzl': 'MZL',

    'b-all': 'B-ALL',
    'b_all': 'B-ALL',

    'aggressive bnhl': 'Aggressive',
    'nhl': 'Aggressive',
    "burkitt's lymphoma": 'Aggressive',
}

# Map and fill unmapped as Indolent
malignancy_per_patient['Diagnosis_group'] = (
    malignancy_per_patient['Malignancy_clean']
    .map(diagnosis_map)
    .fillna('Indolent')
)

# Optional: order categories like paper
order = ['DLBCL','PMBCL','MCL','FL','MZL','B-ALL','Aggressive','Indolent']
malignancy_per_patient['Diagnosis_group'] = pd.Categorical(
    malignancy_per_patient['Diagnosis_group'],
    categories=order,
    ordered=True
)

# Summary
diagnosis_summary = get_n_percent(malignancy_per_patient, 'StudyID', 'Diagnosis_group')
print(diagnosis_summary)

