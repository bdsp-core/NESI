#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# # **Death Cohort**

# In[2]:
current = Path(__file__).resolve()
Death_ROOT = None
for parent in current.parents:
    if parent.name == "Death":
        Death_ROOT = parent
        break

if Death_ROOT is None:
    raise RuntimeError("Death folder not found")

# ------------ BATCH-1 ---------------------
metadata_path_death_batch1 = Death_ROOT / "Cohort" / "Subject_inhospDied_label_batch1_MGB_BIDMC.csv"

ddd_all_label_batch1 = pd.read_csv(metadata_path_death_batch1)

# ------------ BATCH-2 ---------------------
metadata_path_death_batch2 = Death_ROOT / "Cohort" / "Subject_inhospDied_label_batch2_MGB_BIDMC.csv"

ddd_all_label_batch2 = pd.read_csv(metadata_path_death_batch2)

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

print('Total unique patients from cohort metadata == '+str(ddd_all_label_combined['BDSPPatientID'].nunique())) 

# -------------------------------- Death EEG compilation and arranging in order --------------------------

BASE = Death_ROOT / "MorgothActivations"
dir_bs = BASE / "BS"

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

ddd_all_label_combined['BDSPPatientID'] = ddd_all_label_combined['BDSPPatientID'].astype(str)
df_death_eeg_file_compilation['BDSPPatientID'] = df_death_eeg_file_compilation['BDSPPatientID'].astype(str)

# Getting 'DiedInHospital' column from ddd_all_label_combined
df_death_eeg_file_compilation['DiedInHospital'] = (
    df_death_eeg_file_compilation['BDSPPatientID']
    .map(ddd_all_label_combined.set_index('BDSPPatientID')['DiedInHospital'])
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

#  ---------------------------------- Patient EEG duration histogram with label of death -----------------------

# Take unique patients with duration and death label
df_duration_label = df_death_eeg_file_compilation[['BDSPPatientID', 'DurationEEGInHourperSubject', 'DiedInHospital']].drop_duplicates()

# Separate by label
died_yes = df_duration_label[df_duration_label['DiedInHospital'] == 'yes']['DurationEEGInHourperSubject']
died_no  = df_duration_label[df_duration_label['DiedInHospital'] == 'no']['DurationEEGInHourperSubject']

# Histogram bins
bins = range(1, df_duration_label['DurationEEGInHourperSubject'].max()+2)

# Compute counts per bin
counts_no, _ = np.histogram(died_no, bins=bins)
counts_yes, _ = np.histogram(died_yes, bins=bins)

# Print counts per bin
# print("Bin\tAlive\tDied")
# for i in range(len(bins)-1):
#     print(f"{bins[i]}-{bins[i+1]-1}\t{counts_no[i]}\t{counts_yes[i]}")

# Plot
# plt.figure(figsize=(12,6))
# plt.hist([died_no, died_yes], bins=bins, stacked=True, edgecolor='black', label=['Alive','Died in hospital'])
# plt.xlabel('EEG Duration (hours) in a single session')
# plt.ylabel('Number of Patients\n(Each patient has only one randomly\nchosen session of out of multiple admission)')
# plt.title('Stacked Histogram of EEG Duration per Patient by Outcome')
# plt.legend()

# # Sparse xticks
# max_bin = df_duration_label['DurationEEGInHourperSubject'].max()
# plt.xticks(range(0, max_bin+1, max(1, max_bin//10)))
# plt.xlim(0,50)
# plt.show()


# # **Death data helper function**

# In[3]:


# ------------------------------- EEG feature statistics collection -------------------------

BASE = Death_ROOT / "MorgothActivations"

slowing_folder_loc = BASE / "SLOWING"
focgen_folder_loc  = BASE / "FOCGEN"
iiic_folder_loc    = BASE / "IIIC"
nm_folder_loc      = BASE / "NM"
bs_folder_loc      = BASE / "BS"
sleep_folder_loc   = BASE / "SLEEP"


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



def build_patient_wide_sequential_df(df, hour_points):
    patient_rows = []

    grouped = df.groupby('BDSPPatientID')
    total_patients = grouped.ngroups

    for pid, df_pid in tqdm(grouped, total=total_patients, desc='Processing patients'):
        df_pid = df_pid.sort_values('Filename').reset_index(drop=True)

        died_label = df_pid['DiedInHospital'].iloc[0]
        total_hours = int(df_pid['DurationEEGInHourperSubject'].iloc[0])

        hour_features = []
        for _, row in df_pid.iterrows():
            fea = morgoth_10minfea_matrix_stat_for(
                row['Filename']
            )
            hour_features.append(fea)

        hour_features = np.array(hour_features)
        # print(hour_features.shape)
        
        row_dict = {
            'BDSPPatientID': pid,
            'DurationEEGInHourperSubject': total_hours,
            'DiedInHospital': died_label
        }

        for h in hour_points:
            col_name = f'Hour_{h}_features'

            if h <= total_hours and (h-1) < len(hour_features):
                row_dict[col_name] = hour_features[h-1]
            else:
                row_dict[col_name] = np.nan

        patient_rows.append(row_dict)

    return pd.DataFrame(patient_rows)


# # **NESI Functions**

# In[4]:


#-------------------------- ResNet-GAP only model -------------------------------------
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

NESI_ROOT = Path(__file__).resolve().parents[1] / "NESI"

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


# In[5]:


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def generate_nesi_timeseries(df, hour_points, triplet_model, nesi_model, device):
    results = []
    triplet_model.eval()
    nesi_model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating NESI"):

        pid = row['BDSPPatientID']
        nesi_list = []
        
        for h in hour_points:
            features = row[f'Hour_{h}_features']
            if isinstance(features, float) and np.isnan(features):
                nesi_list.append(np.nan)
                continue

            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = triplet_model(x)
                score = nesi_model(emb)
                score = score.squeeze().item()

            nesi_list.append(score)
            
        results.append({
            'BDSPPatientID': pid,
            'NESI_list': nesi_list
        })

    return pd.DataFrame(results)


# # **Extract NESI for Died Patients**

# In[6]:


df_only_death = df_death_eeg_file_compilation[df_death_eeg_file_compilation['DiedInHospital']=='yes']
df_only_death_lt24 = df_only_death[
        df_only_death['DurationEEGInHourperSubject'] < 24
    ].reset_index(drop=True)
print('Total Unique people who died in hospital => ',df_only_death['BDSPPatientID'].nunique())
print('Total Unique people who died in hospital (With EEGs at max 24hr) => ',df_only_death_lt24['BDSPPatientID'].nunique())

hour_points = [2, 6, 10, 14, 18, 20]
df_seq_wide = build_patient_wide_sequential_df(
            df_only_death_lt24,
            hour_points
        )


# In[7]:


hrs = 23
df_died_with_eeg_Nhr = df_seq_wide[df_seq_wide['DurationEEGInHourperSubject']==hrs]
print('Total unique patienst who has at least {hrs} hours of EEG ==> ',
      df_died_with_eeg_Nhr['BDSPPatientID'].nunique())

NESI_death_results = generate_nesi_timeseries(df_died_with_eeg_Nhr, 
                                              hour_points, 
                                              Triplet_model_trained,
                                              NESI_model_score,
                                             device)

nesi_matrix = np.array(NESI_death_results["NESI_list"].tolist())


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_nesi_mean_ci_pro(NESI_death_results, hour_points, selected_hours=None):
    sns.set_style("ticks")
    
    nesi_matrix = np.array(NESI_death_results["NESI_list"].tolist())

    if selected_hours is not None:
        selected_hours = np.array(selected_hours)
        idx = [i for i, h in enumerate(hour_points) if h in selected_hours]
        hour_points_plot = np.array(hour_points)[idx]
        nesi_matrix = nesi_matrix[:, idx]
    else:
        hour_points_plot = np.array(hour_points)

    means = np.nanmean(nesi_matrix, axis=0)
    se = np.nanstd(nesi_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(nesi_matrix), axis=0))
    ci = 1.96 * se

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    ax.fill_between(hour_points_plot, means - ci, means + ci, 
                    color='#3498db', alpha=0.1, label='95% CI')

    ax.errorbar(hour_points_plot, means, yerr=ci, 
                fmt='o', color='#2980b9', ecolor='#2980b9', 
                elinewidth=1.5, capsize=4, capthick=1.5, 
                markersize=8, markerfacecolor='white', 
                markeredgewidth=2, zorder=3)

    ax.plot(hour_points_plot, means, color='#2980b9', 
            lw=2, alpha=0.8, zorder=2)

    ax.set_xlim(left=0)

    ax.set_title("NESI Temporal Trend for Patients Died In-hospital", fontsize=12, fontweight='bold', pad=20)
    ax.set_xlabel("Time (Hours)", fontsize=10)
    ax.set_ylabel("NESI Score", fontsize=10)
    
    plt.xticks(hour_points_plot, fontsize=10)
    plt.yticks(fontsize=10)
    
    sns.despine(offset=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    
    ax.legend(frameon=False, loc='best')

    plt.tight_layout()
    plt.show()

plot_nesi_mean_ci_pro(NESI_death_results, hour_points, selected_hours=[2, 6, 10, 14, 18, 20])


# In[ ]:




