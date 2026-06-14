#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import transforms
import os
import mne
import numpy as np
mne.set_log_level(verbose='WARNING')
from scipy.signal import butter, filtfilt, iirnotch
from scipy.signal import resample
import scipy.io
import subprocess
import glob
from tqdm import tqdm


# # **Load S0001 & S0002 RASS tables and combine them**

# In[2]:
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
df_s0001.insert(0, 'SiteID', 'S0001') 

df_s0002=pd.read_csv(s0002_RASS_datapath)
df_s0002.insert(0, 'SiteID', 'S0002') 

RASS_harvard_metadata=pd.concat([df_s0001, df_s0002], ignore_index=True)
print(RASS_harvard_metadata['BDSPPatientID'].nunique())


# # **AWS key end to end path creation for EEG download**

# In[3]:


import os
from tqdm import tqdm

############################# PASTE THE SUITABLE PATH in "base_path2" WHERE YOU WANNA DOWNLOAD THE EEGs ###############################

# cp for download key, ls for file location checking key
base_path1 = "aws s3 cp s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point/EEG/bids/"
base_path2 = '/home/ayush/Desktop/RASS_EEGs/' # Paste you destination folder where you want to download the EEGs
all_paths = []

for pid in tqdm(RASS_harvard_metadata["BDSPPatientID"].unique(), desc="Building S3 paths"):
    df_pid = RASS_harvard_metadata[RASS_harvard_metadata['BDSPPatientID'] == pid]

    site_id = df_pid['SiteID'].unique()

    base_path1m=base_path1+site_id[0]+'/'
    base_path2m=base_path2+site_id[0]+'/'
    
    # Get unique (BidsFolder, SessionID) pairs
    unique_pairs = df_pid[["BidsFolder", "SessionID"]].drop_duplicates()
    
    # Build full S3 paths
    paths = []
    for row in unique_pairs.itertuples(index=False):
        # ensure local folder exists
        local_folder = os.path.join(base_path2m, str(row.BidsFolder), f"ses-{row.SessionID}")
        os.makedirs(local_folder, exist_ok=True)

        # build the aws s3 cp command
        cmd = (
            base_path1m + str(row.BidsFolder) + "/ses-" + str(row.SessionID)
            + "/ " + local_folder + "/ --recursive"
        )
        paths.append(cmd)
    
    all_paths.extend(paths)



# In[ ]:


# Save the paths
# with open("/home/ayush/Desktop/MGB_EEGs/RASSEEG_download_AWS.txt", "w") as f:
#     for p in all_paths:
#         f.write(p + "\n")


# In[ ]:




