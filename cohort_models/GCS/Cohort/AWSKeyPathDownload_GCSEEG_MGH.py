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
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


# # **Load S0001 GCS tables and combine them**

# In[2]:
current = Path(__file__).resolve()
GCS_ROOT = None
for parent in current.parents:
    if parent.name == "GCS":
        GCS_ROOT = parent
        break

if GCS_ROOT is None:
    raise RuntimeError("GCS folder not found")



s0001_GCS_datapath = GCS_ROOT / "Cohort" / "GCS_MGH_HarvardEEG_metadata.csv" 
df_s0001=pd.read_csv(s0002_GCS_datapath)
df_s0001.insert(0, 'SiteID', 'S0001') 

GCS_harvard_metadata=df_s0001
# # **AWS key end to end path creation for EEG download**

# In[3]:


import os
from tqdm import tqdm

############################# PASTE THE SUITABLE PATH in "base_path2" WHERE YOU WANNA DOWNLOAD THE EEGs ###############################

# cp for download key, ls for file location checking key
base_path1 = "aws s3 cp s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point/EEG/bids/"
base_path2 = '/home/ayush/Desktop/GCS_EEGs/' # Paste you destination folder where you want to download the EEGs
all_paths = []

for pid in tqdm(GCS_harvard_metadata["BDSPPatientID"].unique(), desc="Building S3 paths"):
    df_pid = GCS_harvard_metadata[GCS_harvard_metadata['BDSPPatientID'] == pid]

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
# with open("/home/ayush/Desktop/GitHub-YAMA/GCS/Cohort/MGH_GCS_download_AWS.txt", "w") as f:
#     for p in all_paths:
#         f.write(p + "\n")



