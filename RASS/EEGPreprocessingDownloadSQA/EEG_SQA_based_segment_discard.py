#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from datetime import timedelta
import scipy.io as sio
import mat73
import os
import re
import subprocess
from glob import glob
import scipy.io as sio
from tqdm import tqdm
import pandas as pd
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
from tqdm import tqdm
import shutil


# # **SQA block**

# In[5]:


def compute_nonusable_percentage(eeg_segment, fs):
    """
    eeg_segment: numpy array [channels x samples]
    fs: sampling frequency (Hz)
    Returns: percentage of non-usable data
    """

    window_sec = 60
    window_samples = window_sec * fs
    n_channels, total_samples = eeg_segment.shape

    n_windows = total_samples // window_samples
    flags = []   # 1 = good, 0 = bad

    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples
        segment_60s = eeg_segment[:, start:end]

        # ---------- Flat / no-data detection ----------
        ch_std = np.std(segment_60s, axis=1)
        flat_ratio = np.sum(ch_std < 0.1) / n_channels

        if flat_ratio > 0.40:
            flags.append(0)
            continue

        # ---------- High-noise detection ----------
        NOISE_VAR_THRESH = 300  # tune if needed
        noisy_ratio = np.sum(ch_std > NOISE_VAR_THRESH) / n_channels
        
        if noisy_ratio > 0.40:
            flags.append(0)
        else:
            flags.append(1)

    flags = np.array(flags)
    # ---------- Final non-usable percentage ----------
    nonusable_percentage = (np.sum(flags == 0) / len(flags)) * 100

    return nonusable_percentage


# # **Function that determines which 10 min EEGs to be discraded**

# In[18]:


import os
import numpy as np
import mat73
from tqdm import tqdm

def get_the_best_10mineeg_segs(path, threshold=40):
    """
    Parameters
    ----------
    path : str
        Folder containing .mat files

    threshold : float
        Reject file if non-usable percentage >= threshold

    Returns
    -------
    accepted_files : list
    discarded_files : list
    """

    accepted_files = []
    discarded_files = []

    mat_files = sorted(
        f for f in os.listdir(path)
        if f.endswith(".mat")
    )

    print(f"Found {len(mat_files)} MAT files")

    for fname in tqdm(mat_files, desc="Processing"):

        filepath = os.path.join(path, fname)

        try:
            d = mat73.loadmat(filepath)

            eeg_segment = np.asarray(d["data"])

            # Robust Fs conversion
            fs = int(np.asarray(d["Fs"]).squeeze())

            non_usability_index = compute_nonusable_percentage(
                eeg_segment,
                fs
            )

            # Reject if >=40% unusable
            if non_usability_index >= threshold:
                discarded_files.append(fname)
            else:
                accepted_files.append(fname)

        except Exception as e:
            print(f"\nFAILED: {fname}")
            print(f"ERROR : {e}")
            discarded_files.append(fname)

    print("\n==========================")
    print(f"Total files     : {len(mat_files)}")
    print(f"Accepted files  : {len(accepted_files)}")
    print(f"Discarded files : {len(discarded_files)}")
    print("==========================")

    return accepted_files, discarded_files


# In[19]:
#################################### CAUTION: IMPLEMENTATION ####################################
############# THE PATH WILL BE WHERE yOU HAVE STORED THE 10 MIN INDIVIDUAL EEGS ################
# For sanity check you can download the CAMS 10 min best EEGs and run on them you will see none will be discarded
#  AWS PATH: s3://bdsp-opendata-credentialed/yama/CAMS/GCS_EEG10minSegments
# or any other dataset's like RASS: s3://bdsp-opendata-credentialed/yama/RASS/RASS_EEG10minSegments
# here i have the CAMS best 10 min EEGs stored locally so i am using this path---> please change path as per your requirement!
accepted_files, discarded_files = get_the_best_10mineeg_segs(
    "/home/ayush/Desktop/CAM-S_dataset/CAMS_EEG_modified" ### THIS PATH IS JUST A DUMMY 
)

