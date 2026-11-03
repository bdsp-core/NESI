#!/usr/bin/env python
# coding: utf-8

# In[62]:


import os
import re
import mne
import sys
import subprocess
mne.set_log_level(verbose='WARNING')

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
from scipy.signal import resample
from scipy.io import loadmat, savemat
import shutil

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*"
)
import logging
logging.disable(logging.CRITICAL)

import os


import contextlib
@contextlib.contextmanager
def suppress_all_output():
    devnull = os.open(os.devnull, os.O_WRONLY)

    old_stdout = os.dup(1)
    old_stderr = os.dup(2)

    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)

################### HELPER FUNCTIONS ###########################


# ---------------Preprocessing EDF files------------------------
def preprocessing_edf(edf_path, l_freq=0.5, h_freq=70.0, sfreq=200):
    # 定义 EEG 通道列表
    eeg_channels1 = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                     'T4', 'T6', 'O2']
    eeg_channels2 = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T7', 'P7', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                     'T8', 'P8', 'O2']

    # 检查文件是否存在
    if not os.path.exists(edf_path):
        print(f"{edf_path} does not exist")
        return None, None

    # 读取 EDF 文件
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True)
    except Exception as e:
        print(f"Failed to read {edf_path}: {e}")
        return None, None
    
    # Clean and standardize channel names
    new_channel_names = {
        ch: ch.upper().replace("EEG ", "").replace("-REF", "").strip()
        for ch in raw.ch_names
    }
    raw.rename_channels(new_channel_names)

    # 检查通道是否完整
    channels = raw.ch_names
    if set(channels).issuperset(set(eeg_channels1)):
        selected_channels = eeg_channels1
    elif set(channels).issuperset(set(eeg_channels2)):
        selected_channels = eeg_channels2
    else:
        print(f"{edf_path} does not contain all 19 required channels")
        return None, False

    # 选择通道并处理数据
    fs = raw.info['sfreq']
    

    raw_selected = raw.copy().pick_channels(selected_channels)
    raw_selected = raw_selected.resample(sfreq, n_jobs=5)
    raw_selected = raw_selected.filter(l_freq=l_freq, h_freq=h_freq)
    raw_selected = raw_selected.notch_filter(60.0)
    raw_selected = raw_selected.notch_filter(50.0)
    raw_selected.set_eeg_reference('average')

    # 提取数据和通道名称
    eegData = raw_selected.get_data(units='uV')
    eegData = EEG_clip(eegData)

    selected_channel_names = raw_selected.ch_names
    EEG_start_time = raw.info['meas_date'].replace(tzinfo=None)
    return eegData, selected_channel_names, fs, EEG_start_time

def EEG_clip(eeg_data):
    out_data = np.clip(eeg_data, -500, 500)
    return out_data


########################
# 		EXECUTE		   #
########################

#current = Path(__file__).resolve()
current = Path(__file__).absolute()
# Use this command if get any error related to saving: current = Path(__file__).absolute() instead of the previous line
CONTPRED_ROOT = None

for parent in current.parents:
    if parent.name == "ExternalValidation":
        CONTPRED_ROOT = parent
        break

if CONTPRED_ROOT is None:
    raise RuntimeError("ExternalValidation folder not found.")


EEG_ROOT = CONTPRED_ROOT / "EDF_EEGS"
edf_files = sorted(EEG_ROOT.glob("*.edf"))
if not edf_files:
    raise FileNotFoundError(f"No .edf files found in {EEG_ROOT}")

for i in tqdm(range(len(edf_files)), total=len(edf_files)):
    # GET EDF FILE
    edf_path = edf_files[i]  
    edf_filename = edf_path.stem

    print('*'*80)
    print(f'Working with: {edf_filename}.edf') 
    print('*'*80)

    try:
        # Erika's EEG Metadata
        eeg_metadata_path = CONTPRED_ROOT / "KIMCHI_FINAL_METADATA.xlsx"
        df_eegmetadata = pd.read_excel(eeg_metadata_path)
        df_eegmetadata2 =df_eegmetadata[df_eegmetadata['Filename'] == edf_filename]

        #CREATE MORGOTH ACTIVATION FOLDERS
        Morgoth_activation_root_path = CONTPRED_ROOT / "Products" / "MorgothActivations" 
        eeg_mat_save_path = CONTPRED_ROOT / "ProductsV2" / "ErikaSegmentEEG"
        if not os.path.exists(eeg_mat_save_path):
            os.makedirs(eeg_mat_save_path)

        # EEG PRE-PROCESSING AND SAVE IN MAT
        eegData, selected_channel_names, fs, EEG_start_time = preprocessing_edf(edf_path)
        # eeg_channels1 = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
        #                  'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        channels_cell = np.array(selected_channel_names, dtype=object).reshape(-1,1)
        seg_start_rel_s = df_eegmetadata2['seg_start_rel_s'].iloc[0]
        seg_end_rel_s = df_eegmetadata2['seg_end_rel_s'].iloc[0]

        start_sample = int(seg_start_rel_s * fs)
        end_sample = int(seg_end_rel_s * fs)
        ErikaEEGdata = eegData[:, start_sample:end_sample]
        filename_path = Path(eeg_mat_save_path) / f"{edf_filename}_ErikaSegment_{seg_start_rel_s}.mat"

        savemat(filename_path, {
            'data': ErikaEEGdata,
            'channels': channels_cell,
            'Fs': fs
        })

    except Exception as e:
        print(f"\n[SKIPPED] {edf_filename}.edf")
        print(f"Reason: {e}")
        continue