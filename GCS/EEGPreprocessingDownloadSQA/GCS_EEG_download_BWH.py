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


# Preprocessing EDF files
def preprocessing_edf(edf_path, l_freq=0.5, h_freq=70.0, sfreq=200):
    # 定义 EEG 通道列表
    eeg_channels1 = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                     'T4', 'T6', 'O2']
    eeg_channels2 = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T7', 'P7', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                     'T8', 'P8', 'O2']

    # 检查文件是否存在
    if not os.path.exists(edf_path):
        print (f"{edf_path} does not exist")
        return None, None

    # 读取 EDF 文件
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True)
    except Exception as e:
        print(f"Failed to read {edf_path}: {e}")
        return None, None

    # 数据裁剪
    # if raw.times[-1] > 3600:  # 超过 1 小时，裁剪到 1 小时
    #     raw = raw.crop(tmin=0, tmax=3600)

    # elif raw.times[-1] < 11:  # 少于 10 秒，跳过
    #     print(f"{edf_path} is too short (<= 10 minutes)")
    #     return None, None

    # 统一通道名称为大写
    new_channel_names = {ch_name: ch_name.upper() for ch_name in raw.ch_names}
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
    eegData=EEG_clip(eegData)

    selected_channel_names = raw_selected.ch_names  # 获取处理后的通道名称

    return eegData, selected_channel_names


def EEG_clip(eeg_data):
    out_data = np.clip(eeg_data, -500, 500)
    return out_data


# Extrcat 10 in EEG
def extrcat_10min_EEG_with_GCS(edf_path, df_sub, fs):
    # Load EEG data
    EEG, Channels = preprocessing_edf(edf_path, l_freq=0.5, h_freq=70.0, sfreq=fs)
    n_channels, n_times = EEG.shape
    
    window_sec = 600  # 10 minutes
    samples_per_window = int(window_sec * fs)

    # EEG start and end timestamps
    EEG_begin = pd.to_datetime(df_sub['EEGBeginDTS'].unique()[0])
    EEG_end   = pd.to_datetime(df_sub['EEGExamEndDTS'].unique()[0])

    # Create time index for EEG
    time_index = pd.date_range(start=EEG_begin, periods=n_times, freq=f"{1000/fs}ms")
    time_index = time_index.floor('s')  # round to nearest second

    # GCS times & values
    gcs_times = pd.to_datetime(df_sub['GCSRecordedDTS'].dropna().tolist()).floor('s')
    gcs_values = df_sub['R CPN GLASGOW COMA SCALE SCORE'].dropna().astype(int).tolist()

    # Containers
    X_segments, Y_labels, valid_gcs_times, ch_nms = [], [], [], []

    for t, val in zip(gcs_times, gcs_values):

        # Window: 10 minutes before GCS time
        start_time = t - pd.Timedelta(minutes=10)
        end_time   = t

        # ---- SKIP if window is outside available EEG ----
        if not (start_time >= EEG_begin and end_time <= EEG_end):
            # This avoids cases where GCS is too early (<10 min from start)
            # or too late (crosses EEG_end).
            continue

        # Find indices
        idx_start = np.searchsorted(time_index, start_time)
        idx_end   = np.searchsorted(time_index, end_time)

        # Number of samples available
        n_samples = idx_end - idx_start

        # ---- SKIP if not enough samples (not 10 min) ----
        if n_samples < samples_per_window:
            continue

        # Extract segment (handle small off-by-one)
        if n_samples == samples_per_window:
            EEG_segment = EEG[:, idx_start:idx_end]
        else:
            # Off-by-one case
            EEG_segment = EEG[:, idx_end - samples_per_window : idx_end]

        # Store
        X_segments.append(EEG_segment)
        Y_labels.append(val)
        valid_gcs_times.append(t)
        ch_nms.append(Channels)

    # Convert to arrays 
    X = np.stack(X_segments, axis=0) if len(X_segments) > 0 else np.empty((0,))
    Y = np.array(Y_labels)
    ch_all = np.array(ch_nms, dtype='object')

    return X, Y, ch_all


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


metadata_s0002_path = GCS_ROOT / "Cohort" / "GCS_BWH_HarvardEEG_metadata.csv"

df_s0002=pd.read_csv(metadata_s0002_path)
df_s0002.insert(0, 'SiteID', 'S0002') 

GCS_harvard_metadata= df_s0002
print(GCS_harvard_metadata['BDSPPatientID'].nunique())


AWS_key_path = GCS_ROOT / "Cohort" / "BWH_GCS_download_AWS.txt"
with open(AWS_key_path, "r") as f:
    all_paths = [line.strip() for line in f if line.strip()]



def download_extract_cleanup(all_paths, GCS_harvard_metadata, fs=200):
    """
    Downloads EEG data from AWS S3, extracts 10-min EEG segments around GCS events,
    saves them to .mat files, and deletes the original EDFs to save space.

    Parameters:
    -----------
    all_paths : list of str
        List of AWS S3 cp commands (constructed earlier).
    GCS_harvard_metadata : pd.DataFrame
        Main metadata DataFrame containing BDSPPatientID, SiteID, BidsFolder, SessionID, etc.
    fs : int, optional
        Sampling frequency for EEG extraction (default = 200 Hz)
    """

    for cmd in tqdm(all_paths[:], desc="Processing EEG downloads"): #0:100

        # Extract local folder path (everything after the last space in the command)
        local_folder = cmd.split()[-2]
        if not os.path.exists(local_folder):
            os.makedirs(local_folder, exist_ok=True)

        print(f"\n--- Downloading data to: {local_folder} ---")
        subprocess.run(cmd, shell=True, check=True)

        # Find 'eeg' subfolder inside local_folder
        eeg_folder = os.path.join(local_folder, "eeg")
        if not os.path.exists(eeg_folder):
            print(f"No 'eeg' folder found in {local_folder}, skipping...")
            continue

        # Find all EDF files
        edf_files = glob(os.path.join(eeg_folder, "*.edf"))
        if len(edf_files) == 0:
            print(f"No EDF files found in {eeg_folder}, skipping...")
            continue

        for edf_path in edf_files:
            print(f"Processing EDF: {edf_path}")

            # ------------------------------
            # Extract BDSPPatientID and SessionID from path
            # ------------------------------
            parts = local_folder.split(os.sep)
            sub_part = [p for p in parts if p.startswith("sub-S")][0]  # e.g. sub-S0001121077780
            session_part = [p for p in parts if p.startswith("ses-")][0]  # e.g. ses-7

            
            BDSPPatientID = int(sub_part[len("sub-S0001"):])
            session_id = int(session_part.replace("ses-", ""))

            # ------------------------------
            # Create df_sub for this patient/session
            # ------------------------------
            df_sub = GCS_harvard_metadata[
                (GCS_harvard_metadata["BDSPPatientID"] == BDSPPatientID)
                & (GCS_harvard_metadata["SessionID"] == session_id)
            ]

            if df_sub.empty:
                print(f"No metadata found for BDSPPatientID={BDSPPatientID}, SessionID={session_id}")
                continue

            # ------------------------------
            # Extract 10-min EEG segments using your function
            # ------------------------------

            try:
                EEG_10min_windows, GCS_recordings, all_channels = extrcat_10min_EEG_with_GCS(edf_path, df_sub, fs)
            except Exception as e:
                print(f"Error extracting from {edf_path}: {e}")
                continue

            # ------------------------------
            # Save the extracted data as .mat file
            # Filename format: sub-S0001121077780_ses-7_eeg_10min.mat
            # ------------------------------
            sub_id = sub_part.split("sub-")[-1]
            ses_id = session_part.split("ses-")[-1]
            mat_filename = os.path.join(os.path.dirname(os.path.normpath(local_folder)), f"sub-{sub_id}_ses-{ses_id}_eeg_10min.mat")

            sio.savemat(mat_filename, {
                "EEG_10min_windows": EEG_10min_windows,
                "GCS_recordings": GCS_recordings,
                "Channels_EEG": all_channels
            })

            print(f"Saved {mat_filename}")

            # ------------------------------
            # Delete original EDF file to save space
            # ------------------------------
            try:
            	shutil.rmtree(local_folder)
            	print(f"Deleted folder: {local_folder}")
            except Exception as e:
            	print(f"Could not delete folder {local_folder}: {e}")

        print(f"✅ Completed processing for: {mat_filename}\n")


download_extract_cleanup(all_paths, GCS_harvard_metadata, fs=200)