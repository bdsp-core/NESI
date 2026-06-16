import numpy as np
import pandas as pd
import os
from scipy.io import loadmat, savemat
import mat73
from tqdm import tqdm


######################## CAUTION #################################
base_dir = "/media/ayush/Expansion/BWH_EEGs/S0002/" # Path where the EEGs are downloaded using the code GCS_EEG_download_MGH or BWH.py
save_folder = os.path.expanduser("/media/ayush/Expansion/GCS_EEGs_10min") # Path where you will save the individual 10 min EEGs


SR=200;

# Get all subfolders
subfolders = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
print("Subfolders found:", len(subfolders))

if subfolders:
    for target_folder in tqdm(subfolders, desc="Processing subfolders"):
        mat_files = [f for f in os.listdir(target_folder) if f.endswith('.mat')]
        

        if mat_files:
            for mat_file in mat_files:
                file_path = os.path.join(target_folder, mat_file)
                
                try:
                    data = mat73.loadmat(file_path)
                except:
                    data = loadmat(file_path)

                EEG_all = data['EEG_10min_windows']
                GCS_all = data['GCS_recordings']
                channels_all = data['Channels_EEG']

                n_segments = EEG_all.shape[0]
                base_name = os.path.basename(file_path).replace('.mat', '')

                for i in range(n_segments):
                    EEG_seg = EEG_all[i, :, :]
                    GCS_seg = GCS_all[:, i]
                    channel_seg = channels_all[i,:]
                    channels_seg_cell = np.array(channel_seg, dtype=object).reshape(-1,1)

                    save_name = f"{base_name}_seg{i+1}.mat"
                    save_path = os.path.join(save_folder, save_name)

                    savemat(save_path, {
                        'data': EEG_seg,
                        'GCS_value': GCS_seg,
                        'channels': channels_seg_cell,
                        'Fs':SR
                    })

        else:
            tqdm.write(f"No .mat files found in this folder: {target_folder}")
else:
    print("No subfolders found in the directory.")