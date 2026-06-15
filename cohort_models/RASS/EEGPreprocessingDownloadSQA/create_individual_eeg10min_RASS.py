import os
from scipy.io import loadmat, savemat
import mat73
from tqdm import tqdm


######################## CAUTION #################################
base_dir = "/home/ayush/Desktop/MGB_EEGs/RASS_EEG_Data/S0002" # Path where the EEGs are downloaded using the code GCS_EEG_download_MGH or BWH.py
save_folder = os.path.expanduser("/home/ayush/Desktop/MGB_EEGs/RASS_ALL_EEG_10min/EEG_10min")  # Path where you will save the individual 10 min EEGs

eeg_channels1 = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
                 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
channels_cell = np.array(eeg_channels1, dtype=object).reshape(-1,1)
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
                RASS_all = data['RASS_recordings']

                n_segments = EEG_all.shape[0]
                base_name = os.path.basename(file_path).replace('.mat', '')

                for i in range(n_segments):
                    EEG_seg = EEG_all[i, :, :]
                    RASS_seg = RASS_all[:, i]

                    save_name = f"{base_name}_seg{i+1}.mat"
                    save_path = os.path.join(save_folder, save_name)

                    savemat(save_path, {
                        'data': EEG_seg,
                        'RASS_value': RASS_seg,
                        'channels': channels_cell,
                        'Fs':SR
                    })

                    # tqdm.write(f"Saved: {save_path}")
        else:
            tqdm.write("No .mat files found in this folder.")
else:
    print("No subfolders found in the directory.")