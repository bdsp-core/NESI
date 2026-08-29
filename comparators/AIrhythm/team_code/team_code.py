#!/usr/bin/env python

# @author: Morteza Zabihi (morteza.zabihi@gmail.com) 
# Copyright (C) 2023 Morteza Zabihi  
# =============================================================================
# By accessing the code through the Physionet webpage and/or by installing, 
# copying, or otherwise using this software, you acknowledge and agree to be 
# bound by the terms and conditions of the attached "LICENSE.md" file. If you 
# do not agree to these terms and conditions, do not install, copy or use the 
# software.
# 
# Please note that we make no representation or warranty regarding the 
# suitability or fitness for any particular purpose of this licensed deliverable.
# The software is provided "as is" without any express or implied warranty 
# of any kind.
# 
# Furthermore, any use of the licensed deliverables must include the above
# disclaimer. For more detailed information on the terms and conditions 
# governing the use of this software, please refer to the "readme" file and 
# "LICENSE.md" file located on our GitHub repository.
# 
# Please review the terms and conditions carefully before using the software.
# If you have any questions or concerns about the terms and conditions, 
# please contact Morteza Zabihi (morteza.zabihi@gmail.com)
# =============================================================================

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

###############################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
###############################################################################
import time
import joblib
import numpy as np, os, sys
import mne
import scipy

from utilities_AIrhythm import *
from Compressors_features import *
from tangent_signiture import *
from class_robust import *
from class_model import *
from helper_code import *
from post_ant import post_ant
from frontal_features import frontal_features

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')
###############################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
###############################################################################

# Train your model. (info: about ~25832 EEG files for 72h congig)

def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    counfounder_features = list()
    timestamps = list()
    pids = list()
    outcomes = list()
    cpcs = list()
    
    for i in range(num_patients):
        # if verbose >= 2:
        #     print('    {}/{}...'.format(i+1, num_patients))
        print("*"*35)
        print("[INFO:] patient: {}/{}".format(i, num_patients))
        
        # t1 = time.time()
        current_features, current_counfounder_features1, \
            current_timestamps, current_pid =\
                get_features(data_folder, patient_ids[i])
        
        # print( time.time() - t1)        
        
        features.append(current_features)
        counfounder_features.append(current_counfounder_features1)
        timestamps.append(current_timestamps)
        pids.append(current_pid)
        
        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    counfounder_features = np.vstack(counfounder_features)
    pids = np.vstack(pids)
    
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)
    # -------------------------------------------------------------------------
    # Train and save the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')
        
    build_models_AIrhythm(features, timestamps, outcomes, cpcs, pids, 
                          counfounder_features, model_folder)

    if verbose >= 1:
        print('Done.')
    # -------------------------------------------------------------------------
    
    
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    
    # -------------------------------------------------------------------------
    # Extract features.
    current_features, _, \
        current_timestamps, current_pid = get_features(data_folder, patient_id)
    
    features_test = []
    features_test.append(current_features)
    
    timestamps_test = []
    timestamps_test.append(current_timestamps)
    
    pids = []
    pids.append(current_pid)
    # -------------------------------------------------------------------------    
    outcome, outcome_probability, cpc = helper_run_model_AIrhythm(models,
                                                                  pids, 
                                                                  features_test, 
                                                                  timestamps_test)
    # -------------------------------------------------------------------------
    return outcome, outcome_probability, cpc

###############################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
###############################################################################


# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 45.0]
   
    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    # if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
    #     data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')
    
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency,
                                       utility_frequency, n_jobs=4, 
                                       verbose='error')
    
    if utility_frequency is not None:
        data = mne.filter.notch_filter(data, sampling_frequency,
                                       utility_frequency//2, n_jobs=4, 
                                       verbose='error')
    
    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0],
                                  passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data
    row_means = np.nanmean(data, axis=1, keepdims=True)
    row_stds = np.nanstd(data, axis=1, keepdims=True)
    data = (data - row_means) / row_stds
    return data, int(resampling_frequency)
# =============================================================================


# Extract features.
def get_features(data_folder, patient_id):
    
    pid = int(patient_id.lstrip("0")) 
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    eeg_files = list_eeg_files(data_folder, patient_id)
    
    if len(eeg_files) > 0:
        eeg_files = np.sort(eeg_files)
        eeg_files = sort_strings_by_last_digits(eeg_files)
        num_recordings = len(eeg_files)
    # -------------------------------------------------------------------------
    # Extract patient features ------------------------------------------------
    patient_features = get_patient_features(patient_metadata)
    # age, sex1, hospital, rosc, ohca, shockable_rhythm, ttm
    counfounder_features = patient_features[[1, 2]]      # sex1, hospital
    patient_features = patient_features[[0, 3, 4, 5]]    # age, rosc, ohca, shockable_rhythm
    
    if len(patient_features.shape) == 1:
        patient_features = np.expand_dims(patient_features, axis=0)
    if len(counfounder_features.shape) == 1:
        counfounder_features = np.expand_dims(counfounder_features, axis=0)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    if len(eeg_files) == 0:
        features = np.zeros((1, (6*43)+24+21+21+15+6+1+3+3+3+3+2+2+16)) #***
        timestamps = 0
        patient_features1 = np.repeat(patient_features, features.shape[0], axis=0)
        features = np.hstack((features, patient_features1))
        print("[Warning:] The patient has no valid EEG file!")
        return features, counfounder_features, timestamps, pid
    # -------------------------------------------------------------------------
    # Extract EEG features ----------------------------------------------------
    eeg_channels = ['Fp1', 'F3', 'C3', 'P3', 'Fp2', 'F4', 'C4', 'P4', 'Fz', 'Cz', 
                    'Pz', 'T3', 'T5', 'T4', 'T6', 'O1', 'O2']
    ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    # -------------------------------------------------------------------------
    features = []
    timestamps = []
    k = 0
    delay = 0
    # -------------------------------------------------------------------------
    bandf = [[1, 4], [4, 8], [8, 12], [11, 16], [12, 35], [12, 18], [18, 35]]
    delta_band = [0.5, 4]
    spindle_band = [12, 16]
    beta_band = [16, 25]
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    if num_recordings > 0:
        for iter1 in range(num_recordings):
            
            recording_id = eeg_files[iter1]
            
            parts = recording_id.split(".")
            recording_id = ".".join(parts[:-1])
            
            recording_location_eeg = os.path.join(data_folder, patient_id ,recording_id)
            recording_location_eeg = recording_location_eeg.replace('\\', '/')
            
            recording_location_ecg = os.path.join(data_folder, patient_id, recording_id.replace("EEG", "ECG"))
            recording_location_ecg = recording_location_ecg.replace('\\', '/')
            # -----------------------------------------------------------------
            if os.path.exists(recording_location_eeg + '.hea'):
                utility_frequency_eeg, eeg_start_time_match, _, duration_sec_eeg = extract_info_from_text_file(recording_location_eeg + '.hea')
                # -------------------------------------------------------------
                if eeg_start_time_match is not None:
                    hours, minutes, seconds = map(int, eeg_start_time_match.group(1).split(':'))
                    # ---------------------------------------------------------
                    if hours < 72:                    
                        # -----------------------------------------------------
                        if (duration_sec_eeg /60) >= 15:
                            data_eeg, channels_eeg, fs_eeg = load_recording_data(recording_location_eeg)
                            ind_dc = detect_dc(data_eeg, fs_eeg, channels_eeg)
                            # -------------------------------------------------
                            if not ind_dc:
                                # ---------------------------------------------
                                if all(channel in channels_eeg for channel in eeg_channels):
                                    print("[INFO:] patient: {} *** recording: {}/{} *** length: {:.2f} min".format(patient_id, iter1, num_recordings, duration_sec_eeg / 60))
                                    data_eeg, channels_eeg = reduce_channels(data_eeg, channels_eeg, eeg_channels)
                                    data_eeg, fs_eeg = preprocess_data(data_eeg, fs_eeg, utility_frequency_eeg)
                                    data_eeg1 = np.zeros((21, data_eeg.shape[1]))
                                    data_eeg1[0, :] = data_eeg[1, :] - data_eeg[2, :]    # F3-C3
                                    data_eeg1[1, :] = data_eeg[2, :] - data_eeg[3, :]    # C3-P3 *
                                    data_eeg1[2, :] = data_eeg[5, :] - data_eeg[6, :]    # F4-C4
                                    data_eeg1[3, :] = data_eeg[6, :] - data_eeg[7, :]    # C4-P4 *
                                    data_eeg1[4, :] = data_eeg[8, :] - data_eeg[9, :]    # Fz-Cz *
                                    data_eeg1[5, :] = data_eeg[3, :] - data_eeg[15, :]   # P3-O1
                                    data_eeg1[6, :] = data_eeg[12, :] - data_eeg[15, :]  # T5-O1
                                    data_eeg1[7, :] = data_eeg[7, :] - data_eeg[16, :]   # P4-O2
                                    data_eeg1[8, :] = data_eeg[14, :] - data_eeg[16, :]  # T6-O2
                                    # ref: https://doi.org/10.1016/j.resuscitation.2023.109817
                                    data_eeg1[9, :] = data_eeg[0, :] - data_eeg[4, :]    # Fp1-Fp2 *
                                    data_eeg1[10, :] = data_eeg[11, :] - data_eeg[13, :] # T3-T4 *
                                    #
                                    data_eeg1[11, :] = data_eeg[12, :] - data_eeg[14, :] # T5-T6 *
                                    data_eeg1[12, :] = data_eeg[0, :] - data_eeg[11, :]  # Fp1-T3
                                    data_eeg1[13, :] = data_eeg[4, :] - data_eeg[13, :]  # Fp2-T4
                                    data_eeg1[14, :] = data_eeg[13, :] - data_eeg[15, :] # T4-O1 *
                                    data_eeg1[15, :] = data_eeg[11, :] - data_eeg[16, :] # T3-O2 *
                                    data_eeg1[16, :] = data_eeg[0, :] - data_eeg[9, :]   # Fp1-Cz *
                                    data_eeg1[17, :] = data_eeg[4, :] - data_eeg[9, :]   # Fp2-Cz *
                                    data_eeg1[18, :] = data_eeg[15, :] - data_eeg[16, :] # O1-O2
                                    data_eeg1[19, :] = data_eeg[3, :] - data_eeg[7, :]   # P3-P4
                                    data_eeg1[20, :] = data_eeg[9, :] - data_eeg[10, :]  # Cz-Pz *
                                    # -----------------------------------------
                                    eeg_features, phases, windows_chs = get_eeg_features_AIrhythm(data_eeg1[[4, 1, 3, 9, 10, 20], :], 
                                                                              fs_eeg,
                                                                              bandf, 
                                                                              delta_band, 
                                                                              spindle_band, 
                                                                              beta_band) #6*43
                                    # -----------------------------------------
                                    eig_features, eig_cov_features, reig = eeg_connectivity(data_eeg1, fs_eeg) #21 + 21 + 15
                                    # -----------------------------------------
                                    tangentsigniture = tangent_signiture(data_eeg1[[4], :], fs_eeg) # 6
                                    # -----------------------------------------
                                    ncds = Compressors_features(windows_chs, fs_eeg) #1
                                    # -----------------------------------------
                                    connectivity_features1 = unpack_phases(phases, fs_eeg) #24
                                    # -----------------------------------------
                                    frontals1 = frontal_features(data_eeg1[[12], :], fs_eeg) #3
                                    frontals2 = frontal_features(data_eeg1[[13], :], fs_eeg) #3
                                    frontals3 = frontal_features(data_eeg1[[16], :], fs_eeg) #3
                                    frontals4 = frontal_features(data_eeg1[[17], :], fs_eeg) #3
                                    # -----------------------------------------
                                    postant1 = post_ant(data_eeg1[[9, 18], :], fs_eeg)   #2
                                    postant2 = post_ant(data_eeg1[[12, 13], :], fs_eeg)  #2
                                    # -----------------------------------------                                    
                                    del data_eeg, data_eeg1
                                    # -----------------------------------------
                                    if os.path.exists(recording_location_ecg + '.hea'):
                                        utility_frequency_ecg, _, _, duration_sec_ecg = extract_info_from_text_file(recording_location_ecg + '.hea')
                                        if duration_sec_ecg is not None:
                                            if ((duration_sec_ecg /60) >= 15) and (duration_sec_ecg == duration_sec_eeg):
                                                data_ecg, channels_ecg, fs_ecg = load_recording_data(recording_location_ecg)
                                                ind_dc_ecg = detect_dc(data_ecg, fs_eeg, channels_eeg, eeg=False)
                                                if not ind_dc_ecg:
                                                    data_ecg, channels_ecg = reduce_channels(data_ecg, channels_ecg, ecg_channels)
                                                    data_ecg = data_ecg[0, :]
                                                    if len(data_ecg.shape) == 1:
                                                        data_ecg = np.expand_dims(data_ecg, axis=0)
                                                    data_ecg, fs_ecg = preprocess_data(data_ecg, fs_ecg, utility_frequency_ecg)
                                                    ecg_features = get_ecg_features_AIrhythm(data_ecg[0, :], fs_ecg)
                                                else:
                                                    ecg_features = np.zeros((eeg_features.shape[0], 16))
                                            else:
                                                ecg_features = np.zeros((eeg_features.shape[0], 16))
                                        else:
                                            ecg_features = np.zeros((eeg_features.shape[0], 16))
                                    else:
                                        ecg_features = np.nan*np.ones((eeg_features.shape[0], 16))
                                    # -----------------------------------------
                                    temp = np.hstack((eeg_features, connectivity_features1,
                                                      eig_features, eig_cov_features, reig, 
                                                      tangentsigniture, ncds, 
                                                      frontals1, frontals2, frontals3, frontals4,
                                                      postant1, postant2,
                                                      ecg_features))
                                    # -----------------------------------------
                                    if k == 0:
                                        features = temp
                                        timestamps = np.arange(temp.shape[0])
                                    else:
                                        features = np.vstack((features, temp))                                
                                        timestamps = np.hstack((timestamps, np.arange(temp.shape[0]) + timestamps[-1] + 1 + delay))
                                        delay = 0
                                    
                                    k +=1
                            # -------------------------------------------------
                            else:
                                delay += int(duration_sec_eeg/180)
                        # -----------------------------------------------------
                        else:
                            delay += int(duration_sec_eeg/180)
                    # ---------------------------------------------------------
                    else:
                        break 
    # -------------------------------------------------------------------------
    if len(features) == 0:
        features = np.zeros((1, (6*43)+24+21+21+15+6+1+3+3+3+3+2+2+16))
        timestamps = 0
        print("[Warning:] The patient has no valid EEG file!")
    # -------------------------------------------------------------------------
    patient_features1 = np.repeat(patient_features, features.shape[0], axis=0)
    # -------------------------------------------------------------------------
    features = np.hstack((features, patient_features1))
    print(features.shape)
    # -------------------------------------------------------------------------
    return features, counfounder_features, timestamps, pid
# =============================================================================


# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    
    ttm = get_ttm(data)
    
    if sex == 'Female':
        sex1 = 1
    elif sex == 'Male':
        sex1 = 2
    else:
        sex1 = 3
    
    hospital = get_hospital(data)
    hospital = ord(hospital) - 96
    
    features = np.array((age, sex1, hospital, rosc, ohca, shockable_rhythm, ttm))
    
    return features
# =============================================================================


def detect_dc(data_eeg, fs_eeg, channels_eeg, eeg=True):
    if eeg:
        data1 = eeg_montages(data_eeg, channels_eeg)
        temp = data1[3, :]
    else:
        temp = data_eeg[0, :]
    # -------------------------------------------------------------------------
    window_length = int(180*fs_eeg)
    temp = np.diff(temp)
    windows = rolling_window(temp, window_length)
    
    k = 0
    for window in windows:
        if np.sum(window) == 0:
            k +=1
            
    if k > len(windows)/3:
        ind_dc = True
    else:
        ind_dc = False
    
    return ind_dc
# =============================================================================


def sort_strings_by_last_digits(arrays):
    def last_digits_key(string):
        # Extract the last digits from the string
        last_digits = ''.join(filter(str.isdigit, string))
        # Convert the last digits to an integer for sorting
        return int(last_digits)

    sorted_arrays = sorted(arrays, key=last_digits_key)
    return sorted_arrays
# =============================================================================


def build_models_AIrhythm(features, timestamps, outcomes, cpcs, pids, 
                          counfounder_features, model_folder):
    print()
    print("v.9.1.1")
    print("Author: Morteza Zabihi (morteza.zabihi@gmail.com)") 
    print("Copyright (C) 2023 Morteza Zabihi")
    print()
    
    t1 = time.time()
    
    models_all = []
    models_cpc_all = []
    medians_all = []
    # -------------------------------------------------------------------------
    configs = [ 
                {"clfop": "cat",       "h":6,   "cv": "normal",   'ind_feature':None, 'prepop': "q89"},
                {"clfop": "cat",       "h":6,   "cv": "cluster",  'ind_feature':None, 'prepop': "q89"},
                {"clfop": "cat",       "h":6,   "cv": "ps",       'ind_feature':None, 'prepop': "q89"},
                {"clfop": "cat",       "h":5,   "cv": "normal",   'ind_feature':None, 'prepop': "q89"},
                {"clfop": "cat",       "h":5,   "cv": "cluster",  'ind_feature':None, 'prepop': "q89"},
                {"clfop": "cat",       "h":5,   "cv": "ps",       'ind_feature':None, 'prepop': "q89"},
                {"clfop": "stacking1", "h":6,   "cv": "normal",   'ind_feature':None, 'prepop': "q88"},
                {"clfop": "stacking1", "h":6,   "cv": "cluster",  'ind_feature':None, 'prepop': "q88"},
                {"clfop": "stacking1", "h":6,   "cv": "ps",       'ind_feature':None, 'prepop': "q88"},
                {"clfop": "cat",       "h":6.9, "cv": "normal",   'ind_feature':None, 'prepop': "combine"},
                {"clfop": "cat",       "h":6.9, "cv": "cluster",  'ind_feature':None, 'prepop': "combine"},
                {"clfop": "cat",       "h":6.9, "cv": "ps",       'ind_feature':None, 'prepop': "combine"},
                {"clfop": "stacking2", "h":6.9, "cv": "cluster",  'ind_feature':None, 'prepop': "combine1"}]
    # -------------------------------------------------------------------------
    cr = class_robust()
    all_features = []
    # -------------------------------------------------------------------------
    for iter1 in range(len(configs)):
        
        config = configs[iter1]
        clfop = config["clfop"]
        h = config["h"]
        cvtype = config["cv"]
        prepop = config['prepop']
        
        if cvtype == "normal":
            models, models_cpc, medians, comfeatures, _, _ = cr.cv(features, timestamps, outcomes, cpcs, pids, counfounder_features, h, n_folds=5, clfop=clfop,  ind_feature=None, prepop=prepop, challenge=True)
            all_features.append(comfeatures)
        if cvtype == "cluster":
            models, models_cpc, medians, comfeatures, _, _ = cr.cv_cluster(features, timestamps, outcomes, cpcs, pids, counfounder_features, h, optimal_k=3, modecluster="kmeans", clfop=clfop, ind_feature=None, prepop=prepop, challenge=True)
            all_features.append(comfeatures)
        if cvtype == "ps":
            models, models_cpc, medians, comfeatures, _, _ = cr.cv_ps(features, timestamps, outcomes, cpcs, pids, counfounder_features, h, clfop=clfop, ind_feature=None, prepop=prepop, challenge=True)
            all_features.append(comfeatures)
        # ---------------------------------------------------------------------
        models_all.append(models)
        models_cpc_all.append(models_cpc)
        medians_all.append(medians)
        configs[iter1] = config
    # -------------------------------------------------------------------------
    # Save the models.
    print("[INFO:] models are saving ...")
    cmobj = class_model()
    cmobj.save_challenge_model(model_folder, models_all, models_cpc_all, medians_all, configs)
    # d = {'outcome_model': models_all, 'cpc_model': models_cpc_all, 'medians': medians_all, "configs": configs}
    print("[INFO:] Training time (min): ", (time.time() - t1)/60)   
    # return d, all_features
# =============================================================================


def helper_run_model_AIrhythm(models, pids, features_test, timestamps_test):
    cmobj = class_model()
    # -------------------------------------------------------------------------
    medians = models['medians']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    configs = models['configs']
    # -------------------------------------------------------------------------    
    outcome_probabilitys = []
    cpcs = []
    outcomes = []
    
    for iter1 in range(len(configs)): # number of configurations
        temp_outcome_model = outcome_model[iter1] 
        temp_cpc_model = cpc_model[iter1]
        temp_median = medians[iter1]
        temp_config = configs[iter1]
        
        h = temp_config["h"]
        ind_feature = temp_config['ind_feature']
        prepop = temp_config['prepop']
        
        
        for iter2 in range(len(temp_outcome_model)): # number of models within each config
            clf = temp_outcome_model[iter2]
            clf_cpc = temp_cpc_model[iter2]
            nan_median = temp_median[iter2]            
            # -----------------------------------------------------------------            
            predictions, cpc_predictions, _ = cmobj.inference(pids, 
                                                           features_test, 
                                                           timestamps_test,
                                                           clf, 
                                                           clf_cpc, 
                                                           nan_median,
                                                           h,
                                                           ind_feature,
                                                           prepop)
            if predictions[1] >= 0.5:
                outcomes.append(1)
            else:
                outcomes.append(0)
            
            outcome_probabilitys.append(predictions[1])
            cpcs.append(np.clip(cpc_predictions[0], 1, 5))
    # -------------------------------------------------------------------------
    outcome_probability = np.nanmean(outcome_probabilitys)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    if outcome_probability >= 0.5:
        outcome = 1
    else:
        outcome = 0
    # -------------------------------------------------------------------------
    indices = [i for i, x in enumerate(outcomes) if x == outcome]

    equal_to = []
    # Loop through each element of the list
    for iter1 in range(len(indices)):
        equal_to.append(cpcs[indices[iter1]])
    
    equal_to = np.array(equal_to)
    cpc = np.nanmedian(equal_to)
    # -------------------------------------------------------------------------
    return outcome, outcome_probability, cpc