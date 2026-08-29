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

import os
import re
import numpy as np
import pywt

from mne.time_frequency import morlet
import statsmodels.api as sm

from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.linalg import eigh
from scipy.stats import entropy

from utility_class_qrs_detection import *
from utility_class_ECG_v1 import *

import warnings
warnings.simplefilter('ignore', np.RankWarning)

# =============================================================================
# ====================== Helper functions =====================================
# =============================================================================


def seasonal_autocorrelation(data, seasonal_period):
    
    autocorrelations = []
    for i in range(seasonal_period):
        seasonal_data = data[i::seasonal_period]
        mean = np.mean(seasonal_data)
        variance = np.var(seasonal_data, ddof=1)
        covariance = np.sum((seasonal_data[:-1] - mean) * (seasonal_data[1:] - mean))
        if variance < np.finfo(np.float32).eps:
            autocorrelation = covariance / (variance + np.finfo(np.float32).eps)
        else:    
            autocorrelation = covariance / variance
        autocorrelations.append(autocorrelation)

    autocorrelation_coefficient = np.nanmean(autocorrelations)

    return autocorrelation_coefficient
# =============================================================================


def compute_auto_correlation(signal, delays):
    num_delays = len(delays)
    num_channels, num_samples = signal.shape
    auto_corr_matrices = np.zeros((2*num_channels, 2*num_channels, num_delays))

    for i in range(num_delays):
        delay = delays[i]
        lagged_signal = np.hstack([signal[:, delay:], np.zeros((num_channels, delay))])
        auto_corr_matrices[:, :, i] = np.corrcoef(signal, lagged_signal)

    return auto_corr_matrices
# =============================================================================


def Riemannian_features(eeg_data, fs):
        
    log_spaced_delays = np.array([2, 8, 32]) * fs
    try:       
        auto_corr_matrices = compute_auto_correlation(eeg_data, log_spaced_delays)
        
        reig = []
        for iter3 in range(auto_corr_matrices.shape[2]):
            temp = auto_corr_matrices[:, :, iter3]
            eigenvalues, _ =  np.linalg.eigh(temp)
            reig.append(eigenvalues[-5:])
        reig = np.array(reig)
        reig = reig.flatten()
    except:
        reig = np.zeros((5*len(log_spaced_delays), ))
    
    return reig
# =============================================================================


def spectral_edge_frequency(power, freqs, edges=[0.5, 0.7, 0.8, 0.9, 0.95]):
    out = np.cumsum((power))
    out = out / out[-1]
    ret = []
    if np.sum(np.isnan(out))>0:
        ret = np.zeros((len(edges), ))
    else:
        for edge in edges:
            ret.append(freqs[np.where(out>edge)[0][0]])
        ret = np.array(ret)
    return ret
# =============================================================================


def unpack_phases(phases, fs):
    num_samples = fs * 3 * 60
    num_window = len(phases[0])
    num_channels = len(phases)

    connectivity_channels = np.zeros((num_window, 4 * num_channels))

    for iter0 in range(num_channels):
        ch1 = phases[iter0]

        for iter1 in range(iter0 + 1, num_channels):
            ch2 = phases[iter1]

            for iter2 in range(len(ch1)):
                ch1_1 = ch1[iter2]
                ch2_1 = ch2[iter2]

                phase_1625_ch1 = ch1_1['phase_1625']
                phase_slow_ch1 = ch1_1['phase_slow']

                phase_1625_ch2 = ch2_1['phase_1625']
                phase_slow_ch2 = ch2_1['phase_slow']

                temp_features = np.zeros(4)

                if len(phase_1625_ch1) > 0 and len(phase_1625_ch2) > 0:
                    phases_diff = phase_1625_ch1 - phase_1625_ch2
                    plv = np.abs(np.sum(np.exp(1j * phases_diff))) / num_samples
                    inst_phasei = np.arctan(phase_1625_ch1)
                    inst_phasej = np.arctan(phase_1625_ch2)
                    phaselagindex = np.abs(np.mean(np.sign(inst_phasej - inst_phasei), axis=0))
                else:
                    plv = 0
                    phaselagindex = 0

                temp_features[0] = plv
                temp_features[1] = phaselagindex

                if len(phase_slow_ch1) > 0 and len(phase_slow_ch2) > 0:
                    phases_diff = phase_slow_ch1 - phase_slow_ch2
                    plv = np.abs(np.sum(np.exp(1j * phases_diff))) / num_samples
                    inst_phasei = np.arctan(phase_slow_ch1)
                    inst_phasej = np.arctan(phase_slow_ch2)
                    phaselagindex = np.abs(np.mean(np.sign(inst_phasej - inst_phasei), axis=0))
                else:
                    plv = 0
                    phaselagindex = 0

                temp_features[2] = plv
                temp_features[3] = phaselagindex

                connectivity_channels[iter2, iter0*4:iter0*4+4] = temp_features

    return connectivity_channels
# =============================================================================


def eig_channel(data, fs):
    
    window_length = 180*fs
    num_windows = data.shape[1] // window_length
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    start = 0
    end = window_length
    eig_features = np.zeros((num_windows, data.shape[0]))
    eig_cov_features = np.zeros((num_windows, data.shape[0]))
    reig = np.zeros((num_windows, 15))
    for iter1 in range(num_windows):
        temp = data[:, start:end]
        # ---------------------------------------------------------------------
        correlation_matrix = np.corrcoef(temp)
        eigenvalues, _ = eigh(correlation_matrix)
        eig_features[iter1, :] = eigenvalues
        #----------------------------------------------------------------------
        covar = np.cov(temp)
        eigenvalues, _ = eigh(covar)
        eig_cov_features[iter1, :] = eigenvalues
        # ---------------------------------------------------------------------
        reig[iter1, :] = Riemannian_features(temp, fs)
        # ---------------------------------------------------------------------
        start = end
        end += window_length
    # -------------------------------------------------------------------------
    return eig_features, eig_cov_features, reig
# =============================================================================


def extract_stft_entropy_features(spectrogram, frequencies_r, time_r): #<del>
    
    indf = np.where(frequencies_r <= 45)[0]
    
    freq = frequencies_r[indf]
    X = spectrogram[indf, :]
    ind_0 = np.where(X == 0)
    X[ind_0] = np.finfo(np.float32).eps
    X = 20*np.log10(X)
    X /= np.max(X)
    
    # Calculate Shannon's entropy at dyadic frequency bands
    num_bands = 5  # Number of dyadic frequency bands
    entropy_values = []
    for i in range(num_bands):
        start_freq = 2 ** i
        end_freq = 2 ** (i + 1)
        band_indices = np.logical_and(freq >= start_freq, freq < end_freq)
        band_psd = X[band_indices, :]
        
        # Calculate the probability distribution of the signal
        arr = np.sum(band_psd, axis=0)
        
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = 100 * (arr - min_val) / (max_val - min_val)
            
        hist, _ = np.histogram(normalized_arr, bins=np.arange(0,102,2))
        prob = hist / np.sum(hist)
        band_entropy = entropy(prob)
        entropy_values.append(band_entropy)

    entropy_values = np.array(entropy_values)
    return entropy_values
# =============================================================================


def calculate_plv_cicoh(x, y, fs):  #<del>
    """
    Calculate the phase locking value (PLV) 
    """
    # -------------------------------------------------------------------------
    # Extract the data for the specified channels
    num_samples = len(x)
    # -------------------------------------------------------------------------
    # Calculate the analytic signals
    phases_x = np.angle(hilbert(x))
    phases_y = np.angle(hilbert(y))
    # -------------------------------------------------------------------------
    phases_diff = phases_x - phases_y
    plv = np.abs(np.sum(np.exp(1j * phases_diff))) / num_samples
    # -------------------------------------------------------------------------
    f, Cxy = signal.coherence(x, y, fs, nperseg=256)
    Cxy = np.mean(np.abs(Cxy))
    # -------------------------------------------------------------------------
    inst_phasei = np.arctan(phases_x)
    inst_phasej = np.arctan(phases_y)

    phaselagindex = np.abs(np.mean(np.sign(inst_phasej - inst_phasei), axis=0))
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    connectivity = np.array([plv, Cxy, phaselagindex])
    # -------------------------------------------------------------------------
    return connectivity
# =============================================================================


def spindle(x, sf, cf=13):
    try:
        
        # Parameters
        # cf = 13     # Central spindles frequency in Hz
        nc = 12     # Number of oscillations in the spindles
        
        # Compute the wavelet
        wlt = morlet(sf, [cf], n_cycles=nc)[0]
        
        # Convolve the wavelet and extract magnitude and phase
        analytic = np.convolve(x, wlt, mode='same')
        magnitude = np.abs(analytic)
        phase = np.angle(analytic)
        
        # Square and normalize the magnitude from 0 to 1 (using the min and max)
        power = np.square(magnitude)
        norm_power = (power - power.min()) / (power.max() - power.min())
        
        # Define the threshold
        thresh = 0.25
        
        # Find supra-threshold values
        supra_thresh = np.where(norm_power >= thresh)[0]
        
        # Create vector for plotting purposes
        val_spindles = np.nan * np.zeros(x.size)
        val_spindles[supra_thresh] = x[supra_thresh]
        
        # Extract start and end of each spindles
        sp = np.split(supra_thresh, np.where(np.diff(supra_thresh) != 1)[0] + 1)
        # idx_start_end = np.array([[k[0], k[-1]] for k in sp])
        
        # # Extract the duration (in ms) of each spindles
        # sp_dur = (np.diff(idx_start_end, axis=1) / sf * 1000).flatten()
        
        # # Extract the peak-to-peak amplitude and frequency
        # sp_amp, sp_freq = np.zeros(len(sp)), np.zeros(len(sp))
        
        # for i in range(len(sp)):
        #     # Important: detrend the signal to avoid wrong peak-to-peak amplitude
        #     sp_amp[i] = np.ptp(detrend(x[sp[i]]))
        
        #     # Median of the instantaneous frequency of the spindles, where:
        #     # inst_freq = sf / 2pi * 1st-derivative of the phase of the analytic signal
        #     sp_freq[i] = np.median((sf / (2 * np.pi) * np.diff(phase[sp[i]])))
        
        
        f1 = len(sp) 
        # if f1>0:
        #     f2 = np.mean(sp_dur)  # Duration (ms)
        #     f3 = np.mean(sp_freq) # Frequency (Hz)
        #     f4 = np.mean(sp_amp)  # Amplitude (uV)
        # else:
        #     f2 = 0
        #     f3 = 0 
        #     f4 = 0
        return f1
    except:
        f1 = 0
        # f2 = 0
        # f3 = 0 
        # f4 = 0
        
    return f1
# =============================================================================


def spectral_entropy(signal1, n_short_blocks=10):  #<del>
    """
    ref: https://github.com/tyiannak/pyAudioAnalysis/tree/master
    Computes the spectral entropy
    """
    # number of frame samples
    num_frames = len(signal1)

    # total spectral energy
    total_energy = np.sum(signal1 ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal1 = signal1[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal1.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + np.finfo(np.float32).eps)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + np.finfo(np.float32).eps))

    return entropy
# =============================================================================


def stft_features(X, f1, t, bandf):
    
    ind_0 = np.where(X == 0)
    X[ind_0] = np.finfo(np.float32).eps
    X = 20*np.log10(X)
    # -------------------------------------------------------------------------
    # energies = np.zeros((len(bandf), ))
    # for iter1 in range(len(bandf)):
    #     temp = bandf[iter1]
    #     start = temp[0]
    #     end = temp[1]
    #     fr1 = np.where((f1>=start) & (f1<=end))[0]
    #     temp = np.squeeze(X[fr1, :])
    #     energies[iter1] = np.sum(temp ** 2)    
    
    # -------------------------------------------------------------------------
    fr = np.where((f1>0) & (f1<=40))[0]
    temp = np.squeeze(X[fr, :])
    # -------------------------------------------------------------------------
    indstft = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
    fmax_stft = f1[fr[indstft[0]]]
    # -------------------------------------------------------------------------
    var_stft = np.var(temp)
    # -------------------------------------------------------------------------
    analytic_signal = hilbert(np.sum(temp, axis=0))
    # amplitude_envelope = np.abs(analytic_signal)
    # -------------------------------------------------------------------------
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi))
    mins = np.mean(instantaneous_frequency)
    sins = np.std(instantaneous_frequency)
    # -------------------------------------------------------------------------
    # energy = np.sum(temp ** 2)
    # energies = energies / energy
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    return fmax_stft, var_stft, mins, sins
# =============================================================================


def bandpass_filter(signal, fs, lowcut, highcut, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
# =============================================================================


def zr_cr(signal, fs, low=16, high=25):
    filtered_signal = bandpass_filter(signal, fs, low, high, order=5)
    zero_crossings = np.where(np.diff(np.signbit(filtered_signal)))[0]
    difzc = np.diff(zero_crossings)
    q75, q25 = np.percentile(difzc, [75 ,25])
    iqr_1625 = q75 - q25
    
    return iqr_1625, filtered_signal
# =============================================================================

       
def create_spectrogram(signal, sampling_freq, window_length, overlap):
    """    
    window_length = int(fs_eeg*3)  # Length of the window in samples
    overlap = int(fs_eeg*2)  # Overlap size in samples

    # signal = data_eeg[2, :]
    sampling_freq = fs_eeg
    spectrogram, frequencies, time = create_spectrogram(signal1, sampling_freq,
                                                        window_length, overlap)

    # Plotting the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(20 * np.log10(spectrogram), origin='lower', 
               aspect='auto', 
               cmap='jet',
               extent=[time[0], time[-1], 
               frequencies[0], frequencies[-1]])

    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show()


    """
    n_samples = len(signal)
    window = np.hamming(window_length)
    hop_size = window_length - overlap
    n_fft = next_power_of_2(window_length)

    n_windows = int(np.floor((n_samples - window_length) / hop_size) + 1)
    spectrogram = np.zeros((n_fft // 2 + 1, n_windows))

    for i in range(n_windows):
        start = i * hop_size
        end = start + window_length
        segment = signal[start:end]

        windowed_segment = segment * window
        spectrum = np.fft.fft(windowed_segment, n=n_fft)
        magnitude = np.abs(spectrum[:n_fft // 2 + 1])
        spectrogram[:, i] = magnitude

    frequencies = np.fft.fftfreq(n_fft, d=1/sampling_freq)[:n_fft // 2 + 1]
    time = np.arange(n_windows) * (hop_size / sampling_freq)

    return spectrogram, frequencies, time
# =============================================================================


def next_power_of_2(x):
    return int(2 ** np.ceil(np.log2(x)))
# =============================================================================


def list_eeg_files(data_folder, patient_id):
    eeg_files = []
    for file_name in os.listdir(os.path.join(data_folder, patient_id)):
        if "EEG" in file_name and file_name.endswith(".mat"):
            eeg_files.append(file_name)
    
    filtered_list = []
    for iter1 in range(len(eeg_files)):
        file_name = eeg_files[iter1]
        parts = file_name.split(".")
        file_name_without_format = ".".join(parts[:-1])
        for file_name_without_format in os.listdir(os.path.join(data_folder, patient_id)):
            if "EEG" in file_name_without_format and file_name_without_format.endswith(".hea"):
                filtered_list.append(file_name)
                break
            
    return filtered_list
# =============================================================================


def extract_info_from_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    utility_frequency_match = re.search(r'Utility frequency: (\d+)', content)
    start_time_match = re.search(r'Start time: (\d+:\d+:\d+)', content)
    end_time_match = re.search(r'End time: (\d+:\d+:\d+)', content)
    # -------------------------------------------------------------------------
    if utility_frequency_match is None:
        utility_frequency = None
    else:
        utility_frequency = int(utility_frequency_match.group(1))
        
    # -------------------------------------------------------------------------
    if (start_time_match is None) or (end_time_match is None):
        start_time_match, end_time_match, duration_sec = None, None, None
    else:
        
        hours, minutes, seconds = map(int, start_time_match.group(1).split(':'))
        total_seconds_start = (hours * 60 * 60) + (minutes * 60) + seconds
        
        
        hours, minutes, seconds = map(int, end_time_match.group(1).split(':'))
        total_seconds_end = (hours * 60 * 60) + (minutes * 60) + seconds
        
        # start_time = datetime.strptime(start_time_match.group(1), '%H:%M:%S')
        # end_time = datetime.strptime(end_time_match.group(1), '%H:%M:%S')
        
        duration_sec = total_seconds_end - total_seconds_start
    
    return utility_frequency, start_time_match, end_time_match, duration_sec
# =============================================================================


def rolling_window(array, window_length):
    num_windows = len(array) // window_length
    return np.split(array[:num_windows*window_length], num_windows)
# =============================================================================


def featurize_dynamic_winodows_eeg(row1, fs, bandf, delta_band, spindle_band, beta_band):
    alpha_band = bandf[2]
    # -------------------------------------------------------------------------
    gradient_1 = np.gradient(row1)
    gradient_2 = np.gradient(gradient_1)
    ha = np.var(row1)
    hm = np.sqrt(np.var(gradient_1) / ha)
    hm1 = np.sqrt(np.var(gradient_2) / np.var(gradient_1))
    hc = hm1 / hm    
    # -------------------------------------------------------------------------
    f, psd = signal.welch(row1, fs, window='hamming', nperseg=2*fs)
    # -------------------------------------------------------------------------
    fb0 = np.where(f<=1)
    fb3 = np.where((f>12) & (f<=30))
    fb4 = np.where((f>4) & (f<=12))
    fb5 = np.where((f>8) & (f<=35))
    # -------------------------------------------------------------------------
    if len(fb0) > 0:
        pratio_0 = np.sum(psd[fb0])
    else:
        pratio_0 = 0
    # -------------------------------------------------------------------------
    pratio_3 = np.sum(psd[fb4]) / np.sum(psd[fb3]) #*
    # -------------------------------------------------------------------------
    pratio_5 = np.sum(psd[fb4]) / np.sum(psd[fb5]) #*
    # -------------------------------------------------------------------------
    # Calculate the power in each frequency band
    delta_power = np.sum(psd[(f >= delta_band[0]) & (f < delta_band[1])])
    alpha_power = np.sum(psd[(f >= alpha_band[0]) & (f < alpha_band[1])])
    # -------------------------------------------------------------------------
    # Calculate the band ratios
    alpha_delta_ratio = alpha_power / delta_power #*
    # -------------------------------------------------------------------------
    slow_wave = bandpass_filter(row1, fs, 0.5, 4)
    zero_crossings = np.where(np.diff(np.signbit(slow_wave)))[0]
    difzc = np.diff(zero_crossings)
    q75, q25 = np.percentile(difzc, [75 ,25])
    iqr_slow_wave = q75 - q25 #*
    # -------------------------------------------------------------------------
    coeffs = pywt.wavedec(row1, 'db4', level=5)
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
    # -------------------------------------------------------------------------
    iqr_1625, filtered_1625 = zr_cr(row1, fs)
    # -------------------------------------------------------------------------
    spectrogram, frequencies_r, time_r = create_spectrogram(row1, fs,  int(fs*5), int(fs*3))
    fmax_stft, var_stft, mins, sins = stft_features(spectrogram, frequencies_r, time_r, bandf)
    # -------------------------------------------------------------------------
    sf1 = spindle(row1, fs)
    # -------------------------------------------------------------------------
    rho, sigma2 = sm.regression.linear_model.burg(row1, order=10)
    # -------------------------------------------------------------------------
    ret = spectral_edge_frequency(psd, f, edges=[0.5, 0.7, 0.8, 0.9, 0.95]) # 5
    # -------------------------------------------------------------------------
    sa1 = seasonal_autocorrelation(row1, int(0.3*fs))
    sa2 = seasonal_autocorrelation(row1, int(0.8*fs))
    sa3 = seasonal_autocorrelation(row1, int(1.5*fs))
    # -------------------------------------------------------------------------
    phase_1625 = np.angle(hilbert(filtered_1625))
    phase_slow = np.angle(hilbert(slow_wave))
    phases_x = {'phase_1625' : phase_1625,
                'phase_slow': phase_slow}
    # -------------------------------------------------------------------------
    # Concatenate features
    features = np.array([ha,
                        hm,
                        hc,
                        pratio_0,
                        pratio_3,
                        pratio_5,
                        alpha_delta_ratio,
                        iqr_slow_wave,
                        np.mean(cD1),
                        np.mean(cD2),
                        np.mean(cD3),
                        np.mean(cD4),
                        np.mean(cD5),
                        np.var(cD1), 
                        np.var(cD2),
                        np.var(cD3),
                        np.var(cD4),
                        np.var(cD5),
                        iqr_1625,
                        fmax_stft,
                        var_stft,
                        mins,
                        sins,
                        sf1,
                        sigma2,
                        sa1,
                        sa2,
                        sa3]) #28
    features = np.hstack((features, rho, ret))  #28 + 10 + 5  = 43
    # -------------------------------------------------------------------------
    return features, phases_x

# =============================================================================


def eeg_montages(eegs, ch_names):
    
    list_ch = ['fp1', 'f3', 'c3', 'p3', 'o1', 'fp2', 'f4', 'c4',\
           'p4', 'o2', 'f7', 't3', 't5', 'f8', 't4', 't6', 'fz',\
           'cz', 'pz']
    
    ind_chs = []
    for iter1 in range(len(list_ch)):
        string1 = list_ch[iter1]
        string1 = np.array2string(np.squeeze(string1))
        string1 = string1.replace("'", "")
        string1 = string1.replace(" ", "")
        for iter2 in range(len(ch_names)):
            string2 = ch_names[iter2]
            string2 = np.array2string(np.squeeze(string2))
            string2 = string2.replace("'", "")
            string2 = string2.replace(" ", "")
            if string1.lower() == string2.lower():
                ind_chs.append(iter2)
                break
    # longitudinal montage ----------------------------------------------------
    # longitudinal (anterior to posterior)
    # fp1-f7, f7-t3, t3-t5, t5-o1
    # fp1-f3, f3-c3, c3-p3, p3-o1
    # fp2-f4, f4-c4, c4-p4, p4-02
    # fp2-f8, f8-t4, t4-t6, t6-02
    # fz-cz, cz-pz
    eegs_new = np.zeros((18, eegs.shape[1], ), dtype=np.float32)
    eegs_new[0, :] = eegs[ind_chs[0], :] - eegs[ind_chs[10], :]   # 'fp1-f7'
    eegs_new[1, :] = eegs[ind_chs[10], :] - eegs[ind_chs[11], :]  # 'f7-t3'
    eegs_new[2, :] = eegs[ind_chs[11], :] - eegs[ind_chs[12], :]  # 't3-t5'
    eegs_new[3, :] = eegs[ind_chs[12], :] - eegs[ind_chs[4], :]   # 't5-o1'
    eegs_new[4, :] = eegs[ind_chs[0], :] - eegs[ind_chs[1], :]    # 'fp1-f3'
    eegs_new[5, :] = eegs[ind_chs[1], :] - eegs[ind_chs[2], :]    # 'f3-c3'
    eegs_new[6, :] = eegs[ind_chs[2], :] - eegs[ind_chs[3], :]    # 'c3-p3'
    eegs_new[7, :] = eegs[ind_chs[3], :] - eegs[ind_chs[4], :]    # 'p3-o1'
    eegs_new[8, :] = eegs[ind_chs[5], :] - eegs[ind_chs[6], :]    # 'fp2-f4'
    eegs_new[9, :] = eegs[ind_chs[6], :] - eegs[ind_chs[7], :]    # 'f4-c4'
    eegs_new[10, :] = eegs[ind_chs[7], :] - eegs[ind_chs[8], :]   # 'c4-p4'
    eegs_new[11, :] = eegs[ind_chs[8], :] - eegs[ind_chs[9], :]   # 'p4-o2'
    eegs_new[12, :] = eegs[ind_chs[5], :] - eegs[ind_chs[13], :]  # 'fp2-f8'
    eegs_new[13, :] = eegs[ind_chs[13], :] - eegs[ind_chs[14], :] # 'f8-t4'
    eegs_new[14, :] = eegs[ind_chs[14], :] - eegs[ind_chs[15], :] # 't4-t6'
    eegs_new[15, :] = eegs[ind_chs[15], :] - eegs[ind_chs[9], :]  # 't6-o2'
    eegs_new[16, :] = eegs[ind_chs[16], :] - eegs[ind_chs[17], :] # 'fz-cz'
    eegs_new[17, :] = eegs[ind_chs[17], :] - eegs[ind_chs[18], :] # 'cz-pz'

    return eegs_new
# =============================================================================
# ============================= Main Functions ================================
# =============================================================================


def featurizing_core_eeg(ch1, fs, bandf, delta_band, spindle_band, beta_band):
    
    # windowing ---------------------------------------------------------------
    window_length = 180*fs
    windows = rolling_window(ch1, window_length)
    k = 0
    
    q75, q25 = np.percentile(ch1, [75 ,25])
    iqr = q75 - q25
    thr1 = 5*iqr
    
    temp_phases = []
    for window in windows:
        
        if np.var(window) > np.finfo(np.float32).eps: # check for all dc values  
            
            if len(np.where(np.abs(window)>thr1)[0])>0: # check for artifacts
                temp = np.zeros((1, 43)) #**
                temp_phase = {'phase_1625' : [],
                            'phase_slow': []}
            else:
                temp, temp_phase = featurize_dynamic_winodows_eeg(window, fs, bandf, 
                                                      delta_band, spindle_band,
                                                      beta_band)
        else:
            temp = np.zeros((1, 43)) #**
            temp_phase = {'phase_1625' : [],
                        'phase_slow': []}
        # ---------------------------------------------------------------------
        if k == 0:
            segment_features = np.zeros((len(windows), 43)) #**
            temp_phase = {'phase_1625' : [],
                        'phase_slow': []}
        # ---------------------------------------------------------------------  
        segment_features[k, :] = temp
        temp_phases.append(temp_phase)
        k += 1
    # -------------------------------------------------------------------------
    return segment_features, temp_phases, windows
# =============================================================================


def featurizing_wrapper_perpatient_eeg(data, fs, bandf, delta_band, spindle_band, beta_band):
    """
    Parameters
    ----------
    data : each row is a channel, the first channel should be frontal or centeral
    fs : sampling frequency
    plot : The default is False

    Returns
    -------
    features : features for one patient

    """
    temp_phases_chs = []
    windows_chs = []
    for iter1 in range(data.shape[0]): # featurizing each channel
        
        segment_features, temp_phases, windows = featurizing_core_eeg(data[iter1, :], fs, bandf,
                                                delta_band, spindle_band, beta_band)
        if iter1 == 0:
            features = segment_features
        else:
            features = np.hstack((features, segment_features))

        temp_phases_chs.append(temp_phases)
        windows_chs.append(windows)
    return features, temp_phases_chs, windows_chs
# =============================================================================
# =============================================================================
# =============================================================================


def eeg_connectivity(data, fs):
    
    window_length = 180*fs
    num_windows = data.shape[1] // window_length
    try:
        eig_features, eig_cov_features, reig = eig_channel(data, fs)
    except:
        eig_features = np.zeros((num_windows, data.shape[0]))
        eig_cov_features = np.zeros((num_windows, data.shape[0]))
        reig = np.zeros((num_windows, 15))
    
    return eig_features, eig_cov_features, reig
# =============================================================================
# =============================================================================
# =============================================================================


# Extract features from the EEG data ==========================================
def get_eeg_features_AIrhythm(data, sampling_frequency, bandf, delta_band, spindle_band, beta_band):
    
    try:
        features, phases, windows_chs = featurizing_wrapper_perpatient_eeg(data, sampling_frequency, 
                                                      bandf, delta_band, 
                                                      spindle_band, beta_band)
        return features, phases, windows_chs
    except:
        # features = np.nan * np.ones((int(num_samples / (180*sampling_frequency)), 59*data.shape[0])) #**
        raise TypeError("Check the EEG features!!.")
# =============================================================================


def get_ecg_features_AIrhythm(ch_ecg, fs_ecg):
    
    qrs = utility_class_qrs_detection()
    ecgf = utility_class_ECG_v1()
    # -------------------------------------------------------------------------
    cutoff_freq = 0.05  # Cutoff frequency in Hz
    nyquist_freq = 0.5 * fs_ecg
    cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(1, cutoff, btype='highpass', analog=False)
    ch_ecg = signal.filtfilt(b, a, ch_ecg)
    # -------------------------------------------------------------------------
    q75, q25 = np.percentile(np.abs(ch_ecg), [75 ,25])
    iqr = q75 - q25
    thr1 = 5*iqr
    med = np.median(ch_ecg)
    ind1 = np.where(np.abs(ch_ecg) >= thr1)[0]
    ch_ecg[ind1] = med
    # windowing ---------------------------------------------------------------
    window_length = 180*fs_ecg
    windows = rolling_window(ch_ecg, window_length)
    k = 0
        
    for window in windows:
        
        if np.var(window) > np.finfo(np.float32).eps: # check for all dc values  
            # -----------------------------------------------------------------
            windows1 = rolling_window(np.diff(window), int(30*fs_ecg))
            k1 = 0
            for window1 in windows1:
                if np.sum(window1) < np.finfo(np.float32).eps:
                    k1 +=1
            if k1 > 0.7*len(windows1):
                ind_dc = True
            else:
                ind_dc = False
            # -----------------------------------------------------------------
            if not ind_dc:
                ris = qrs.qrs_detection_wrapper(window, fs_ecg, wd=20, plot=0, preprocess=False)
                if len(ris) > 50:
                    f1_ecg = ecgf.hrv_features_v1(ris, fs_ecg) #10
                    f2_ecg = ecgf.comput_AFEv(ris, fs_ecg) #3
                    f3_ecg = ecgf.sh_vs_nshself(window, fs_ecg) #3
                    temp = np.hstack((f1_ecg, f2_ecg, f3_ecg))
                else:
                    temp = np.zeros((1, 16))
            else:
                temp = np.zeros((1, 16))
        else:
            temp = np.zeros((1, 16)) #**
        # ---------------------------------------------------------------------
        if k == 0:
            segment_features = np.zeros((len(windows), 16)) #**
        # ---------------------------------------------------------------------  
        segment_features[k, :] = temp
        k += 1
    # -------------------------------------------------------------------------
    return segment_features  
      
    
    