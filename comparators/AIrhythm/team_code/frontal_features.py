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

import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import welch
from scipy.stats import linregress
# =============================================================================


def apply_window_and_fft(epoch):
    windowed_epoch = epoch * np.blackman(len(epoch))
    return np.fft.rfft(windowed_epoch)
# =============================================================================


def BIS_core(channel1_eeg_waveform, fs):
        
    detrended_waveform = savgol_filter(channel1_eeg_waveform, window_length=51, polyorder=3)
    detrended_waveform = np.squeeze(detrended_waveform)
    epoch_length = 2 * fs  # 2 seconds at fs Hz
    overlap = int(epoch_length * 0.75)
    epochs = []
    for i in range(0, len(detrended_waveform)-epoch_length+1, overlap):
              epochs.append(detrended_waveform[i:i+epoch_length])
    
    spectral_powers = [np.abs(apply_window_and_fft(epoch))**2 for epoch in epochs]
    
    P_30_47Hz = np.mean([np.sum(spectral_power[30:48]) for spectral_power in spectral_powers])
    P_11_20Hz = np.mean([np.sum(spectral_power[11:21]) for spectral_power in spectral_powers])
    
    RBR = np.log10(P_30_47Hz / P_11_20Hz)
    
    return RBR
# =============================================================================


def alpha_ratio(temp, sampling_rate):
    temp = np.squeeze(temp)
    fft_result = np.fft.fft(temp)
    power_spectrum = np.abs(fft_result) ** 2

    frequencies = np.fft.fftfreq(len(temp), d=1/sampling_rate)

    low_alpha_band = (6, 12)       # 6–12 Hz
    high_alpha_band = (30, 42.5)   # 30–42.5 Hz

    low_alpha_indices = np.where((frequencies >= low_alpha_band[0]) & (frequencies <= low_alpha_band[1]))[0]
    high_alpha_indices = np.where((frequencies >= high_alpha_band[0]) & (frequencies <= high_alpha_band[1]))[0]

    spectral_energy_low_alpha = np.sum(power_spectrum[low_alpha_indices])
    spectral_energy_high_alpha = np.sum(power_spectrum[high_alpha_indices])

    alpha_ratio = np.log(spectral_energy_high_alpha / spectral_energy_low_alpha)

    return alpha_ratio
# =============================================================================


def calculate_psd_slope(eeg_channel, sampling_rate):
    
    f, psd = welch(eeg_channel, fs=sampling_rate, nperseg=256)

    log_f = np.log10(f[1:])
    log_psd = np.log10(psd[0, 1:])
    slope, _, _, _, _ = linregress(log_f, log_psd)
    slope_dB_per_Hz = 10 * slope

    return slope_dB_per_Hz
# =============================================================================


def frontal_features(data, fs):
    window_length = 180*fs
    num_windows = data.shape[1] // window_length
    bis_features = np.zeros((num_windows, 3))
    # -------------------------------------------------------------------------
    start = 0
    end = window_length
    for iter1 in range(num_windows):
        temp = data[:, start:end]
        # ---------------------------------------------------------------------
        try:
            bis_features[iter1, 0] = BIS_core(temp, fs)
            bis_features[iter1, 1] = alpha_ratio(temp, fs)
            bis_features[iter1, 2] = calculate_psd_slope(temp, fs)
            # -----------------------------------------------------------------
        except:
            pass
        start = end
        end += window_length
    # -------------------------------------------------------------------------
    return bis_features
