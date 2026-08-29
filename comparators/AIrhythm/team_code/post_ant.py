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
from scipy import signal
from scipy.stats import linregress


def post_ant_core(a1, a2, fs):
    
    f1, psd1 = signal.welch(a1, fs, window='hamming', nperseg=2*fs)
    fb1 = np.where((f1>=8) & (f1<=13))
    alpha1 = np.log10(np.sum(psd1[fb1]))
    # -------------------------------------------------------------------------
    f2, psd2 = signal.welch(a2, fs, window='hamming', nperseg=2*fs)
    fb2 = np.where((f2>=8) & (f2<=13))
    alpha2 = np.log10(np.sum(psd2[fb2]))
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    log_f1 = np.log10(f1[1:])
    log_psd1 = np.log10(psd1[1:])
    slope1, _, _, _, _ = linregress(log_f1, log_psd1)
    slope_dB_per_Hz1 = 10 * slope1
    # -------------------------------------------------------------------------
    log_f2 = np.log10(f2[1:])
    log_psd2 = np.log10(psd2[1:])
    slope2, _, _, _, _ = linregress(log_f2, log_psd2)
    slope_dB_per_Hz2 = 10 * slope2
    
    return alpha1/alpha2, slope_dB_per_Hz1/slope_dB_per_Hz2
# =============================================================================


def post_ant(data, fs):
    window_length = 180*fs
    num_windows = data.shape[1] // window_length
    bis_features = np.zeros((num_windows, 2))
    # -------------------------------------------------------------------------
    start = 0
    end = window_length
    for iter1 in range(num_windows):
        temp = data[:, start:end]
        # ---------------------------------------------------------------------
        try:
            bis_features[iter1, :] = post_ant_core(temp[0, :], temp[1, :], fs)
            # -----------------------------------------------------------------
        except:
            pass
        start = end
        end += window_length
    # -------------------------------------------------------------------------
    return bis_features