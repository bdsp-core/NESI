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


import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal



class utility_class_qrs_detection:
    def __init__(self):
        self.seed = 0    
    # #########################################################################
    def qrs_detection_wrapper(self, ecg_signal, fs, wd=10, plot=0, preprocess=False):
        """
        Tested on at least 20 min ECG sig

        Parameters
        ----------
        ecg_signal : 1 channel ECG
        fs : Samling frequency
        wd: in seconds
        plot : The option to plot, not recommended if the signal is long.
        The default is 0.

        Returns
        -------
        peaks_all_array : the numpy array of contains the location of Rs
        
        example:
        ecg_signal = dataset['value']
        fs = dataset['sample_freq']
        fs = fs[0]
        fs = int(fs[0])
        # ecg_signal = ecg_signal[15*60*fs:35*60*fs]

        """
        peaks_all_array = []
        # ---------------------------------------------------------------------
        # preprocessing -------------------------------------------------------
        if preprocess:
            level = 7
            Wtype = 'db4'
            
            coeffs  = pywt.wavedec(ecg_signal, Wtype, level=level)
            cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs        
            
            Mapp = np.zeros_like(cA7)
            McD2 = np.zeros_like(cD2)
            McD1 = np.zeros_like(cD1)
            
            Mc = [Mapp, cD7, cD6, cD5, cD4, cD3, McD2, McD1]
            recD = pywt.waverec(Mc, Wtype)
        else:
            recD = ecg_signal
        # thr detection -------------------------------------------------------
        thrs = []
        
        l = wd*fs
        # Iterate over windows
        for i in range(0, len(recD), l):
            window = recD[i:i+l]  # Extract windowed data
            if np.var(window) > np.finfo(np.float32).eps:
                thr = np.percentile(np.abs(window), 98)
                thrs.append(thr)
        
        thr = np.percentile(thrs, 1)
        
        # peak detection ------------------------------------------------------
        peaks_all = []
        min_peak_distance = int(0.4*fs)
        
        # Iterate over windows
        for i in range(0, len(recD), l):
            window = recD[i:i+l]  # Extract windowed data
            if np.var(window) > 0.000001:
                peaks, _ = find_peaks(np.abs(window), height=thr, distance=min_peak_distance)
                peaks_all.append(peaks + i)
                
        peaks_all_array = np.concatenate(peaks_all)
        # post processing -----------------------------------------------------
        for iter1 in range(1, len(peaks_all_array)-1):
            peak = peaks_all_array[iter1]
            
            llim = peak - int(0.125*fs)
            hlim = peak + int(0.125*fs)
            if llim >0 & hlim <len(ecg_signal):
                if llim>peaks_all_array[iter1-1] & hlim<peaks_all_array[iter1+1]:
                    window = ecg_signal[llim:hlim]
                    window = np.abs(window)
                    
                    peak_new = np.argmax(window)
                    peak_new += llim
                    peaks_all_array[iter1] = peak_new
        # post processing 1 (if the peaks inteval is more than 1.6s)-----------
        intervals = np.diff(peaks_all_array)
        missedlocs = np.where(intervals > fs*1.6)[0]
        if len(missedlocs) > 0:
            peaks_new = []
            for iter1 in range(len(missedlocs)):
                llim = peaks_all_array[missedlocs[iter1]]
                hlim = peaks_all_array[missedlocs[iter1]+1]
                window = ecg_signal[llim:hlim]
                window = np.abs(window)
                peak_new = np.argmax(window)
                peak_new += llim
                peaks_new.append(peak_new)
            
            peaks_new = np.array(peaks_new)
            peaks_all_array = np.append(peaks_all_array, peaks_new)
            peaks_all_array = np.sort(peaks_all_array)
        # post processing 2 (if the peaks are closer than 0.099s)--------------
        intervals = np.diff(peaks_all_array)
        tooclose = np.where(intervals < 0.099*fs)[0]
        peaks_all_array = np.delete(peaks_all_array, tooclose)
        # post processing -----------------------------------------------------
        for iter1 in range(1, len(peaks_all_array)-1):
            peak = peaks_all_array[iter1]
            
            llim = peak - int(0.125*fs)
            hlim = peak + int(0.125*fs)
            if llim >0 & hlim <len(ecg_signal):
                if llim>peaks_all_array[iter1-1] & hlim<peaks_all_array[iter1+1]:
                    window = ecg_signal[llim:hlim]
                    window = np.abs(window)
                    
                    peak_new = np.argmax(window)
                    peak_new += llim
                    peaks_all_array[iter1] = peak_new
        # plot ----------------------------------------------------------------
        if plot:
            plt.figure()
            plt.plot(ecg_signal)
            plt.plot(peaks_all_array, ecg_signal[peaks_all_array], 'ro')
            plt.title('Signal with Peaks')
            plt.show()
            
        return peaks_all_array
