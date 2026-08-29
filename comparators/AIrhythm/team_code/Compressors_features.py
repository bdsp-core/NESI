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

import gzip
import numpy as np

from pyts.approximation import SymbolicAggregateApproximation
# =============================================================================


def replace_nan_with_median_1(arr, default_value=0):
    # Check if any NaN is present in the array
    has_nan = np.isnan(arr).any()

    # If there are no NaNs, return the original array and flag=False
    if not has_nan:
        return arr, False

    # Check if all elements are NaN
    all_nan = np.isnan(arr).all()

    if all_nan:
        return [], True

    # Calculate the median of non-NaN elements
    non_nan_values = arr[~np.isnan(arr)]
    median_value = np.median(non_nan_values)

    # Replace NaNs with the median value
    arr[np.isnan(arr)] = median_value

    return arr, False
# =============================================================================


def Compressors_features(windows_chs, fs_eeg, n_bins=10):
    num_channels = 1#len(windows_chs)
    # -------------------------------------------------------------------------
    for iter0 in range(num_channels):
        ch1 = windows_chs[0] #windows_chs[iter0]
        num_windows = len(ch1)
        # ---------------------------------------------------------------------
        if iter0 == 0:
            ncds = np.zeros((num_windows, num_channels))
        # ---------------------------------------------------------------------        
        for iter1 in range(1, num_windows):
            temp0 = ch1[iter1-1]
            temp0 = temp0[:60*fs_eeg]
            temp0, flag0 = replace_nan_with_median_1(temp0)
            
            
            temp1 = ch1[iter1]
            temp1 = temp1[:60*fs_eeg]
            temp1, flag1 = replace_nan_with_median_1(temp1)
            # -----------------------------------------------------------------
            if (flag0 == False) and (flag1 == False):
                
                sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='uniform')
                eeg_sax = sax.fit_transform(temp0.reshape(1, -1))
                Cx = len(gzip.compress(eeg_sax))
                
                
                sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='uniform')
                eeg_sax = sax.fit_transform(temp1.reshape(1, -1))
                Cy = len(gzip.compress(eeg_sax)) 
                
                temp = np.hstack((temp0, temp1))
                sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='uniform')
                eeg_sax = sax.fit_transform(temp.reshape(1, -1))
                Cxy = len(gzip.compress(eeg_sax))
                
                ncd = (Cxy - np.min(np.array([Cx, Cy]))) / np.max(np.array([Cx, Cy]))
                # -------------------------------------------------------------
            else:
                ncd = 0
            # -----------------------------------------------------------------
            ncds[iter1, iter0] = ncd
    # -------------------------------------------------------------------------   
    ncds[0, :] = ncds[1, :]
    # -------------------------------------------------------------------------
    return ncds
