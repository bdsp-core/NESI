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
import esig

# =============================================================================
# =============================================================================


def phase_space_reconstruction(time_series, embedding_dimension, embedding_delay):
    num_points = len(time_series) - (embedding_dimension - 1) * embedding_delay
    indices = np.arange(0, embedding_dimension * embedding_delay, embedding_delay)
    phase_space = time_series[indices + np.arange(num_points).reshape(-1, 1)]
    return phase_space
# =============================================================================


def tangent_signiture(data, fs):
        
    window_length = 180*fs
    num_windows = data.shape[1] // window_length
    # -------------------------------------------------------------------------
    lag = int(0.3*fs)
    embedding_dim = 3
    lwind = 6
    depth = 2
    start = 0
    end = window_length
    
    sig_features = np.zeros((num_windows, 6*data.shape[0]))
    for iter1 in range(num_windows):
        temp = data[:, start:end]
        temp = temp[:, :fs*90]
        # ---------------------------------------------------------------------
        
        start1 = 0
        end1  = 6
        for iter2 in range(temp.shape[0]):
            temp1 = phase_space_reconstruction(temp[iter2, :], embedding_dim, lag)
            sig = esig.stream2sig(temp1, depth)
            sigdiff = np.diff(sig)
                        
            sig_features[iter1, start1:end1] = np.quantile(sigdiff, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95])
            start1 = end1
            end1 += lwind
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        start = end
        end += window_length
    # -------------------------------------------------------------------------
    
    return sig_features
    
    
    
