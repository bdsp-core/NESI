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
import pywt
from scipy import signal
from scipy.stats import entropy
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from sklearn.decomposition import PCA


class utility_class_ECG_v1:
    def __init__(self):
        self.seed = 0
    # #########################################################################
    def sh_vs_nshself(self, row, fs):
        
        lowcut = 6.5
        highcut = 30
        order = 5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, row)
        
        # Normalize to amplitude one
        normalized_ecg = (filtered_signal - np.mean(filtered_signal)) /  np.std(filtered_signal)
        f1 = np.max(normalized_ecg) - np.min(normalized_ecg)
        # ---------------------------------------------------------------------
        diff_ecg = np.diff(row)
        # Square the first-difference and normalize to amplitude one
        xd = (diff_ecg ** 2) / np.max(diff_ecg ** 2)
        
        # Compute the proportion of time xd is below the threshold (ThS)
        ThS = 0.1
        f2 = np.mean(xd < ThS)
        # ---------------------------------------------------------------------
        # Calculate the 1024-point FFT of the ECG segment
        power_proportion = 0.2
        fft_result = fft(row, 1024)
        # Calculate the power spectrum
        power_spectrum = np.abs(fft_result) ** 2
    
        # Normalize the power spectrum to unit area under the curve
        normalized_spectrum = power_spectrum / np.sum(power_spectrum)
    
        # Calculate the cumulative sum of the normalized spectrum
        cumulative_spectrum = np.cumsum(normalized_spectrum)
    
        # Find the frequency above which the given proportion of power is contained
        fH_index = np.argmax(cumulative_spectrum >= power_proportion)
        fH = fH_index / 1024
    
        # Find the frequency below which the given proportion of power is contained
        fL_index = np.argmax(cumulative_spectrum >= (1 - power_proportion))
        fL = fL_index / 1024
    
        # Calculate the bandwidth
        f3 = fH - fL
        
        fshnsh = np.hstack((f1, f2, f3))
        # ---------------------------------------------------------------------
        return fshnsh     
    # #########################################################################
    def calculate_regularity(self, row, fs):
        window_size = int(fs // 4)
        smoothed_signal = np.convolve(row, np.ones(window_size)/window_size, mode='same')
        sorted_signal = np.sort(smoothed_signal)[::-1]  # sort in descending order
        std_norm = np.std(sorted_signal) / np.mean(sorted_signal)  # normalized standard deviation
        return std_norm    
    # #########################################################################
    def autocorrelation_fx(self, x):
        r = np.correlate(x, x, mode='full')[len(x)-1:]
        r = r[500:]
        analytic_signal = r - np.mean(r)
        zero_crossings = np.where(np.diff(np.sign(analytic_signal)))[0]
        num_zero_crossings = len(zero_crossings)/2
        return num_zero_crossings
    # #########################################################################
    def estimate_C0_complexity(self, row):
        # DOI: 10.1109/IEMBS.2008.4649615
    
        X = np.fft.fft(row)
        M = np.mean(np.abs(X)**2)
        Y = X.copy()
        Y[np.abs(Y)**2 < M] = 0
        y = np.fft.ifft(Y)
        
        A1 = np.sum(np.power(np.abs(row - y), 2))
        A0 = np.sum(np.power(np.abs(row), 2))
        Co = A1/A0
        return Co
    # #########################################################################
    def wavelet_features(self, row, fs, scale=4):
        coeffs = pywt.wavedec(row, 'db4', level=scale)
        cA4, cD4, cD3, cD2, cD1 = coeffs
        c_var = [np.std(cD1), np.std(cD2), np.std(cD3), np.std(cD4),
                 np.mean(cD1), np.mean(cD2), np.mean(cD3), np.mean(cD4)]
        return c_var
    # #########################################################################
    def calc_shannon_entropy(self, row, levels=30):
        z = (row - np.mean(row)) / np.std(row)
        x_scaled = (z - np.min(z)) / (np.max(z) - np.min(z))
        scale = 1/levels
        x_quantized = np.floor(x_scaled / scale)
        edges = np.arange(levels+1)
        counts, _ = np.histogram(x_quantized, edges)
        counts = counts / len(x_quantized)
        shen = -np.sum(counts * np.log10(counts + 0.00000001))
        return shen
    # #########################################################################
    def signal_power(self, row):
        signal_length = len(row)
        power = np.sum(np.abs(row)**2) / signal_length
        return power    
    # #########################################################################
    def hilbert_features(self, row, fs):
        analytic = signal.hilbert(row)
        inst_phase = np.angle(analytic)
        inst_pow = np.square(np.abs(analytic))
        inst_freq = fs / (2 * np.pi) * np.diff(inst_phase, axis=-1)
        # ---------------------------------------------------------------------
        sp_abs = np.median(np.log10(inst_pow[inst_pow > 0]))
        sp_freq = np.median(inst_freq[inst_freq > 0])
        return sp_abs, sp_freq
    # #########################################################################
    def hurst(self, row):
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(row[lag:], row[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    # #########################################################################
    def skw_zc(self, row):
        analytic_signal = hilbert(row)
        analytic_signal = analytic_signal - np.mean(analytic_signal)
        zero_crossings = np.where(np.diff(np.sign(analytic_signal)))[0]
        num_zero_crossings = len(zero_crossings)
        if num_zero_crossings == 0:
            return 0
        mean_zero_crossing = np.mean(zero_crossings)
        skewness = np.sum((zero_crossings - mean_zero_crossing)**3) / (num_zero_crossings * np.std(zero_crossings)**3)
        return skewness
    # #########################################################################
    def psd_complexity(self, row, fs):
        # ---------------------------------------------------------------------
        fx, psdx = signal.welch(row, fs, nperseg=fs*2, noverlap=fs, scaling='density')
        # ---------------------------------------------------------------------
        N = len(psdx)
        # ---------------------------------------------------------------------
        diff1 = np.diff(psdx)
        nzc_deriv = np.diff(np.signbit(diff1)).sum()
        pfd_psdx = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * nzc_deriv)))
        # ---------------------------------------------------------------------
        spec_centroid = np.sum(psdx * fx) / np.sum(psdx)
        # ---------------------------------------------------------------------
        Pxx_norm = psdx / np.sum(psdx)
        # ---------------------------------------------------------------------
        spec_entropy = -np.sum(Pxx_norm * np.log2(Pxx_norm))        
        # ---------------------------------------------------------------------
        cumsum_Pxx_norm = np.cumsum(Pxx_norm)
        spec_roll_off = fx[np.where(cumsum_Pxx_norm >= 0.85)[0][0]]
        # ---------------------------------------------------------------------
        mean_norm_freq = np.sum(fx * Pxx_norm)
        # ---------------------------------------------------------------------
        return pfd_psdx, spec_centroid, spec_entropy, spec_roll_off, mean_norm_freq
    # #########################################################################
    def stft_features(self, row, fs, bandf, epsilon):
        [f1, t, X] = signal.spectrogram(row, fs=fs, window='hann',
                                 nperseg=int(5*fs), noverlap=int(4*fs),
                                 detrend=False,
                                 return_onesided=True, scaling='spectrum')
        ind_0 = np.where(X == 0)
        X[ind_0] = epsilon
        X = 10*np.log10(X)
        energy_stft = np.zeros([1, (len(bandf)-1)])
        for i in range(len(bandf)-1):
            fr = np.where((f1>bandf[i]) & (f1<=bandf[i+1]))
            energy_stft[0, i] = np.sum(np.power(X[fr], 2))

        fr1 = np.where((f1>bandf[0]) & (f1<=bandf[-1]))
        energy_stft = energy_stft / np.sum(np.power(X[fr1], 2))
        # ---------------------------------------------------------------------
        fr1 = np.where((f1>11) & (f1<=16))
        semi_sigma_index = np.max(np.power(X[fr1], 2))
        # ---------------------------------------------------------------------
        return energy_stft, semi_sigma_index
    # #########################################################################
    def Hjorth_q(self, row):
        ha = np.var(row)
        hm = np.sqrt(np.var(np.gradient(row)) / ha)
        hm1 = np.sqrt(np.var(np.gradient(np.gradient(row))) /
                      np.var(np.gradient(row)))
        hc = hm1 / hm
        return ha, hm, hc
    # #########################################################################
    def decay_psd(self, f, psd, bandf):
        dp = []
        for i in range(len(bandf)-1):
            ind = np.where((f>bandf[i]) & (f<=bandf[i+1]))[0]
            w = f[ind]
            F = np.log10(psd[ind])
            m = np.polyfit(w, F, 1)
            dp.append(m[0])
        dp = np.array(dp)
        return dp
    # #########################################################################
    def dimension(self, row):
        # ref: https://github.com/raphaelvallat/entropy/blob/master/entropy/fractal.py
        N = len(row)
        # ---------------------------------------------------------------------
        diff1 = np.diff(row)
        nzc_deriv = np.diff(np.signbit(diff1)).sum()
        pfd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * nzc_deriv)))
        # ---------------------------------------------------------------------
        dists = np.abs(np.diff(row))
        ll = dists.sum()
        ln = np.log10(ll / dists.mean())
        aux_d = row - np.take(row, indices=[0])
        d = np.max(np.abs(aux_d))
        kfd = np.squeeze(ln / (ln + np.log10(d / ll)))
        return pfd, kfd
    # #########################################################################
    def wavelet_entropy(self, row):
        coeffs = pywt.wavedec(row, 'db4', level=7)
        # ---------------------------------------------------------------------
        energyd = []
        for iter1 in range(len(coeffs)):
            energyd.append(np.sum(np.power(coeffs[iter1], 2)))

        energyd = np.array(energyd)
        energyd = energyd / np.sum(energyd)
        energyd += 1
        wen = entropy(energyd)
        return wen
    # #########################################################################
    # ########################## ECG specific #################################
    # #########################################################################
    def hrv_features_v1(self, R_i2, fs):
        
        # ---------------------------------------------------------------------
        diff_nni = np.diff(R_i2) # in sample
        diff_nni = diff_nni / fs # in seconds
        diff_nni = diff_nni * 1000 # in millisecond 
        
        NNx = sum(np.abs(diff_nni) > 50)
        # ---------------------------------------------------------------------
        length_int = len(R_i2)
        diff_nni = np.diff(R_i2)
        nni_50 = sum(np.abs(diff_nni) > 50)
        pNNx = 100 * nni_50 / length_int
        # ---------------------------------------------------------------------
        diff_nn_intervals = np.diff(R_i2)
        SD1 =  np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
        # ---------------------------------------------------------------------
        SD2 =  np.sqrt(2 * np.std(R_i2, ddof=1) ** 2 - 0.5 * np.std(\
                       diff_nn_intervals, ddof=1) ** 2)
        # ---------------------------------------------------------------------
        ratio_sd2_sd1 = SD2 / (SD1 + np.finfo(np.float32).eps)
        # ---------------------------------------------------------------------       
        L = 4 * SD1
        T = 4 * SD2
        CSI = L/T
        # ---------------------------------------------------------------------
        CVI = np.log10((L * T) + np.finfo(np.float32).eps)
        # ---------------------------------------------------------------------     
        modifiedCVI =  L ** 2 / T
        # ---------------------------------------------------------------------
        ff21 = np.mean(diff_nn_intervals)
        ff22 = np.std(diff_nn_intervals)
        f1 = np.sqrt(np.mean(diff_nn_intervals ** 2))
        f2 = ff22 / ff21
        # ---------------------------------------------------------------------
        feature_HRV_1 = [NNx, pNNx, SD1, SD2, ratio_sd2_sd1, CSI, CVI, modifiedCVI, f1, f2]
        feature_HRV_1 = np.array(feature_HRV_1)
        return feature_HRV_1    
    # #########################################################################
    def comput_AFEv(self, segment, fs):
        """
        %%
        % //This software is licensed under the BSD 3 Clause license: http://opensource.org/licenses/BSD-3-Clause 
        % 
        % //Copyright (c) 2013, University of Oxford
        % //All rights reserved.
        % 
        % //Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
        % 
        % //Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        % //Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
        % //Neither the name of the University of Oxford nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
        % //THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        %
        %   The method implemented in this file has been patented by their original
        %   authors. Commercial use of this code is thus strongly not
        %   recocomended.
        %
        % //Authors: 	Gari D Clifford - 
        % //            Roberta Colloca -
        % //			Julien Oster	-
        """
        
        segment = np.diff(segment)/fs
        # Compute dRR intervals series
        dRR = self.comp_dRR(segment)
        
        # Compute metrics
        OriginCount, IrrEv, PACEv, DensityEvidence, AnisotropyEvidence = self.metrics(dRR)
        
        # Compute AFEvidence
        AFEv = IrrEv - OriginCount - 2 * PACEv
        
        # Compute ATEvidence
        RegularityEvidence = 0
        ATEvidence = IrrEv + AnisotropyEvidence + DensityEvidence + RegularityEvidence - 4 * PACEv
        
        # Compute OrgIndex
        OrgIndex = OriginCount + AnisotropyEvidence + DensityEvidence + RegularityEvidence - 2 * IrrEv
        
        feature_HRV_2 = [AFEv, ATEvidence, OrgIndex]
        feature_HRV_2 = np.array(feature_HRV_2)
        return feature_HRV_2
    # #########################################################################
    def Pbased_BLACKSWAN(self, x, ris, sampleRate):
            
        PWAVE = self.CheckPWAVE(x, ris, sampleRate)
    
        if PWAVE.shape[0] > 0:
            W1 = np.copy(PWAVE)
            NN, MM = W1.shape
            
            # Normalization of the input time series
            W1 = (W1 - np.mean(W1.flatten())) / np.std(W1.flatten())
            N = MM
            
            # Scale between 0-1
            W1 = (W1 - np.min(W1.flatten())) / (np.max(W1.flatten()) - np.min(W1.flatten()))
            
            # Quantize the signal
            qLevels = 100
            scalingFactor = 1 / qLevels
            W1 = np.floor(W1 / scalingFactor)
            
            # Compute the probability
            edges = np.arange(101)
            counts, edges1 = np.histogram(W1.flatten(), edges)
            
            pr = np.zeros((NN, MM))
            for iter1 in range(NN):
                for iter2 in range(MM):
                    indcount = np.where(edges1 == W1[iter1, iter2])[0]
                    if indcount > 0:
                        indcount = indcount - 1
                    pr[iter1, iter2] = counts[indcount] / N
            
            pr[pr == 0] = 1e-17
            Renyi = (1 / (2 - 1)) * np.log(np.sum(np.power(pr.flatten(), 2)))
            F22 = np.mean(np.std(PWAVE, axis=1))
            
            U = np.corrcoef(PWAVE)
            U = np.triu(U)
            U1 = np.sum(U) - U.shape[1]
            F33 = U1 / PWAVE.shape[0]
            
            U = U.flatten()
            U = U[U != 0]
            F44 = np.std(U)
            
            F55 = np.max(np.mean(PWAVE, axis=1))
        else:
            Renyi = 0
            F22 = 0
            F33 = 0
            F44 = 0
            F55 = 0
        
        Pbased_FEATURE = [F22, F33, F44, F55, Renyi]
        Pbased_FEATURE = np.array(Pbased_FEATURE)
        return Pbased_FEATURE
    # #########################################################################
    def beat_based_feature(self, x, R_i2, fs):
        
        TW  = 0.5*fs
        
        if len(R_i2) <= 3:
            X_FeaturesSpace1 = np.zeros(10)
            X_FeaturesSpace2 = np.zeros(10)
            f7 = np.zeros(5)
        else:
            f = 1
            b, a = butter(3, 2 * f / fs, 'high')
            x = filtfilt(b, a, x)
                
            data1 = np.zeros((1, int(TW * 2 )))
            k = 0
            for i in range(1, len(R_i2) - 1):
                lim1 = int(R_i2[i] - TW)
                lim2 = int(R_i2[i] + TW)
                if (lim1 >= 0) & (lim2 <= len(x) - 1):
                    if k == 0:
                        data1[k, :] = x[lim1:lim2]
                    else:
                        data1 = np.vstack((data1, x[lim1:lim2]))
                     
                    k +=1
        
            MEANdata1 = np.mean(data1, axis=0)
            STDdata1 = np.std(data1, axis=0)
            X_FeaturesSpace1 = self.wavelet_features(MEANdata1, fs, scale=4)
            X_FeaturesSpace2 = self.wavelet_features(STDdata1, fs, scale=4)
            # -----------------------------------------------------------------
            corr_coef = np.zeros(data1.shape[0])
            for iter1 in range(data1.shape[0]):
                corr_coef[iter1] = np.corrcoef(MEANdata1, data1[iter1, :])[0, 1]

            corr_coef = 1 - corr_coef
            ind_abnormal = np.where(np.abs(corr_coef)>0.4)[0]
            # data1_abnormal = data1[ind_abnormal, :]
            
            f1 = len(ind_abnormal) / data1.shape[0]
            f2 = np.max(corr_coef)
            
            p_data1_converge = data1[:, 40:45]
            r_data1_converge = data1[:, 59:67]
            s_data1_converge = data1[:, 67:72]
            t_data1_converge = data1[:, 96:102]
            
            f3 = np.quantile(p_data1_converge.flatten(), [0.25, 0.5, 0.75, 0.95])
            f4 = np.quantile(r_data1_converge.flatten(), [0.25, 0.5, 0.75, 0.95])
            f5 = np.quantile(s_data1_converge.flatten(), [0.25, 0.5, 0.75, 0.95])
            f6 = np.quantile(t_data1_converge.flatten(), [0.25, 0.5, 0.75, 0.95])
            
            f7 = np.append(np.array([f1, f2]), np.concatenate((f3, f4, f5, f6)))
            # -----------------------------------------------------------------
        if len(R_i2) <= 10:
            feature_eig = np.zeros(7)
        else:
            pca = PCA(n_components=8)
            pca.fit(data1)
            PC_val = pca.explained_variance_
            eigen_val = np.square(PC_val)
            eigen_val1 = eigen_val / eigen_val[0]
            feature_eig = eigen_val1[1:8]
        
        feature_pca = np.concatenate((feature_eig, X_FeaturesSpace1, X_FeaturesSpace2, f7))
        
        return feature_pca
    # #########################################################################
    # ############################### Helper functions ########################
    # #########################################################################
    def comp_dRR(self, data):
        """
        % //This software is licensed under the BSD 3 Clause license: http://opensource.org/licenses/BSD-3-Clause 
        % 
        % 
        % //Copyright (c) 2013, University of Oxford
        % //All rights reserved.
        % 
        % //Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
        % 
        % //Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        % //Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
        % //Neither the name of the University of Oxford nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
        % //THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        %
        %   The method implemented in this file has been patented by their original
        %   authors. Commercial use of this code is thus strongly not
        %   recocomended.
        %
        % //Authors: 	Gari D Clifford - 
        % //            Roberta Colloca -
        % //			Julien Oster	-
        """
        # RR_s[:,0] = RR[i] and RR_s[:,1] = RR[i-1]
        RR_s = np.column_stack((data[1:], data[:-1]))
        dRR_s = np.zeros((len(data)-1, 1))
    
        # Normalization factors (normalize according to the heart rate)
        k1 = 2
        k2 = 0.5
    
        for i in range(len(RR_s)):
            if np.sum(RR_s[i, :] < 0.500) >= 1:
                dRR_s[i, 0] = k1 * (RR_s[i, 0] - RR_s[i, 1])
            elif np.sum(RR_s[i, :] > 1) >= 1:
                dRR_s[i, 0] = k2 * (RR_s[i, 0] - RR_s[i, 1])
            else:
                dRR_s[i, 0] = RR_s[i, 0] - RR_s[i, 1]
    
        return dRR_s
    # #########################################################################
    def metrics(self, dRR):
        # dRR={dRR(i),dRR(i-1)}
        dRR = np.column_stack((dRR[1:], dRR[:-1]))
        
        # COMPUTE OriginCount
        OCmask = 0.02
        os = np.sum(np.abs(dRR) <= OCmask, axis=1)
        OriginCount = np.sum(os == 2)
    
        # DELETE OUTLIERS |dRR|>=1.5
        OLmask = 1.5
        dRRnew = []
        for i in range(dRR.shape[0]):
            if np.sum(np.abs(dRR[i]) >= OLmask) == 0:
                dRRnew.append(dRR[i])
        dRRnew = np.array(dRRnew)
        if dRRnew.shape[0] == 0:
            dRRnew = np.array([[0, 0]])
    
        # BUILD HISTOGRAM
        bin_c = np.arange(-0.58, 0.59, 0.04)
    
        # Three dimensional histogram of bivariate data
        Z, _, _ = np.histogram2d(dRRnew[:, 0], dRRnew[:, 1], bins=[bin_c, bin_c])
    
        # Clear SegmentZero
        Z[13:16, 14:17] = 0
        Z[14:17, 13:18] = 0
        Z[16:19, 14:17] = 0
    
        # Z2 contains all the bins belonging to the II quadrant of Z
        Z2 = Z[15:30, 15:30]
        BC12, PC12, sZ2 = self.BPcount(Z2)
        Z[15:30, 15:30] = sZ2
    
        # Z3 contains points belonging to the III quadrant of Z
        Z3 = Z[15:30, 0:15]
        Z3 = np.fliplr(Z3)
        BC11, PC11, sZ3 = self.BPcount(Z3)
        Z[15:30, 0:15] = np.fliplr(sZ3)
    
        # Z4 contains points belonging to the IV quadrant of Z
        Z4 = Z[0:15, 0:15]
        BC10, PC10, sZ4 = self.BPcount(Z4)
        Z[0:15, 0:15] = sZ4
    
        # COMPUTE BinCount9, PointCount9
        # Z1 contains points belonging to the I quadrant of Z
        Z1 = Z[0:15, 15:30]
        Z1 = np.fliplr(Z1)
        BC9, PC9, sZ1 = self.BPcount(Z1)
        Z[0:15, 15:30] = np.fliplr(sZ1)
    
        # COMPUTE BinCount5, PointCount5
        BC5 = np.sum(Z[0:15, 13:17] != 0)
        PC5 = np.sum(Z[0:15, 13:17])
    
        # COMPUTE BinCount7, PointCount7
        BC7 = np.sum(Z[15:30, 13:17] != 0)
        PC7 = np.sum(Z[15:30, 13:17])
    
        # COMPUTE BinCount6, PointCount6
        BC6 = np.sum(Z[13:17, 0:15] != 0)
        PC6 = np.sum(Z[13:17, 0:15])
    
        # COMPUTE BinCount8, PointCount8
        BC8 = np.sum(Z[13:17, 15:30] != 0)
        PC8 = np.sum(Z[13:17, 15:30])
    
        # CLEAR SEGMENTS 5, 6, 7, 8
        Z[13:17, :] = 0
        Z[:, 13:17] = 0
    
        # COMPUTE BinCount2, PointCount2
        BC2 = np.sum(Z[0:13, 0:13] != 0)
        PC2 = np.sum(Z[0:13, 0:13])
    
        # COMPUTE BinCount1, PointCount1
        BC1 = np.sum(Z[0:13, 17:30] != 0)
        PC1 = np.sum(Z[0:13, 17:30])
    
        # COMPUTE BinCount3, PointCount3
        BC3 = np.sum(Z[17:30, 0:13] != 0)
        PC3 = np.sum(Z[17:30, 0:13])
    
        # COMPUTE BinCount4, PointCount4
        BC4 = np.sum(Z[17:30, 17:30] != 0)
        PC4 = np.sum(Z[17:30, 17:30])
    
        # COMPUTE IrregularityEvidence
        IrrEv = BC1 + BC2 + BC3 + BC4 + BC5 + BC6 + BC7 + BC8 + BC9 + BC10 + BC11 + BC12
    
        # COMPUTE PACEvidence
        PACEv = (PC1 - BC1) + (PC2 - BC2) + (PC3 - BC3) + (PC4 - BC4) + (PC5 - BC5) + (PC6 - BC6) + (PC10 - BC10) - (PC7 - BC7) - (PC8 - BC8) - (PC12 - BC12)
        
        # COMPUTE DensityEvidence
        DensityEvidence = (PC5 - BC5) + (PC6 - BC6) + (PC10 - BC10) - (PC7 - BC7) - (PC8 - BC8) - (PC12 - BC12)
    
        # COMPUTE AnisotropyEvidence
        AnisotropyEvidence = abs((BC9 + BC11) - (BC10 + BC12)) + abs((BC6 + BC7) - (BC5 + BC8))
    
        return OriginCount, IrrEv, PACEv, DensityEvidence, AnisotropyEvidence
    
    # #########################################################################
    def BPcount(self, sZ):
        """
        % //This software is licensed under the BSD 3 Clause license: http://opensource.org/licenses/BSD-3-Clause 
        % 
        % 
        % //Copyright (c) 2013, University of Oxford
        % //All rights reserved.
        % 
        % //Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
        % 
        % //Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        % //Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
        % //Neither the name of the University of Oxford nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
        % //THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        %
        %   The method implemented in this file has been patented by their original
        %   authors. Commercial use of this code is thus strongly not
        %   recocomended.
        %
        % //Authors: 	Gari D Clifford - 
        % //            Roberta Colloca -
        % //			Julien Oster	-
        """
        # BPcount counts the number of non empty bins (BC) in the matrix Z and
        # the number of {dRR(i),dRR(i-1)} in the same region (PC) and deletes
        # the counted points from the matrix Z
    
        # bdc is the BIN diagonal count: number of non empty bins contained in
        # the i-th diagonal of Z
        bdc = 0
        BC = 0
        # pdc is the POINTS diagonal count: number of {dRR(i),dRR(i-1)} contained in
        # the i-th diagonal of Z
        pdc = 0
        PC = 0
        
        for i in range(-2, 3):
            diag_vals = np.diag(sZ, i)
            bdc = np.sum(diag_vals != 0)
            pdc = np.sum(diag_vals)
            BC += bdc
            PC += pdc
            np.fill_diagonal(sZ[i:], 0)
        return BC, PC, sZ
    
    # #########################################################################
    def CheckPWAVE(self, AAA, R_i2, sampleRate):
        
        
        PWAVE = np.empty((0, 128))  # Initialize empty array for PWAVE
        
        if len(R_i2) > 2:
            k = 0
            for i1 in range(1, len(R_i2)-1):
                Endd = R_i2[i1] - int(0.06*sampleRate)
                Startt = Endd - int((R_i2[i1] - R_i2[i1-1])/2) - int(0.06*sampleRate)
                
                Pwave = AAA[Startt:Endd]
                xj = np.arange(1, len(Pwave)+1)
                xp = np.linspace(xj[0], xj[-1], 128)
                PWAVE = np.vstack((PWAVE, np.interp(xp, xj, Pwave)))
                k += 1
                Endd, Startt, Pwave, xj, xp = None, None, None, None, None  # Clear variables
            
        else:
            PWAVE = np.array([])
    
        return PWAVE
    # #########################################################################
    
    