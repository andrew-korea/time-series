#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

from scipy.signal import find_peaks, peak_prominences
from scipy.signal import savgol_filter

from var_and_z_score.zScore import z_outlier


def variance_detection(data_list,split):
    
    #we apply a Fourier Transform on the data and convert the time-series to the frequency domain
    tmp = data_list
    tmp -=np.mean(tmp)
    s = np.fft.fft(tmp)
    s =s[0:1000]
    
    #the FFT graph is smoothened using the Savitzky-Golay filter.
    s_abs = abs(s)
    s_abs[np.argmax(s_abs)] = 0
    s_sav = savgol_filter(s_abs, 31, 5)


    #prominent peaks are scanned for.
    prom = (np.amax(abs(s_sav)))//4
    peaks, _ = find_peaks(s_sav, prominence = prom)
    
    #the first peak index is the periodicity of the time-series
    period = peaks[0]
    
    
    #Rolling Variance
    window = period
    v = pd.Series(data_list)
    v = v.rolling(window).var()

    
    #we clean up NaN values
    v = v.to_numpy()
    v = np.nan_to_num(v,nan=np.nanmean(v))
    
    #find the outlier using the z-score of the rolling variance graph
    #print("+++++++++Variance Score+++++++++")
    outlier, conf = z_outlier(v, split)
    
    return outlier, conf

