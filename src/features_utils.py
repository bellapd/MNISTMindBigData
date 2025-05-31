# src/features.py

import numpy as np
from scipy.fft import rfft, rfftfreq

def extract_fft_bandpower(x_clean, fs, bands=None):
    """
    Compute average bandpower (FFT) per channel for each trial.
    
    Parameters
    ----------
    x_clean : np.ndarray, shape (n_trials, n_channels, n_times)
        Cleaned EEG (µV).
    fs : float
        Sampling rate in Hz.
    bands : dict, optional
        Mapping { 'delta': (1,4), 'theta': (4,8), ... }. 
        If None, defaults to {'delta':(1,4), 'theta':(4,8), 'alpha':(8,12), 'beta':(12,30)}.
    
    Returns
    -------
    features : np.ndarray, shape (n_trials, n_channels * n_bands)
        Each row is concatenated bandpower features per channel.
    freqs : np.ndarray, shape (n_freqs,)
        The frequency bins from rfftfreq.
    """
    if bands is None:
        bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}
    
    n_trials, n_channels, n_times = x_clean.shape
    freqs = rfftfreq(n_times, 1.0 / fs)
    power = np.abs(rfft(x_clean, axis=2))**2

    features = np.zeros((n_trials, n_channels * len(bands)), dtype=np.float32)

    for t in range(n_trials):
        for ch in range(n_channels):
            for b, (fmin, fmax) in enumerate(bands.values()):
                idx_band = np.where((freqs >= fmin) & (freqs < fmax))[0]
                if len(idx_band) == 0:
                    features[t, ch * len(bands) + b] = 0
                else:
                    features[t, ch * len(bands) + b] = power[t, ch, idx_band].mean()
    
    return features, freqs
