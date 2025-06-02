import numpy as np
from scipy.fft import rfft, rfftfreq
import pywt

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

def extract_wavelet_on_fft(x_clean, fs, bands=None, wavelet='morl'):
    """
    Compute wavelet bandpower on FFT power spectrum per channel for each trial.
    
    Parameters
    ----------
    x_clean : np.ndarray, shape (n_trials, n_channels, n_times)
        Cleaned EEG (µV).
    fs : float
        Sampling rate in Hz.
    bands : dict, optional
        Mapping { 'delta': (1,4), 'theta': (4,8), ... }. 
        If None, defaults to {'delta':(1,4), 'theta':(4,8), 'alpha':(8,12), 'beta':(12,30)}.
    wavelet : str, optional
        Wavelet type for CWT (default: 'morl' for Morlet wavelet).
    
    Returns
    -------
    features : np.ndarray, shape (n_trials, n_channels * n_bands)
        Each row is concatenated bandpower features per channel.
    scales : np.ndarray
        Scales used in CWT.
    """
    if bands is None:
        bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}
    
    n_trials, n_channels, n_times = x_clean.shape
    freqs = rfftfreq(n_times, 1.0 / fs)
    power = np.abs(rfft(x_clean, axis=2))**2  # FFT power spectrum: (n_trials, n_channels, n_freqs)
    
    # Define scales for CWT (treating frequency axis as "time")
    freq_range = np.linspace(1, fs / 2, len(freqs))  # Pseudo-frequency for wavelet scales
    scales = pywt.scale2frequency(wavelet, np.ones_like(freq_range)) * fs / freq_range
    scales = scales[::-1]  # Reverse to align with increasing frequencies
    
    features = np.zeros((n_trials, n_channels * len(bands)), dtype=np.float32)
    
    for t in range(n_trials):
        for ch in range(n_channels):
            # Apply CWT to FFT power spectrum along frequency axis
            coeffs, freqs_cwt = pywt.cwt(power[t, ch], scales, wavelet, sampling_period=1.0/fs)
            wavelet_power = np.abs(coeffs)**2  # Wavelet power spectrum
            
            for b, (fmin, fmax) in enumerate(bands.values()):
                idx_band = np.where((freqs_cwt >= fmin) & (freqs_cwt < fmax))[0]
                if len(idx_band) == 0:
                    features[t, ch * len(bands) + b] = 0
                else:
                    features[t, ch * len(bands) + b] = wavelet_power[idx_band].mean()
    
    return features, scales