import numpy as np
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt


def butter_bandpass(lowcut, highcut, fs, order=8):
    """
    Create a normalized bandpass filter (SOS) with given order.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def apply_bandpass_filter(signal, lowcut, highcut, fs, order=8):
    """
    Apply zero-phase bandpass to a 1D signal.
    """
    sos = butter_bandpass(lowcut, highcut, fs, order)
    return sosfiltfilt(sos, signal, axis=-1)


def bandpass_all_trials(x_data, fs=128, lowcut=1, highcut=40, order=8):
    """
    Apply bandpass filter to each channel of every trial.

    Parameters
    ----------
    x_data : np.ndarray, shape (n_trials, n_channels, n_samples)
    fs : float, sampling rate
    lowcut : float
    highcut : float
    order : int, filter order

    Returns
    -------
    x_filtered : np.ndarray, same shape as x_data
    """
    n_trials, n_channels, n_samples = x_data.shape
    x_filtered = np.zeros_like(x_data, dtype=np.float32)

    for i in tqdm(range(n_trials), desc="Filtering Trials"):
        for ch in range(n_channels):
            x_filtered[i, ch] = apply_bandpass_filter(
                x_data[i, ch], lowcut, highcut, fs, order
            )
    return x_filtered
