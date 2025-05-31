# src/asr_utils.py

import os
import numpy as np
from tqdm import tqdm
from scipy.stats import kurtosis
from meegkit.asr import ASR  # type: ignore

from utils import plot_all_channels_eeg

# Channel names (must match your dataset)
CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

def run_asr_pipeline(
    x_filtered,
    fs=128,
    calib_size=500,
    cutoff=3.0,
    asr_path="data/filtered_asr.npy",
    plot_trial_idx=None,
    x_raw=None,
    x_detrended=None,
    y_labels=None,
):
    """
    1) Select a subset of “good” trials from x_filtered for ASR calibration.
    2) Fit ASR(cutoff=...) on that subset.
    3) Apply ASR to every trial in x_filtered, saving the cleaned data to `asr_path`.
    4) If plot_trial_idx is not None, plot raw vs filtered vs ASR for that index.

    Parameters
    ----------
    x_filtered : np.ndarray, shape (n_trials, n_channels, n_times) [µV]
    fs : float — sampling rate (Hz)
    calib_size : int — number of trials for ASR calibration
    cutoff : float — ASR cutoff parameter
    asr_path : str — where to save the ASR‐cleaned output
    plot_trial_idx : int or None — if int, plot that trial after cleaning
    x_raw : np.ndarray, shape (n_trials, n_channels, n_times) [µV]
        Required if plot_trial_idx is not None
    x_detrended : np.ndarray, shape (n_trials, n_channels, n_times) [µV]
        Required if plot_trial_idx is not None
    y_labels : np.ndarray, shape (n_trials,)
        Required if plot_trial_idx is not None

    Returns
    -------
    asr : the fitted ASR object
    x_data_asr : np.ndarray, shape (n_trials, n_channels, n_times) [µV]
    """
    n_trials, n_ch, n_times = x_filtered.shape

    # 1) Identify “good” trials by simple amplitude/kurtosis screening
    flat = x_filtered.reshape(n_trials, -1)
    stds = np.std(flat, axis=1)
    ptp = np.ptp(flat, axis=1)
    kurt_vals = kurtosis(flat, axis=1)

    good_idx = np.where(
        (stds > 1.0)
        & (stds < 100.0)
        & (ptp < 500.0)
        & (np.abs(kurt_vals) < 5.0)
    )[0]
    print(f"Found {len(good_idx)} good trials for ASR calibration.")

    if len(good_idx) < calib_size:
        raise RuntimeError(
            f"Not enough calibration trials: found {len(good_idx)} < {calib_size}"
        )

    chosen = np.random.choice(good_idx, size=calib_size, replace=False)
    calib_data = np.concatenate([x_filtered[i] for i in chosen], axis=1)

    # 2) Fit ASR
    asr = ASR(sfreq=fs, cutoff=cutoff)
    print(f"Calibrating ASR (cutoff={cutoff}) on {calib_size} trials…")
    asr.fit(calib_data)
    print("ASR calibration complete.")

    # 3) Apply ASR to every trial
    x_data_asr = []
    deltas = []
    print("Applying ASR to all filtered trials…")
    for i in tqdm(range(n_trials), desc="ASR Cleaning"):
        try:
            cleaned = asr.transform(x_filtered[i])
            x_data_asr.append(cleaned)
            delta = np.mean(np.abs(cleaned - x_filtered[i]))
            deltas.append(delta)
        except Exception as e:
            print(f"⚠ Trial {i} failed: {e} → copying filtered trial")
            x_data_asr.append(x_filtered[i].copy())
            deltas.append(0.0)

    x_data_asr = np.stack(x_data_asr)
    os.makedirs(os.path.dirname(asr_path) or ".", exist_ok=True)
    np.save(asr_path, x_data_asr)
    print(f"Saved ASR-cleaned EEG → '{asr_path}'")
    print(f"Mean ASR Δ = {np.mean(deltas):.2f} µV | Max Δ = {np.max(deltas):.2f} µV")

    # 4) Optionally plot one trial: raw vs filtered vs ASR
    if plot_trial_idx is not None:
        if x_raw is None or x_detrended is None or y_labels is None:
            raise ValueError(
                "To plot trial, you must pass x_raw, x_detrended, and y_labels."
            )
        idx = plot_trial_idx
        print(f"\nPlotting trial {idx}: raw vs filtered vs ASR")
        # Scale from µV → µV (plot expects µV)
        x_raw_s = x_detrended[idx][None] * 1e6
        x_filt_s = x_filtered[idx][None] * 1e6
        x_asr_s = x_data_asr[idx][None] * 1e6

        plot_all_channels_eeg(
            x_raw=x_raw_s,
            x_filt=x_filt_s,
            x_asr=x_asr_s,
            trial_index=0,
            channel_names=CHANNEL_NAMES,
            fs=fs,
            duration_sec=2,
            y_labels=y_labels,
        )

    return asr, x_data_asr
