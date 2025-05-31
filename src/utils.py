# src/utils.py

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import use_log_level

def to_mne_raw(trial, ch_names, fs=128):

    # Remove DC offset
    trial_centered = trial - np.mean(trial, axis=1, keepdims=True)
    # Convert µV → V
    trial_volt = trial_centered * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')

    # Temporarily set MNE’s log level to ERROR to suppress the “Creating RawArray…” messages
    with use_log_level("ERROR"):
        raw = mne.io.RawArray(trial_volt, info)
        raw.set_montage("standard_1020")

    return raw


def plot_all_channels_eeg(
    x_raw,
    x_filt=None,
    x_asr=None,
    trial_index=0,
    channel_names=None,
    fs=128,
    duration_sec=2,
    y_labels=None,
    scale=1.0
):
    """
    Overlays raw / filtered / ASR-cleaned traces for one trial, all channels stacked vertically.
    Each panel shows time on the x-axis and signal (µV) on the y-axis.
    """
    # Pick correct trial (supports both shape (1,n_ch,n_times) or (n_ch,n_times))
    if x_raw.ndim == 3:
        trial_raw = x_raw[trial_index] * scale
    else:
        trial_raw = x_raw * scale
    n_channels, n_samples = trial_raw.shape
    end = min(int(duration_sec * fs), n_samples)
    time = np.arange(end) / fs

    trial_filt = (x_filt[trial_index] * scale) if x_filt is not None else None
    trial_asr  = (x_asr[trial_index] * scale)  if x_asr  is not None else None

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(n_channels)]

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 1.1), sharex=True)
    title = f"EEG Trial {trial_index}"
    if y_labels is not None:
        title = f"Imagined Digit: {y_labels[trial_index]}"
    suffixes = []
    if x_filt  is not None: suffixes.append("Filtered")
    if x_asr   is not None: suffixes.append("ASR")
    if suffixes:
        title += " — Raw vs " + " vs ".join(suffixes)

    fig.suptitle(title, fontsize=14)
    for i, ax in enumerate(axes):
        ax.plot(time, trial_raw[i, :end], label="Raw", color="steelblue", lw=0.8, alpha=0.6)
        if trial_filt is not None:
            ax.plot(time, trial_filt[i, :end], label="Filtered", color="orangered", lw=0.9, alpha=0.8)
        if trial_asr is not None:
            ax.plot(time, trial_asr[i, :end], label="ASR", color="green", lw=1.0, alpha=0.9)

        ax.set_ylabel(channel_names[i], rotation=0, labelpad=20, fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_psd_comparison(raw_raw, raw_filt, raw_asr, raw_ica, fs=128, fmin=0.5, fmax=60.0):
    """
    For a single trial, compute and plot the PSD (Welch) of raw, bandpass‐filtered, ASR‐cleaned, and ICA‐cleaned.
    Uses a log‐scale y‐axis and highlights delta/beta bands.
    """
    # Helper to compute PSD via MNE’s compute_psd (v1.0+ API)
    def compute_psd(raw_inst):
        psd_obj = raw_inst.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=256, verbose=False)
        return psd_obj.get_data(), psd_obj.freqs

    raws = [raw_raw, raw_filt, raw_asr, raw_ica]
    labels = ["Raw", "Filtered", "ASR Cleaned", "ICA Cleaned"]
    psds, freqs = [], None

    for inst in raws:
        psd_data, freqs = compute_psd(inst)
        psds.append(psd_data)

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        mean_psd = np.mean(psds[i], axis=0) * 1e12  # V²/Hz → µV²/Hz
        plt.plot(freqs, mean_psd, label=label)

    # Highlight artifact bands
    plt.axvspan(0.5, 4, color='red', alpha=0.1, label='Delta (Eye Blink)')
    plt.axvspan(13, 30, color='orange', alpha=0.1, label='Beta (Muscle)')
    plt.axvspan(30, 40, color='orange', alpha=0.1)

    plt.title("PSD Comparison Before & After Cleaning")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (µV²/Hz)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
