import os
import csv
import numpy as np


CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]


def load_data(filepath, required_channels=14, sample_length=256, max_samples_per_digit=6500):
    """
    Load EEG data from a tab-delimited file where each row corresponds to a channel sample for a given digit.

    Parameters
    ----------
    filepath : str
        Path to the EEG data file (tab-delimited, with channel data in column 6).
    required_channels : int
        Number of EEG channels expected per trial (default 14).
    sample_length : int
        Number of time samples to read per channel (default 256).
    max_samples_per_digit : int
        Maximum number of trials to load per digit (0-9).

    Returns
    -------
    eeg_data : np.ndarray, shape (n_trials, channels, time)
    digit_labels : np.ndarray, shape (n_trials,)
    samples_per_class : list of int
        Counts of loaded trials per digit.
    """
    eeg_data = []
    digit_labels = []
    samples_per_class = [0] * 10

    current_channels = {}
    current_digit = None

    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) < 7:
                continue

            try:
                digit_code = int(row[4])
                channel_name = row[3].strip()
                signal_values = row[6].split(',')
            except (ValueError, IndexError):
                continue

            # filter out-of-range digits and over-sampled classes
            if digit_code < 0 or digit_code > 9:
                continue
            if samples_per_class[digit_code] >= max_samples_per_digit:
                continue
            if channel_name not in CHANNEL_NAMES:
                continue
            if len(signal_values) < sample_length:
                continue

            signal_array = np.array(signal_values[:sample_length], dtype=np.float32)

            # start a new trial when current_channels is empty
            if not current_channels:
                current_digit = digit_code

            # collect channels for this trial
            if digit_code == current_digit and channel_name not in current_channels:
                current_channels[channel_name] = signal_array

            # once we have all channels, store the trial
            if len(current_channels) == required_channels:
                # ensure correct order
                if all(ch in current_channels for ch in CHANNEL_NAMES):
                    ordered = [current_channels[ch] for ch in CHANNEL_NAMES]
                    eeg_data.append(np.stack(ordered))
                    digit_labels.append(current_digit)
                    samples_per_class[current_digit] += 1
                current_channels = {}

                # optional progress print
                total_loaded = sum(samples_per_class)
                if total_loaded % 1000 == 0:
                    print(f"{total_loaded} samples collected...")

            # break early if all digits have enough samples
            if all(count >= max_samples_per_digit for count in samples_per_class):
                break

    eeg_data = np.array(eeg_data)
    digit_labels = np.array(digit_labels)

    print(f"\nLoaded EEG shape: {eeg_data.shape} (trials, channels, time)")
    for d in range(10):
        print(f" - Digit {d}: {samples_per_class[d]} samples")

    return eeg_data, digit_labels, samples_per_class
