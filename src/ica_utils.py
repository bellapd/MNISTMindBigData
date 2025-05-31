# src/ica_utils.py
import os
import numpy as np
import mne
from joblib import Parallel, delayed
from mne.preprocessing import ICA
from mne_icalabel import label_components # type: ignore
from mne import use_log_level
# import to_mne_raw from src/utils
from utils import to_mne_raw

def apply_ICA(data, n_components=0.99, method='picard', random_state=1, plot_picks=[0]):
    print("Fitting ICA...")
    ica = ICA(n_components=n_components, method=method, max_iter='auto',
              random_state=random_state, fit_params=dict(extended=True))
    ica.fit(data)

    print("Plotting ICA sources...")
    ica.plot_sources(data, show=True)

    print("Plotting ICA components...")
    ica.plot_components(inst=data, show=True)

    print("Labeling components via ICLabel...")
    ic_labels = label_components(inst=data, ica=ica, method="iclabel")
    labels = ic_labels["labels"]

    # Display index-label mapping
    print("\n ICA Component Index ↔ Label:")
    for idx, lbl in enumerate(labels):
        print(f" - ICA{idx:03}: {lbl}")

    # Build label-to-indices dictionary
    label_dict = {}
    for idx, lbl in enumerate(labels):
        label_dict.setdefault(lbl, []).append(idx)

    print("\n ICA Component Label Breakdown:")
    for label, indices in label_dict.items():
        print(f" - {label:<10}: {len(indices)} components")

    return ica, labels, label_dict

def apply_ica_all_trials(
    x_asr,
    ica_obj,
    ic_labels,
    ch_names,
    fs,
    n_jobs=4,
    output_path=None,
):
    """
    Apply a fitted ICA object to every ASR‐cleaned trial in parallel,
    but EXCLUDE all components *that are not* labeled 'brain'.

    Parameters
    ----------
    x_asr : np.ndarray, shape (n_trials, n_channels, n_times)
        ASR‐cleaned EEG in µV.
    ica_obj : instance of mne.preprocessing.ICA (already fitted)
        The ICA object you want to apply.
    ic_labels : dict
        Mapping from label→list of component indices, e.g.
          {'brain': [2,5,8], 'eye blink': [0,1], 'heart beat': [3,4], ... }
        This function will keep only indices in `ic_labels['brain']`
        and will exclude every other index.
    ch_names : list of str
        Length = n_channels. Used by to_mne_raw.
    fs : float
        Sampling frequency (e.g. 128 Hz).
    n_jobs : int, default 4
        How many parallel jobs to use (passed to joblib.Parallel).
    output_path : str or None
        If a filepath is provided, the resulting cleaned array will be
        saved via np.save(output_path). If None, it is not saved.

    Returns
    -------
    x_ica_cleaned : np.ndarray, shape (n_trials, n_channels, n_times)
        The fully ICA‐cleaned data (µV) for all trials.
    """
    keep_brain = ic_labels.get("brain", [])
    all_indices = []
    for lbl, idxs in ic_labels.items():
        if lbl != "brain":
            all_indices.extend(idxs)
    exclude_idx = sorted(set(all_indices))

    n_trials = x_asr.shape[0]
    print(f"Applying ICA to all {n_trials} trials (excluding non‐brain ICs)…")

    def _apply_one(trial_data, idx):
        """
        Apply ICA to a single trial (in µV). Returns the cleaned data in µV.
        If anything fails, returns the original ASR‐cleaned trial.
        """
        try:
            with use_log_level("WARNING"):
                raw = to_mne_raw(trial_data, ch_names, fs)
                raw.set_eeg_reference("average", projection=True)
                raw.apply_proj()
                cleaned = ica_obj.apply(raw.copy(), exclude=exclude_idx)
                return cleaned.get_data() * 1e6  # Convert V→µV
        except Exception as e:
            print(f"⚠ Trial {idx:05d} ICA failed ({e}); returning ASR‐cleaned fallback")
            return trial_data  # already in µV

    # 2) Launch parallel loop
    results = Parallel(n_jobs=n_jobs)(
        delayed(_apply_one)(x_asr[i], i) for i in range(n_trials)
    )

    x_ica_cleaned = np.stack(results)

    # 3) Optionally save to disk
    if output_path is not None:
        arr_folder = os.path.dirname(output_path) or "."
        os.makedirs(arr_folder, exist_ok=True)
        np.save(output_path, x_ica_cleaned)
        print(f"Saved ICA‐cleaned EEG → '{output_path}'")

    return x_ica_cleaned
