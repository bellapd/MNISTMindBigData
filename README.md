# MindBigData - MNIST Brain Digit Classification Project
---
## 1. Introduction

The MindBigData - “MNIST Brain Digit” dataset contains EEG signals recorded while a subject is visually exposed to digits ranging from 0 to 9. The main objective is to classify these EEG signals into the corresponding digit categories.

While advancements in machine learning—such as in image recognition, natural language processing, and speech recognition—have shown remarkable success, applying similar techniques to EEG data remains highly challenging. Despite progress in neural networks, decoding EEG signals continues to be difficult due to factors like high noise levels and variability across recording sessions.

This project aims to investigate the feasibility of using machine learning techniques to classify EEG signals into their respective digit classes.

## 2. Dataset Information
#### Overview
- Dataset: **MindBigData-EP-v1.0**
- Subject: David Vivancos
- Total Trials: 64344 trials
- Signal: 2-second EEG trials per digit (0–9)
- Sampling rate: **128 Hz**
- Channels used: **14** ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                 "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
#### Format
The data has no headers in the files and each corresponding field is separated by a tab character. Here is the explanation cited from the MindBigData documentation[^1]:
**[id]**: a numeric, only for reference purposes.
**[event]** id, a integer, used to distinguish the same event captured at different brain locations, used only by multichannel devices (all except MW).
**[device]**: a 2 character string, to identify the device used to capture the signals, "MW" for MindWave, "EP" for Emotive Epoc, "MU" for Interaxon Muse & "IN" for Emotiv Insight.
**[channel]**: a string, to indentify the 10/20 brain location of the signal, with possible values:
EPOC	"AF3, "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
**[code]**: a integer, to indentify the digit been thought/seen, with possible values 0,1,2,3,4,5,6,7,8,9 or -1 for random captured signals not related to any of the digits.
**[size]**: a integer, to identify the size in number of values captured in the 2 seconds of this signal, since the Hz of each device varies, in "theory" the value is close to 512Hz for MW, 128Hz for EP, 220Hz for MU & 128Hz for IN, for each of the 2 seconds.
**[data]**: a coma separated set of numbers, with the time-series amplitude of the signal, each device uses a different precision to identify the electrical potential captured from the brain: integers in the case of MW & MU or real numbers in the case of EP & IN.

## 3. Environment Setup
- **Python**: 3.12.2  
- **CUDA**: 12.4  
- **Operating System**: Ubuntu 22.04.2 LTS  
- **GPU**: NVIDIA GeForce RTX 4060  

### Installation Instructions
Before running the code, You can run the following command:

1. Install all dependencies from `requirements.txt` into your Conda environment:

   ```
   conda env create -f requirements.txt
   ```
2. Activate the new environment:

   ```
   conda activate MBC_BCI
   ```
3. Create a folder named `data/`:

   ```
   mkdir data
   ```
4. Download the MindBigData EEG dataset:

   ```
   wget https://www.mindbigdata.com/opendb/MindBigData-EP-v1.0.zip
   ```
5. Unzip the downloaded archive:

   ```
   unzip MindBigData-EP-v1.0.zip -d data/
   ```
6. Verify that the unzipped files now reside under `data/`.

Once these steps are complete, your EEG files will be available in the `data/` folder and your Conda environment will be ready.
### Usage Instructions
The main entry point to run the project is in `main.ipynb`. Activate the conda environment in the kernel and you are good to go. Here is the file structure of the project:
```
├── img/                        # Figures used in README or reports (e.g., flowcharts)
│
├── models/                    # Saved trained models
│   ├── model_cnn.pth          # Trained weights of CNN model
│   └── model_mlp.pth          # Trained weights of MLP model
│
├── src/                       # Source code
│   ├── classifier/            # Model definitions and evaluation
│   │   ├── cnn.py             # CNN model architecture
│   │   ├── mlp.py             # MLP model architecture
│   │   └── eval.py            # Evaluation functions (accuracy, confusion matrix, etc.)
│   │
│   ├── asr_utils.py           # Artifact Subspace Reconstruction (ASR) preprocessing functions
│   ├── data_prep.py           # Data loading, formatting, and conversion utilities
│   ├── features_utils.py      # Feature extraction from EEG signals
│   ├── filter_utils.py        # Bandpass filtering and related signal preprocessing
│   ├── ica_utils.py           # ICA decomposition and ICLabel integration
│   └── utils.py               # Miscellaneous helper functions such as plotting
│
├── main.ipynb                 # Notebook for full training & evaluation workflow
├── requirements.txt           # Required Python packages for the project
├── .gitignore                 # Files and folders to ignore in version control
└── README.md                  # Project documentation
```



## 4. Model Framework
### Outline of the architecture
![Model Architecture Flowchart](img/flowchart.png)

**1. Data Loading `data_prep.py`**:
EEG data is loaded from a tab-delimited text file using the `load_data()` function. Each trial is reconstructed by grouping 14 specific EEG channels (AF3, F7, ..., AF4) into a (14, 256) array. Only valid trials with sufficient length and balanced digit classes (0–9) are kept, up to 6,500 samples per class. The function returns:
`eeg_data`: EEG trials of shape (n_trials, 14, 256)
`digit_labels`: corresponding digit labels
`samples_per_class`: number of samples loaded per digit

**2. Data Preprocessing**:
The EEG data undergoes several preprocessing steps:
- **Bandpass Filtering `filter_utils.py`**: 
To remove low-frequency drift and high-frequency noise, a zero-phase 8th-order Butterworth bandpass filter is applied to each channel in every EEG trial. The frequency range is set to 1–40 Hz, which captures most of the meaningful EEG activity (e.g., alpha, beta bands).The function `bandpass_all_trials()` processes all trials using SciPy’s `sosfiltfilt` for zero-phase filtering, preserving signal shape.
- **ASR `asr_utils.py`**:
Then ASR is then applied to the bandpass-filtered EEG data.
The `run_asr_pipeline()` function follows a 4-step procedure:
    - Calibration:
    From the bandpass-filtered EEG data, we select a subset of "clean" trials using thresholds on standard deviation, peak-to-peak amplitude, and kurtosis. These trials are concatenated to train (fit) the ASR model.

    - Fitting the ASR:
    The ASR algorithm is calibrated using the selected clean calibration set. A cutoff value (default: 3.0) controls how aggressively artifacts are removed.

    - Denoising All Trials:
    The calibrated ASR model is applied to every trial in the dataset. If a trial fails to clean (e.g., due to shape issues), it falls back to the bandpass-filtered version.

- **ICA `ica_utils.py`**:
To remove structured artifacts (e.g., eye blinks, heartbeats), we apply Independent Component Analysis (ICA) followed by ICLabel-based classification:
    - ICA Fitting on Subset
    We first fit the ICA model using a small subset of ASR-cleaned EEG trials. This step extracts independent components (ICs), which are then automatically labeled using ICLabel into categories like brain, eye blink, muscle, etc.

    - Apply to All Trials
    The fitted ICA model is then applied to all trials in parallel. All ICs not labeled as brain are removed before reconstructing the cleaned EEG signals. This step ensures that only neural-related activity remains.

**3. Feature Extraction `features_utils.py`**:

#### The change in the number of recognized ICs for the following EEG datasets:
| EEG (32 channels, 1 dataset) | Bandpass filter | ASR | Brain | Muscle | Eye | Heartbeat | Line | Channel noise | Other | 
|-------------------------------|----------------|-----|-------|--------|-----|-------|------|---------------|-------|
| Raw                           |                |     |    3  |    0   |   1 |   0   |   0  |      0       |    10   | 
| Filtered                      | ✓              |    |    7  |    0   |  2 |    1  |    0 |       0        |    4  |   
| ASR                           | ✓              | ✓   |   6   |    1   |  2  |   0   |  0  |        0       |   5  |  

### 
## Reference
[^1]: Mindbigdata the mnist of brain digits https://www.mindbigdata.com/opendb/index.html 