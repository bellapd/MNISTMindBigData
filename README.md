# MindBigData - MNIST Brain Digit Classification 
**Group 11**  Annabella Putri Dirgo 林柏亞 夏高上 
---
## 1.Introduction
The MindBigData - "MNIST Brain Digit" is a dataset that is collected by EEG signal when the subject is exposed to the digit ranging from 0 to 9 as the visual stimuli. The objective is to classify EEG signals into the corresponding digit category. 

The advancement of Machine Learning techniques has increased a lot, starting from normal The goal of this project is to explore the feasibility of machine learning techniques in classifying EEG signals into the corresponding digit category.

## Model Framework

The architecture of our BCI system consists of the following components:

### Input
- Dataset: **MindBigData-EP-v1.0**
- Signal: 2-second EEG trials per digit (0–9)
- Sampling rate: **128 Hz**
- Channels used: **14 **

### Signal Preprocessing
- **Bandpass filter** (1–40 Hz) using a Butterworth filter
- **Detrending** to remove DC offset
- **ASR (Artifact Subspace Reconstruction)** via `asrpy` for automatic artifact removal
- **ICA (Infomax)** for decomposing sources
- **ICLabel** to automatically classify ICs as brain, eye blink, muscle, etc.

### Data Segmentation
- Each EEG trial lasts 2 seconds (256 samples)
- All trials are kept as individual samples (shape: trials × channels × time)

### Feature Extraction
- **Frequency-domain features** via FFT
- Optional: Power in EEG bands (Delta, Theta, Alpha, Beta, Gamma)
- Features flattened per trial → ready for classification

### Machine Learning Model
- Classifier: **Multilayer Perceptron (MLP)** with 2 hidden layers
- Loss: Cross-entropy
- Evaluation: Accuracy, F1-score

### Output
- Predicted digit (0–9) for each EEG trial


## Validation

We validate the system in two key ways:

### 1. **ICLabel Analysis**
- ICA is applied to three data types:
  - Raw EEG
  - Filtered EEG
  - ASR-cleaned EEG
- Components are labeled using ICLabel into categories: Brain, Eye Blink, Muscle, Heartbeat, Line Noise, etc.
- We compare the number of “brain” vs. “artifact” components across conditions to evaluate preprocessing effectiveness.

### 2. **Classification Performance**
- Evaluation metrics:
  - Accuracy
  - F1-score
  - Confusion matrix
- We assess model reliability using train/test splits (or k-fold cross-validation).


## Usage 

### Requirements
```bash
pip install numpy mne matplotlib scikit-learn asrpy
```

Running the pipeline
```bash
python main_pipeline.py
```
Or run step-by-step in main_pipeline.ipynb.

Configurable options
FS: Sampling frequency

lowcut, highcut: Bandpass filter range

cutoff: ASR aggressiveness

ICA components: default 14

Classifier settings (e.g., learning rate, hidden layer size)

Files and Structure

├── main_pipeline.py
├── utils/
│   ├── filter.py
│   ├── asr.py
│   ├── ica.py
│   └── plot.py
├── data/
│   └── EP1.01.txt
├── results/
│   ├── ica_labels_raw.txt
│   ├── confusion_matrix.png
│   └── accuracy_log.csv
└── README.md

To get the the data from the website, run this in command prompt
```
wget https://mindbigdata.com/opendb/MindBigData-EP-v1.0.zip
unzip MindBigData-EP-v1.0.zip
```

**File Format**

[id]: a numeric, only for reference purposes.

[event] id, a integer, used to distinguish the same event captured at different brain locations, used only by multichannel devices (all except MW).

[device]: a 2 character string, to identify the device used to capture the signals, "MW" for MindWave, "EP" for Emotive Epoc, "MU" for Interaxon Muse & "IN" for Emotiv Insight.

[channel]: a string, to indentify the 10/20 brain location of the signal, with possible values:
 
EPOC	"AF3, "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"

[code]: a integer, to indentify the digit been thought/seen, with possible values 0,1,2,3,4,5,6,7,8,9 or -1 for random captured signals not related to any of the digits.

[size]: a integer, to identify the size in number of values captured in the 2 seconds of this signal, since the Hz of each device varies, in "theory" the value is close to 512Hz for MW, 128Hz for EP, 220Hz for MU & 128Hz for IN, for each of the 2 seconds.

[data]: a coma separated set of numbers, with the time-series amplitude of the signal, each device uses a different precision to identify the electrical potential captured from the brain: integers in the case of MW & MU or real numbers in the case of EP & IN.

There is no headers in the files,  every line is  a signal, and the fields are separated by a tab character.