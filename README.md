# Parkinson Disease Detection using CNN and CNN-LSTM

Final Year Project

## Research Question

**Does temporal modeling improve Parkinson's speech detection from spectrogram features under controlled subject-independent evaluation?**

---

## Objectives

1. Build a CNN baseline model for Parkinson's speech detection.
2. Build a hybrid CNN-LSTM model that incorporates temporal modeling.
3. Compare both models under identical training and evaluation conditions.
4. Evaluate performance using subject-independent train/validation/test splits.
5. Investigate cross-dataset generalization if time permits.
6. Deploy the best-performing model as a simple web application.

---

## Models

### CNN (Baseline)

Learns spectral patterns from Mel spectrograms.

### CNN-LSTM (Proposed Model)

Combines CNN-based feature extraction with LSTM-based temporal modeling to capture sequential speech information.

---

## Datasets

### NeuroVoz

Language: Spanish

Dataset Structure Notes:

* Labels: HC (Healthy Control) / PD (Parkinson's Disease)
* Metadata Files:

  * `data_hc.csv`
  * `data_pd.csv`
* Audio Files:

  * `audios/*.wav`
* Task Type:

  * Extracted from filename middle token

Examples:

```text
HC_SOMBRA_0122.wav
HC_U1_0122.wav
PD_A1_0078.wav
```

* Subject ID:

  * Obtained from the `ID` column in metadata files
* Age:

  * Obtained from metadata files
* Gender:

  * Obtained from metadata files
* Additional Files:

  * `audio_features.csv` (not used)
  * `transcriptions/` (not used for this project)

---

### IPVS

Language: Italian

Dataset Structure Notes:

* Labels:

  * Determined from folder hierarchy
  * Healthy Control folders
  * Parkinson's Disease folders
* Metadata Files:

  * `15 YHC.xlsx`
  * `Tab 3.xlsx`
  * `TAB 5.xlsx`
* Task Code Reference:

  * `FILE CODES.xlsx`
* Subject ID:

  * Participant name extracted from folder structure
* Age:

  * Obtained from metadata spreadsheets
* Gender:

  * Obtained from metadata spreadsheets
* Task Type:

  * Extracted from filename prefixes and interpreted using `FILE CODES.xlsx`

Examples:

```text
B1LBULCAAS94M100120171015.wav
D1XXXXXXXXXXXX.wav
VA1XXXXXXXXXXX.wav
```

Task Code Examples:

```text
B1  -> First reading of phonemically balanced text
B2  -> Second reading of phonemically balanced text
D1  -> Repetition of syllable "pa"
D2  -> Repetition of syllable "ta"
VA1 -> Sustained vowel "a"
VE1 -> Sustained vowel "e"
VI1 -> Sustained vowel "i"
VO1 -> Sustained vowel "o"
VU1 -> Sustained vowel "u"
```

---

## Data Processing Pipeline

```text
Raw Audio
    ↓
Metadata Extraction
    ↓
raw_manifest.csv
    ↓
Corruption Detection
    ↓
Silence Trimming
    ↓
Resampling
    ↓
Amplitude Normalization
    ↓
processed_audio/
    ↓
processed_manifest.csv
    ↓
Subject-Level Stratified Split
    ↓
train.csv
val.csv
test.csv
    ↓
Windowing
    ↓
Mel Spectrogram Extraction
    ↓
Compute Spectrogram Statistics (Train Only)
    ↓
Apply Spectrogram Normalization
    ↓
Train CNN
    ↓
Train CNN-LSTM
    ↓
Evaluation
```

---

## Evaluation Strategy

### Primary Evaluation

Subject-independent train/validation/test split.

A subject may appear in only one split to prevent data leakage.

### Benchmark Comparison

* CNN
* CNN-LSTM

Both models will use:

* The same datasets
* The same preprocessing pipeline
* The same train/validation/test split
* The same spectrogram representation

This ensures a scientifically valid comparison.

### Optional Evaluation

Cross-dataset generalization:

* Train on NeuroVoz
* Test on IPVS

and/or

* Train on NeuroVoz + IPVS
* Test on MDVR-KCL

---

## Manifest Schema

### raw_manifest.csv

| Column       | Description                      |
| ------------ | -------------------------------- |
| recording_id | Unique recording identifier      |
| file_path    | Path to audio file               |
| subject_id   | Dataset-aware subject identifier |
| label        | HC or PD                         |
| dataset      | Dataset source                   |
| task         | Speech task                      |
| language     | Recording language               |
| duration     | Audio duration (seconds)         |
| sample_rate  | Audio sample rate                |
| gender       | Subject gender                   |
| age          | Subject age                      |

Examples:

```text
recording_id = neurovoz_HC_ABLANDADA_0034
subject_id   = neurovoz_34

recording_id = ipvs_B1LBULCAAS94M100120171015
subject_id   = ipvs_Alberto_R
```

---

## Author

Abdul-Hameed
