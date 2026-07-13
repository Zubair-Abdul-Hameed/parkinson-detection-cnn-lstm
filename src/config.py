# src/config.py

"""
Central configuration for the Parkinson's FYP pipeline.

All filesystem paths are defined here using pathlib.Path so that
no other module ever contains a hardcoded string path.  Change the
project root once (PROJECT_ROOT) and everything else follows.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root – everything is relative to this
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # parkinson-fyp/

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR          = PROJECT_ROOT / "data"
RAW_AUDIO_DIR     = DATA_DIR    / "raw_audio"
PROCESSED_DIR     = DATA_DIR    / "processed_audio"
SPLITS_DIR        = DATA_DIR    / "splits"
STATISTICS_DIR    = DATA_DIR    / "statistics"

# ---------------------------------------------------------------------------
# Per-dataset raw audio roots
# ---------------------------------------------------------------------------
NEUROVOZ_DIR = RAW_AUDIO_DIR / "neurovoz" / "zenodo_upload"
IPVS_DIR     = RAW_AUDIO_DIR / "ipvs"

# NeuroVoz sub-paths
NEUROVOZ_AUDIO_DIR    = NEUROVOZ_DIR / "audios"
NEUROVOZ_METADATA_DIR = NEUROVOZ_DIR / "metadata"
NEUROVOZ_HC_CSV       = NEUROVOZ_METADATA_DIR / "data_hc.csv"
NEUROVOZ_PD_CSV       = NEUROVOZ_METADATA_DIR / "data_pd.csv"

# IPVS sub-paths
IPVS_YHC_DIR   = IPVS_DIR / "15 Young Healthy Control"
IPVS_EHC_DIR   = IPVS_DIR / "22 Elderly Healthy Control"
IPVS_PD_DIR    = IPVS_DIR / "28 People with Parkinson's disease"

IPVS_YHC_META  = IPVS_YHC_DIR / "15 YHC.xlsx"
IPVS_EHC_META  = IPVS_EHC_DIR / "Tab 3.xlsx"
IPVS_PD_META   = IPVS_PD_DIR  / "TAB 5.xlsx"

# IPVS PD sub-group folders (each contains per-subject sub-folders)
IPVS_PD_SUBGROUPS = [
    IPVS_PD_DIR / "1-5",
    IPVS_PD_DIR / "6-10",
    IPVS_PD_DIR / "11-16",
    IPVS_PD_DIR / "17-28",
]

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUTS_DIR      = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR  = OUTPUTS_DIR  / "checkpoints"
LOGS_DIR         = OUTPUTS_DIR  / "logs"
RESULTS_DIR      = OUTPUTS_DIR  / "results"
REPORTS_DIR      = PROJECT_ROOT / "report"

# ---------------------------------------------------------------------------
# Manifest paths
# ---------------------------------------------------------------------------
RAW_MANIFEST_CSV       = DATA_DIR / "raw_manifest.csv"
PROCESSED_MANIFEST_CSV = DATA_DIR / "processed_manifest.csv"
CORRUPTION_REPORT_CSV  = REPORTS_DIR / "corruption_report.csv"
QUALITY_REPORT_CSV     = REPORTS_DIR / "quality_assessment_report.csv"
RMS_REPORT_CSV         = REPORTS_DIR / "high_rms_report.csv"
WINDOWS_MANIFEST_CSV   = DATA_DIR / "windows_manifest.csv"

# ---------------------------------------------------------------------------
# Split ratios  (subject-level, stratified by dataset × label)
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15

# Split CSVs
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV   = SPLITS_DIR / "val.csv"
TEST_CSV  = SPLITS_DIR / "test.csv"

# Spectrogram statistics
SPECTROGRAM_STATS_JSON = STATISTICS_DIR / "spectrogram_mean_std.json"
 

# ---------------------------------------------------------------------------
# Audio preprocessing constants
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE: int   = 16_000   # Hz
TARGET_CHANNELS:    int   = 1        # mono
SILENCE_TOP_DB:     float = 30.0     # dB threshold for silence trimming

# ---------------------------------------------------------------------------
# Windowing constants
# ---------------------------------------------------------------------------
WINDOW_DURATION_SEC: float = 2.0    # seconds
HOP_DURATION_SEC:    float = 1.0    # seconds (50 % overlap)

# ---------------------------------------------------------------------------
# Mel spectrogram constants
# ---------------------------------------------------------------------------
N_MELS:    int = 128
N_FFT:     int = 1024
HOP_LENGTH: int = 512
F_MIN:     float = 0.0
F_MAX:     float = 8_000.0          # Nyquist for 16 kHz

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Dataset identifiers  (used to namespace subject and recording IDs)
# ---------------------------------------------------------------------------
DATASET_NEUROVOZ = "neurovoz"
DATASET_IPVS     = "ipvs"

# ---------------------------------------------------------------------------
# Raw manifest column order  (schema contract for all downstream code)
# ---------------------------------------------------------------------------
MANIFEST_COLUMNS = [
    "recording_id",
    "file_path",
    "subject_id",
    "label",
    "dataset",
    "task",
    "language",
    "duration",
    "sample_rate",
    "gender",
    "age",
]

# ---------------------------------------------------------------------------
# Auto-create output directories on import
# (raw_audio and its sub-dirs are *not* created – they must already exist)
# ---------------------------------------------------------------------------
_DIRS_TO_CREATE = [
    PROCESSED_DIR,
    SPLITS_DIR,
    STATISTICS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
]

for _d in _DIRS_TO_CREATE:
    _d.mkdir(parents=True, exist_ok=True)