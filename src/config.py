# src/config.py

"""
Central configuration for the Parkinson's FYP pipeline.

All filesystem paths are defined here using pathlib.Path so that
no other module ever contains a hardcoded string path.  Change the
project root once (PROJECT_ROOT) and everything else follows.

Environment awareness
---------------------
The same codebase runs in two places:

  Local (Windows)  — everything lives under PROJECT_ROOT.
  Google Colab     — the repo arrives via `git clone`, so code, manifests,
                     splits, and statistics are all present under
                     PROJECT_ROOT and need no special handling.  But two
                     things must live on Google Drive instead:

                       1. processed_audio/ — too large for git (gitignored),
                          uploaded to Drive once and mounted at runtime.
                       2. outputs/ — Colab's local disk is wiped when the
                          session ends, so a checkpoint written there would
                          be lost.  Writing to Drive survives disconnects.

Detection is by attempting `import google.colab`, which only succeeds inside
a Colab runtime.  This is more robust than checking for the existence of
/content, which can exist in other container environments.

Drive is NOT mounted from this module — mounting is an explicit notebook
step the user performs before importing config.  This module only assumes a
mount is already in place and fails loudly if it isn't (see _assert_drive_mounted).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _is_colab() -> bool:
    """Return True when running inside a Google Colab runtime.

    Detection is by import: the ``google.colab`` package exists only in a
    Colab runtime.  This is preferred over checking for a path such as
    /content, which may exist in unrelated container environments.
    """
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IS_COLAB: bool = _is_colab()

# Single source of truth for the Drive location.  Rename the Drive folder and
# only this constant changes.
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/parkinson-fyp-data")


def _assert_drive_mounted() -> None:
    """Fail loudly if Drive is not mounted before config is imported.

    Without this check the failure would be silent rather than obvious:
    /content/drive is an ordinary local directory until Drive is mounted over
    it, so the _DIRS_TO_CREATE loop below would happily create the entire tree
    on Colab's ephemeral disk.  Training would then run to completion and every
    checkpoint would disappear when the session ended.

    Raises
    ------
    RuntimeError
        If /content/drive/MyDrive does not exist, indicating Drive has not
        been mounted yet.
    """
    mount_point = Path("/content/drive/MyDrive")
    if not mount_point.is_dir():
        raise RuntimeError(
            f"Running on Colab but Google Drive is not mounted at {mount_point}.\n"
            f"Mount it BEFORE importing config, e.g.:\n\n"
            f"    from google.colab import drive\n"
            f"    drive.mount('/content/drive')\n\n"
            f"Without a mount, outputs would be written to Colab's ephemeral "
            f"disk and lost when the session ends."
        )


# ---------------------------------------------------------------------------
# Project root – everything is relative to this
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # parkinson-fyp/

# ---------------------------------------------------------------------------
# Data directories
#
# DATA_DIR, SPLITS_DIR, and STATISTICS_DIR stay rooted at PROJECT_ROOT in both
# environments — their contents (manifests, split CSVs, the statistics JSON)
# are small and travel with the git clone.
#
# PROCESSED_DIR is the exception: the audio is gitignored, so on Colab it must
# resolve to Drive.
# ---------------------------------------------------------------------------
DATA_DIR          = PROJECT_ROOT / "data"
RAW_AUDIO_DIR     = DATA_DIR    / "raw_audio"
SPLITS_DIR        = DATA_DIR    / "splits"
STATISTICS_DIR    = DATA_DIR    / "statistics"

if IS_COLAB:
    _assert_drive_mounted()
    PROCESSED_DIR = COLAB_DRIVE_ROOT / "processed_audio"
else:
    PROCESSED_DIR = DATA_DIR / "processed_audio"

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
# Manifest paths
# ---------------------------------------------------------------------------
RAW_MANIFEST_CSV       = DATA_DIR / "raw_manifest.csv"
PROCESSED_MANIFEST_CSV = DATA_DIR / "processed_manifest.csv"
WINDOWS_MANIFEST_CSV   = DATA_DIR / "windows_manifest.csv"
CORRUPTION_REPORT_CSV  = PROJECT_ROOT / "report" / "corruption_report.csv"

# Split CSVs
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV   = SPLITS_DIR / "val.csv"
TEST_CSV  = SPLITS_DIR / "test.csv"

# Spectrogram statistics
SPECTROGRAM_STATS_JSON = STATISTICS_DIR / "spectrogram_mean_std.json"

# ---------------------------------------------------------------------------
# Output directories
#
# On Colab these are re-rooted to Drive so that checkpoints, logs, and plots
# survive a session disconnect.  The sub-structure is identical in both
# environments — only OUTPUTS_DIR itself moves, and everything derives from it.
#
# REPORTS_DIR is deliberately NOT re-rooted: its contents are small analysis
# artefacts produced during local preprocessing, not training outputs.
# ---------------------------------------------------------------------------
if IS_COLAB:
    OUTPUTS_DIR = COLAB_DRIVE_ROOT / "outputs"
else:
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CHECKPOINTS_DIR  = OUTPUTS_DIR  / "checkpoints"
LOGS_DIR         = OUTPUTS_DIR  / "logs"
RESULTS_DIR      = OUTPUTS_DIR  / "results"
PLOTS_DIR        = RESULTS_DIR  / "plots"
REPORTS_DIR      = PROJECT_ROOT / "report"

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
# Split ratios  (subject-level, stratified by dataset × label)
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15

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
#
# This loop is environment-agnostic: it iterates over whichever paths were
# resolved above.  On Colab the Drive paths are safe to mkdir once Drive is
# mounted — and _assert_drive_mounted() above guarantees it is, since an
# unmounted Drive would otherwise cause this loop to silently build the tree
# on ephemeral local disk.
# ---------------------------------------------------------------------------
_DIRS_TO_CREATE = [
    PROCESSED_DIR,
    SPLITS_DIR,
    STATISTICS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
]

for _d in _DIRS_TO_CREATE:
    _d.mkdir(parents=True, exist_ok=True)