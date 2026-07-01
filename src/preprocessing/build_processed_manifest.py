# src/preprocessing/build_processed_manifest.py
"""
Full-dataset preprocessing orchestration for the Parkinson's FYP pipeline.

Reads data/raw_manifest.csv, processes every recording through the
already-implemented preprocessing chain, writes the processed audio under
data/processed_audio/ (mirroring the raw_audio/ directory structure), and
builds data/processed_manifest.csv.

Pipeline applied to each recording:

    load waveform → trim_silence → resample → normalize → save

This module only ORCHESTRATES.  It contains no preprocessing logic of its
own — every transformation is delegated to the existing pure functions in
trim_silence.py, resample_audio.py, and amplitude_normalization.py.

processed_manifest.csv is the source of truth for all later stages.
raw_manifest.csv is read once here and never read again downstream.

Usage
-----
    python src/preprocessing/build_processed_manifest.py
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C
from preprocessing.trim_silence import trim_silence
from preprocessing.resample_audio import resample_audio
from preprocessing.amplitude_normalization import normalize_amplitude

logger = logging.getLogger(__name__)

# ── Optional tqdm ─────────────────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)

except ImportError:
    def _progress(iterable, **kw):   # type: ignore[misc]
        return iterable

# Recordings whose trimmed duration falls below this are skipped (consistent
# with the 0.5s floor used in the quality-assessment stage).
MIN_PROCESSED_DURATION_S: float = 0.50

# Columns carried over unchanged from the raw manifest.
_CARRY_COLUMNS = [
    'recording_id', 'subject_id', 'label', 'dataset',
    'task', 'language', 'gender', 'age',
]


# =============================================================================
# Path mapping
# =============================================================================

def _processed_path_for(raw_file_path: str) -> Path:
    """
    Map a raw audio path to its processed-audio destination, preserving the
    directory structure beneath raw_audio/.

    Example
    -------
    raw_audio/neurovoz/zenodo_upload/audios/HC_A1_0034.wav
      → processed_audio/neurovoz/zenodo_upload/audios/HC_A1_0034.wav

    Falls back to a flat dataset-free placement only if the raw path is not
    located under RAW_AUDIO_DIR (should not happen for valid manifests).
    """
    raw = Path(raw_file_path)
    try:
        relative = raw.relative_to(C.RAW_AUDIO_DIR)
    except ValueError:
        # Path is not under raw_audio/ — fall back to basename to avoid crashing.
        logger.warning("Path not under raw_audio/: %s", raw)
        relative = Path(raw.name)
    return C.PROCESSED_DIR / relative


# =============================================================================
# Single-recording processing
# =============================================================================

def _process_one(row: pd.Series) -> Tuple[Optional[dict], Optional[str]]:
    """
    Process a single manifest row through the full preprocessing chain.

    Returns
    -------
    (manifest_record, None)      on success
    (None, failure_reason)       on skip or failure

    The waveform is loaded exactly once and passed by value through each
    pure preprocessing function.
    """
    recording_id = row['recording_id']
    raw_path     = str(row['file_path'])

    # ── Load once ─────────────────────────────────────────────────────────
    try:
        import librosa
        waveform, sr = librosa.load(raw_path, sr=None, mono=True)
        waveform = waveform.astype(np.float32)
    except Exception as exc:
        logger.error("Load failed for %s: %s", recording_id, exc)
        return None, 'load_error'

    if waveform.size == 0:
        return None, 'empty_waveform'

    # ── Preprocessing chain ───────────────────────────────────────────────
    try:
        trimmed,    sr_t, _ = trim_silence(waveform, sr)
        resampled,  sr_r, _ = resample_audio(trimmed, sr_t)
        normalized, _       = normalize_amplitude(resampled)
    except Exception as exc:
        logger.error("Preprocessing failed for %s: %s", recording_id, exc)
        return None, 'preprocessing_error'

    # ── Skip recordings that became too short after trimming ──────────────
    final_duration = len(normalized) / sr_r if sr_r else 0.0
    if final_duration < MIN_PROCESSED_DURATION_S:
        return None, 'too_short_after_trim'

    # ── Save processed audio (mirroring raw structure) ────────────────────
    out_path = _processed_path_for(raw_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), normalized, sr_r)
    except Exception as exc:
        logger.error("Save failed for %s: %s", recording_id, exc)
        return None, 'save_error'

    # ── Build the processed-manifest record ───────────────────────────────
    record = {col: row.get(col) for col in _CARRY_COLUMNS}
    record['processed_file_path'] = str(out_path)
    record['duration']    = round(final_duration, 6)
    record['sample_rate'] = int(sr_r)

    return record, None


# =============================================================================
# Main orchestration
# =============================================================================

def build_processed_manifest(
    raw_manifest_path : Path = C.RAW_MANIFEST_CSV,
    processed_manifest_path : Path = C.PROCESSED_MANIFEST_CSV,
) -> pd.DataFrame:
    """
    Process the entire dataset and build processed_manifest.csv.

    Parameters
    ----------
    raw_manifest_path : read-only source manifest
    processed_manifest_path : destination for the processed manifest

    Returns
    -------
    The processed manifest as a DataFrame (also written to disk).
    """
    start_time = time.time()

    if not raw_manifest_path.exists():
        logger.error("Raw manifest not found: %s", raw_manifest_path)
        return pd.DataFrame()

    df = pd.read_csv(raw_manifest_path)
    total = len(df)
    logger.info("Loaded raw manifest — %d recordings.", total)

    records: list[dict] = []
    failures: Counter   = Counter()
    n_success = 0

    for _, row in _progress(df.iterrows(), total=total, desc="Preprocessing"):
        record, reason = _process_one(row)
        if record is not None:
            records.append(record)
            n_success += 1
        else:
            failures[reason] += 1

    # ── Build and save processed manifest ─────────────────────────────────
    manifest_columns = _CARRY_COLUMNS + [
        'processed_file_path', 'duration', 'sample_rate',
    ]
    processed_df = pd.DataFrame(records, columns=manifest_columns)

    processed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_manifest_path, index=False)
    logger.info("Saved → %s  (%d rows)", processed_manifest_path, len(processed_df))

    elapsed = time.time() - start_time
    _print_summary(total, n_success, failures, processed_manifest_path, elapsed)

    return processed_df


# =============================================================================
# Summary
# =============================================================================

def _print_summary(
    total    : int,
    success  : int,
    failures : Counter,
    manifest_path : Path,
    elapsed  : float,
) -> None:
    n_failed = sum(failures.values())
    sep = '=' * 56

    print(f"\n{sep}")
    print("PREPROCESSING SUMMARY")
    print(sep)
    print(f"Total recordings      : {total:,}")
    print(f"Successfully processed: {success:,}")
    print(f"Failed / skipped      : {n_failed:,}")

    if failures:
        print("\nFailures by reason:")
        for reason, count in failures.most_common():
            print(f"  {reason:<24s}: {count:,}")

    print(f"\nOutput directory      : {C.PROCESSED_DIR}")
    print(f"Processed manifest    : {manifest_path}")

    mins, secs = divmod(elapsed, 60)
    print(f"Elapsed time          : {int(mins)}m {secs:.1f}s")
    print(f"{sep}\n")


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    build_processed_manifest()