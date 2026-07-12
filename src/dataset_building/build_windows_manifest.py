# src/dataset_building/build_windows_manifest.py
"""
Windows-manifest builder for the Parkinson's FYP pipeline.

Reads data/processed_manifest.csv, applies the pure windowing function from
windowing.py to each processed recording, and writes data/windows_manifest.csv
— one row per window, metadata only (no audio data).

The manifest is SPLIT-AGNOSTIC: it has no split column.  Split membership is
resolved downstream by joining subject_id against train/val/test.csv.

Each recording's waveform is loaded once — not to slice it (no audio is
written), but to (a) obtain its true length for windowing and (b) verify the
processed file is present and readable before committing window rows that
reference it.

Usage
-----
    python src/dataset_building/build_windows_manifest.py
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C
from dataset_building.windowing import compute_windows

logger = logging.getLogger(__name__)

# ── Optional tqdm ─────────────────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)

except ImportError:
    def _progress(iterable, **kw):   # type: ignore[misc]
        return iterable

# Columns carried over unchanged from the processed manifest into each
# window row (downstream spectrogram extraction, training, and evaluation
# all read from these).
_CARRY_COLUMNS = [
    'recording_id', 'subject_id', 'label', 'dataset',
    'task', 'language', 'gender', 'age',
    'processed_file_path', 'sample_rate',
]

# Final column order for windows_manifest.csv.
_WINDOW_COLUMNS = _CARRY_COLUMNS + [
    'window_index', 'start_sample', 'end_sample',
]


# =============================================================================
# Single-recording windowing
# =============================================================================

def _window_one(row: pd.Series) -> Tuple[List[dict], Optional[str]]:
    """
    Produce window-manifest rows for a single processed recording.

    Returns
    -------
    (window_records, None)   on success (list may be empty only for empty audio)
    ([], failure_reason)     on load failure

    The waveform is loaded once to obtain its true length and to verify the
    file is readable; no audio samples are written anywhere.
    """
    recording_id   = row['recording_id']
    processed_path = str(row['processed_file_path'])
    sample_rate    = int(row['sample_rate'])

    # ── Load once (length + readability check) ─────────────────────────────
    try:
        import librosa
        waveform, sr = librosa.load(processed_path, sr=None, mono=True)
    except Exception as exc:
        logger.error("Load failed for %s: %s", recording_id, exc)
        return [], 'load_error'

    if waveform.size == 0:
        return [], 'empty_waveform'

    # Guard: processed audio should already be at TARGET_SAMPLE_RATE.  If the
    # on-disk rate disagrees with the manifest, trust the file and log it.
    if sr != sample_rate:
        logger.warning(
            "%s: sample_rate on disk (%d) != manifest (%d); using disk value.",
            recording_id, sr, sample_rate,
        )
        sample_rate = sr

    # ── Compute window boundaries (pure function, reused) ─────────────────
    windows, _info = compute_windows(waveform, sample_rate)

    # ── Build one record per window ───────────────────────────────────────
    carried = {col: row.get(col) for col in _CARRY_COLUMNS}
    carried['sample_rate'] = sample_rate   # reflect the value actually used

    records: List[dict] = []
    for idx, (start_sample, end_sample) in enumerate(windows):
        record = dict(carried)
        record['window_index'] = idx
        record['start_sample'] = int(start_sample)
        record['end_sample']   = int(end_sample)
        records.append(record)

    return records, None


# =============================================================================
# Main orchestration
# =============================================================================

def build_windows_manifest(
    processed_manifest_path: Path = C.PROCESSED_MANIFEST_CSV,
    windows_manifest_path:   Path = C.WINDOWS_MANIFEST_CSV,
) -> pd.DataFrame:
    """
    Build windows_manifest.csv from processed_manifest.csv.

    Parameters
    ----------
    processed_manifest_path : read-only source manifest
    windows_manifest_path   : destination for the windows manifest

    Returns
    -------
    The windows manifest as a DataFrame (also written to disk).
    """
    start_time = time.time()

    if not processed_manifest_path.exists():
        logger.error("Processed manifest not found: %s", processed_manifest_path)
        return pd.DataFrame()

    df = pd.read_csv(processed_manifest_path)
    total = len(df)
    logger.info("Loaded processed manifest — %d recordings.", total)

    all_records: List[dict] = []
    failures: Counter       = Counter()
    n_success = 0

    for _, row in _progress(df.iterrows(), total=total, desc="Windowing"):
        records, reason = _window_one(row)
        if reason is not None:
            failures[reason] += 1
        else:
            all_records.extend(records)
            n_success += 1

    windows_df = pd.DataFrame(all_records, columns=_WINDOW_COLUMNS)

    windows_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    windows_df.to_csv(windows_manifest_path, index=False)
    logger.info("Saved → %s  (%d window rows)",
                windows_manifest_path, len(windows_df))

    elapsed = time.time() - start_time
    _print_summary(windows_df, total, n_success, failures, elapsed)

    return windows_df


# =============================================================================
# Summary
# =============================================================================

def _distribution(counts: pd.Series, label: str) -> None:
    """Print min/max/mean/median of a count series."""
    if counts.empty:
        print(f"\nWindows per {label}: (none)")
        return
    print(f"\nWindows per {label}:")
    print(f"  min    : {int(counts.min())}")
    print(f"  max    : {int(counts.max())}")
    print(f"  mean   : {counts.mean():.2f}")
    print(f"  median : {counts.median():.1f}")


def _print_summary(
    windows_df : pd.DataFrame,
    total      : int,
    success    : int,
    failures   : Counter,
    elapsed    : float,
) -> None:
    n_failed = sum(failures.values())
    sep = '=' * 60

    print(f"\n{sep}")
    print("WINDOWS MANIFEST SUMMARY")
    print(sep)
    print(f"Total recordings       : {total:,}")
    print(f"Successfully windowed   : {success:,}")
    print(f"Failed / skipped        : {n_failed:,}")
    print(f"Total windows produced  : {len(windows_df):,}")

    if failures:
        print("\nFailures by reason:")
        for reason, count in failures.most_common():
            print(f"  {reason:<22s}: {count:,}")

    if not windows_df.empty:
        # Windows per subject
        per_subject = windows_df.groupby('subject_id').size()
        _distribution(per_subject, "subject")

        # Windows per label
        print(f"\nWindows per label (totals):")
        for lbl, n in windows_df['label'].value_counts().items():
            print(f"  {lbl}: {n:,}")

    mins, secs = divmod(elapsed, 60)
    print(f"\nElapsed time            : {int(mins)}m {secs:.1f}s")
    print(f"{sep}\n")


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    build_windows_manifest()