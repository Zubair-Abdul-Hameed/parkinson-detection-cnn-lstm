# src/dataset_building/compute_spectrogram_stats.py
"""
Compute mel spectrogram normalization statistics — TRAIN WINDOWS ONLY.

Reads data/windows_manifest.csv and data/splits/train.csv, keeps only the
windows whose subject_id is in the training split, computes each window's
mel spectrogram via spectrogram.py, and accumulates a running global mean
and standard deviation over every mel value.  The result is written to
config.SPECTROGRAM_STATS_JSON.

Data-leakage safety
-------------------
Validation and test windows are NEVER loaded, sliced, or processed here.
Filtering to train subjects happens before any audio is opened.

Memory safety
-------------
Spectrograms are folded into running scalar accumulators (sum,
sum-of-squares, count) one at a time.  No array of all train spectrograms
is ever held in memory.  Each processed recording is loaded exactly once
(windows are grouped by file) with only one waveform resident at a time.

Statistics choice
-----------------
A single GLOBAL scalar mean and std are computed (not per-mel-bin).  The
mel spectrogram is a single-channel CNN input; the mel bins are the image
height, not separate channels.  Global standardization gives zero-mean/
unit-variance input while preserving the relative energy structure across
mel bins that the CNN can exploit.

Output
------
  data/statistics/spectrogram_mean_std.json

Usage
-----
    python src/dataset_building/compute_spectrogram_stats.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C
from dataset_building.spectrogram import compute_mel_spectrogram

logger = logging.getLogger(__name__)

# ── Optional tqdm ─────────────────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)

except ImportError:
    def _progress(iterable, **kw):   # type: ignore[misc]
        return iterable


# =============================================================================
# Running accumulator
# =============================================================================

class _RunningStats:
    """
    Streaming global mean/std accumulator.

    Accumulates in float64 for numerical stability, folding in any-shaped
    array of values at a time.  Mean = Σx / N; Std = sqrt(Σx²/N − mean²).
    """

    def __init__(self) -> None:
        self.sum_x   : float = 0.0
        self.sum_x2  : float = 0.0
        self.count   : int   = 0

    def update(self, values: np.ndarray) -> None:
        """Fold an array of values into the running accumulation."""
        v = values.astype(np.float64, copy=False).ravel()
        self.sum_x  += float(v.sum())
        self.sum_x2 += float(np.dot(v, v))   # Σx² without building x² array
        self.count  += v.size

    def finalize(self) -> Tuple[float, float]:
        """Return (mean, std).  std uses the population formula (÷ N)."""
        if self.count == 0:
            raise ValueError("No values accumulated — cannot compute statistics.")
        mean = self.sum_x / self.count
        var  = self.sum_x2 / self.count - mean * mean
        var  = max(var, 0.0)                 # guard tiny negative from rounding
        return float(mean), float(np.sqrt(var))


# =============================================================================
# Window slicing (matches windowing.py padding convention)
# =============================================================================

def _slice_window(
    waveform: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    """
    Slice [start_sample, end_sample) from waveform, zero-padding the tail if
    end_sample exceeds the waveform length (the windowing.py convention).

    The returned array is always exactly (end_sample - start_sample) long.
    """
    window_len = end_sample - start_sample
    available  = waveform[start_sample:end_sample]
    if available.shape[0] < window_len:
        out = np.zeros(window_len, dtype=np.float32)
        out[:available.shape[0]] = available
        return out
    return available.astype(np.float32, copy=False)


# =============================================================================
# Main computation
# =============================================================================

def compute_spectrogram_stats(
    windows_manifest_path: Path = C.WINDOWS_MANIFEST_CSV,
    train_csv_path:        Path = C.TRAIN_CSV,
    output_path:           Path = C.SPECTROGRAM_STATS_JSON,
) -> dict:
    """
    Compute global mel spectrogram mean/std over train windows and save JSON.

    Parameters
    ----------
    windows_manifest_path : path to windows_manifest.csv (read-only)
    train_csv_path        : path to train.csv (used to select train subjects)
    output_path           : destination JSON path

    Returns
    -------
    The statistics dict that was written to disk.
    """
    start_time = time.time()

    if not windows_manifest_path.exists():
        logger.error("Windows manifest not found: %s", windows_manifest_path)
        return {}
    if not train_csv_path.exists():
        logger.error("Train split not found: %s", train_csv_path)
        return {}

    windows = pd.read_csv(windows_manifest_path)
    train   = pd.read_csv(train_csv_path)

    # ── Train-only filter (before any audio is opened) ────────────────────
    train_subjects = set(train['subject_id'])
    train_windows  = windows[windows['subject_id'].isin(train_subjects)].copy()
    logger.info(
        "Total windows: %d | train windows: %d (from %d train subjects).",
        len(windows), len(train_windows), len(train_subjects),
    )

    if train_windows.empty:
        logger.error("No train windows to process — aborting.")
        return {}

    # ── Group windows by file so each recording is loaded once ────────────
    train_windows.sort_values(
        ['processed_file_path', 'window_index'], inplace=True
    )

    stats     = _RunningStats()
    failures  : Counter = Counter()
    n_windows = 0
    recordings_used: set = set()

    # Cache one waveform at a time, keyed by the current file path.
    current_path : Optional[str] = None
    current_wave : Optional[np.ndarray] = None
    current_sr   : Optional[int] = None

    import librosa

    for _, row in _progress(train_windows.iterrows(),
                            total=len(train_windows),
                            desc="Accumulating stats"):
        path = str(row['processed_file_path'])

        # (Re)load only when the file changes.
        if path != current_path:
            try:
                current_wave, current_sr = librosa.load(path, sr=None, mono=True)
                current_wave = current_wave.astype(np.float32)
                current_path = path
            except Exception as exc:
                logger.error("Load failed for %s: %s", row['recording_id'], exc)
                failures['load_error'] += 1
                current_path = None       # force reload attempt next time
                continue

        window = _slice_window(
            current_wave,
            int(row['start_sample']),
            int(row['end_sample']),
        )

        try:
            spec, _info = compute_mel_spectrogram(window, int(current_sr))
        except Exception as exc:
            logger.error("Spectrogram failed for %s (win %s): %s",
                         row['recording_id'], row['window_index'], exc)
            failures['spectrogram_error'] += 1
            continue

        stats.update(spec)
        n_windows += 1
        recordings_used.add(row['recording_id'])

    # ── Finalize ──────────────────────────────────────────────────────────
    mean, std = stats.finalize()

    result = {
        'mean': mean,
        'std':  std,
        'num_train_windows':    n_windows,
        'num_train_recordings': len(recordings_used),
        'num_values_accumulated': stats.count,
        'mel_config': {
            'n_mels':     C.N_MELS,
            'n_fft':      C.N_FFT,
            'hop_length': C.HOP_LENGTH,
            'f_min':      C.F_MIN,
            'f_max':      C.F_MAX,
            'sample_rate': C.TARGET_SAMPLE_RATE,
        },
        'computed_at': datetime.now().isoformat(timespec='seconds'),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    logger.info("Saved → %s", output_path)

    elapsed = time.time() - start_time
    _print_summary(result, failures, elapsed, output_path)
    return result


# =============================================================================
# Summary
# =============================================================================

def _print_summary(
    result   : dict,
    failures : Counter,
    elapsed  : float,
    output_path : Path,
) -> None:
    sep = '=' * 60

    print(f"\n{sep}")
    print("SPECTROGRAM STATISTICS SUMMARY  (train only)")
    print(sep)
    print(f"Train recordings used  : {result['num_train_recordings']:,}")
    print(f"Train windows processed: {result['num_train_windows']:,}")
    print(f"Values accumulated     : {result['num_values_accumulated']:,}")

    n_failed = sum(failures.values())
    print(f"Failed / skipped       : {n_failed:,}")
    if failures:
        print("\nFailures by reason:")
        for reason, count in failures.most_common():
            print(f"  {reason:<22s}: {count:,}")

    print(f"\nGlobal mean (dB)       : {result['mean']:.6f}")
    print(f"Global std  (dB)       : {result['std']:.6f}")

    print(f"\nMel config             : n_mels={result['mel_config']['n_mels']}, "
          f"n_fft={result['mel_config']['n_fft']}, "
          f"hop={result['mel_config']['hop_length']}")
    print(f"Output                 : {output_path}")

    mins, secs = divmod(elapsed, 60)
    print(f"Elapsed time           : {int(mins)}m {secs:.1f}s")
    print(f"{sep}\n")


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    compute_spectrogram_stats()