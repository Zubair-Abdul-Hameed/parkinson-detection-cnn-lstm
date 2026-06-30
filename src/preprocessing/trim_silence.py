# src/preprocessing/trim_silence.py
"""
Silence trimming for the Parkinson's FYP preprocessing pipeline.

Provides a single pure function that removes leading and trailing silence
from an already-loaded waveform.  It does not load, save, plot, or print —
it transforms a waveform array and returns the result, so it slots cleanly
into the eventual pipeline:

    load waveform → trim_silence → resample → normalize → save

Uses librosa.effects.trim under the hood.
"""

from __future__ import annotations

from typing import Dict, Tuple

import librosa
import numpy as np

import config as C


def trim_silence(
    waveform: np.ndarray,
    sample_rate: int,
    top_db: float = C.SILENCE_TOP_DB,
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    """Remove leading and trailing silence from a waveform.

    Only the silent head and tail are removed; interior silence (pauses
    between words, for example) is preserved.  The waveform is returned as
    float32.  The sample rate is returned unchanged — this function does
    not resample.

    Parameters
    ----------
    waveform : np.ndarray
        Mono audio samples as a 1-D array.  Any dtype is accepted but the
        result is cast to float32.
    sample_rate : int
        Sample rate in Hz.  Returned unchanged; used only to convert the
        sample counts in the returned info dict into seconds.
    top_db : float, optional
        The threshold (in decibels below the signal peak) under which audio
        is considered silence.  Defaults to ``config.SILENCE_TOP_DB``.  A
        smaller value trims more aggressively.

    Returns
    -------
    trimmed : np.ndarray
        The silence-trimmed waveform as float32.
    sample_rate : int
        The same sample rate that was passed in.
    info : Dict[str, float]
        Diagnostic information:
          - ``original_samples``  : sample count before trimming
          - ``trimmed_samples``   : sample count after trimming
          - ``samples_removed``   : total samples removed from both ends
          - ``leading_removed``   : samples removed from the start
          - ``trailing_removed``  : samples removed from the end
          - ``original_duration`` : duration in seconds before trimming
          - ``trimmed_duration``  : duration in seconds after trimming

    Notes
    -----
    If the waveform is empty, or is entirely silence (so librosa would
    return an empty array), the original waveform is returned unchanged
    (as float32) and ``samples_removed`` is 0.  Downstream stages remain
    responsible for deciding what to do with such recordings.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    original_samples = int(waveform.shape[0])

    # Guard: empty input — nothing to trim.
    if original_samples == 0:
        info = _build_info(0, 0, 0, 0, sample_rate)
        return waveform, sample_rate, info

    trimmed, index = librosa.effects.trim(waveform, top_db=top_db)

    # Guard: librosa returned nothing (waveform was entirely silence).
    # Preserve the original rather than emit an empty array.
    if trimmed.shape[0] == 0:
        info = _build_info(original_samples, original_samples, 0, 0, sample_rate)
        return waveform, sample_rate, info

    trimmed = trimmed.astype(np.float32, copy=False)

    # index is (start_sample, end_sample) of the kept region.
    leading_removed  = int(index[0])
    trailing_removed = int(original_samples - index[1])

    info = _build_info(
        original_samples=original_samples,
        trimmed_samples=int(trimmed.shape[0]),
        leading_removed=leading_removed,
        trailing_removed=trailing_removed,
        sample_rate=sample_rate,
    )
    return trimmed, sample_rate, info


def _build_info(
    original_samples: int,
    trimmed_samples: int,
    leading_removed: int,
    trailing_removed: int,
    sample_rate: int,
) -> Dict[str, float]:
    """Assemble the diagnostic info dict returned by ``trim_silence``."""
    sr = float(sample_rate) if sample_rate else 1.0
    return {
        'original_samples':  float(original_samples),
        'trimmed_samples':   float(trimmed_samples),
        'samples_removed':   float(leading_removed + trailing_removed),
        'leading_removed':   float(leading_removed),
        'trailing_removed':  float(trailing_removed),
        'original_duration': original_samples / sr,
        'trimmed_duration':  trimmed_samples / sr,
    }