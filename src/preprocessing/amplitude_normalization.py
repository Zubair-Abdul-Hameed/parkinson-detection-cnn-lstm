# src/preprocessing/amplitude_normalization.py
"""
Amplitude normalization for the Parkinson's FYP preprocessing pipeline.

Provides a single pure function that peak-normalizes an already-loaded
waveform.  It does not load, save, plot, or print — it transforms a
waveform array and returns the result, so it slots cleanly into the
eventual pipeline:

    load waveform → trim_silence → resample → normalize → save
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def normalize_amplitude(
    waveform: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Peak-normalize a waveform so its loudest sample sits at ±1.0.

    Computes ``peak = max(abs(waveform))`` and, when the peak is positive,
    divides the waveform by it.  A waveform that is entirely zeros is
    returned unchanged.  The result is always float32.

    Because every sample is divided by the maximum absolute value, the
    output is bounded within [-1.0, 1.0] by construction — peak
    normalization cannot introduce clipping.  Sample rate and duration are
    not affected (this function does not see or alter either).

    Parameters
    ----------
    waveform : np.ndarray
        Mono audio samples as a 1-D array.  Any dtype is accepted but the
        result is cast to float32.

    Returns
    -------
    normalized : np.ndarray
        The peak-normalized waveform as float32.
    info : Dict[str, float]
        Diagnostic information:
          - ``normalized``      : 1.0 if scaling occurred, else 0.0
          - ``original_peak``   : max absolute amplitude before normalization
          - ``normalized_peak`` : max absolute amplitude after normalization
          - ``num_samples``     : sample count (unchanged by this operation)

    Notes
    -----
    If the waveform is empty or entirely silent (peak == 0), it is returned
    unchanged and ``normalized`` is 0.0.  Downstream stages remain
    responsible for deciding what to do with such recordings.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    num_samples = int(waveform.shape[0])

    # Guard: empty input — nothing to normalize.
    if num_samples == 0:
        info = _build_info(False, 0.0, 0.0, 0)
        return waveform, info

    original_peak = float(np.max(np.abs(waveform)))

    # Guard: all-zero (or empty-of-signal) waveform — return unchanged.
    if original_peak <= 0.0:
        info = _build_info(False, original_peak, original_peak, num_samples)
        return waveform, info

    normalized = (waveform / original_peak).astype(np.float32, copy=False)
    normalized_peak = float(np.max(np.abs(normalized)))

    info = _build_info(True, original_peak, normalized_peak, num_samples)
    return normalized, info


def _build_info(
    normalized: bool,
    original_peak: float,
    normalized_peak: float,
    num_samples: int,
) -> Dict[str, float]:
    """Assemble the diagnostic info dict returned by ``normalize_amplitude``."""
    return {
        'normalized':      1.0 if normalized else 0.0,
        'original_peak':   float(original_peak),
        'normalized_peak': float(normalized_peak),
        'num_samples':     float(num_samples),
    }