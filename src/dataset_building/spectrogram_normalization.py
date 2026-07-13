# src/dataset_building/spectrogram_normalization.py
"""
Z-score normalization for mel spectrograms in the Parkinson's FYP pipeline.

Two responsibilities, kept deliberately separate:

  normalize_spectrogram()   — PURE function.  (spectrogram, mean, std) in,
                              normalized spectrogram out.  No file I/O.
                              Called on-the-fly inside Dataset.__getitem__,
                              once per window per epoch.

  load_spectrogram_stats()  — convenience helper that reads the mean/std
                              JSON produced by compute_spectrogram_stats.py.
                              Meant to be called ONCE (e.g. at Dataset
                              __init__), NOT per window.

The mean/std are the GLOBAL scalar train-only statistics.  The same values
are applied to every window regardless of split — val/test are normalized
using train statistics, which is standard and leakage-free.

Usage
-----
    from dataset_building.spectrogram_normalization import (
        load_spectrogram_stats, normalize_spectrogram,
    )
    mean, std = load_spectrogram_stats()          # once, at init
    norm, info = normalize_spectrogram(spec, mean, std)   # per window
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import config as C

logger = logging.getLogger(__name__)

# A std below this is treated as degenerate; normalization falls back to
# mean-centering only (divide by 1.0) rather than producing inf/NaN.
_MIN_VALID_STD: float = 1e-8


def normalize_spectrogram(
    spectrogram: np.ndarray,
    mean: float,
    std: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply z-score normalization to a mel spectrogram.

    Computes ``(spectrogram - mean) / std`` using the global scalar
    train-only statistics passed in.  The result is float32.

    Std guard
    ---------
    If ``std`` is below ``_MIN_VALID_STD`` (degenerate — should not happen
    with real data), the function falls back to dividing by 1.0, i.e. it
    only mean-centers the spectrogram.  This avoids inf/NaN while still
    producing a usable, finite output.  ``fallback_used`` in the info dict
    is set to 1.0 when this happens.

    Parameters
    ----------
    spectrogram : np.ndarray
        Mel spectrogram of shape (n_mels, n_frames), dB scale, as produced
        by spectrogram.py.  Any dtype accepted; cast to float32 internally.
    mean : float
        Global scalar mean (train-only) to subtract.
    std : float
        Global scalar std (train-only) to divide by.

    Returns
    -------
    normalized : np.ndarray
        The z-score normalized spectrogram as float32, same shape as input.
    info : Dict[str, float]
        Diagnostics:
          - ``output_mean``   : mean of the normalized array
          - ``output_std``    : std of the normalized array
          - ``input_mean``    : mean of the input array (pre-normalization)
          - ``fallback_used`` : 1.0 if the std guard was triggered, else 0.0
          - ``stat_mean``     : the mean value used
          - ``stat_std``      : the std value used
    """
    spectrogram = np.asarray(spectrogram, dtype=np.float32)

    input_mean = float(spectrogram.mean()) if spectrogram.size else 0.0

    # ── Std guard ─────────────────────────────────────────────────────────
    if std < _MIN_VALID_STD:
        logger.warning(
            "std=%.3e below minimum valid threshold; falling back to "
            "mean-centering only (divide by 1.0).", std,
        )
        effective_std = 1.0
        fallback_used = True
    else:
        effective_std = float(std)
        fallback_used = False

    normalized = ((spectrogram - float(mean)) / effective_std).astype(
        np.float32, copy=False
    )

    output_mean = float(normalized.mean()) if normalized.size else 0.0
    output_std  = float(normalized.std())  if normalized.size else 0.0

    info: Dict[str, float] = {
        'output_mean':   output_mean,
        'output_std':    output_std,
        'input_mean':    input_mean,
        'fallback_used': 1.0 if fallback_used else 0.0,
        'stat_mean':     float(mean),
        'stat_std':      float(std),
    }
    return normalized, info


def load_spectrogram_stats(
    stats_path: Path = C.SPECTROGRAM_STATS_JSON,
    validate_mel_config: bool = True,
) -> Tuple[float, float]:
    """Load the global (mean, std) from the spectrogram statistics JSON.

    Intended to be called ONCE (e.g. at Dataset __init__), not per window.
    The returned scalars are then passed into ``normalize_spectrogram`` for
    each window.

    Parameters
    ----------
    stats_path : Path
        Path to spectrogram_mean_std.json.  Defaults to
        config.SPECTROGRAM_STATS_JSON.
    validate_mel_config : bool, optional
        If True (default), warn when the mel_config stored in the JSON does
        not match the current config.py values — this would mean the stats
        were computed under different settings and should be recomputed.

    Returns
    -------
    (mean, std) : Tuple[float, float]
        The global scalar statistics.

    Raises
    ------
    FileNotFoundError
        If the statistics file does not exist.
    KeyError
        If the JSON is missing the required 'mean' or 'std' fields.
    ValueError
        If the loaded std is non-positive.
    """
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Spectrogram statistics not found: {stats_path}. "
            f"Run compute_spectrogram_stats.py first."
        )

    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    if 'mean' not in stats or 'std' not in stats:
        raise KeyError(
            f"Statistics file {stats_path} missing 'mean' or 'std'."
        )

    mean = float(stats['mean'])
    std  = float(stats['std'])

    if std <= 0.0:
        raise ValueError(
            f"Loaded std must be positive, got {std}. Statistics may be corrupt."
        )

    # ── Optional mel-config sanity check ──────────────────────────────────
    if validate_mel_config and 'mel_config' in stats:
        expected = {
            'n_mels':      C.N_MELS,
            'n_fft':       C.N_FFT,
            'hop_length':  C.HOP_LENGTH,
            'f_min':       C.F_MIN,
            'f_max':       C.F_MAX,
            'sample_rate': C.TARGET_SAMPLE_RATE,
        }
        stored = stats['mel_config']
        mismatches = {
            k: (stored.get(k), v)
            for k, v in expected.items()
            if stored.get(k) != v
        }
        if mismatches:
            logger.warning(
                "mel_config in %s does not match config.py: %s. "
                "Statistics may be stale — consider recomputing.",
                stats_path.name, mismatches,
            )

    return mean, std