# src/dataset_building/windowing.py
"""
Fixed-length windowing for the Parkinson's FYP pipeline.

Provides a single pure function that converts a preprocessed waveform into
a list of (start_sample, end_sample) window boundaries.  It does not slice
audio, load, save, plot, or print — it returns index boundaries and a
diagnostic info dict, mirroring the pattern of trim_silence, resample_audio,
and normalize_amplitude.

Boundaries index into the ORIGINAL waveform.  Where zero-padding is
required (short recordings, kept trailing partial windows), the boundary
end index may exceed len(waveform); the consumer that materialises the
window is responsible for padding the overhang with zeros up to the fixed
window length.  This keeps this function memory-light and disk-free.

Window length and hop are derived from config.WINDOW_DURATION_SEC,
config.HOP_DURATION_SEC, and the waveform's sample rate (expected to be
config.TARGET_SAMPLE_RATE after preprocessing).

Usage
-----
    from dataset_building.windowing import compute_windows
    windows, info = compute_windows(waveform, sample_rate)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import config as C

# A trailing partial window is kept only if it introduces at least this
# fraction of a hop's worth of new samples beyond the previous window start.
# Prevents manufacturing near-all-silence windows from tiny remainders.
_TRAILING_KEEP_FRACTION: float = 0.5


def compute_windows(
    waveform: np.ndarray,
    sample_rate: int,
    window_duration_sec: float = C.WINDOW_DURATION_SEC,
    hop_duration_sec: float = C.HOP_DURATION_SEC,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """Compute fixed-length window boundaries over a waveform.

    Windows are ``window_duration_sec`` long and advance by
    ``hop_duration_sec`` each step (so a 2.0s window with a 1.0s hop gives
    50 % overlap).  Boundaries are returned as (start_sample, end_sample)
    index pairs into ``waveform``; the audio itself is never sliced here.

    Edge-case behaviour
    -------------------
    Recording shorter than one window:
        A single window ``(0, window_samples)`` is returned.  Its end index
        exceeds ``len(waveform)``; the consumer zero-pads the overhang.
        Rationale: skipping would silently discard recordings (including the
        short clinical clips near the 0.5s floor).  One padded window keeps
        every recording and yields a fixed CNN input shape.  ``padded`` is
        set True.

    Trailing partial window:
        After emitting all full windows, a final partial window is kept only
        if it advances at least ``_TRAILING_KEEP_FRACTION`` of a hop beyond
        the last full window's start.  If kept, its end index exceeds
        ``len(waveform)`` and is zero-padded by the consumer; ``padded`` is
        set True.  Otherwise the remainder is dropped.
        Rationale: preserves recording tails without manufacturing
        near-all-silence duplicate windows from tiny remainders.

    Parameters
    ----------
    waveform : np.ndarray
        1-D preprocessed audio (trimmed, resampled, normalized).
    sample_rate : int
        Sample rate in Hz.  Expected to equal config.TARGET_SAMPLE_RATE.
    window_duration_sec : float, optional
        Window length in seconds.  Defaults to config.WINDOW_DURATION_SEC.
    hop_duration_sec : float, optional
        Hop between window starts in seconds.  Defaults to
        config.HOP_DURATION_SEC.

    Returns
    -------
    windows : List[Tuple[int, int]]
        (start_sample, end_sample) pairs.  end_sample - start_sample always
        equals window_samples.  end_sample may exceed len(waveform) when
        padding is required.
    info : Dict[str, float]
        Diagnostics:
          - ``num_windows``       : number of windows produced
          - ``window_samples``    : samples per window
          - ``hop_samples``       : hop in samples
          - ``num_samples``       : input waveform length
          - ``padded``            : 1.0 if any window needs zero-padding
          - ``trailing_kept``     : 1.0 if a trailing partial window was kept
          - ``full_windows``      : count of windows fully inside the waveform

    Raises
    ------
    ValueError
        If sample_rate <= 0, or if the derived window/hop sizes are
        non-positive, or if hop_samples > window_samples (which would leave
        gaps between windows).
    """
    waveform = np.asarray(waveform)
    num_samples = int(waveform.shape[0]) if waveform.ndim >= 1 else 0

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")

    window_samples = int(round(window_duration_sec * sample_rate))
    hop_samples    = int(round(hop_duration_sec * sample_rate))

    if window_samples <= 0:
        raise ValueError(
            f"window_samples must be positive (window_duration_sec="
            f"{window_duration_sec}, sample_rate={sample_rate})."
        )
    if hop_samples <= 0:
        raise ValueError(
            f"hop_samples must be positive (hop_duration_sec="
            f"{hop_duration_sec}, sample_rate={sample_rate})."
        )
    if hop_samples > window_samples:
        raise ValueError(
            f"hop_samples ({hop_samples}) exceeds window_samples "
            f"({window_samples}); this would leave gaps between windows."
        )

    # ── Empty input ───────────────────────────────────────────────────────
    if num_samples == 0:
        return [], _build_info(0, window_samples, hop_samples, 0,
                               padded=False, trailing_kept=False, full_windows=0)

    # ── Shorter than one window → one zero-padded window ──────────────────
    if num_samples < window_samples:
        windows = [(0, window_samples)]
        return windows, _build_info(
            1, window_samples, hop_samples, num_samples,
            padded=True, trailing_kept=False, full_windows=0,
        )

    # ── Emit all full windows ─────────────────────────────────────────────
    windows: List[Tuple[int, int]] = []
    start = 0
    while start + window_samples <= num_samples:
        windows.append((start, start + window_samples))
        start += hop_samples

    full_windows = len(windows)

    # ── Consider a trailing partial window ────────────────────────────────
    # `start` now points just past the last full window's start.  The last
    # full window began at start - hop_samples.  Remaining unwindowed audio
    # exists if start < num_samples.
    trailing_kept = False
    padded = False
    if start < num_samples:
        new_samples = num_samples - start
        if new_samples >= _TRAILING_KEEP_FRACTION * hop_samples:
            windows.append((start, start + window_samples))
            trailing_kept = True
            padded = True   # end index exceeds num_samples → consumer pads

    return windows, _build_info(
        len(windows), window_samples, hop_samples, num_samples,
        padded=padded, trailing_kept=trailing_kept, full_windows=full_windows,
    )


def _build_info(
    num_windows: int,
    window_samples: int,
    hop_samples: int,
    num_samples: int,
    padded: bool,
    trailing_kept: bool,
    full_windows: int,
) -> Dict[str, float]:
    """Assemble the diagnostic info dict returned by ``compute_windows``."""
    return {
        'num_windows':   float(num_windows),
        'window_samples': float(window_samples),
        'hop_samples':   float(hop_samples),
        'num_samples':   float(num_samples),
        'padded':        1.0 if padded else 0.0,
        'trailing_kept': 1.0 if trailing_kept else 0.0,
        'full_windows':  float(full_windows),
    }