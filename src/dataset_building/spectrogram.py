# src/dataset_building/spectrogram.py
"""
Mel spectrogram extraction for the Parkinson's FYP pipeline.

Provides a single pure function that converts one window's waveform into a
log-scaled (dB) mel spectrogram.  It does not load, save, read CSVs, or
know anything about train/val/test splits — it transforms an in-memory
array and returns the result plus a diagnostic info dict, mirroring the
pattern of trim_silence, resample_audio, normalize_amplitude, and
compute_windows.

This function computes NO statistics and applies NO normalization; those
are separate downstream stages.  It will be called on-the-fly inside the
PyTorch Dataset.__getitem__, once per window per epoch.

Mel parameters default to the constants in config.py
(N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX).

Usage
-----
    from dataset_building.spectrogram import compute_mel_spectrogram
    spec, info = compute_mel_spectrogram(window_waveform, sample_rate)
"""

from __future__ import annotations

from typing import Dict, Tuple

import librosa
import numpy as np

import config as C

# Dynamic-range floor for power_to_db.  Values more than this many dB below
# the reference are clamped, which also converts the -inf that log(0) would
# produce for silent (zero-padded) windows into a finite floor value.
# 80 dB is librosa's default and standard for speech spectrograms.
_TOP_DB: float = 80.0

# A window whose peak absolute amplitude is below this is treated as silent
# (for diagnostic reporting only — the computation handles it either way).
_SILENCE_PEAK_THRESHOLD: float = 1e-6


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = C.N_MELS,
    n_fft: int = C.N_FFT,
    hop_length: int = C.HOP_LENGTH,
    f_min: float = C.F_MIN,
    f_max: float = C.F_MAX,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute a log-scaled (dB) mel spectrogram for one window waveform.

    The power mel spectrogram is converted to decibels with a fixed
    reference of 1.0 (not per-window max).  A fixed reference keeps dB
    values comparable across windows — the same acoustic energy maps to the
    same dB value everywhere — which matters because the waveforms are
    already peak-normalized upstream and a later stage computes dataset-wide
    statistics.  Using ``ref=np.max`` would re-normalize every window to its
    own peak and destroy that cross-window comparability.

    Silent (e.g. zero-padded) windows are handled by ``power_to_db``'s
    ``top_db`` flooring: ``log(0)`` would be ``-inf``, but ``top_db`` clamps
    everything to within ``_TOP_DB`` dB of the reference, so a silent window
    becomes a finite, uniform floor-valued spectrogram rather than producing
    ``-inf`` / ``NaN`` that would poison training.

    Parameters
    ----------
    waveform : np.ndarray
        1-D window waveform, already the fixed window length.  Any dtype is
        accepted; it is cast to float32 internally.
    sample_rate : int
        Sample rate in Hz.  Expected to equal config.TARGET_SAMPLE_RATE.
    n_mels : int, optional
        Number of mel bands.  Defaults to config.N_MELS.
    n_fft : int, optional
        FFT window size.  Defaults to config.N_FFT.
    hop_length : int, optional
        Hop length between STFT frames.  Defaults to config.HOP_LENGTH.
    f_min : float, optional
        Lowest frequency (Hz).  Defaults to config.F_MIN.
    f_max : float, optional
        Highest frequency (Hz).  Defaults to config.F_MAX.

    Returns
    -------
    mel_db : np.ndarray
        Log-scaled mel spectrogram as a float32 2-D array of shape
        (n_mels, n_frames).
    info : Dict[str, float]
        Diagnostics:
          - ``n_mels``       : number of mel bands (rows)
          - ``n_frames``     : number of time frames (columns)
          - ``min_db``       : minimum dB value in the output
          - ``max_db``       : maximum dB value in the output
          - ``is_silent``    : 1.0 if the window was detected as silent
          - ``has_nan``      : 1.0 if any NaN slipped through (should be 0)
          - ``has_inf``      : 1.0 if any inf slipped through (should be 0)

    Raises
    ------
    ValueError
        If sample_rate <= 0 or the waveform is not 1-D.
    """
    waveform = np.asarray(waveform, dtype=np.float32)

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if waveform.ndim != 1:
        raise ValueError(
            f"waveform must be 1-D, got shape {waveform.shape}."
        )

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    is_silent = peak < _SILENCE_PEAK_THRESHOLD

    # ── Power mel spectrogram ─────────────────────────────────────────────
    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
    )

    # ── Convert to dB with a fixed reference and a finite floor ───────────
    # ref=1.0 keeps values comparable across windows; top_db floors -inf.
    mel_db = librosa.power_to_db(mel_power, ref=1.0, top_db=_TOP_DB)
    mel_db = mel_db.astype(np.float32, copy=False)

    # ── Diagnostics ───────────────────────────────────────────────────────
    has_nan = bool(np.isnan(mel_db).any())
    has_inf = bool(np.isinf(mel_db).any())

    info: Dict[str, float] = {
        'n_mels':    float(mel_db.shape[0]),
        'n_frames':  float(mel_db.shape[1]),
        'min_db':    float(mel_db.min()),
        'max_db':    float(mel_db.max()),
        'is_silent': 1.0 if is_silent else 0.0,
        'has_nan':   1.0 if has_nan else 0.0,
        'has_inf':   1.0 if has_inf else 0.0,
    }
    return mel_db, info