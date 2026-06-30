# src/preprocessing/resample_audio.py
"""
Resampling for the Parkinson's FYP preprocessing pipeline.

Provides a single pure function that resamples an already-loaded waveform
to a configurable target sample rate.  It does not load, save, plot, or
print — it transforms a waveform array and returns the result, so it slots
cleanly into the eventual pipeline:

    load waveform → trim_silence → resample → normalize → save

Uses librosa.resample under the hood.
"""

from __future__ import annotations

from typing import Dict, Tuple

import librosa
import numpy as np

import config as C


def resample_audio(
    waveform: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = C.TARGET_SAMPLE_RATE,
    res_type: str = "soxr_hq",
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    """Resample a waveform to a target sample rate.

    If the waveform is already at the target rate, it is returned unchanged
    (cast to float32) without invoking the resampler.  The result is always
    float32.

    Parameters
    ----------
    waveform : np.ndarray
        Mono audio samples as a 1-D array.  Any dtype is accepted but the
        result is cast to float32.
    sample_rate : int
        The current sample rate of ``waveform`` in Hz.
    target_sample_rate : int, optional
        The desired output sample rate in Hz.  Defaults to
        ``config.TARGET_SAMPLE_RATE`` (16 000 Hz).
    res_type : str, optional
        Resampling method passed to librosa.  Defaults to ``"soxr_hq"``,
        a high-quality, fast resampler.

    Returns
    -------
    resampled : np.ndarray
        The resampled waveform as float32.
    target_sample_rate : int
        The output sample rate.  Equals ``target_sample_rate`` on success;
        equals the input ``sample_rate`` only in the no-op case (which only
        occurs when they were already equal).
    info : Dict[str, float]
        Diagnostic information:
          - ``resampled``         : 1.0 if resampling occurred, else 0.0
          - ``original_sr``       : input sample rate
          - ``target_sr``         : output sample rate
          - ``original_samples``  : sample count before resampling
          - ``resampled_samples`` : sample count after resampling
          - ``original_duration`` : duration in seconds before
          - ``resampled_duration``: duration in seconds after

    Notes
    -----
    Duration is preserved by resampling (only the sample count changes).
    If the waveform is empty, an empty float32 array is returned at the
    target sample rate.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    original_samples = int(waveform.shape[0])
    original_sr = int(sample_rate)
    target_sr = int(target_sample_rate)

    # Guard: empty input — nothing to resample.
    if original_samples == 0:
        info = _build_info(False, original_sr, target_sr, 0, 0)
        return waveform, target_sr, info

    # Fast path: already at the target rate — no computation needed.
    if original_sr == target_sr:
        info = _build_info(
            False, original_sr, target_sr,
            original_samples, original_samples,
        )
        return waveform, target_sr, info

    resampled = librosa.resample(
        waveform,
        orig_sr=original_sr,
        target_sr=target_sr,
        res_type=res_type,
    ).astype(np.float32, copy=False)

    info = _build_info(
        True, original_sr, target_sr,
        original_samples, int(resampled.shape[0]),
    )
    return resampled, target_sr, info


def _build_info(
    resampled: bool,
    original_sr: int,
    target_sr: int,
    original_samples: int,
    resampled_samples: int,
) -> Dict[str, float]:
    """Assemble the diagnostic info dict returned by ``resample_audio``."""
    o_sr = float(original_sr) if original_sr else 1.0
    t_sr = float(target_sr) if target_sr else 1.0
    return {
        'resampled':          1.0 if resampled else 0.0,
        'original_sr':        float(original_sr),
        'target_sr':          float(target_sr),
        'original_samples':   float(original_samples),
        'resampled_samples':  float(resampled_samples),
        'original_duration':  original_samples / o_sr,
        'resampled_duration': resampled_samples / t_sr,
    }