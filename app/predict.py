# app/predict.py
"""
Single-file inference wrapper for the Parkinson's FYP web demo.

Runs an arbitrary uploaded .wav through the exact same preprocessing and
model chain used in training and evaluation, and returns a recording-level
prediction.  Lives under app/ alongside the Flask app; all genuinely shared
pipeline logic is imported from src/ and never reimplemented here.

Chain (identical to build_processed_manifest.py + evaluate.py):

    load waveform
      -> trim_silence        (src/preprocessing/trim_silence.py)
      -> resample_audio       (src/preprocessing/resample_audio.py)
      -> normalize_amplitude  (src/preprocessing/amplitude_normalization.py)
      -> compute_windows      (src/dataset_building/windowing.py)
      -> per window:
           compute_mel_spectrogram    (src/dataset_building/spectrogram.py)
           normalize_spectrogram      (src/dataset_building/spectrogram_normalization.py)
           model forward -> sigmoid -> per-window P(PD)
      -> average window probabilities, threshold at 0.5

No manifest entry is required — this works on a brand-new file.

Usage
-----
    from predict import predict_recording
    result = predict_recording(Path("some_upload.wav"))
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

# ── Path bootstrap ────────────────────────────────────────────────────────
# This file lives at <project_root>/app/predict.py, so the project root is
# one level up, and src/ is a sibling of app/.  Add src/ to the path so the
# shared pipeline modules import exactly as they do elsewhere in the project.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import librosa
import numpy as np
import torch

import config as C
# Preprocessing (imported from src/, reused unchanged).
from preprocessing.trim_silence import trim_silence
from preprocessing.resample_audio import resample_audio
from preprocessing.amplitude_normalization import normalize_amplitude
# Dataset building (imported from src/, reused unchanged).
from dataset_building.windowing import compute_windows
from dataset_building.spectrogram import compute_mel_spectrogram
from dataset_building.spectrogram_normalization import (
    load_spectrogram_stats,
    normalize_spectrogram,
)
# Model + checkpoint loading (imported from src/, reused — not rewritten).
from models.cnn_lstm import ParkinsonCNNLSTM
from models.cnn import ParkinsonCNN  # noqa: F401  (available for CNN checkpoints)
from evaluation.evaluate import load_checkpoint

logger = logging.getLogger(__name__)

_DECISION_THRESHOLD: float = 0.5
_LABELS = {0: 'HC', 1: 'PD'}

# Load the train-derived spectrogram statistics ONCE at import, exactly as the
# Dataset does — the same mean/std used in training and evaluation.
_MEAN, _STD = load_spectrogram_stats()

_DEFAULT_MODEL_CLASS = ParkinsonCNNLSTM


def _slice_window(
    waveform: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    """Slice [start, end), zero-padding the tail if end exceeds the length.

    Replicates the windowing.py padding convention used everywhere else in
    the pipeline, so window tensors are always exactly the window length.
    """
    window_len = end_sample - start_sample
    available  = waveform[start_sample:end_sample]
    if available.shape[0] < window_len:
        out = np.zeros(window_len, dtype=np.float32)
        out[:available.shape[0]] = available
        return out
    return available.astype(np.float32, copy=False)


def predict_recording(
    audio_path: Path,
    run_name: str = "cnn_lstm_v1",
    model_class: type = _DEFAULT_MODEL_CLASS,
) -> Dict[str, object]:
    """Predict HC/PD for a single arbitrary .wav file.

    Runs the full preprocessing-and-inference chain and aggregates every
    window's probability into one recording-level decision by averaging
    (the same soft-vote aggregation used in evaluate.py).

    Parameters
    ----------
    audio_path : Path
        Path to a .wav file.  Need not appear in any manifest.
    run_name : str, optional
        Checkpoint run name, loaded from
        ``config.CHECKPOINTS_DIR / f"{run_name}_best.pt"``.  Defaults to
        ``"cnn_lstm_v1"``.
    model_class : type, optional
        The architecture matching the checkpoint.  Defaults to
        ParkinsonCNNLSTM.  Pass ParkinsonCNN to evaluate a CNN checkpoint.

    Returns
    -------
    Dict[str, object]
        ``{"label": "PD"|"HC", "confidence": float 0-100,
           "probability_pd": float 0-1, "num_windows": int}``.
        ``confidence`` is confidence in the PREDICTED label: probability_pd
        when PD is predicted, else (1 - probability_pd).

    Raises
    ------
    FileNotFoundError
        If the audio file or the checkpoint does not exist.
    RuntimeError
        If the file cannot be decoded, or yields no analysable windows.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ── Load once, native sample rate, mono float32 ───────────────────────
    try:
        waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)
        waveform = waveform.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Could not decode audio {audio_path.name}: {exc}") from exc

    if waveform.size == 0:
        raise RuntimeError(f"Audio file is empty: {audio_path.name}")

    # ── Preprocessing — SAME ORDER as build_processed_manifest.py ─────────
    trimmed,    sr_t, _ = trim_silence(waveform, sr)
    resampled,  sr_r, _ = resample_audio(trimmed, sr_t)   # -> TARGET_SAMPLE_RATE
    normalized, _       = normalize_amplitude(resampled)

    # ── Windowing ─────────────────────────────────────────────────────────
    windows, _info = compute_windows(normalized, sr_r)
    if not windows:
        raise RuntimeError(
            f"No windows produced for {audio_path.name} — the recording may "
            f"be too short after silence trimming."
        )

    # ── Load model ────────────────────────────────────────────────────────
    # NOTE: loaded per call — fine for a demo (one prediction at a time). If
    # this were ever looped over many files, cache the model outside the loop.
    model = load_checkpoint(run_name, model_class)
    device = next(model.parameters()).device

    # ── Per-window spectrogram -> normalize -> logit -> P(PD) ─────────────
    window_probs = []
    with torch.no_grad():
        for start, end in windows:
            window = _slice_window(normalized, start, end)
            spec, _ = compute_mel_spectrogram(window, sr_r)
            norm_spec, _ = normalize_spectrogram(spec, _MEAN, _STD)

            # (N_MELS, T) -> (1, 1, N_MELS, T): add channel + batch dims.
            tensor = torch.from_numpy(norm_spec).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(tensor)                      # (1,)
            prob = torch.sigmoid(logit).item()         # scalar P(PD)
            window_probs.append(prob)

    # ── Recording-level aggregation ───────────────────────────────────────
    # Same soft-vote logic as evaluate.aggregate_to_recording_level: average
    # window probabilities, threshold at 0.5. Done directly on the array here
    # because there is a single recording with no recording_id to group on.
    probability_pd = float(np.mean(window_probs))
    prediction = int(probability_pd >= _DECISION_THRESHOLD)
    label = _LABELS[prediction]

    # Confidence in the PREDICTED label (not always P(PD)).
    confidence = probability_pd if prediction == 1 else (1.0 - probability_pd)

    logger.info(
        "%s -> %s (P(PD)=%.3f, confidence=%.1f%%, %d windows)",
        audio_path.name, label, probability_pd, confidence * 100, len(windows),
    )

    return {
        'label':          label,
        'confidence':     round(confidence * 100, 2),
        'probability_pd': round(probability_pd, 4),
        'num_windows':    len(windows),
    }