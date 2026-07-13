# save as test_spectrogram_normalization.py in the project root, run from there
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import librosa

import config as C
from dataset_building.spectrogram import compute_mel_spectrogram
from dataset_building.spectrogram_normalization import (
    load_spectrogram_stats, normalize_spectrogram,
)

# ── 1. Load the stats (once) ─────────────────────────────────────────────────
mean, std = load_spectrogram_stats()
print(f"Loaded stats: mean={mean:.4f}, std={std:.4f}\n")

# ── 2. Synthetic sanity check ────────────────────────────────────────────────
# A spectrogram whose values equal `mean` everywhere should normalize to ~0.
flat = np.full((C.N_MELS, 63), mean, dtype=np.float32)
norm, info = normalize_spectrogram(flat, mean, std)
print("=== flat-at-mean spectrogram ===")
print(f"  output mean : {info['output_mean']:.6f}  (expect ~0)")
print(f"  output std  : {info['output_std']:.6f}   (expect ~0)")
assert abs(info['output_mean']) < 1e-4, "mean-centering failed!"
print()

# ── 3. Std guard ─────────────────────────────────────────────────────────────
norm_z, info_z = normalize_spectrogram(flat, mean, 0.0)
print("=== std=0 fallback ===")
print(f"  fallback_used : {bool(info_z['fallback_used'])}  (expect True)")
print(f"  has NaN?      : {np.isnan(norm_z).any()}  (expect False)")
print(f"  has inf?      : {np.isinf(norm_z).any()}  (expect False)")
assert info_z['fallback_used'] == 1.0
assert np.isfinite(norm_z).all()
print()

# ── 4. Real window: full chain spectrogram → normalize ───────────────────────
w = pd.read_csv(C.WINDOWS_MANIFEST_CSV).iloc[0]
wav, sr = librosa.load(w['processed_file_path'], sr=None, mono=True)
wav = wav.astype(np.float32)
start, end = int(w['start_sample']), int(w['end_sample'])
window = wav[start:end]
if len(window) < (end - start):
    window = np.pad(window, (0, (end - start) - len(window)))

spec, _ = compute_mel_spectrogram(window, sr)
norm, info = normalize_spectrogram(spec, mean, std)

print(f"=== real window: {w['recording_id']} ===")
print(f"  spec shape         : {spec.shape}")
print(f"  pre-norm  mean/std : {spec.mean():.4f} / {spec.std():.4f}")
print(f"  post-norm mean/std : {info['output_mean']:.4f} / {info['output_std']:.4f}")
print(f"  dtype              : {norm.dtype}")
print(f"  all finite?        : {np.isfinite(norm).all()}")
assert norm.dtype == np.float32
assert norm.shape == spec.shape
print("\nAll assertions passed.")