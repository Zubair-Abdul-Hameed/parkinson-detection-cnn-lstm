# save as test_spectrogram.py in the project root, run from there
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import librosa

import config as C
from dataset_building.windowing import compute_windows
from dataset_building.spectrogram import compute_mel_spectrogram

SR = C.TARGET_SAMPLE_RATE
W  = int(round(C.WINDOW_DURATION_SEC * SR))
print(f"config: n_mels={C.N_MELS}, n_fft={C.N_FFT}, hop_length={C.HOP_LENGTH}, "
      f"f_min={C.F_MIN}, f_max={C.F_MAX}")
print(f"window length: {W} samples ({C.WINDOW_DURATION_SEC}s at {SR} Hz)\n")

def check(name, wav):
    spec, info = compute_mel_spectrogram(wav.astype(np.float32), SR)
    print(f"=== {name} ===")
    print(f"  shape      : {spec.shape}  (n_mels × n_frames)")
    print(f"  dtype      : {spec.dtype}")
    print(f"  dB range   : [{info['min_db']:.2f}, {info['max_db']:.2f}]")
    print(f"  is_silent  : {bool(info['is_silent'])}")
    print(f"  has_nan    : {bool(info['has_nan'])}")
    print(f"  has_inf    : {bool(info['has_inf'])}")
    assert spec.dtype == np.float32,          "dtype not float32!"
    assert not info['has_nan'],               "NaN in output!"
    assert not info['has_inf'],               "inf in output!"
    assert spec.shape[0] == C.N_MELS,         "wrong mel-bin count!"
    print()
    return spec.shape

# ── Edge cases ───────────────────────────────────────────────────────────────
shapes = set()
shapes.add(check("silent (all zeros)",  np.zeros(W)))                    # padded-window case
shapes.add(check("white noise",         np.random.uniform(-1, 1, W)))
shapes.add(check("sine 440Hz",
                 0.5 * np.sin(2*np.pi*440*np.arange(W)/SR)))
shapes.add(check("quiet sine (1e-4)",
                 1e-4 * np.sin(2*np.pi*220*np.arange(W)/SR)))

print(f"All windows produced identical shape? {len(shapes) == 1}  ({shapes})")

# ── Real window from a processed recording ───────────────────────────────────
w = pd.read_csv(C.WINDOWS_MANIFEST_CSV).iloc[0]
wav, sr = librosa.load(w['processed_file_path'], sr=None, mono=True)
wav = wav.astype(np.float32)

start, end = int(w['start_sample']), int(w['end_sample'])
window = wav[start:end]
# Zero-pad the overhang exactly as the future Dataset will
if len(window) < (end - start):
    window = np.pad(window, (0, (end - start) - len(window)))

spec, info = compute_mel_spectrogram(window, sr)
print(f"\n=== REAL window from {w['recording_id']} ===")
print(f"  window samples : {len(window)} (expected {W})")
print(f"  spec shape     : {spec.shape}")
print(f"  dB range       : [{info['min_db']:.2f}, {info['max_db']:.2f}]")
print(f"  finite?        : {np.isfinite(spec).all()}")
print("\nAll assertions passed.")