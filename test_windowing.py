# save as test_windowing.py in the project root, run from there
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import librosa

import config as C
from dataset_building.windowing import compute_windows

SR = C.TARGET_SAMPLE_RATE
W  = int(round(C.WINDOW_DURATION_SEC * SR))   # window samples
H  = int(round(C.HOP_DURATION_SEC * SR))      # hop samples
print(f"config: window={C.WINDOW_DURATION_SEC}s ({W} samples), "
      f"hop={C.HOP_DURATION_SEC}s ({H} samples), sr={SR}\n")

def show(name, n_samples):
    wav = np.zeros(n_samples, dtype=np.float32)
    windows, info = compute_windows(wav, SR)
    print(f"=== {name}: {n_samples} samples ({n_samples/SR:.3f}s) ===")
    print(f"  windows produced : {int(info['num_windows'])} "
          f"(full={int(info['full_windows'])}, "
          f"trailing_kept={bool(info['trailing_kept'])})")
    print(f"  padded?          : {bool(info['padded'])}")
    if windows:
        print(f"  first boundary   : {windows[0]}")
        print(f"  last boundary    : {windows[-1]}  "
              f"(waveform ends at {n_samples})")
    # Sanity: every window is exactly W samples wide
    assert all((e - s) == W for s, e in windows), "window width mismatch!"
    print()

# ── Synthetic edge cases ─────────────────────────────────────────────────────
show("empty",                    0)
show("half a window",           W // 2)          # shorter than one window → 1 padded
show("exactly one window",      W)               # → 1 full, no padding
show("one window + tiny tail",  W + H // 10)     # tail too small → dropped
show("one window + big tail",   W + H // 2)      # tail big enough → kept + padded
show("exactly two hops",        W + H)           # → 2 full windows, clean
show("5 seconds",               5 * SR)          # typical NeuroVoz length
show("38 seconds",              38 * SR)         # typical IPVS length

# ── Real processed recording ─────────────────────────────────────────────────
df = pd.read_csv(C.PROCESSED_MANIFEST_CSV)
sample_path = df.iloc[0]['processed_file_path']
wav, sr = librosa.load(sample_path, sr=None, mono=True)
windows, info = compute_windows(wav.astype(np.float32), sr)
print(f"=== REAL: {Path(sample_path).name} ===")
print(f"  sample rate      : {sr} Hz (expected {SR})")
print(f"  duration         : {len(wav)/sr:.3f}s")
print(f"  windows produced : {int(info['num_windows'])}")
print(f"  padded?          : {bool(info['padded'])}")
print(f"  boundaries valid : "
      f"{all(0 <= s < e for s, e in windows)}")

# Verify overlap: consecutive starts differ by exactly hop_samples
starts = [s for s, _ in windows]
if len(starts) > 1:
    diffs = set(np.diff(starts))
    print(f"  start-to-start gaps: {diffs} (should be {{{H}}})")
print("\nAll assertions passed.")