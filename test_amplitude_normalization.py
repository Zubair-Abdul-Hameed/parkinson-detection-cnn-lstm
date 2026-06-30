# save as test_amplitude_normalization.py in the project root, run from there
import sys
from pathlib import Path

sys.path.insert(0, "src")

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import config as C
from preprocessing.amplitude_normalization import normalize_amplitude

# ── Three recordings: mix of quiet and loud, both datasets ───────────────────
TEST_FILES = [
    C.NEUROVOZ_AUDIO_DIR / "PD_A1_0078.wav",                 # one of the quietest (low peak)
    C.NEUROVOZ_AUDIO_DIR / "HC_A1_0034.wav",                 # 44100 Hz
    next(iter(C.IPVS_YHC_DIR.rglob("*.wav"))),               # 16000 Hz IPVS
]

OUT_DIR = Path("test_outputs")
OUT_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(len(TEST_FILES), 2, figsize=(14, 4 * len(TEST_FILES)))
if len(TEST_FILES) == 1:
    axes = axes.reshape(1, 2)

for i, path in enumerate(TEST_FILES):
    if not path.exists():
        print(f"  ✗ missing, skipping: {path}")
        continue

    # Load once at native sample rate as float32 mono.
    waveform, sr = librosa.load(str(path), sr=None, mono=True)
    waveform = waveform.astype(np.float32)
    duration = len(waveform) / sr

    normalized, info = normalize_amplitude(waveform)
    norm_duration = len(normalized) / sr   # unchanged, computed for display

    print(f"\n=== {path.name} ===")
    print(f"  original sr        : {sr} Hz")
    print(f"  normalized sr      : {sr} Hz  (unchanged — not touched by this stage)")
    print(f"  original duration  : {duration:.3f}s")
    print(f"  normalized duration: {norm_duration:.3f}s")
    print(f"  original peak      : {info['original_peak']:.6f}")
    print(f"  normalized peak    : {info['normalized_peak']:.6f}")
    print(f"  normalized?        : {'yes' if info['normalized'] else 'no (silent/empty)'}")
    print(f"  dtype              : {normalized.dtype}")

    t = np.arange(len(waveform)) / sr
    axes[i, 0].plot(t, waveform, linewidth=0.5)
    axes[i, 0].set_title(f"{path.name} — original (peak {info['original_peak']:.3f})")
    axes[i, 0].set_xlabel("time (s)")
    axes[i, 0].set_ylim(-1.05, 1.05)
    axes[i, 1].plot(t, normalized, linewidth=0.5, color="tab:purple")
    axes[i, 1].set_title(f"{path.name} — normalized (peak {info['normalized_peak']:.3f})")
    axes[i, 1].set_xlabel("time (s)")
    axes[i, 1].set_ylim(-1.05, 1.05)

    sf.write(OUT_DIR / f"normalized_{path.name}", normalized, sr)

plt.tight_layout()
plt.savefig(OUT_DIR / "normalize_comparison.png", dpi=120)
print(f"\nSaved plots + normalized audio to: {OUT_DIR.resolve()}")
plt.show()