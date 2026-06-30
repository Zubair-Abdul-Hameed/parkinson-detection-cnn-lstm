# save as test_trim_silence.py in the project root, run from there
import sys
from pathlib import Path

sys.path.insert(0, "src")

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import config as C
from preprocessing.trim_silence import trim_silence

# ── Pick three recordings to inspect ─────────────────────────────────────────
# Two NeuroVoz with known leading/trailing silence, one IPVS. Swap freely.
TEST_FILES = [
    C.NEUROVOZ_AUDIO_DIR / "HC_A1_0034.wav",
    C.NEUROVOZ_AUDIO_DIR / "PD_PERRO_0058.wav",
    next(iter(C.IPVS_YHC_DIR.rglob("*.wav"))),   # first IPVS file found
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

    # Load once, at native sample rate, as float32 mono.
    waveform, sr = librosa.load(str(path), sr=None, mono=True)
    waveform = waveform.astype(np.float32)

    trimmed, sr_out, info = trim_silence(waveform, sr)

    print(f"\n=== {path.name} ===")
    print(f"  sample rate       : {sr} Hz  (returned: {sr_out} Hz)")
    print(f"  original duration : {info['original_duration']:.3f}s "
          f"({int(info['original_samples'])} samples)")
    print(f"  trimmed duration  : {info['trimmed_duration']:.3f}s "
          f"({int(info['trimmed_samples'])} samples)")
    print(f"  leading removed   : {info['leading_removed']/sr:.3f}s")
    print(f"  trailing removed  : {info['trailing_removed']/sr:.3f}s")
    print(f"  dtype             : {trimmed.dtype}")

    # Plot original vs trimmed on a shared time axis scale.
    t_orig = np.arange(len(waveform)) / sr
    t_trim = np.arange(len(trimmed)) / sr
    axes[i, 0].plot(t_orig, waveform, linewidth=0.5)
    axes[i, 0].set_title(f"{path.name} — original ({info['original_duration']:.2f}s)")
    axes[i, 0].set_xlabel("time (s)")
    axes[i, 1].plot(t_trim, trimmed, linewidth=0.5, color="tab:orange")
    axes[i, 1].set_title(f"{path.name} — trimmed ({info['trimmed_duration']:.2f}s)")
    axes[i, 1].set_xlabel("time (s)")

    # Save trimmed audio for listening (verification only — NOT the pipeline).
    sf.write(OUT_DIR / f"trimmed_{path.name}", trimmed, sr)

plt.tight_layout()
plt.savefig(OUT_DIR / "trim_comparison.png", dpi=120)
print(f"\nSaved plots + trimmed audio to: {OUT_DIR.resolve()}")
plt.show()