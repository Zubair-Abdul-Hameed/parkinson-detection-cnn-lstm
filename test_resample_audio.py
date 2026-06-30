# save as test_resample_audio.py in the project root, run from there
import sys
from pathlib import Path

sys.path.insert(0, "src")

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import config as C
from preprocessing.resample_audio import resample_audio

# ── Three recordings: a 44.1 kHz, a 16 kHz, and one more ─────────────────────
TEST_FILES = [
    C.NEUROVOZ_AUDIO_DIR / "HC_A1_0034.wav",                 # 44100 Hz
    next(iter(C.IPVS_YHC_DIR.rglob("*.wav"))),               # 16000 Hz (already target)
    C.NEUROVOZ_AUDIO_DIR / "PD_PERRO_0058.wav",              # 44100 Hz, your choice
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

    resampled, sr_out, info = resample_audio(waveform, sr)

    print(f"\n=== {path.name} ===")
    print(f"  original sr        : {int(info['original_sr'])} Hz")
    print(f"  new sr             : {sr_out} Hz")
    print(f"  resampled?         : {'yes' if info['resampled'] else 'no (already target)'}")
    print(f"  original duration  : {info['original_duration']:.3f}s")
    print(f"  new duration       : {info['resampled_duration']:.3f}s")
    print(f"  original samples   : {int(info['original_samples'])}")
    print(f"  new samples        : {int(info['resampled_samples'])}")
    print(f"  dtype              : {resampled.dtype}")

    t_orig = np.arange(len(waveform)) / sr
    t_new  = np.arange(len(resampled)) / sr_out
    axes[i, 0].plot(t_orig, waveform, linewidth=0.5)
    axes[i, 0].set_title(f"{path.name} — original ({int(sr)} Hz)")
    axes[i, 0].set_xlabel("time (s)")
    axes[i, 1].plot(t_new, resampled, linewidth=0.5, color="tab:green")
    axes[i, 1].set_title(f"{path.name} — resampled ({sr_out} Hz)")
    axes[i, 1].set_xlabel("time (s)")

    sf.write(OUT_DIR / f"resampled_{path.name}", resampled, sr_out)

plt.tight_layout()
plt.savefig(OUT_DIR / "resample_comparison.png", dpi=120)
print(f"\nSaved plots + resampled audio to: {OUT_DIR.resolve()}")
plt.show()