# save as test_pipeline_integration.py in the project root, run from there
"""
Pipeline integration test for the Parkinson's FYP preprocessing chain.

Applies the full preprocessing pipeline to six recordings:

    load waveform → trim_silence → resample → normalize → return

Each recording is loaded exactly once.  Processed audio and before/after
plots are written to test_outputs/ for verification only.  Nothing is
written to processed_audio/ and no existing module is modified.

Run from the project root:

    python test_pipeline_integration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, "src")

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import config as C
from preprocessing.trim_silence import trim_silence
from preprocessing.resample_audio import resample_audio
from preprocessing.amplitude_normalization import normalize_amplitude


# ── Six recordings spanning the required variety ─────────────────────────────
# NeuroVoz (44100 Hz, es) — both PD and HC, including a silence-trim case
# IPVS     (16000 Hz, it) — both PD and HC, including a no-trim case
def _first_ipvs(*parts: str) -> Path:
    """Return the first .wav under an IPVS subgroup path (None if none found)."""
    base = C.IPVS_DIR.joinpath(*parts)
    wavs = sorted(base.rglob("*.wav")) if base.exists() else []
    return wavs[0] if wavs else None


TEST_FILES = [
    C.NEUROVOZ_AUDIO_DIR / "HC_A1_0034.wav",      # NeuroVoz HC, 44100, no trim expected
    C.NEUROVOZ_AUDIO_DIR / "PD_PERRO_0058.wav",   # NeuroVoz PD, 44100, trims samples
    C.NEUROVOZ_AUDIO_DIR / "PD_A1_0078.wav",      # NeuroVoz PD, 44100, very quiet (normalize)
    _first_ipvs("15 Young Healthy Control"),      # IPVS HC, 16000
    _first_ipvs("28 People with Parkinson's disease", "1-5"),  # IPVS PD, 16000
    _first_ipvs("22 Elderly Healthy Control"),    # IPVS HC, 16000
]

OUT_DIR = Path("test_outputs")
OUT_DIR.mkdir(exist_ok=True)


def _run_pipeline(waveform: np.ndarray, sr: int):
    """
    Apply trim → resample → normalize to one waveform.

    Returns the processed waveform, its sample rate, and a merged dict of
    per-stage diagnostics.
    """
    trimmed, sr_t, trim_info       = trim_silence(waveform, sr)
    resampled, sr_r, resample_info = resample_audio(trimmed, sr_t)
    normalized, norm_info          = normalize_amplitude(resampled)

    return normalized, sr_r, {
        'trim':      trim_info,
        'resample':  resample_info,
        'normalize': norm_info,
    }


valid_files = [p for p in TEST_FILES if p is not None and p.exists()]

fig, axes = plt.subplots(len(valid_files), 2, figsize=(14, 4 * len(valid_files)))
if len(valid_files) == 1:
    axes = axes.reshape(1, 2)

for i, path in enumerate(valid_files):
    # ── Load once, native sample rate, float32 mono ──────────────────────────
    waveform, sr = librosa.load(str(path), sr=None, mono=True)
    waveform = waveform.astype(np.float32)

    original_peak     = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    original_duration = len(waveform) / sr

    processed, proc_sr, info = _run_pipeline(waveform, sr)

    processed_peak     = float(np.max(np.abs(processed))) if processed.size else 0.0
    processed_duration = len(processed) / proc_sr

    trimmed_samples = int(info['trim']['samples_removed'])
    did_trim        = trimmed_samples > 0
    did_resample    = bool(info['resample']['resampled'])
    did_normalize   = bool(info['normalize']['normalized'])

    print(f"\n=== {path.name} ===")
    print(f"  original sr        : {sr} Hz")
    print(f"  processed sr       : {proc_sr} Hz")
    print(f"  original duration  : {original_duration:.3f}s")
    print(f"  processed duration : {processed_duration:.3f}s")
    print(f"  original samples   : {len(waveform)}")
    print(f"  processed samples  : {len(processed)}")
    print(f"  original peak      : {original_peak:.6f}")
    print(f"  processed peak     : {processed_peak:.6f}")
    print(f"  dtype              : {processed.dtype}")
    print(f"  silence trimmed?   : {'yes' if did_trim else 'no'}"
          f"  ({trimmed_samples} samples removed)")
    print(f"  resampled?         : {'yes' if did_resample else 'no (already 16 kHz)'}")
    print(f"  normalized?        : {'yes' if did_normalize else 'no (silent/empty)'}")

    # ── Plot original (native sr) vs processed (16 kHz) ──────────────────────
    t_orig = np.arange(len(waveform)) / sr
    t_proc = np.arange(len(processed)) / proc_sr
    axes[i, 0].plot(t_orig, waveform, linewidth=0.5)
    axes[i, 0].set_title(f"{path.name} — original ({sr} Hz, {original_duration:.2f}s)")
    axes[i, 0].set_xlabel("time (s)")
    axes[i, 0].set_ylim(-1.05, 1.05)
    axes[i, 1].plot(t_proc, processed, linewidth=0.5, color="tab:red")
    axes[i, 1].set_title(f"{path.name} — processed ({proc_sr} Hz, {processed_duration:.2f}s)")
    axes[i, 1].set_xlabel("time (s)")
    axes[i, 1].set_ylim(-1.05, 1.05)

    # ── Save processed audio for listening (verification only) ───────────────
    sf.write(OUT_DIR / f"pipeline_{path.name}", processed, proc_sr)

plt.tight_layout()
plt.savefig(OUT_DIR / "pipeline_comparison.png", dpi=120)
print(f"\nProcessed {len(valid_files)} recording(s).")
print(f"Saved plots + processed audio to: {OUT_DIR.resolve()}")
plt.show()