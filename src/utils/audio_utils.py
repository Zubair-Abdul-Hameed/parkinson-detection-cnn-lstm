# src/utils/audio_utils.py

"""
Lightweight audio file inspection utilities.

Intentionally avoids librosa and torchaudio — only soundfile and the
stdlib wave module are used.  These are fast enough for metadata
extraction over thousands of files without loading full sample arrays.
"""

import wave
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import soundfile once at module load.
# If it is not installed we fall back to the stdlib wave module.
try:
    import soundfile as sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:                          # pragma: no cover
    _SOUNDFILE_AVAILABLE = False
    logger.warning(
        "soundfile is not installed.  Falling back to stdlib wave module. "
        "Only uncompressed PCM .wav files will be readable."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_audio_info(file_path: Path) -> Tuple[Optional[float], Optional[int]]:
    """Return (duration_seconds, sample_rate_hz) for a WAV file.

    Tries soundfile first (handles more formats and edge-cases).
    Falls back to the stdlib wave module for plain PCM WAV files.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the audio file.

    Returns
    -------
    duration : float or None
        Length of the audio in seconds.  None if the file could not be read.
    sample_rate : int or None
        Sample rate in Hz (e.g. 16000, 44100).  None on failure.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning("File not found: %s", file_path)
        return None, None

    if not file_path.is_file():
        logger.warning("Path is not a file: %s", file_path)
        return None, None

    # --- Attempt 1: soundfile ---
    if _SOUNDFILE_AVAILABLE:
        result = _info_via_soundfile(file_path)
        if result != (None, None):
            return result
        logger.debug(
            "soundfile failed for %s, trying wave fallback.", file_path.name
        )

    # --- Attempt 2: stdlib wave (PCM WAV only) ---
    return _info_via_wave(file_path)


def is_valid_audio(file_path: Path) -> bool:
    """Return True if get_audio_info succeeds and duration > 0.

    Convenience wrapper used by corruption-detection steps.
    """
    duration, sample_rate = get_audio_info(file_path)
    return (
        duration is not None
        and sample_rate is not None
        and duration > 0.0
        and sample_rate > 0
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _info_via_soundfile(file_path: Path) -> Tuple[Optional[float], Optional[int]]:
    """Extract info using soundfile.info() — no audio data is loaded."""
    try:
        info = sf.info(str(file_path))
        duration    = info.frames / info.samplerate if info.samplerate > 0 else None
        sample_rate = int(info.samplerate)
        return duration, sample_rate
    except Exception as exc:          # SoundFileError, OSError, etc.
        logger.debug("soundfile.info failed for %s: %s", file_path.name, exc)
        return None, None


def _info_via_wave(file_path: Path) -> Tuple[Optional[float], Optional[int]]:
    """Extract info using the stdlib wave module (PCM WAV only)."""
    try:
        with wave.open(str(file_path), "rb") as wf:
            frames      = wf.getnframes()
            sample_rate = wf.getframerate()
            duration    = frames / sample_rate if sample_rate > 0 else None
            return duration, int(sample_rate)
    except wave.Error as exc:
        logger.debug("wave module failed for %s: %s", file_path.name, exc)
        return None, None
    except Exception as exc:
        logger.debug(
            "Unexpected error reading %s with wave: %s", file_path.name, exc
        )
        return None, None