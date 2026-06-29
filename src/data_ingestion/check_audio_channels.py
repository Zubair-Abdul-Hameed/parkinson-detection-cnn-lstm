# src/data_ingestion/check_audio_channels.py
"""
One-time exploratory script: inspect audio channel counts (mono vs stereo)
across the entire raw manifest.

This is NOT part of the corruption-detection or preprocessing pipeline.
It exists purely to inform preprocessing decisions (e.g. whether a
stereo-to-mono downmix step is needed in clean_audio.py).

Reads data/raw_manifest.csv only. Does not modify the manifest, any
audio files, or any existing pipeline code.

Usage
-----
    python src/data_ingestion/check_audio_channels.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent   # …/parkinson-fyp/src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C

logger = logging.getLogger(__name__)

# ── Optional tqdm for progress bars ──────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)

except ImportError:
    def _progress(iterable, **kw):   # type: ignore[misc]
        return iterable


def check_audio_channels(manifest_path: Path = C.RAW_MANIFEST_CSV) -> None:
    """
    Inspect every audio file referenced in the raw manifest and report
    channel-count statistics (mono / stereo / >2 channels / unreadable).

    Uses soundfile.info() so only the file header is read — no audio
    samples are decoded.
    """
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return

    df = pd.read_csv(manifest_path)
    total = len(df)
    logger.info("Loaded manifest with %d recordings.", total)

    mono_count       = 0
    stereo_count     = 0
    multi_count      = 0
    unreadable_count = 0

    stereo_records:     list[tuple[str, str]]      = []
    multi_records:      list[tuple[str, str, int]] = []
    unreadable_records: list[tuple[str, str, str]] = []

    for _, row in _progress(df.iterrows(), total=total, desc="Inspecting channels"):
        recording_id = row['recording_id']
        file_path    = Path(row['file_path'])

        try:
            info     = sf.info(str(file_path))
            channels = info.channels
        except Exception as exc:
            unreadable_count += 1
            unreadable_records.append((recording_id, str(file_path), str(exc)))
            continue

        if channels == 1:
            mono_count += 1
        elif channels == 2:
            stereo_count += 1
            stereo_records.append((recording_id, str(file_path)))
        else:
            multi_count += 1
            multi_records.append((recording_id, str(file_path), channels))

    # ── Summary ─────────────────────────────────────────────────────────
    sep = '=' * 64
    print(f"\n{sep}")
    print("AUDIO CHANNEL INSPECTION SUMMARY")
    print(sep)
    print(f"Total recordings checked : {total:,}")
    print(f"Mono (1 channel)         : {mono_count:,}")
    print(f"Stereo (2 channels)      : {stereo_count:,}")
    print(f"More than 2 channels     : {multi_count:,}")
    print(f"Could not be inspected   : {unreadable_count:,}")

    if stereo_records:
        print(f"\n--- Stereo recordings ({len(stereo_records)}) ---")
        for rid, fp in stereo_records:
            print(f"  {rid}  |  {fp}")

    if multi_records:
        print(f"\n--- Recordings with >2 channels ({len(multi_records)}) ---")
        for rid, fp, ch in multi_records:
            print(f"  {rid}  |  {fp}  |  channels={ch}")

    if unreadable_records:
        print(f"\n--- Unreadable recordings ({len(unreadable_records)}) ---")
        for rid, fp, err in unreadable_records:
            print(f"  {rid}  |  {fp}  |  error={err}")

    print(f"\n{sep}\n")


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    check_audio_channels()