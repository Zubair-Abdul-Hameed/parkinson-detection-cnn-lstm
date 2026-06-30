# src/preprocessing/assess_audio_quality.py
"""
Audio Quality Assessment — frame-wise RMS energy analysis.

Detects recordings containing abnormally high-energy distortion by
analysing per-frame RMS energy after temporarily removing leading and
trailing silence (in memory only — no files are modified).

Detection strategy
------------------
A frame is flagged as abnormal when BOTH of the following are true:
  1. frame RMS  >  RELATIVE_MULTIPLIER × recording's own median frame RMS
  2. frame RMS  >  ABSOLUTE_RMS_FLOOR

Using AND rather than OR avoids flagging:
  • quiet recordings where one frame is slightly louder than the rest
  • uniformly loud but clean recordings

Consecutive abnormal frames are merged into regions; regions are then
used to classify the recording and recommend a preprocessing action.

This stage is analysis and reporting only.
No audio files are modified.
raw_manifest.csv is never modified.

Output
------
  report/quality_assessment_report.csv

Usage
-----
    python src/preprocessing/assess_audio_quality.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C

logger = logging.getLogger(__name__)

# ── Optional tqdm ─────────────────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)

except ImportError:
    def _progress(iterable, **kw):   # type: ignore[misc]
        return iterable


# =============================================================================
# Configuration  — all thresholds in one place, no magic numbers elsewhere
# =============================================================================

# Frame analysis
FRAME_DURATION_S : float = 0.050   # 50 ms per frame
HOP_DURATION_S   : float = 0.025   # 25 ms hop (50 % overlap)

# In-memory silence removal
# Frames whose RMS falls below this are treated as silence at the edges.
SILENCE_RMS_THRESHOLD : float = 0.01

# Abnormal-frame detection  (BOTH conditions must hold)
RELATIVE_MULTIPLIER  : float = 4.0    # frame RMS must exceed N × recording median
ABSOLUTE_RMS_FLOOR   : float = 0.10   # and must also exceed this absolute value

# Region merging
MIN_ABNORMAL_FRAMES : int = 3   # discard runs shorter than this (transient noise)
MERGE_GAP_FRAMES    : int = 4   # bridge gaps of ≤ this many clean frames

# Classification geometry
EDGE_FRACTION             : float = 0.15   # first/last 15 % of duration = "edge"
FULLY_DISTORTED_THRESHOLD : float = 0.60   # ≥ 60 % abnormal → FULLY_DISTORTED

# Recordings shorter than this are skipped (matches corruption-stage decision)
MIN_ANALYSIS_DURATION_S : float = 0.50


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class AbnormalRegion:
    start_sec    : float
    end_sec      : float
    duration_sec : float
    location     : str   # 'start' | 'end' | 'middle'


@dataclass
class QualityResult:
    recording_id           : str
    file_path              : str
    filename               : str
    dataset                : str
    duration_seconds       : float
    rms_mean               : float
    rms_median             : float
    rms_max                : float
    rms_std                : float
    abnormal_region_count  : int
    abnormal_region_locations : str   # JSON
    abnormal_duration_seconds : float
    abnormal_percentage    : float
    classification         : str
    recommended_action     : str


# =============================================================================
# Audio loading and in-memory silence trimming
# =============================================================================

def _load_audio(file_path: Path) -> Optional[Tuple[np.ndarray, int]]:
    """
    Load a mono audio file as a float32 numpy array.

    Returns (audio, sample_rate) or None if the file cannot be read.
    """
    try:
        audio, sr = sf.read(str(file_path), dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", file_path.name, exc)
        return None


def _trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Remove leading and trailing silence in memory.  Returns the trimmed array.

    A region is silence when its frame-level RMS is below SILENCE_RMS_THRESHOLD.
    The original array is never modified; a slice is returned.

    If trimming would reduce the audio to nothing, the original is returned.
    """
    frame_size = max(1, int(FRAME_DURATION_S * sr))
    hop_size   = max(1, int(HOP_DURATION_S   * sr))

    n        = len(audio)
    n_frames = max(0, 1 + (n - frame_size) // hop_size)
    if n_frames == 0:
        return audio

    rms = np.array([
        np.sqrt(np.mean(audio[i * hop_size : i * hop_size + frame_size] ** 2))
        for i in range(n_frames)
    ], dtype=np.float32)

    above = np.where(rms >= SILENCE_RMS_THRESHOLD)[0]
    if above.size == 0:
        return audio   # entirely silent — return as-is, will be classified later

    first_frame = int(above[0])
    last_frame  = int(above[-1])

    start_sample = first_frame * hop_size
    end_sample   = min(n, last_frame * hop_size + frame_size)

    trimmed = audio[start_sample:end_sample]
    return trimmed if len(trimmed) > 0 else audio


# =============================================================================
# Frame-level RMS
# =============================================================================

def _compute_frame_rms(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame RMS energy and the centre time of each frame.

    Returns
    -------
    rms_values   : float32 array, shape (n_frames,)
    frame_times  : float32 array, centre time in seconds, shape (n_frames,)
    """
    frame_size = max(1, int(FRAME_DURATION_S * sr))
    hop_size   = max(1, int(HOP_DURATION_S   * sr))

    n        = len(audio)
    n_frames = max(0, 1 + (n - frame_size) // hop_size)

    rms_values  = np.zeros(n_frames, dtype=np.float32)
    frame_times = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start : start + frame_size]
        rms_values[i]  = float(np.sqrt(np.mean(frame ** 2)))
        frame_times[i] = float(start + frame_size / 2) / sr

    return rms_values, frame_times


# =============================================================================
# Abnormal-frame detection and region extraction
# =============================================================================

def _detect_abnormal_frames(rms_values: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask — True where a frame is abnormal.

    A frame is abnormal when BOTH conditions hold:
      1. rms > RELATIVE_MULTIPLIER × recording median
      2. rms > ABSOLUTE_RMS_FLOOR

    Using AND keeps false-positive rate low.  A single loud unvoiced
    phoneme will not trigger the absolute floor; an entire quiet recording
    will not trigger the relative threshold just because one frame is louder.
    """
    median = float(np.median(rms_values))
    relative_threshold = RELATIVE_MULTIPLIER * median

    relative_flag = rms_values > relative_threshold
    absolute_flag = rms_values > ABSOLUTE_RMS_FLOOR

    return relative_flag & absolute_flag


def _extract_regions(
    abnormal_mask : np.ndarray,
    frame_times   : np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Convert a per-frame boolean mask to merged (start_sec, end_sec) pairs.

    Steps:
      1. Collect contiguous True runs.
      2. Drop runs shorter than MIN_ABNORMAL_FRAMES.
      3. Bridge gaps of ≤ MERGE_GAP_FRAMES clean frames.
      4. Convert frame indices to seconds.
    """
    n = len(abnormal_mask)
    if n == 0:
        return []

    # Step 1 — contiguous runs
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if abnormal_mask[i]:
            j = i + 1
            while j < n and abnormal_mask[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1

    # Step 2 — drop short runs
    runs = [(s, e) for s, e in runs if (e - s + 1) >= MIN_ABNORMAL_FRAMES]
    if not runs:
        return []

    # Step 3 — merge close runs
    merged: List[Tuple[int, int]] = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        if (s - prev_e - 1) <= MERGE_GAP_FRAMES:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    # Step 4 — frame indices → seconds
    half_frame = FRAME_DURATION_S / 2.0
    result: List[Tuple[float, float]] = []
    for s_idx, e_idx in merged:
        start_s = max(0.0, float(frame_times[s_idx]) - half_frame)
        end_s   = float(frame_times[e_idx]) + half_frame
        result.append((start_s, end_s))

    return result


def _region_location(start_s: float, end_s: float, total_s: float) -> str:
    """Return 'start', 'end', or 'middle' for a region."""
    edge_s = EDGE_FRACTION * total_s
    edge_e = (1.0 - EDGE_FRACTION) * total_s
    at_start = start_s < edge_s
    at_end   = end_s   > edge_e
    if at_start:
        return 'start'
    if at_end:
        return 'end'
    return 'middle'


# =============================================================================
# Classification
# =============================================================================

def _classify(
    regions       : List[AbnormalRegion],
    abnormal_pct  : float,
) -> Tuple[str, str]:
    """
    Return (classification, recommended_action).

    Priority:
      FULLY_DISTORTED → MANUAL_REVIEW  (too damaged to trim cleanly)
      single region   → edge trim recommendations where applicable
      multiple regions → trim if all on one edge; else MANUAL_REVIEW
    """
    if not regions:
        return 'CLEAN', 'KEEP'

    if abnormal_pct >= FULLY_DISTORTED_THRESHOLD * 100:
        return 'FULLY_DISTORTED', 'MANUAL_REVIEW'

    locations  = [r.location for r in regions]
    has_start  = any(loc == 'start'  for loc in locations)
    has_end    = any(loc == 'end'    for loc in locations)
    has_middle = any(loc == 'middle' for loc in locations)

    if len(regions) == 1:
        loc = locations[0]
        if loc == 'start':  return 'EDGE_DISTORTION_START', 'TRIM_START'
        if loc == 'end':    return 'EDGE_DISTORTION_END',   'TRIM_END'
        if loc == 'middle': return 'MID_RECORDING_DISTORTION', 'MANUAL_REVIEW'

    # Multiple regions
    if has_middle:
        return 'MULTIPLE_DISTORTED_REGIONS', 'MANUAL_REVIEW'
    if has_start and has_end:
        return 'EDGE_DISTORTION_BOTH', 'TRIM_BOTH'
    if has_start:
        return 'MULTIPLE_DISTORTED_REGIONS', 'TRIM_START'
    if has_end:
        return 'MULTIPLE_DISTORTED_REGIONS', 'TRIM_END'

    return 'REVIEW_REQUIRED', 'MANUAL_REVIEW'


# =============================================================================
# Per-recording analysis
# =============================================================================

def _analyse_recording(row: pd.Series) -> Optional[QualityResult]:
    """
    Run full RMS quality assessment for one manifest row.

    Returns None if the recording is too short or unloadable.
    """
    recording_id = row['recording_id']
    file_path    = Path(row['file_path'])
    duration     = float(row['duration'])

    if duration < MIN_ANALYSIS_DURATION_S:
        logger.debug("Skipping %s — too short (%.3fs)", recording_id, duration)
        return None

    loaded = _load_audio(file_path)
    if loaded is None:
        return None
    audio, sr = loaded

    # ── In-memory silence removal (does not modify any file) ───────────────
    audio_trimmed = _trim_silence(audio, sr)

    # ── Frame-level RMS on silence-trimmed audio ───────────────────────────
    rms_values, frame_times = _compute_frame_rms(audio_trimmed, sr)
    if rms_values.size == 0:
        return None

    # ── Detect and merge abnormal regions ─────────────────────────────────
    abnormal_mask = _detect_abnormal_frames(rms_values)
    raw_regions   = _extract_regions(abnormal_mask, frame_times)

    regions: List[AbnormalRegion] = []
    total_abnormal_s = 0.0
    for start_s, end_s in raw_regions:
        dur_s    = end_s - start_s
        location = _region_location(start_s, end_s, float(len(audio_trimmed) / sr))
        regions.append(AbnormalRegion(
            start_sec    = round(start_s, 3),
            end_sec      = round(end_s,   3),
            duration_sec = round(dur_s,   3),
            location     = location,
        ))
        total_abnormal_s += dur_s

    trimmed_duration = len(audio_trimmed) / sr
    abnormal_pct     = 100.0 * total_abnormal_s / max(trimmed_duration, 1e-6)
    largest_region_s = max((r.duration_sec for r in regions), default=0.0)

    classification, recommended_action = _classify(regions, abnormal_pct)

    return QualityResult(
        recording_id              = recording_id,
        file_path                 = str(file_path),
        filename                  = file_path.name,
        dataset                   = str(row.get('dataset', '')),
        duration_seconds          = round(duration, 3),
        rms_mean                  = round(float(rms_values.mean()),   6),
        rms_median                = round(float(np.median(rms_values)), 6),
        rms_max                   = round(float(rms_values.max()),    6),
        rms_std                   = round(float(rms_values.std()),    6),
        abnormal_region_count     = len(regions),
        abnormal_region_locations = json.dumps([asdict(r) for r in regions]),
        abnormal_duration_seconds = round(total_abnormal_s, 3),
        abnormal_percentage       = round(abnormal_pct, 2),
        classification            = classification,
        recommended_action        = recommended_action,
    )


# =============================================================================
# Main runner
# =============================================================================

def run_quality_assessment(
    manifest_path : Path = C.RAW_MANIFEST_CSV,
    report_path   : Path = C.QUALITY_REPORT_CSV,
) -> pd.DataFrame:
    """
    Run RMS-based quality assessment over all recordings in the manifest.

    Parameters
    ----------
    manifest_path : read-only path to raw_manifest.csv
    report_path   : destination for quality_assessment_report.csv

    Returns
    -------
    Quality report as a DataFrame.
    """
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return pd.DataFrame()

    df    = pd.read_csv(manifest_path)
    total = len(df)
    logger.info("Loaded manifest — %d recordings.", total)

    results : List[QualityResult] = []
    skipped : int = 0

    for _, row in _progress(df.iterrows(), total=total, desc="Quality assessment"):
        result = _analyse_recording(row)
        if result is None:
            skipped += 1
        else:
            results.append(result)

    if not results:
        logger.warning("No results — check that audio files are accessible.")
        return pd.DataFrame()

    report_df = pd.DataFrame([asdict(r) for r in results])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)
    logger.info("Saved → %s  (%d rows)", report_path, len(report_df))

    _print_summary(report_df, total, skipped)
    return report_df


# =============================================================================
# Summary printer
# =============================================================================

def _print_summary(
    report_df : pd.DataFrame,
    total     : int,
    skipped   : int,
) -> None:
    sep  = '=' * 72
    dash = '─' * 72

    print(f"\n{sep}")
    print("AUDIO QUALITY ASSESSMENT SUMMARY")
    print(sep)
    print(f"Total recordings in manifest : {total:,}")
    print(f"Skipped (too short < {MIN_ANALYSIS_DURATION_S}s)   : {skipped:,}")
    print(f"Analysed                     : {len(report_df):,}")

    print("\nClassification breakdown:")
    counts = report_df['classification'].value_counts()
    for label in [
        'CLEAN',
        'EDGE_DISTORTION_START',
        'EDGE_DISTORTION_END',
        'EDGE_DISTORTION_BOTH',
        'MID_RECORDING_DISTORTION',
        'MULTIPLE_DISTORTED_REGIONS',
        'FULLY_DISTORTED',
        'REVIEW_REQUIRED',
    ]:
        print(f"  {label:<34s}: {counts.get(label, 0):,}")

    print("\nRecommended actions:")
    action_counts = report_df['recommended_action'].value_counts()
    for action in ['KEEP', 'TRIM_START', 'TRIM_END', 'TRIM_BOTH', 'MANUAL_REVIEW']:
        print(f"  {action:<18s}: {action_counts.get(action, 0):,}")

    flagged = report_df[report_df['recommended_action'] != 'KEEP']

    if flagged.empty:
        print(f"\n✓  No recordings flagged — all assessed recordings are CLEAN.")
    else:
        print(f"\n{dash}")
        print(f"FLAGGED RECORDINGS ({len(flagged):,})  "
              f"— sorted by abnormal percentage descending")
        print(dash)
        for _, row in flagged.sort_values(
            'abnormal_percentage', ascending=False
        ).iterrows():
            print(f"\n  {row['filename']}")
            print(f"    path       : {row['file_path']}")
            print(f"    dataset    : {row['dataset']}")
            print(f"    duration   : {row['duration_seconds']:.2f}s")
            print(f"    class      : {row['classification']}")
            print(f"    action     : {row['recommended_action']}")
            print(f"    abnormal   : {row['abnormal_percentage']:.1f}%  "
                  f"({row['abnormal_duration_seconds']:.2f}s)")
            print(f"    rms        : mean={row['rms_mean']:.4f}  "
                  f"max={row['rms_max']:.4f}  "
                  f"std={row['rms_std']:.4f}")
            regions = json.loads(row['abnormal_region_locations'])
            for r in regions:
                print(
                    f"    region     : {r['start_sec']:.2f}s – "
                    f"{r['end_sec']:.2f}s  "
                    f"({r['duration_sec']:.2f}s, {r['location']})"
                )

    print(f"\n{sep}\n")


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    run_quality_assessment()