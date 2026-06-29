# src/preprocessing/clean_audio.py
"""
Corruption detection stage of the Parkinson's FYP preprocessing pipeline.

Reads raw_manifest.csv (never modifies it) and classifies every recording
as corrupted or valid based on four technical criteria only:

  Criterion                    Corruption reason
  ───────────────────────────  ───────────────────────
  File missing on disk         file_not_found
  Header unreadable            unreadable_audio
  Duration ≤ 0 or NaN          invalid_duration
  Sample rate ≤ 0 or NaN       invalid_sample_rate

Short recordings are NOT corrupted.  They are flagged in the terminal
summary for windowing review only and do not appear in the report CSV.

Output artifact
---------------
  report/corruption_report.csv
    columns: recording_id, is_corrupted, corruption_reason

Usage
-----
    python src/preprocessing/clean_audio.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import soundfile as sf

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent   # …/parkinson-fyp/src/
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

# ── Short-recording thresholds (windowing review only) ────────────────────
_CAT1_MAX = 1.0   # duration < 1.0 s
_CAT2_MAX = 2.0   # 1.0 s ≤ duration < 2.0 s


# =============================================================================
# Core per-recording checks
# =============================================================================

def _check_corruption(
    file_path:          Path,
    manifest_duration:  float,
    manifest_sample_rate: float,
) -> Tuple[bool, Optional[str]]:
    """
    Apply the four corruption checks for a single recording.

    Checks are applied in the order mandated by the spec, stopping at the
    first failure so the reason reflects the most fundamental problem:

      1. File existence  (no file I/O — just a stat call)
      2. Readability     (sf.info() reads the header only; no sample data loaded)
      3. Duration        (uses manifest value — no file open)
      4. Sample rate     (uses manifest value — no file open)

    Returns
    -------
    (is_corrupted, reason)
      reason is None when is_corrupted is False.
    """
    # ── 1. Existence ──────────────────────────────────────────────────────
    if not file_path.exists():
        return True, 'file_not_found'

    # ── 2. Readability (header only) ──────────────────────────────────────
    try:
        sf.info(str(file_path))
    except Exception:
        return True, 'unreadable_audio'

    # ── 3. Duration (from manifest — no re-open needed) ───────────────────
    if pd.isna(manifest_duration) or manifest_duration <= 0.0:
        return True, 'invalid_duration'

    # ── 4. Sample rate (from manifest — no re-open needed) ────────────────
    if pd.isna(manifest_sample_rate) or manifest_sample_rate <= 0:
        return True, 'invalid_sample_rate'

    return False, None


def _short_category(duration: float) -> Optional[str]:
    """
    Return the windowing-review category for a short but valid recording.

    Returns None when the recording is long enough to require no special
    handling, 'category_1' when duration < 1.0 s, 'category_2' when
    1.0 s ≤ duration < 2.0 s.
    """
    if pd.isna(duration):
        return None
    if duration < _CAT1_MAX:
        return 'category_1'
    if duration < _CAT2_MAX:
        return 'category_2'
    return None


# =============================================================================
# Main runner
# =============================================================================

def run_corruption_detection(
    manifest_path: Path = C.RAW_MANIFEST_CSV,
    report_path:   Path = C.CORRUPTION_REPORT_CSV,
) -> pd.DataFrame:
    """
    Run corruption detection over every recording in the raw manifest.

    Parameters
    ----------
    manifest_path : path to raw_manifest.csv  (read-only)
    report_path   : destination for corruption_report.csv

    Returns
    -------
    The corruption report as a DataFrame.
    """
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return pd.DataFrame()

    df = pd.read_csv(manifest_path)
    total = len(df)
    logger.info("Loaded manifest — %d recordings.", total)

    report_rows: List[dict]                   = []
    short_cat1:  List[Tuple[str, float, str]] = []
    short_cat2:  List[Tuple[str, float, str]] = []

    for _, row in _progress(df.iterrows(), total=total, desc="Corruption check"):
        recording_id = row['recording_id']
        file_path    = Path(row['file_path'])
        duration     = row['duration']
        sample_rate  = row['sample_rate']

        is_corrupted, reason = _check_corruption(file_path, duration, sample_rate)

        report_rows.append({
            'recording_id':      recording_id,
            'is_corrupted':      is_corrupted,
            'corruption_reason': reason,
        })

        # Short-recording flags apply only to valid recordings
        if not is_corrupted:
            cat = _short_category(duration)
            if cat == 'category_1':
                short_cat1.append((recording_id, float(duration), str(file_path)))
            elif cat == 'category_2':
                short_cat2.append((recording_id, float(duration), str(file_path)))

    report_df = pd.DataFrame(
        report_rows,
        columns=['recording_id', 'is_corrupted', 'corruption_reason'],
    )

    # ── Save report ───────────────────────────────────────────────────────
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)
    logger.info("Saved → %s", report_path)

    _print_summary(report_df, short_cat1, short_cat2, total)

    return report_df


# =============================================================================
# Summary printer
# =============================================================================

def _print_summary(
    report_df:  pd.DataFrame,
    short_cat1: List[Tuple[str, float, str]],
    short_cat2: List[Tuple[str, float, str]],
    total:      int,
) -> None:
    corrupted   = report_df[report_df['is_corrupted']]
    n_corrupted = len(corrupted)

    sep  = '=' * 64
    dash = '─' * 64

    print(f"\n{sep}")
    print("CORRUPTION DETECTION SUMMARY")
    print(sep)
    print(f"Total recordings     : {total:,}")
    print(f"Corrupted recordings : {n_corrupted:,}")

    print("\nCorruption reasons:")
    for reason in ('file_not_found', 'unreadable_audio',
                   'invalid_duration', 'invalid_sample_rate'):
        count = (corrupted['corruption_reason'] == reason).sum()
        print(f"  {reason:<25s}: {count:,}")

    print(f"\n{dash}")
    print("SHORT RECORDINGS (FOR WINDOWING REVIEW)")
    print(dash)

    print(f"\n  Category 1 — duration < {_CAT1_MAX:.1f}s"
          f"  [{len(short_cat1):,} recording(s)]")
    if short_cat1:
        for rid, dur, fp in short_cat1:
            print(f"    {rid}  |  {dur:.3f}s  |  {fp}")

    print(f"\n  Category 2 — {_CAT1_MAX:.1f}s ≤ duration < {_CAT2_MAX:.1f}s"
          f"  [{len(short_cat2):,} recording(s)]")
    if short_cat2:
        for rid, dur, fp in short_cat2:
            print(f"    {rid}  |  {dur:.3f}s  |  {fp}")

    print(f"\n{sep}\n")


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    run_corruption_detection()