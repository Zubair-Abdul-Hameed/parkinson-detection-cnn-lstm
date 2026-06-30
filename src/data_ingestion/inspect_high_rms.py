# src/data_ingestion/inspect_high_rms.py
"""
Standalone inspection script.

Reads raw_manifest.csv (never modifies it) and reports every recording
whose rms_energy is at or above a configurable threshold.

The rms_energy column was already computed by inspect_rms_energy.py and
stored in raw_manifest.csv — this script only reads it, no audio is opened.

Output
------
  report/high_rms_report.csv   (recording_id, file_path, dataset, label,
                                 duration, sample_rate, rms_energy)

Usage
-----
    python src/data_ingestion/inspect_high_rms.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C

logger = logging.getLogger(__name__)

# ── Threshold ─────────────────────────────────────────────────────────────
RMS_HIGH_THRESHOLD: float = 0.3


def inspect_high_rms(
    manifest_path : Path = C.RAW_MANIFEST_CSV,
    report_path   : Path = C.REPORTS_DIR / "high_rms_report.csv",
    threshold     : float = RMS_HIGH_THRESHOLD,
) -> pd.DataFrame:
    """
    Filter and report recordings with rms_energy >= threshold.

    Parameters
    ----------
    manifest_path : path to raw_manifest.csv  (read-only)
    report_path   : destination for high_rms_report.csv
    threshold     : rms_energy cutoff (inclusive)

    Returns
    -------
    DataFrame of flagged recordings.
    """
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return pd.DataFrame()

    df = pd.read_csv(manifest_path)

    if 'rms_energy' not in df.columns:
        logger.error(
            "'rms_energy' column not found in manifest. "
            "Run inspect_rms_energy.py first."
        )
        return pd.DataFrame()

    total   = len(df)
    flagged = df[df['rms_energy'] >= threshold].copy()
    flagged = flagged.sort_values('rms_energy', ascending=False).reset_index(drop=True)

    # ── Save report ───────────────────────────────────────────────────────
    report_cols = [
        'recording_id', 'file_path', 'dataset', 'label',
        'duration', 'sample_rate', 'rms_energy',
    ]
    # Keep only columns that actually exist (rms_energy may not be in
    # MANIFEST_COLUMNS, but it is written by inspect_rms_energy.py)
    report_cols = [c for c in report_cols if c in flagged.columns]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    flagged[report_cols].to_csv(report_path, index=False)
    logger.info("Saved → %s  (%d rows)", report_path, len(flagged))

    # ── Terminal output ───────────────────────────────────────────────────
    sep  = '=' * 72
    dash = '─' * 72

    print(f"\n{sep}")
    print(f"HIGH RMS ENERGY REPORT  (threshold ≥ {threshold})")
    print(sep)
    print(f"Total recordings in manifest : {total:,}")
    print(f"Recordings at/above threshold: {len(flagged):,}  "
          f"({100 * len(flagged) / max(total, 1):.1f}%)")

    if flagged.empty:
        print("\nNo recordings above threshold.")
        print(f"\n{sep}\n")
        return flagged

    # ── RMS distribution among flagged recordings ─────────────────────────
    rms = flagged['rms_energy']
    print(f"\nRMS energy among flagged recordings:")
    print(f"  min    : {rms.min():.6f}")
    print(f"  max    : {rms.max():.6f}")
    print(f"  mean   : {rms.mean():.6f}")
    print(f"  median : {rms.median():.6f}")

    # ── Per-dataset breakdown ─────────────────────────────────────────────
    if 'dataset' in flagged.columns:
        print(f"\nBreakdown by dataset:")
        for ds, grp in flagged.groupby('dataset'):
            print(f"  {ds:<20s}: {len(grp):,} recording(s)")

    # ── Per-label breakdown ───────────────────────────────────────────────
    if 'label' in flagged.columns:
        print(f"\nBreakdown by label:")
        for lbl, grp in flagged.groupby('label'):
            print(f"  {lbl:<10s}: {len(grp):,} recording(s)")

    # ── Individual recording details ──────────────────────────────────────
    print(f"\n{dash}")
    print(f"INDIVIDUAL RECORDINGS (sorted by rms_energy descending)")
    print(dash)

    for _, row in flagged.iterrows():
        print(
            f"\n  {row['recording_id']}"
            f"\n    rms_energy : {row['rms_energy']:.6f}"
            f"\n    duration   : {row['duration']:.3f}s"
            f"\n    dataset    : {row.get('dataset', 'N/A')}"
            f"\n    label      : {row.get('label', 'N/A')}"
            f"\n    path       : {row['file_path']}"
        )

    print(f"\n{sep}\n")
    return flagged


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    inspect_high_rms()