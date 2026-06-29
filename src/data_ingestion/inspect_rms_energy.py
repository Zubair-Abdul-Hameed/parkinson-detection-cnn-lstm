# src/data_ingestion/inspect_rms_energy.py
"""
One-time exploratory script: compute RMS energy for every recording in
the raw manifest, to help choose a near-silence threshold empirically
(rather than guessing a number).

This is NOT the corruption-detection pipeline. It does not classify
recordings as near-silent and does not apply any threshold. It only
measures and reports.

Adds an `rms_energy` column to data/raw_manifest.csv (overwrites the
file in place). Note: re-running build_raw_manifest.py will remove this
column again, since rms_energy is not part of MANIFEST_COLUMNS.

Usage
-----
    python src/data_ingestion/inspect_rms_energy.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
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


def _compute_rms(file_path: Path) -> float:
    """
    Compute RMS energy of an audio file.

    RMS = sqrt(mean(sample^2))

    Reads the full waveform (all files are confirmed mono, so this is
    a 1-D float array). Returns NaN if the file cannot be read.
    """
    try:
        data, _ = sf.read(str(file_path), dtype='float32')
        if data.size == 0:
            return float('nan')
        # Use float64 accumulation for numerical stability
        rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
        return rms
    except Exception as exc:
        logger.debug("Failed to read %s: %s", file_path, exc)
        return float('nan')


def inspect_rms_energy(manifest_path: Path = C.RAW_MANIFEST_CSV) -> pd.DataFrame:
    """
    Compute RMS energy for every recording in the manifest, save the
    enriched manifest, and print summary statistics + the 20 quietest
    recordings.

    Returns
    -------
    The manifest DataFrame with the new `rms_energy` column.
    """
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return pd.DataFrame()

    df = pd.read_csv(manifest_path)
    total = len(df)
    logger.info("Loaded manifest with %d recordings.", total)

    rms_values: list[float] = []
    failed: list[tuple[str, str]] = []

    for _, row in _progress(df.iterrows(), total=total, desc="Computing RMS energy"):
        file_path = Path(row['file_path'])
        rms = _compute_rms(file_path)
        if np.isnan(rms):
            failed.append((row['recording_id'], str(file_path)))
        rms_values.append(rms)

    df['rms_energy'] = rms_values

    # ── Save enriched manifest ──────────────────────────────────────────
    df.to_csv(manifest_path, index=False)
    logger.info(
        "Saved enriched manifest with rms_energy column → %s", manifest_path
    )

    # ── Summary statistics ───────────────────────────────────────────────
    valid = df['rms_energy'].dropna()
    sep = '=' * 64
    print(f"\n{sep}")
    print("RMS ENERGY SUMMARY")
    print(sep)
    print(f"Total recordings        : {total:,}")
    print(f"Valid RMS computations   : {len(valid):,}")
    print(f"Failed to read           : {len(failed):,}")

    if not valid.empty:
        print(f"\nmin              : {valid.min():.6f}")
        print(f"max              : {valid.max():.6f}")
        print(f"mean             : {valid.mean():.6f}")
        print(f"median           : {valid.median():.6f}")
        print(f"1st percentile   : {valid.quantile(0.01):.6f}")
        print(f"5th percentile   : {valid.quantile(0.05):.6f}")
        print(f"10th percentile  : {valid.quantile(0.10):.6f}")
    else:
        print("\nNo valid RMS values computed.")

    if failed:
        print(f"\n--- Files that failed to read ({len(failed)}) ---")
        for rid, fp in failed:
            print(f"  {rid}  |  {fp}")

    # ── 20 quietest recordings ───────────────────────────────────────────
    print(f"\n--- 20 quietest recordings (lowest RMS energy) ---")
    lowest = df.dropna(subset=['rms_energy']).nsmallest(20, 'rms_energy')
    cols = ['recording_id', 'file_path', 'duration', 'rms_energy']
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(lowest[cols].to_string(index=False))

    print(f"\n{sep}\n")

    return df


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    inspect_rms_energy()