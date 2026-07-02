# src/dataset_building/split_subjects.py
"""
Subject-level stratified train/val/test split for the Parkinson's FYP pipeline.

Reads data/processed_manifest.csv and writes train.csv, val.csv, test.csv
under data/splits/.

Guarantees
----------
1. Subject-level integrity: every recording from a given subject_id lands
   in exactly one split.  No subject is ever shared across splits.
2. Stratification by dataset × label: each stratum (e.g. ipvs_PD,
   neurovoz_HC) is split independently so the train/val/test proportions
   hold within every stratum, not just globally.

Gender is NOT used for stratification (cells are too small); instead a
per-split gender breakdown is printed for manual sanity-checking.

Usage
-----
    python src/dataset_building/split_subjects.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C

logger = logging.getLogger(__name__)


# =============================================================================
# Subject-table construction
# =============================================================================

def _build_subject_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the recording-level manifest to one row per subject.

    Each subject row carries the subject's dataset, label, and the stratum
    key (dataset_label).  Fails hard if any subject maps to more than one
    (dataset, label) pair, since subject-level splitting depends on that
    invariant.

    Returns
    -------
    pd.DataFrame indexed by position with columns:
        subject_id, dataset, label, stratum, n_recordings
    """
    # Verify each subject has exactly one (dataset, label) combination.
    combos = df.groupby('subject_id')[['dataset', 'label']].nunique()
    bad = combos[(combos['dataset'] > 1) | (combos['label'] > 1)]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} subject(s) map to multiple dataset/label values; "
            f"subject-level split cannot proceed. Offenders: {list(bad.index)}"
        )

    grouped = (
        df.groupby('subject_id')
          .agg(dataset=('dataset', 'first'),
               label=('label', 'first'),
               n_recordings=('recording_id', 'count'))
          .reset_index()
    )
    grouped['stratum'] = grouped['dataset'] + '_' + grouped['label']
    return grouped


# =============================================================================
# Core split logic
# =============================================================================

def _split_one_stratum(
    subject_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a single stratum's subject IDs into (train, val, test) lists.

    Shuffles deterministically using the supplied RNG, then partitions by
    rounded counts.  Rounding is handled so that small strata still place
    at least one subject in val and test when the stratum has ≥ 3 subjects.
    """
    ids = subject_ids.copy()
    rng.shuffle(ids)
    n = len(ids)

    if n == 0:
        return [], [], []
    if n == 1:
        # Only one subject — must go somewhere; training is the safest choice.
        return list(ids), [], []
    if n == 2:
        # One train, one val, none for test.
        return [ids[0]], [ids[1]], []

    n_train = int(round(train_ratio * n))
    n_val   = int(round(val_ratio * n))

    # Guardrails: ensure val and test each receive at least one subject.
    n_train = min(n_train, n - 2)          # leave ≥ 2 for val + test
    n_train = max(n_train, 1)
    n_val   = max(n_val, 1)
    n_val   = min(n_val, n - n_train - 1)  # leave ≥ 1 for test

    train = list(ids[:n_train])
    val   = list(ids[n_train:n_train + n_val])
    test  = list(ids[n_train + n_val:])
    return train, val, test


def split_subjects(
    manifest_path: Path = C.PROCESSED_MANIFEST_CSV,
    train_ratio: float = C.TRAIN_RATIO,
    val_ratio: float = C.VAL_RATIO,
    test_ratio: float = C.TEST_RATIO,
    seed: int = C.RANDOM_SEED,
) -> Dict[str, pd.DataFrame]:
    """
    Produce subject-level stratified train/val/test splits.

    Parameters
    ----------
    manifest_path : path to processed_manifest.csv (read-only)
    train_ratio, val_ratio, test_ratio : split proportions (should sum to 1)
    seed : RNG seed for reproducibility

    Returns
    -------
    Dict with keys 'train', 'val', 'test' mapping to recording-level
    DataFrames.  Also writes each to its configured CSV path.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio})."
        )

    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    logger.info("Loaded processed manifest — %d recordings.", len(df))

    subjects = _build_subject_table(df)
    logger.info("Collapsed to %d unique subjects across %d strata.",
                len(subjects), subjects['stratum'].nunique())

    rng = np.random.default_rng(seed)

    train_ids: List[str] = []
    val_ids:   List[str] = []
    test_ids:  List[str] = []

    # Split each stratum independently, in a deterministic stratum order.
    for stratum in sorted(subjects['stratum'].unique()):
        stratum_ids = subjects.loc[subjects['stratum'] == stratum, 'subject_id'].to_numpy()
        tr, va, te = _split_one_stratum(stratum_ids, train_ratio, val_ratio, rng)
        train_ids.extend(tr)
        val_ids.extend(va)
        test_ids.extend(te)
        logger.info("  [%s] %d subjects → train=%d val=%d test=%d",
                    stratum, len(stratum_ids), len(tr), len(va), len(te))

    # Map subject IDs back to full recording-level rows.
    train_df = df[df['subject_id'].isin(train_ids)].copy()
    val_df   = df[df['subject_id'].isin(val_ids)].copy()
    test_df  = df[df['subject_id'].isin(test_ids)].copy()

    # ── Hard integrity check: no subject shared across splits ─────────────
    _assert_no_leakage(train_df, val_df, test_df)

    # ── Save ──────────────────────────────────────────────────────────────
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(C.TRAIN_CSV, index=False)
    val_df.to_csv(C.VAL_CSV, index=False)
    test_df.to_csv(C.TEST_CSV, index=False)
    logger.info("Saved splits → %s", C.SPLITS_DIR)

    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    _print_summary(splits, subjects)
    return splits


# =============================================================================
# Validation & summary
# =============================================================================

def _assert_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Raise if any subject_id appears in more than one split."""
    s_train = set(train_df['subject_id'])
    s_val   = set(val_df['subject_id'])
    s_test  = set(test_df['subject_id'])

    overlaps = {
        'train∩val':  s_train & s_val,
        'train∩test': s_train & s_test,
        'val∩test':   s_val & s_test,
    }
    for name, shared in overlaps.items():
        if shared:
            raise AssertionError(
                f"Subject leakage detected in {name}: {sorted(shared)}"
            )
    logger.info("✓ Leakage check passed — no subject shared across splits.")


def _print_summary(
    splits: Dict[str, pd.DataFrame],
    subjects: pd.DataFrame,
) -> None:
    """Print subject/recording counts per split and per stratum + gender check."""
    sep  = '=' * 68
    dash = '─' * 68

    print(f"\n{sep}")
    print("SUBJECT-LEVEL STRATIFIED SPLIT SUMMARY")
    print(sep)

    total_subjects = sum(s['subject_id'].nunique() for s in splits.values())
    total_records  = sum(len(s) for s in splits.values())

    print(f"\nOverall: {total_subjects} subjects, {total_records} recordings\n")
    print(f"{'Split':<8}{'Subjects':>10}{'Recordings':>14}"
          f"{'Subj %':>9}{'Rec %':>9}")
    print(dash)
    for name in ['train', 'val', 'test']:
        s = splits[name]
        n_subj = s['subject_id'].nunique()
        n_rec  = len(s)
        print(f"{name:<8}{n_subj:>10}{n_rec:>14}"
              f"{100*n_subj/total_subjects:>8.1f}%{100*n_rec/total_records:>8.1f}%")

    # ── Per-stratum subject counts ────────────────────────────────────────
    print(f"\n{dash}")
    print("SUBJECTS PER STRATUM (dataset × label)")
    print(dash)
    print(f"{'Stratum':<18}{'Train':>8}{'Val':>8}{'Test':>8}{'Total':>8}")
    print(dash)
    for stratum in sorted(subjects['stratum'].unique()):
        counts = {}
        for name in ['train', 'val', 'test']:
            s = splits[name]
            subj_in_stratum = s[s['dataset'] + '_' + s['label'] == stratum]
            counts[name] = subj_in_stratum['subject_id'].nunique()
        total = counts['train'] + counts['val'] + counts['test']
        print(f"{stratum:<18}{counts['train']:>8}{counts['val']:>8}"
              f"{counts['test']:>8}{total:>8}")

    # ── Gender distribution per split (report only) ───────────────────────
    print(f"\n{dash}")
    print("GENDER DISTRIBUTION PER SPLIT  (subjects — sanity-check only)")
    print(dash)
    print(f"{'Split':<8}{'M':>6}{'F':>6}{'Unknown':>10}")
    print(dash)
    for name in ['train', 'val', 'test']:
        s = splits[name]
        # One row per subject to avoid counting a subject once per recording.
        subj_gender = s.drop_duplicates('subject_id')['gender']
        n_m = (subj_gender == 'M').sum()
        n_f = (subj_gender == 'F').sum()
        n_u = subj_gender.isna().sum()
        print(f"{name:<8}{n_m:>6}{n_f:>6}{n_u:>10}")

    print(f"\n{sep}\n")


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    split_subjects()