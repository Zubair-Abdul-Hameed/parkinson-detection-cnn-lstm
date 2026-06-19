# src/data_ingestion/build_raw_manifest.py
"""
Builds raw_manifest.csv by parsing NeuroVoz and IPVS datasets.

Public API
----------
parse_neurovoz()      -> pd.DataFrame
parse_ipvs()          -> pd.DataFrame
build_raw_manifest()  -> pd.DataFrame  (concatenates both and saves CSV)

Usage
-----
    python src/data_ingestion/build_raw_manifest.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Ensures `import config` and `from utils.X import Y` resolve when this file
# is run directly as a script.
_SRC_DIR = Path(__file__).resolve().parent.parent   # …/parkinson-fyp/src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C
from utils.audio_utils import get_audio_info

logger = logging.getLogger(__name__)

# ── Optional tqdm for progress bars ──────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)

except ImportError:
    def _progress(iterable, **kw):   # type: ignore[misc]
        return iterable


# =============================================================================
# Task-extraction helpers
# =============================================================================

def _extract_neurovoz_task(stem: str) -> str:
    """
    Return the task code embedded in a NeuroVoz filename stem.

    Format:  {GROUP}_{TASK}_{ID:04d}
    Examples
    --------
    HC_ABLANDADA_0034  →  ABLANDADA
    PD_A1_0078         →  A1
    HC_U1_0122         →  U1
    """
    parts = stem.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[1:-1])
    logger.warning("Cannot parse task from NeuroVoz stem: '%s'", stem)
    return 'UNKNOWN'


# 3-char prefixes must be listed before 2-char ones so the longest match wins.
_IPVS_PREFIXES_3 = [
    'FB1', 'FB2', 'PR1',
    'VA1', 'VA2',
    'VE1', 'VE2',
    'VI1', 'VI2',
    'VO1', 'VO2',
    'VU1', 'VU2',
]
_IPVS_PREFIXES_2 = ['B1', 'B2', 'D1', 'D2']
# ---------------------------------------------------------------------------
# IPVS special-case constants
# ---------------------------------------------------------------------------

# Manual gender corrections for subjects absent from all Excel metadata files.
# Source: verified by listening to the recordings.
_IPVS_GENDER_OVERRIDE: Dict[str, str] = {
    'antonietta_p': 'F',
    'lisco_g':      'M',
    'porcelli_a':   'M',
    'summo_l':      'F',
}

# Subject-ID overrides for folder-name collisions.
# Key  : (normalise(subgroup_dir_name), normalise(subject_folder_name))
# Value: full subject_id string (including "ipvs_" prefix)
#
# Roberto R (1-5 / 11-16) and Nicola S (6-10 / 11-16) need NO entry here —
# their folders normalise to the same string in every subgroup, so they
# already receive the same subject_id and are treated as one person.
#
# Vito S and Vito L each have two physically distinct individuals sharing
# the same folder name, so they require explicit disambiguation.
_IPVS_SUBJECT_ID_OVERRIDE: Dict[Tuple[str, str], str] = {
    # Vito S — two distinct people in the PD group
    ('6_10',  'vito_s'): 'ipvs_vito_s_71',   # M, 71 — same person as 11-16
    ('11_16', 'vito_s'): 'ipvs_vito_s_71',   # M, 71 — same person as 6-10
    ('17_28', 'vito_s'): 'ipvs_vito_s',      # M, 70 — different person; keeps base name

    # Vito L — one person in PD (17-28), a different person in EHC
    ('17_28',                      'vito_l'): 'ipvs_vito_l',      # M, 70, PD; keeps base name
    ('22_elderly_healthy_control', 'vito_l'): 'ipvs_vito_l_hc',  # HC, unknown age; renamed
}
# Age overrides for subjects whose metadata lookup returns the wrong value
# due to duplicate name collisions in the Excel files.
# Key: final subject_id string. Value: correct age as float.
_IPVS_AGE_OVERRIDE: Dict[str, float] = {
    'ipvs_vito_s_71': 71.0,   # Vito S (6-10 / 11-16) — lookup returns 70
                               # because the 17-28 entry overwrites in the dict
}


def _extract_ipvs_task(stem: str) -> str:
    """
    Return the task code from the leading prefix of an IPVS filename stem.

    Examples
    --------
    B1LBULCAAS94M100120171015  →  B1
    FB1AGNIGNEE54F230320171025 →  FB1
    PR1XYZ...                  →  PR1
    VA1...                     →  VA1
    """
    u = stem.upper()
    for p in _IPVS_PREFIXES_3:
        if u.startswith(p):
            return p
    for p in _IPVS_PREFIXES_2:
        if u.startswith(p):
            return p
    logger.warning("Cannot extract task from IPVS stem: '%s'", stem)
    return 'UNKNOWN'


# =============================================================================
# IPVS metadata helpers
# =============================================================================

def _normalise(s: str) -> str:
    """Lowercase, strip, collapse whitespace / underscores / hyphens → '_'."""
    return re.sub(r'[\s_\-]+', '_', str(s).strip().lower())


def _load_ipvs_metadata() -> pd.DataFrame:
    """
    Load and combine subject metadata from all three IPVS Excel files.

    Returns a DataFrame with columns:
        name, surname, sex, age, label, subgroup

    ``sex``  is normalised to 'M' / 'F' (None if unrecognised).
    ``age``  is float (NaN on parse failure).
    """
    frames: list[pd.DataFrame] = []

    # ── YHC (15 YHC.xlsx) ────────────────────────────────────────────────
    try:
        raw = pd.read_excel(C.IPVS_YHC_META, header=1)
        raw.columns = [str(c).strip() for c in raw.columns]
        df = raw[['name', 'surname', 'sex', 'age']].copy()
        df = df[df['name'].notna()].copy()
        # Drop trailing "average" summary row
        df = df[~df['name'].astype(str).str.strip().str.lower()
                  .isin(['average', ''])].copy()
        df['label'] = 'HC'
        df['subgroup'] = 'YHC'
        frames.append(df)
        logger.debug("YHC metadata: %d subjects", len(df))
    except Exception as exc:
        logger.error("Cannot load YHC metadata (%s): %s", C.IPVS_YHC_META, exc)

    # ── EHC (Tab 3.xlsx) ─────────────────────────────────────────────────
    try:
        raw = pd.read_excel(C.IPVS_EHC_META, header=1)
        raw.columns = [str(c).strip() for c in raw.columns]
        df = raw[['name', 'surname', 'sex', 'age']].copy()
        df = df[df['name'].notna()].copy()
        df = df[~df['name'].astype(str).str.strip().str.lower()
                  .isin([''])].copy()
        df['label'] = 'HC'
        df['subgroup'] = 'EHC'
        frames.append(df)
        logger.debug("EHC metadata: %d subjects", len(df))
    except Exception as exc:
        logger.error("Cannot load EHC metadata (%s): %s", C.IPVS_EHC_META, exc)

    # ── PD (TAB 5.xlsx) ──────────────────────────────────────────────────
    try:
        raw = pd.read_excel(C.IPVS_PD_META, header=1)
        raw.columns = [str(c).strip() for c in raw.columns]
        # Column 0 is the subgroup range label (1-5, 6-10, …) — drop it by
        # selecting only the named columns we need.
        df = raw[['name', 'surname', 'sex', 'age']].copy()
        df = df[df['name'].notna()].copy()
        df = df[~df['name'].astype(str).str.strip().str.lower()
                  .isin(['', 'average'])].copy()
        df['label'] = 'PD'
        df['subgroup'] = 'PD'
        frames.append(df)
        logger.debug("PD metadata: %d subjects", len(df))
    except Exception as exc:
        logger.error("Cannot load PD metadata (%s): %s", C.IPVS_PD_META, exc)

    if not frames:
        return pd.DataFrame(
            columns=['name', 'surname', 'sex', 'age', 'label', 'subgroup']
        )

    combined = pd.concat(frames, ignore_index=True)
    combined['sex'] = (
        combined['sex'].astype(str).str.strip().str.upper()
        .map({'M': 'M', 'F': 'F'})          # anything else → NaN
    )
    combined['age'] = pd.to_numeric(combined['age'], errors='coerce')
    return combined


def _build_lookup(meta_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Return ``{normalise(name)_normalise(surname) → row}`` for fast matching.
    """
    return {
        f"{_normalise(r['name'])}_{_normalise(r['surname'])}": r
        for _, r in meta_df.iterrows()
    }


def _match_to_metadata(
    folder_name: str,
    lookup: Dict[str, pd.Series],
) -> Tuple[Optional[str], Optional[float]]:
    """
    Try to match an IPVS subject folder name to a metadata row.

    Matching strategies (in order):
      1. Exact normalised key  e.g. 'Alberto_R' → key 'alberto_r'
      2. First-name-only match (only when the result is unambiguous)

    Returns
    -------
    (sex, age) on success, (None, None) on failure.
    """
    norm = _normalise(folder_name)

    # Strategy 1 – exact
    if norm in lookup:
        row = lookup[norm]
        age = None if pd.isna(row['age']) else float(row['age'])
        return row['sex'], age

    # Strategy 2 – first name token (unambiguous only)
    first_token = norm.split('_')[0]
    candidates  = [(k, v) for k, v in lookup.items()
                   if k.split('_')[0] == first_token]
    if len(candidates) == 1:
        row = candidates[0][1]
        age = None if pd.isna(row['age']) else float(row['age'])
        return row['sex'], age

    logger.warning(
        "No metadata match for IPVS folder '%s' (normalised='%s'). "
        "age/gender will be None.",
        folder_name, norm,
    )
    return None, None


def _iter_wav(directory: Path):
    """
    Yield all WAV files under *directory* (case-insensitive suffix).
    Using explicit suffix check so the code works on case-sensitive file
    systems (Linux) as well as Windows.
    """
    return sorted(
        f for f in directory.rglob('*')
        if f.is_file() and f.suffix.lower() == '.wav'
    )


# =============================================================================
# NeuroVoz parser
# =============================================================================

def parse_neurovoz() -> pd.DataFrame:
    """
    Parse the NeuroVoz dataset and return a manifest DataFrame.

    Reads ``metadata/data_hc.csv`` and ``metadata/data_pd.csv``.
    The ``Audio`` column in each CSV contains a relative path whose
    basename is the actual filename under ``audios/``.

    Returns
    -------
    pd.DataFrame with columns defined by ``config.MANIFEST_COLUMNS``.
    """
    logger.info("Parsing NeuroVoz …")
    records: list[dict] = []

    for csv_path, label in [
        (C.NEUROVOZ_HC_CSV, 'HC'),
        (C.NEUROVOZ_PD_CSV, 'PD'),
    ]:
        if not csv_path.exists():
            logger.error("CSV not found: %s", csv_path)
            continue

        meta = pd.read_csv(csv_path)
        meta.columns = meta.columns.str.strip()   # remove accidental whitespace
        logger.info("  [NeuroVoz %s] %d rows in %s", label, len(meta), csv_path.name)

        for _, row in _progress(meta.iterrows(), total=len(meta),
                                desc=f"NeuroVoz {label}", leave=False):
            audio_rel    = str(row.get('Audio', '')).strip()
            filename     = Path(audio_rel).name
            file_path    = C.NEUROVOZ_AUDIO_DIR / filename
            stem         = file_path.stem                    # e.g. HC_ABLANDADA_0034

            subject_id   = f"neurovoz_{int(float(row['ID']))}"
            recording_id = f"neurovoz_{stem}"
            task         = _extract_neurovoz_task(stem)

            try:
                gender = 'M' if float(row['Sex']) == 1.0 else 'F'
            except (ValueError, TypeError, KeyError):
                gender = None

            age_raw = row.get('Age')
            age = None if pd.isna(age_raw) else float(age_raw)

            duration, sample_rate = get_audio_info(file_path)

            records.append({
                'recording_id': recording_id,
                'file_path':    str(file_path),
                'subject_id':   subject_id,
                'label':        label,
                'dataset':      C.DATASET_NEUROVOZ,
                'task':         task,
                'language':     'es',
                'duration':     duration,
                'sample_rate':  sample_rate,
                'gender':       gender,
                'age':          age,
            })

    return pd.DataFrame(records, columns=C.MANIFEST_COLUMNS)


# =============================================================================
# IPVS parser
# =============================================================================

def parse_ipvs() -> pd.DataFrame:
    """
    Parse the IPVS dataset and return a manifest DataFrame.

    Directory structure
    -------------------
    15 Young Healthy Control/   <group>/<subject_folder>/*.wav  → HC
    22 Elderly Healthy Control/ <group>/<subject_folder>/*.wav  → HC
    28 People with Parkinson's disease/
        {1-5, 6-10, 11-16, 17-28}/<subject_folder>/*.wav       → PD

    Returns
    -------
    pd.DataFrame with columns defined by ``config.MANIFEST_COLUMNS``.
    """
    logger.info("Parsing IPVS …")

    meta_df    = _load_ipvs_metadata()
    lookup_yhc = _build_lookup(meta_df[meta_df['subgroup'] == 'YHC'])
    lookup_ehc = _build_lookup(meta_df[meta_df['subgroup'] == 'EHC'])
    lookup_pd  = _build_lookup(meta_df[meta_df['subgroup'] == 'PD'])

    records: list[dict] = []

    # ── HC groups (YHC + EHC) ────────────────────────────────────────────
    for group_dir, lookup in [
        (C.IPVS_YHC_DIR, lookup_yhc),
        (C.IPVS_EHC_DIR, lookup_ehc),
    ]:
        if not group_dir.exists():
            logger.warning("HC group dir not found: %s", group_dir)
            continue

        subject_dirs = sorted(d for d in group_dir.iterdir() if d.is_dir())
        logger.info("  [IPVS %s] %d subject folders", group_dir.name, len(subject_dirs))

        for subj_dir in _progress(subject_dirs, desc=group_dir.name, leave=False):
            norm_folder = _normalise(subj_dir.name)
            gender, age = _match_to_metadata(subj_dir.name, lookup)

            # Apply manual gender correction for subjects absent from metadata
            if gender is None:
                gender = _IPVS_GENDER_OVERRIDE.get(norm_folder)

            # Use explicit override for name-collision cases, else derive normally
            override_key = (_normalise(group_dir.name), norm_folder)
            subject_id   = _IPVS_SUBJECT_ID_OVERRIDE.get(
                override_key, f"ipvs_{norm_folder}"
            )
            
            # Correct age for known metadata-collision cases
            if subject_id in _IPVS_AGE_OVERRIDE:
                age = _IPVS_AGE_OVERRIDE[subject_id]

            for wav in _iter_wav(subj_dir):
                stem         = wav.stem
                recording_id = f"ipvs_{stem}"
                task         = _extract_ipvs_task(stem)
                duration, sr = get_audio_info(wav)

                records.append({
                    'recording_id': recording_id,
                    'file_path':    str(wav),
                    'subject_id':   subject_id,
                    'label':        'HC',
                    'dataset':      C.DATASET_IPVS,
                    'task':         task,
                    'language':     'it',
                    'duration':     duration,
                    'sample_rate':  sr,
                    'gender':       gender,
                    'age':          age,
                })

    # ── PD group ─────────────────────────────────────────────────────────
    for subgroup_dir in C.IPVS_PD_SUBGROUPS:
        if not subgroup_dir.exists():
            logger.warning("PD subgroup dir not found: %s", subgroup_dir)
            continue

        subject_dirs = sorted(d for d in subgroup_dir.iterdir() if d.is_dir())
        logger.info(
            "  [IPVS PD/%s] %d subject folders",
            subgroup_dir.name, len(subject_dirs),
        )

        for subj_dir in _progress(subject_dirs,
                                   desc=f"PD/{subgroup_dir.name}", leave=False):
            norm_folder = _normalise(subj_dir.name)
            gender, age = _match_to_metadata(subj_dir.name, lookup_pd)

            # Apply manual gender correction (unlikely in PD group, but consistent)
            if gender is None:
                gender = _IPVS_GENDER_OVERRIDE.get(norm_folder)

            # Use explicit override for name-collision cases, else derive normally
            override_key = (_normalise(subgroup_dir.name), norm_folder)
            subject_id   = _IPVS_SUBJECT_ID_OVERRIDE.get(
                override_key, f"ipvs_{norm_folder}"
            )

            # Correct age for known metadata-collision cases
            if subject_id in _IPVS_AGE_OVERRIDE:
                age = _IPVS_AGE_OVERRIDE[subject_id]

            for wav in _iter_wav(subj_dir):
                stem         = wav.stem
                recording_id = f"ipvs_{stem}"
                task         = _extract_ipvs_task(stem)
                duration, sr = get_audio_info(wav)

                records.append({
                    'recording_id': recording_id,
                    'file_path':    str(wav),
                    'subject_id':   subject_id,
                    'label':        'PD',
                    'dataset':      C.DATASET_IPVS,
                    'task':         task,
                    'language':     'it',
                    'duration':     duration,
                    'sample_rate':  sr,
                    'gender':       gender,
                    'age':          age,
                })

    return pd.DataFrame(records, columns=C.MANIFEST_COLUMNS)


# =============================================================================
# Summary & validation helpers
# =============================================================================

def _print_summary(df: pd.DataFrame, title: str) -> None:
    """Print a concise summary of a manifest DataFrame."""
    sep = '=' * 64
    print(f"\n{sep}")
    print(title)
    print(sep)
    print(f"Total recordings : {len(df):,}")
    print(f"Unique subjects  : {df['subject_id'].nunique():,}")

    if df['dataset'].nunique() > 1:
        print("\nDataset distribution:")
        for ds, n in df['dataset'].value_counts().items():
            print(f"  {ds}: {n:,}")

    print("\nLabel distribution:")
    for lbl, n in df['label'].value_counts().items():
        print(f"  {lbl}: {n:,}")

    print("\nTask distribution:")
    for task, n in df['task'].value_counts().items():
        print(f"  {task}: {n:,}")

    print("\nMissing values per column:")
    for col, n in df.isnull().sum().items():
        flag = '  ← check' if n > 0 else ''
        print(f"  {col}: {n}{flag}")

    dups = df['recording_id'].duplicated().sum()
    print(f"\nDuplicate recording IDs: {dups}")

    missing_files = sum(not Path(p).exists() for p in df['file_path'])
    print(f"Missing files          : {missing_files}")

    print(f"\nRandom sample (seed={C.RANDOM_SEED}, 10 rows):")
    sample = df.sample(min(10, len(df)), random_state=C.RANDOM_SEED)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(sample.to_string(index=False))


def _validate_label_consistency(df: pd.DataFrame, ctx: str) -> None:
    """Assert every subject_id maps to exactly one label; print any violations."""
    multi      = df.groupby('subject_id')['label'].nunique()
    violations = multi[multi > 1]
    if violations.empty:
        print(f"\n✓  [{ctx}] Label consistency OK — every subject maps to one label.")
    else:
        print(f"\n⚠️  [{ctx}] LABEL VIOLATIONS ({len(violations)} subjects):")
        for sid in violations.index:
            labels = df.loc[df['subject_id'] == sid, 'label'].unique().tolist()
            print(f"    {sid}  →  {labels}")


def _print_recordings_per_subject(df: pd.DataFrame, ctx: str) -> None:
    """Print min / max / mean / median recordings per subject."""
    rps = df.groupby('subject_id').size()
    print(f"\nRecordings per subject [{ctx}]:")
    print(f"  min    : {rps.min()}")
    print(f"  max    : {rps.max()}")
    print(f"  mean   : {rps.mean():.2f}")
    print(f"  median : {rps.median():.1f}")


# =============================================================================
# Main builder
# =============================================================================

def build_raw_manifest() -> pd.DataFrame:
    """
    Orchestrate manifest building.

    1. parse_neurovoz()  →  neurovoz_df
    2. parse_ipvs()      →  ipvs_df
    3. Print per-dataset summaries and run validations.
    4. Concatenate, enforce column order, save raw_manifest.csv.
    5. Print final combined summary.

    Returns
    -------
    The combined pd.DataFrame (also saved to disk).
    """
    # ── NeuroVoz ─────────────────────────────────────────────────────────
    neurovoz_df = parse_neurovoz()
    _print_summary(neurovoz_df, "NEUROVOZ SUMMARY")
    _validate_label_consistency(neurovoz_df, "NEUROVOZ")
    _print_recordings_per_subject(neurovoz_df, "NEUROVOZ")

    # ── IPVS ─────────────────────────────────────────────────────────────
    ipvs_df = parse_ipvs()
    _print_summary(ipvs_df, "IPVS SUMMARY")
    _validate_label_consistency(ipvs_df, "IPVS")
    _print_recordings_per_subject(ipvs_df, "IPVS")

    # ── Concatenate & enforce column order ────────────────────────────────
    combined = pd.concat([neurovoz_df, ipvs_df], ignore_index=True)
    combined = combined[C.MANIFEST_COLUMNS]

    # ── Save ─────────────────────────────────────────────────────────────
    combined.to_csv(C.RAW_MANIFEST_CSV, index=False)
    logger.info(
        "Saved → %s  (%d rows)", C.RAW_MANIFEST_CSV, len(combined)
    )

    # ── Final summary ─────────────────────────────────────────────────────
    _print_summary(combined, "FINAL RAW MANIFEST SUMMARY")
    _validate_label_consistency(combined, "FINAL RAW MANIFEST")
    _print_recordings_per_subject(combined, "FINAL RAW MANIFEST")

    return combined


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    build_raw_manifest()