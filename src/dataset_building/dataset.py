# src/dataset_building/dataset.py
"""
PyTorch Dataset for the Parkinson's FYP pipeline.

Reads windows_manifest.csv, filters to one split's subjects, and serves
normalized mel spectrograms computed on-the-fly, one window at a time.

Nothing is precomputed to disk: each __getitem__ loads the recording's
waveform, slices the window, computes its mel spectrogram, and normalizes
it using the train-derived global statistics.

Path portability
----------------
The manifests store ABSOLUTE Windows paths, which do not resolve on Colab.
This class never uses the stored path string as a path — it extracts the
portion after ``processed_audio/`` and rejoins it onto config.PROCESSED_DIR,
which points at the correct root for the current environment.  Backslash and
forward-slash separators are both handled, so a Windows-authored manifest
resolves correctly on Linux.

Usage
-----
    from dataset_building.dataset import PDWindowDataset
    train_ds = PDWindowDataset(split='train')
    spec, label = train_ds[0]           # (1, N_MELS, n_frames), scalar long
    subject     = train_ds.subject_ids[0]   # cheap, no audio loaded
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as C
from dataset_building.spectrogram import compute_mel_spectrogram
from dataset_building.spectrogram_normalization import (
    load_spectrogram_stats,
    normalize_spectrogram,
)

logger = logging.getLogger(__name__)

# Label encoding.  PD is the positive class (1) so that precision / recall /
# AUC read naturally as "how well do we detect the disease".
LABEL_MAP: Dict[str, int] = {'HC': 0, 'PD': 1}

# Marker used to split a stored absolute path into its portable remainder.
_PROCESSED_MARKER: str = 'processed_audio/'

# Split name → split CSV path.
_SPLIT_CSV: Dict[str, Path] = {
    'train': C.TRAIN_CSV,
    'val':   C.VAL_CSV,
    'test':  C.TEST_CSV,
}


# =============================================================================
# Path reconstruction
# =============================================================================

def reconstruct_processed_path(stored_path: str) -> Path:
    """Rebuild a portable path to a processed audio file.

    The manifests store absolute paths from the machine that generated them
    (Windows, backslash-separated).  Those strings are not valid paths on
    another machine, and pathlib cannot even split a Windows path correctly
    when running on Linux.  This function therefore treats the stored value
    as a plain string: it normalises separators, locates the
    ``processed_audio/`` marker, and rejoins the remainder onto
    ``config.PROCESSED_DIR``.

    Example
    -------
    ``C:\\Users\\USER\\...\\processed_audio\\neurovoz\\audios\\HC_A1_0034.wav``
    becomes ``<config.PROCESSED_DIR>/neurovoz/audios/HC_A1_0034.wav``.

    Parameters
    ----------
    stored_path : str
        The ``processed_file_path`` value from a manifest row.

    Returns
    -------
    Path
        The path rooted at ``config.PROCESSED_DIR`` for the current
        environment.

    Raises
    ------
    ValueError
        If the marker cannot be found in ``stored_path``, which would mean
        the manifest was not produced by build_processed_manifest.py.
    """
    normalized = str(stored_path).replace('\\', '/')

    # rfind + lowercase: case-insensitive, and picks the last occurrence in
    # the unlikely event the marker appears more than once.
    idx = normalized.lower().rfind(_PROCESSED_MARKER)
    if idx == -1:
        raise ValueError(
            f"Cannot locate '{_PROCESSED_MARKER}' in stored path: "
            f"{stored_path!r}. Path is not relative to the processed audio "
            f"root and cannot be made portable."
        )

    relative = normalized[idx + len(_PROCESSED_MARKER):]
    return C.PROCESSED_DIR / relative


# =============================================================================
# Dataset
# =============================================================================

class PDWindowDataset(Dataset):
    """Serves normalized mel spectrograms for one split, one window per item.

    Each item is a single windowed segment of a recording, converted to a
    log-mel spectrogram and z-score normalized with the train-derived global
    statistics.  Spectrograms are computed on-the-fly in ``__getitem__``;
    nothing is cached to disk.

    Split membership is resolved by joining ``windows_manifest.csv`` against
    the split's CSV on ``subject_id``.  The windows manifest itself is
    split-agnostic.

    Attributes
    ----------
    split : str
        One of 'train', 'val', 'test'.
    windows : pd.DataFrame
        The filtered window rows for this split, positionally indexed.
    subject_ids : List[str]
        ``subject_ids[i]`` is the subject for item ``i``.  Exposed for
        DataLoader-level samplers (e.g. WeightedRandomSampler) that need
        per-item subject membership without loading any audio.
    labels : List[int]
        ``labels[i]`` is the encoded label (see ``LABEL_MAP``) for item ``i``.
        Also exposed cheaply for sampler weight computation.
    mean, std : float
        The global train-only normalization statistics, loaded once at init.
    """

    def __init__(
        self,
        split: str,
        windows_manifest_path: Path = C.WINDOWS_MANIFEST_CSV,
        stats_path: Path = C.SPECTROGRAM_STATS_JSON,
    ) -> None:
        """Build a Dataset for one split.

        Parameters
        ----------
        split : str
            'train', 'val', or 'test'.
        windows_manifest_path : Path, optional
            Path to windows_manifest.csv.  Defaults to the config constant.
        stats_path : Path, optional
            Path to spectrogram_mean_std.json.  Defaults to the config
            constant.

        Raises
        ------
        ValueError
            If ``split`` is not a recognised split name, or if the filtered
            manifest is empty, or if an unrecognised label is present.
        FileNotFoundError
            If the windows manifest or the split CSV does not exist.
        """
        if split not in _SPLIT_CSV:
            raise ValueError(
                f"split must be one of {sorted(_SPLIT_CSV)}, got {split!r}."
            )
        self.split = split

        split_csv = _SPLIT_CSV[split]
        if not windows_manifest_path.exists():
            raise FileNotFoundError(
                f"Windows manifest not found: {windows_manifest_path}. "
                f"Run build_windows_manifest.py first."
            )
        if not split_csv.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {split_csv}. Run split_subjects.py first."
            )

        # ── Filter windows to this split's subjects ────────────────────────
        all_windows    = pd.read_csv(windows_manifest_path)
        split_subjects = set(pd.read_csv(split_csv)['subject_id'])
        windows = all_windows[
            all_windows['subject_id'].isin(split_subjects)
        ].reset_index(drop=True)

        if windows.empty:
            raise ValueError(
                f"No windows found for split {split!r}. Check that "
                f"{split_csv.name} and the windows manifest share subject_ids."
            )
        self.windows = windows

        # ── Validate labels up front rather than failing mid-epoch ─────────
        unknown = set(windows['label']) - set(LABEL_MAP)
        if unknown:
            raise ValueError(
                f"Unrecognised label(s) in windows manifest: {sorted(unknown)}. "
                f"Expected only {sorted(LABEL_MAP)}."
            )

        # ── Cheap per-index metadata (no audio touched) ────────────────────
        # Materialised as plain lists so a sampler can read them without
        # going through pandas indexing for every item.
        self.subject_ids: List[str] = windows['subject_id'].tolist()
        self.recording_ids: List[str] = windows['recording_id'].tolist()
        self.datasets: List[str] = windows['dataset'].tolist()
        self.labels: List[int] = [LABEL_MAP[l] for l in windows['label']]

        # ── Normalization statistics: loaded ONCE, not per item ────────────
        self.mean, self.std = load_spectrogram_stats(stats_path)

        # ── Single-item waveform cache ─────────────────────────────────────
        # Helps only when consecutive items share a recording (sequential
        # access, e.g. evaluation with shuffle=False).  Under shuffling the
        # hit rate is ~0, which is fine — the cost is one waveform per worker.
        self._cached_path: Optional[str] = None
        self._cached_wave: Optional[np.ndarray] = None
        self._cached_sr:   Optional[int] = None

        logger.info(
            "PDWindowDataset[%s]: %d windows from %d subjects "
            "(mean=%.4f, std=%.4f).",
            split, len(self.windows), len(split_subjects), self.mean, self.std,
        )

    # ── Required Dataset interface ────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of windows in this split."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one normalized mel spectrogram and its label.

        Parameters
        ----------
        idx : int
            Positional index into this split's windows.

        Returns
        -------
        spectrogram : torch.Tensor
            float32, shape ``(1, N_MELS, n_frames)``.  The leading 1 is the
            channel dimension expected by ``nn.Conv2d``; the DataLoader adds
            the batch dimension in front of it.
        label : torch.Tensor
            Scalar ``torch.long`` — 0 for HC, 1 for PD (see ``LABEL_MAP``).

        Raises
        ------
        RuntimeError
            If the audio file cannot be found or read.  This should not
            happen: build_windows_manifest.py already verified every
            recording loads.
        """
        row = self.windows.iloc[idx]

        waveform, sample_rate = self._get_waveform(row)

        window = self._slice_window(
            waveform,
            int(row['start_sample']),
            int(row['end_sample']),
        )

        spec, _info = compute_mel_spectrogram(window, sample_rate)
        normalized, _norm_info = normalize_spectrogram(spec, self.mean, self.std)

        # (N_MELS, n_frames) → (1, N_MELS, n_frames): add the channel axis.
        spectrogram = torch.from_numpy(normalized).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return spectrogram, label

    # ── Public metadata accessors (no audio loaded) ───────────────────────

    def get_subject_id(self, idx: int) -> str:
        """Return the subject_id for an item without loading any audio."""
        return self.subject_ids[idx]
    
    def get_recording_id(self, idx: int) -> str:
        """Return the recording_id for an item without loading any audio.

        Used by evaluate.py to group window-level predictions back into their
        parent recordings for recording-level aggregation.
        """
        return self.recording_ids[idx]

    def get_dataset(self, idx: int) -> str:
        """Return the dataset (neurovoz/ipvs) for an item without loading audio."""
        return self.datasets[idx]

    def get_label(self, idx: int) -> int:
        """Return the encoded label for an item without loading any audio."""
        return self.labels[idx]

    # ── Internals ─────────────────────────────────────────────────────────

    def _get_waveform(self, row: pd.Series) -> Tuple[np.ndarray, int]:
        """Load (or reuse) the waveform for a window row's recording.

        Reconstructs a portable path from the manifest's stored path, then
        loads the file.  A single-item cache short-circuits the load when the
        previous item came from the same recording.
        """
        stored_path = str(row['processed_file_path'])

        if stored_path == self._cached_path:
            return self._cached_wave, self._cached_sr   # type: ignore[return-value]

        audio_path = reconstruct_processed_path(stored_path)

        try:
            waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio for {row['recording_id']}.\n"
                f"  stored path      : {stored_path}\n"
                f"  reconstructed to : {audio_path}\n"
                f"  config.PROCESSED_DIR = {C.PROCESSED_DIR}\n"
                f"  underlying error : {exc}"
            ) from exc

        waveform = waveform.astype(np.float32)

        self._cached_path = stored_path
        self._cached_wave = waveform
        self._cached_sr   = int(sr)

        return waveform, int(sr)

    @staticmethod
    def _slice_window(
        waveform: np.ndarray,
        start_sample: int,
        end_sample: int,
    ) -> np.ndarray:
        """Slice ``[start_sample, end_sample)``, zero-padding a short tail.

        Replicates the windowing.py convention: ``end_sample`` may exceed the
        waveform length, in which case the overhang is filled with zeros so
        the returned array is always exactly the full window length.  This
        keeps the CNN input shape fixed.
        """
        window_len = end_sample - start_sample
        available  = waveform[start_sample:end_sample]

        if available.shape[0] < window_len:
            out = np.zeros(window_len, dtype=np.float32)
            out[:available.shape[0]] = available
            return out

        return available.astype(np.float32, copy=False)