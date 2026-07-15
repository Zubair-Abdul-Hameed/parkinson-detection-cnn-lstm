# src/dataset_building/data_loader.py
"""
DataLoader construction for the Parkinson's FYP pipeline.

Wraps the per-split PDWindowDataset from dataset.py in a properly configured
torch DataLoader.

Subject imbalance
-----------------
Windows per subject in the train split range from 36 to 757 (~21x skew).
Left uncorrected, a model would see one subject's voice 21x more often than
another's and could learn speaker identity rather than pathology.  The train
DataLoader therefore uses a WeightedRandomSampler whose per-sample weight is
inversely proportional to that sample's subject's window count, so every
subject carries equal total probability mass per epoch.

Validation and test DataLoaders are deliberately NOT weighted — their metrics
must reflect the true, unweighted data distribution.

Usage
-----
    from dataset_building.data_loader import build_dataloader

    train_dl = build_dataloader('train', batch_size=32)
    val_dl   = build_dataloader('val',   batch_size=32)
    test_dl  = build_dataloader('test',  batch_size=32)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as C
from dataset_building.dataset import PDWindowDataset

logger = logging.getLogger(__name__)

# Splits that receive subject-balanced weighted sampling.
_WEIGHTED_SPLITS = {'train'}


# =============================================================================
# Sampler weights
# =============================================================================

def compute_subject_balanced_weights(subject_ids: List[str]) -> torch.Tensor:
    """Compute per-sample weights that equalise subject representation.

    Each sample's weight is ``1 / n_windows_for_that_subject``.  Since a
    subject contributing *n* windows has *n* samples each weighted ``1/n``,
    every subject's weights sum to exactly 1.0 — so all subjects carry the
    same total probability mass regardless of how many windows they
    contributed.

    Note
    ----
    This balances SUBJECTS, not CLASSES.  If the split contains unequal
    numbers of HC and PD *subjects*, the sampled class ratio will reflect
    that subject ratio rather than being 50/50.

    Parameters
    ----------
    subject_ids : List[str]
        ``subject_ids[i]`` is the subject for sample ``i``.  Obtained from
        ``PDWindowDataset.subject_ids`` — no audio is loaded.

    Returns
    -------
    torch.Tensor
        float64 tensor of shape ``(len(subject_ids),)`` with one weight per
        sample, aligned to the dataset's positional index.

    Raises
    ------
    ValueError
        If ``subject_ids`` is empty.
    """
    if not subject_ids:
        raise ValueError("subject_ids is empty — cannot compute weights.")

    counts: Counter = Counter(subject_ids)
    weights = [1.0 / counts[sid] for sid in subject_ids]

    # float64: WeightedRandomSampler normalises internally, and float64
    # avoids precision loss when summing tens of thousands of small weights.
    return torch.tensor(weights, dtype=torch.float64)


def _build_train_sampler(
    dataset: PDWindowDataset,
    seed: int,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Build the subject-balanced WeightedRandomSampler for the train split.

    ``replacement=True`` is required: without it the sampler degenerates into
    a plain permutation in which weights only affect ordering, never
    frequency — defeating the purpose.  With replacement, under-represented
    subjects are oversampled and over-represented ones undersampled, so some
    windows recur within an epoch and others are skipped.

    Parameters
    ----------
    dataset : PDWindowDataset
        The train dataset.
    seed : int
        Seed for the sampler's generator, for reproducible epochs.
    num_samples : int, optional
        Number of draws per epoch.  Defaults to ``len(dataset)`` so an epoch
        remains a comparable unit of work to unweighted sampling.
    """
    weights = compute_subject_balanced_weights(dataset.subject_ids)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples if num_samples is not None else len(dataset),
        replacement=True,
        generator=generator,
    )


# =============================================================================
# DataLoader construction
# =============================================================================

def build_dataloader(
    split: str,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    seed: int = C.RANDOM_SEED,
    num_samples: Optional[int] = None,
) -> DataLoader:
    """Build a configured DataLoader for one split.

    The train split receives a subject-balanced ``WeightedRandomSampler``;
    val and test are served sequentially and unweighted so their metrics
    reflect the true data distribution.

    Parameters
    ----------
    split : str
        'train', 'val', or 'test'.
    batch_size : int, optional
        Samples per batch.  Default 32.
    num_workers : int, optional
        Subprocesses for data loading.  Default 2.  Use 0 on Windows if
        worker processes cause problems locally; 2–4 is reasonable on Colab.
    pin_memory : bool, optional
        Pin host memory for faster GPU transfer.  Defaults to True when CUDA
        is available, False otherwise — so the same call works on a CPU-only
        laptop and on a Colab GPU without change.
    seed : int, optional
        Seed for the train sampler's generator.  Defaults to
        ``config.RANDOM_SEED``.  Ignored for val/test (no sampling).
    num_samples : int, optional
        Draws per epoch for the train sampler.  Defaults to the dataset
        length.  Ignored for val/test.

    Returns
    -------
    DataLoader
        Yields ``(spectrograms, labels)`` batches of shape
        ``(batch_size, 1, N_MELS, n_frames)`` and ``(batch_size,)``.

    Raises
    ------
    ValueError
        If ``split`` is not a recognised split name (raised by the Dataset).
    """
    dataset = PDWindowDataset(split=split)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # persistent_workers is only valid when workers exist; it avoids
    # re-spawning them every epoch.
    persistent_workers = num_workers > 0

    if split in _WEIGHTED_SPLITS:
        sampler = _build_train_sampler(dataset, seed=seed, num_samples=num_samples)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,          # mutually exclusive with shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,           # a size-1 final batch breaks BatchNorm
            persistent_workers=persistent_workers,
        )
        logger.info(
            "DataLoader[%s]: %d windows, subject-balanced weighted sampling "
            "(%d draws/epoch, batch_size=%d).",
            split, len(dataset), len(sampler), batch_size,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,            # full sequential coverage for evaluation
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,          # keep every sample when evaluating
            persistent_workers=persistent_workers,
        )
        logger.info(
            "DataLoader[%s]: %d windows, sequential unweighted "
            "(batch_size=%d).",
            split, len(dataset), batch_size,
        )

    return loader


def build_all_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    seed: int = C.RANDOM_SEED,
) -> Dict[str, DataLoader]:
    """Build train, val, and test DataLoaders in one call.

    Convenience wrapper around ``build_dataloader`` for training scripts that
    need all three.  Parameters are passed through unchanged.

    Returns
    -------
    Dict[str, DataLoader]
        Keys 'train', 'val', 'test'.
    """
    return {
        split: build_dataloader(
            split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        for split in ['train', 'val', 'test']
    }