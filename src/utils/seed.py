# src/utils/seed.py
"""
Reproducibility utilities for the Parkinson's FYP pipeline.

A single entry point that seeds every source of randomness relevant to
training, so that two runs with the same seed produce identical results.

Usage
-----
    from utils.seed import set_seed
    set_seed()            # uses config.RANDOM_SEED
    set_seed(1234)        # explicit override
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

import config as C

logger = logging.getLogger(__name__)


def set_seed(seed: int = C.RANDOM_SEED) -> int:
    """Seed all RNGs used by this project and enable deterministic cuDNN.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU and, when present,
    CUDA).  On a CUDA machine it also constrains cuDNN to deterministic
    algorithms.

    cuDNN flags and their tradeoff
    ------------------------------
    ``torch.backends.cudnn.benchmark`` (PyTorch default: True) lets cuDNN
    autotune convolution algorithms by timing several candidates and caching
    the fastest for each input shape.  The winner can differ between runs,
    and different algorithms sum floating-point values in different orders —
    so results drift slightly even with identical seeds.

    ``torch.backends.cudnn.deterministic = True`` restricts cuDNN to
    algorithms whose output is bit-for-bit reproducible.

    Setting ``deterministic=True`` and ``benchmark=False`` therefore buys
    reproducibility at the cost of roughly 5–15 % slower training.  For this
    project that is a good trade: reported results must be reproducible, and
    the model's input shape is fixed, so autotuning had little to offer
    anyway (it pays off mainly when input shapes vary between batches).

    Scope
    -----
    This seeds the CALLING process.  DataLoader worker subprocesses maintain
    their own RNG state and are not covered here.  That is fine for this
    pipeline — ``Dataset.__getitem__`` performs no random operations, and
    ``data_loader.py`` seeds its sampler's generator explicitly.  If random
    augmentation is added inside ``__getitem__`` later, a ``worker_init_fn``
    will be needed as well.

    Parameters
    ----------
    seed : int, optional
        The seed value.  Defaults to ``config.RANDOM_SEED``, so the whole
        project shares one seed unless a caller overrides it.

    Returns
    -------
    int
        The seed that was applied, for convenient logging by the caller.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Seed set to %d (CPU + CUDA, deterministic cuDNN).", seed)
    else:
        logger.info("Seed set to %d (CPU only — no CUDA device present).", seed)

    return seed