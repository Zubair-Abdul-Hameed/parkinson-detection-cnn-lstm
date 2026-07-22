# src/training/train_cnn_lstm.py
"""
Training entry point for the CNN-LSTM model.

Runs the full CNN-LSTM training pipeline end-to-end:

    seed → build dataloaders → build model → train → plot curves

Touches only the train and validation splits.  The test set is not read
here — that is evaluate.py's job, after model selection is complete.

Controlled comparison
---------------------
This script mirrors train_cnn.py's training protocol exactly (seed, dataloaders, 
batch size, epoch budget, Trainer defaults) so that any performance difference 
reflects the architectural choice — an independently designed conv+LSTM 
backbone versus the plain CNN — rather than training conditions

Outputs
-------
  config.CHECKPOINTS_DIR / cnn_lstm_v1_best.pt        — best-val-loss checkpoint
  config.LOGS_DIR        / cnn_lstm_v1_log.csv        — per-epoch metrics
  config.PLOTS_DIR       / cnn_lstm_v1_accuracy.png   — accuracy curves
  config.PLOTS_DIR       / cnn_lstm_v1_loss.png       — loss curves

Usage
-----
    python src/training/train_cnn_lstm.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C
from dataset_building.data_loader import build_dataloader
from models.cnn_lstm import ParkinsonCNNLSTM
from training.trainer import Trainer
from utils.plotting import plot_training_curves
from utils.seed import set_seed

logger = logging.getLogger(__name__)

# ── Run configuration ─────────────────────────────────────────────────────
# Every value below is identical to train_cnn.py except RUN_NAME.  Trainer's
# own hyperparameter defaults (learning_rate, weight_decay, patience values,
# grad clipping) are used as-is and deliberately not overridden here — the
# same way train_cnn.py leaves them alone.
RUN_NAME:    str = "cnn_lstm_v1"
NUM_EPOCHS:  int = 50
BATCH_SIZE:  int = 32
NUM_WORKERS: int = 2


def main(
    run_name: str = RUN_NAME,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Trainer:
    """Run CNN-LSTM training end-to-end.

    Seeds all RNGs first, then builds the train/val DataLoaders, the model,
    and the Trainer, runs the training loop, and plots the resulting curves.

    Mirrors train_cnn.py's ``main`` exactly, with ParkinsonCNNLSTM in place
    of ParkinsonCNN.

    Parameters
    ----------
    run_name : str, optional
        Identifier for checkpoint, log, and plot filenames.
    num_epochs : int, optional
        Maximum epochs.  Early stopping may end the run sooner.
    batch_size : int, optional
        Samples per batch for both splits.
    num_workers : int, optional
        DataLoader worker subprocesses.  Set to 0 if worker processes cause
        problems locally on Windows.

    Returns
    -------
    Trainer
        The trainer after fitting, so a caller (e.g. a notebook) can inspect
        ``trainer.history`` or reload the best checkpoint.
    """
    # Seed BEFORE constructing anything — covers model weight init and the
    # train sampler's generator.  Same call, same seed as train_cnn.py, so
    # both models see identically ordered batches.
    set_seed()

    logger.info("Building dataloaders …")
    train_loader = build_dataloader(
        'train', batch_size=batch_size, num_workers=num_workers,
    )
    val_loader = build_dataloader(
        'val', batch_size=batch_size, num_workers=num_workers,
    )

    logger.info("Building model …")
    model = ParkinsonCNNLSTM()
    print(f"\n{model.summary()}\n")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=run_name,
        # Trainer's own defaults apply for all hyperparameters — matching
        # train_cnn.py exactly.
    )

    trainer.fit(num_epochs=num_epochs)

    logger.info("Plotting training curves …")
    plot_paths = plot_training_curves(
        log_csv_path=trainer.log_path,
        run_name=run_name,
    )
    for p in plot_paths:
        print(f"  plot → {p}")

    return trainer


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    main()