# src/training/trainer.py
"""
Architecture-agnostic training loop for the Parkinson's FYP pipeline.

Trains any nn.Module following the project's model contract:

    input : (batch, 1, N_MELS, n_frames)
    output: (batch,) raw logits — no sigmoid applied in-model

Both ParkinsonCNN and the later CNN-LSTM satisfy this contract, so this
Trainer is reused unchanged by train_cnn.py and train_cnn_lstm.py.  It
never imports a specific model class.

Validation scope
----------------
Validation is WINDOW-LEVEL: every window is scored independently and
metrics are computed over windows, not recordings or subjects.  This is a
deliberate, documented tradeoff to keep the training loop simple.  Because
``validate()`` returns raw per-window logits/probabilities/labels alongside
the scalar metrics, a recording- or subject-level aggregation step can be
layered on later (in evaluate.py) without rewriting this loop.

Outputs
-------
  config.CHECKPOINTS_DIR / {run_name}_best.pt   — best-val-loss checkpoint
  config.LOGS_DIR        / {run_name}_log.csv   — per-epoch metrics

Usage
-----
    from training.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, run_name='cnn_v1')
    history = trainer.fit(num_epochs=50)
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import config as C

logger = logging.getLogger(__name__)

# Probability threshold for converting sigmoid outputs to class predictions.
_DECISION_THRESHOLD: float = 0.5

# Minimum improvement in val loss to count as progress (guards against
# declaring victory on floating-point noise).
_MIN_DELTA: float = 1e-4


# =============================================================================
# Result containers
# =============================================================================

@dataclass
class ValidationResult:
    """Outcome of one validation pass.

    Carries both the aggregate metrics and the raw per-window arrays.  The
    raw arrays are what make recording-level aggregation possible later
    without changing the validation loop.

    Attributes
    ----------
    loss : float
        Mean BCE loss over all validation windows.
    accuracy, precision, recall, f1 : float
        Window-level metrics.  PD (1) is the positive class.
    logits, probabilities, labels : np.ndarray
        Raw per-window arrays, aligned by index and ordered as the val
        DataLoader yielded them (which is sequential — shuffle=False).
    """

    loss:       float
    accuracy:   float
    precision:  float
    recall:     float
    f1:         float
    logits:        np.ndarray = field(repr=False)
    probabilities: np.ndarray = field(repr=False)
    labels:        np.ndarray = field(repr=False)

    def scalar_metrics(self) -> Dict[str, float]:
        """Return just the scalar metrics, without the raw arrays."""
        return {
            'val_loss':      self.loss,
            'val_accuracy':  self.accuracy,
            'val_precision': self.precision,
            'val_recall':    self.recall,
            'val_f1':        self.f1,
        }


@dataclass
class EpochRecord:
    """One row of the per-epoch training log."""

    epoch:         int
    train_loss:    float
    train_accuracy: float
    val_loss:      float
    val_accuracy:  float
    val_precision: float
    val_recall:    float
    val_f1:        float
    learning_rate: float
    epoch_seconds: float
    is_best:       bool


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """Trains any model satisfying the project's input/output contract.

    Handles the epoch loop, LR scheduling, early stopping, best-checkpoint
    saving, and per-epoch metric logging.  Knows nothing about the specific
    architecture it is training.

    Attributes
    ----------
    model : nn.Module
        The model being trained (already moved to ``device``).
    device : torch.device
        Where computation happens.
    optimizer : AdamW
        The optimizer.
    scheduler : ReduceLROnPlateau
        LR scheduler monitoring validation loss.
    best_val_loss : float
        Lowest validation loss seen so far.
    history : List[EpochRecord]
        Per-epoch records accumulated during ``fit``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_name: str,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        early_stopping_patience: int = 8,
        grad_clip_max_norm: Optional[float] = 5.0,
        checkpoints_dir: Path = C.CHECKPOINTS_DIR,
        logs_dir: Path = C.LOGS_DIR,
    ) -> None:
        """Configure the trainer.

        Parameters
        ----------
        model : nn.Module
            Any model taking ``(B, 1, N_MELS, n_frames)`` and returning
            ``(B,)`` raw logits.
        train_loader, val_loader : DataLoader
            From data_loader.py.  train is weighted-sampled; val is not.
        run_name : str
            Identifier used to name the checkpoint and log files, e.g.
            'cnn_v1'.  Keeps CNN and CNN-LSTM runs from overwriting
            each other.
        device : torch.device, optional
            Defaults to CUDA when available, else CPU — so the same call
            works locally and on Colab.
        learning_rate : float, optional
            Initial LR for AdamW.  Default 1e-3.
        weight_decay : float, optional
            AdamW weight decay.  Default 1e-4 — modest regularisation,
            appropriate given the small subject count.
        scheduler_patience : int, optional
            Epochs without val-loss improvement before the LR is reduced.
            Default 3.
        scheduler_factor : float, optional
            Multiplier applied to the LR when the scheduler fires.
            Default 0.5.
        early_stopping_patience : int, optional
            Epochs without val-loss improvement before training stops.
            Default 8 — deliberately longer than ``scheduler_patience`` so a
            LR reduction has several epochs to demonstrate whether it helped
            before the run is abandoned.
        grad_clip_max_norm : float or None, optional
            Max gradient norm.  Default 5.0.  Pass None to disable.  Cheap
            insurance for the plain CNN; more important for the CNN-LSTM.
        checkpoints_dir, logs_dir : Path, optional
            Output directories.  Default to the config constants.

        Raises
        ------
        ValueError
            If early_stopping_patience is not greater than
            scheduler_patience, which would stop training before the
            scheduler could ever help.
        """
        if early_stopping_patience <= scheduler_patience:
            raise ValueError(
                f"early_stopping_patience ({early_stopping_patience}) must "
                f"exceed scheduler_patience ({scheduler_patience}); "
                f"otherwise training stops before a LR reduction can take "
                f"effect."
            )

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_name = run_name

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # 'min' — we are minimising validation loss.
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

        self.early_stopping_patience = early_stopping_patience
        self.grad_clip_max_norm = grad_clip_max_norm

        self.checkpoints_dir = Path(checkpoints_dir)
        self.logs_dir = Path(logs_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.checkpoints_dir / f"{run_name}_best.pt"
        self.log_path = self.logs_dir / f"{run_name}_log.csv"

        self.best_val_loss: float = float('inf')
        self.epochs_without_improvement: int = 0
        self.history: List[EpochRecord] = []

        logger.info(
            "Trainer[%s]: device=%s, lr=%.1e, weight_decay=%.1e, "
            "scheduler_patience=%d, early_stopping_patience=%d, grad_clip=%s",
            run_name, self.device, learning_rate, weight_decay,
            scheduler_patience, early_stopping_patience,
            grad_clip_max_norm if grad_clip_max_norm else 'off',
        )

    # ── Single epochs ─────────────────────────────────────────────────────

    def train_one_epoch(self) -> Tuple[float, float]:
        """Run one training epoch.

        Performs forward, loss, backward, optional gradient clipping, and an
        optimizer step for every batch.

        Returns
        -------
        Tuple[float, float]
            Mean training loss and training accuracy over the epoch, both
            weighted by batch size so a smaller final batch doesn't skew them.

        Note
        ----
        Training accuracy is measured with dropout ACTIVE and under weighted
        sampling, so it reads slightly pessimistically compared to val
        accuracy. Its value is the train-vs-val gap, not its absolute level.
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for spectrograms, labels in self.train_loader:
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            # BCEWithLogitsLoss requires float targets matching the logits'
            # shape; the DataLoader yields torch.long labels.
            labels = labels.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(spectrograms)          # (B,)
            loss = self.criterion(logits, labels)      # scalar

            loss.backward()

            if self.grad_clip_max_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Accuracy on the fly — no extra forward pass needed.
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= _DECISION_THRESHOLD).float()
                total_correct += (preds == labels).sum().item()

        n = max(total_samples, 1)
        return total_loss / n, total_correct / n

    @torch.no_grad()
    def validate(self) -> ValidationResult:
        """Run one validation pass over the val DataLoader.

        No gradients are computed.  Metrics are WINDOW-LEVEL: each window is
        scored independently.  Raw per-window arrays are returned alongside
        the scalars so that a recording- or subject-level aggregation can be
        added later without touching this method.

        Returns
        -------
        ValidationResult
            Scalar metrics plus the raw logits, probabilities, and labels.
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_logits: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for spectrograms, labels in self.val_loader:
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            labels_float = labels.to(self.device, non_blocking=True).float()

            logits = self.model(spectrograms)
            loss = self.criterion(logits, labels_float)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        logits_arr = np.concatenate(all_logits) if all_logits else np.array([])
        labels_arr = np.concatenate(all_labels) if all_labels else np.array([])

        # Sigmoid is applied here rather than in the model, so the same
        # logits can feed BCEWithLogitsLoss (which applies it internally).
        probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
        preds_arr = (probs_arr >= _DECISION_THRESHOLD).astype(int)

        # zero_division=0: an epoch predicting a single class yields 0.0
        # rather than raising.  Early epochs can legitimately do this.
        return ValidationResult(
            loss=total_loss / max(total_samples, 1),
            accuracy=float(accuracy_score(labels_arr, preds_arr)),
            precision=float(precision_score(labels_arr, preds_arr, zero_division=0)),
            recall=float(recall_score(labels_arr, preds_arr, zero_division=0)),
            f1=float(f1_score(labels_arr, preds_arr, zero_division=0)),
            logits=logits_arr,
            probabilities=probs_arr,
            labels=labels_arr,
        )

    # ── Full training loop ────────────────────────────────────────────────

    def fit(self, num_epochs: int = 50) -> List[EpochRecord]:
        """Run the full training loop.

        Each epoch: train, validate, step the scheduler, checkpoint if the
        validation loss improved, log, and check the early-stopping
        condition.

        Parameters
        ----------
        num_epochs : int, optional
            Maximum epochs.  Training may stop earlier via early stopping.

        Returns
        -------
        List[EpochRecord]
            The per-epoch history, also written to ``self.log_path``.
        """
        logger.info(
            "Starting training for up to %d epochs (%d train batches, "
            "%d val batches).",
            num_epochs, len(self.train_loader), len(self.val_loader),
        )
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_one_epoch()
            val_result = self.validate()

            lr_before = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_result.loss)
            lr_after = self.optimizer.param_groups[0]['lr']

            # ── Best-checkpoint tracking ──────────────────────────────────
            is_best = val_result.loss < (self.best_val_loss - _MIN_DELTA)
            if is_best:
                self.best_val_loss = val_result.loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_result)
            else:
                self.epochs_without_improvement += 1

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_result.loss,
                val_accuracy=val_result.accuracy,
                val_precision=val_result.precision,
                val_recall=val_result.recall,
                val_f1=val_result.f1,
                learning_rate=lr_before,
                epoch_seconds=time.time() - epoch_start,
                is_best=is_best,
            )
            self.history.append(record)
            self._append_log(record)
            # Console shows the four you care about day-to-day; the CSV keeps
            # precision/recall/F1 for reporting.

            logger.info(
                "Epoch %3d/%d | train %.4f (acc %.3f) | val %.4f (acc %.3f) "
                "| lr %.1e | %.1fs%s",
                epoch, num_epochs, train_loss, train_acc,
                val_result.loss, val_result.accuracy,
                lr_before, record.epoch_seconds,
                '  ← best' if is_best else '',
            )
            
            if lr_after < lr_before:
                logger.info(
                    "  LR reduced: %.2e → %.2e (val loss plateaued).",
                    lr_before, lr_after,
                )

            # ── Early stopping ────────────────────────────────────────────
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d — val loss has not improved "
                    "for %d epochs (best %.4f).",
                    epoch, self.early_stopping_patience, self.best_val_loss,
                )
                break

        elapsed = time.time() - start_time
        self._print_summary(elapsed)
        return self.history

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_result: ValidationResult) -> None:
        """Save a full training checkpoint at the current best val loss.

        Stores optimizer and scheduler state alongside the weights so a
        dropped Colab session can resume rather than restart.
        """
        torch.save(
            {
                'epoch':                epoch,
                'model_state_dict':     self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss':             val_result.loss,
                'val_metrics':          val_result.scalar_metrics(),
                'run_name':             self.run_name,
            },
            self.checkpoint_path,
        )
        logger.debug("Checkpoint saved → %s", self.checkpoint_path)

    def load_best_checkpoint(self) -> dict:
        """Load the best checkpoint's weights into the model.

        Returns
        -------
        dict
            The full checkpoint dict, so the caller can inspect the epoch
            and metrics it was saved at.

        Raises
        ------
        FileNotFoundError
            If no checkpoint exists for this run name.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {self.checkpoint_path}. Has fit() run?"
            )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(
            "Loaded best checkpoint from epoch %d (val loss %.4f).",
            checkpoint['epoch'], checkpoint['val_loss'],
        )
        return checkpoint

    # ── Logging ───────────────────────────────────────────────────────────

    def _append_log(self, record: EpochRecord) -> None:
        """Append one epoch's metrics to the run's CSV log.

        Written incrementally rather than at the end, so a run interrupted
        by a Colab disconnect still leaves a usable log.
        """
        row = asdict(record)
        write_header = not self.log_path.exists()
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _print_summary(self, elapsed: float) -> None:
        """Print an end-of-training summary."""
        sep = '=' * 64
        print(f"\n{sep}")
        print(f"TRAINING SUMMARY — {self.run_name}")
        print(sep)
        print(f"Epochs run          : {len(self.history)}")
        print(f"Best val loss       : {self.best_val_loss:.4f}")

        best = min(self.history, key=lambda r: r.val_loss) if self.history else None
        if best is not None:
            print(f"Best epoch          : {best.epoch}")
            print(f"  train accuracy    : {best.train_accuracy:.4f}")
            print(f"  val accuracy      : {best.val_accuracy:.4f}")
            print(f"  val precision     : {best.val_precision:.4f}")
            print(f"  val recall        : {best.val_recall:.4f}")
            print(f"  val F1            : {best.val_f1:.4f}")

        print(f"\nCheckpoint          : {self.checkpoint_path}")
        print(f"Log                 : {self.log_path}")
        mins, secs = divmod(elapsed, 60)
        print(f"Elapsed time        : {int(mins)}m {secs:.1f}s")
        print(f"{sep}\n")