# src/utils/plotting.py
"""
Training-curve plotting for the Parkinson's FYP pipeline.

Reads the per-epoch CSV log written by trainer.py and produces report-ready
graphs.  Architecture-agnostic: both the CNN and CNN-LSTM runs write logs in
the same schema, so the same function serves both.

Outputs
-------
  config.PLOTS_DIR / {run_name}_accuracy.png
  config.PLOTS_DIR / {run_name}_loss.png

Usage
-----
    from utils.plotting import plot_training_curves
    plot_training_curves(trainer.log_path, run_name='cnn_v1')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib

# Select a non-interactive backend BEFORE importing pyplot.  This script
# saves files rather than displaying them, and Colab / CI machines have no
# display for an interactive backend to attach to.
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

import config as C               # noqa: E402

logger = logging.getLogger(__name__)


def plot_training_curves(
    log_csv_path: Path,
    run_name: str,
    output_dir: Path = C.PLOTS_DIR,
) -> List[Path]:
    """Plot accuracy and loss curves from a training log.

    Produces two separate figures — one comparing train vs. validation
    accuracy across epochs, one comparing train vs. validation loss — each
    with per-epoch markers so individual values are readable in a report.

    Parameters
    ----------
    log_csv_path : Path
        Path to the per-epoch CSV written by ``Trainer`` (typically
        ``trainer.log_path``).  Must contain the columns ``epoch``,
        ``train_accuracy``, ``val_accuracy``, ``train_loss``, ``val_loss``.
    run_name : str
        Run identifier, used in the figure titles and output filenames so
        CNN and CNN-LSTM plots don't overwrite each other.
    output_dir : Path, optional
        Destination directory.  Defaults to ``config.PLOTS_DIR``.

    Returns
    -------
    List[Path]
        The paths of the saved figures, accuracy first then loss.

    Raises
    ------
    FileNotFoundError
        If the log CSV does not exist.
    ValueError
        If the log is empty or is missing a required column.
    """
    log_csv_path = Path(log_csv_path)
    if not log_csv_path.exists():
        raise FileNotFoundError(
            f"Training log not found: {log_csv_path}. Has fit() run?"
        )

    df = pd.read_csv(log_csv_path)
    if df.empty:
        raise ValueError(f"Training log is empty: {log_csv_path}")

    required = {'epoch', 'train_accuracy', 'val_accuracy',
                'train_loss', 'val_loss'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Training log {log_csv_path.name} is missing column(s): "
            f"{sorted(missing)}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = _plot_pair(
        df,
        train_col='train_accuracy',
        val_col='val_accuracy',
        ylabel='Accuracy',
        title=f'{run_name} — Training vs. Validation Accuracy',
        output_path=output_dir / f'{run_name}_accuracy.png',
    )
    loss_path = _plot_pair(
        df,
        train_col='train_loss',
        val_col='val_loss',
        ylabel='Loss (BCE)',
        title=f'{run_name} — Training vs. Validation Loss',
        output_path=output_dir / f'{run_name}_loss.png',
    )

    logger.info("Saved plots → %s, %s", accuracy_path.name, loss_path.name)
    return [accuracy_path, loss_path]


def _plot_pair(
    df: pd.DataFrame,
    train_col: str,
    val_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    figsize: Tuple[int, int] = (9, 5),
) -> Path:
    """Plot one train/val metric pair against epoch and save it.

    Parameters
    ----------
    df : pd.DataFrame
        The parsed training log.
    train_col, val_col : str
        Column names for the training and validation series.
    ylabel : str
        Y-axis label.
    title : str
        Figure title.
    output_path : Path
        Where to write the PNG.
    figsize : Tuple[int, int], optional
        Figure size in inches.

    Returns
    -------
    Path
        ``output_path``, for convenience.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df['epoch'], df[train_col],
            marker='o', markersize=4, linewidth=1.5, label='Train')
    ax.plot(df['epoch'], df[val_col],
            marker='o', markersize=4, linewidth=1.5, label='Validation')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    # Close explicitly: pyplot keeps figures alive otherwise, and repeated
    # calls would accumulate them and trigger a memory warning.
    plt.close(fig)

    return output_path


# ── Append to src/utils/plotting.py ──────────────────────────────────────────
# (these imports go at the top of the file, alongside the existing ones)
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Class-label display names, HC=0 / PD=1 to match dataset.py's LABEL_MAP.
_CLASS_NAMES = ['HC', 'PD']


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    run_name: str,
    output_dir: Path = C.PLOTS_DIR,
) -> Path:
    """Plot a 2x2 confusion matrix with count annotations.

    Rows are the true class, columns the predicted class, labelled HC/PD.

    Parameters
    ----------
    y_true, y_pred : Sequence[int]
        True and predicted labels (0=HC, 1=PD).
    run_name : str
        Identifier for the title and output filename.
    output_dir : Path, optional
        Destination directory.  Defaults to ``config.PLOTS_DIR``.

    Returns
    -------
    Path
        The saved figure path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1]); ax.set_xticklabels(_CLASS_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(_CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{run_name} — Confusion Matrix (recording-level)')

    # Annotate each cell; white text on dark cells for contrast.
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > threshold else 'black',
                    fontsize=14)

    fig.tight_layout()
    out = output_dir / f'{run_name}_confusion_matrix.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out.name)
    return out


def plot_roc_curve(
    results_by_model: Dict[str, Tuple[Sequence[int], Sequence[float]]],
    output_dir: Path = C.PLOTS_DIR,
    filename: str = 'roc_comparison.png',
) -> Path:
    """Overlay ROC curves for one or more models on a single axis.

    Parameters
    ----------
    results_by_model : Dict[str, Tuple[Sequence[int], Sequence[float]]]
        Maps a model display name to ``(y_true, y_prob)``, where y_prob is
        the positive-class (PD) probability.
    output_dir : Path, optional
        Destination directory.  Defaults to ``config.PLOTS_DIR``.
    filename : str, optional
        Output filename.

    Returns
    -------
    Path
        The saved figure path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_prob) in results_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.8, label=f'{name} (AUC = {roc_auc:.3f})')

    # Chance diagonal.
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1,
            label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — recording-level')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve → %s", out.name)
    return out


def plot_precision_recall_curve(
    results_by_model: Dict[str, Tuple[Sequence[int], Sequence[float]]],
    output_dir: Path = C.PLOTS_DIR,
    filename: str = 'pr_comparison.png',
) -> Path:
    """Overlay precision-recall curves for one or more models.

    Parameters
    ----------
    results_by_model : Dict[str, Tuple[Sequence[int], Sequence[float]]]
        Maps a model display name to ``(y_true, y_prob)`` for the PD class.
    output_dir : Path, optional
        Destination directory.  Defaults to ``config.PLOTS_DIR``.
    filename : str, optional
        Output filename.

    Returns
    -------
    Path
        The saved figure path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_prob) in results_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, linewidth=1.8,
                label=f'{name} (AUC = {pr_auc:.3f})')

    # Baseline = prevalence of the positive class (uses the last model's
    # labels; all models share the same test set so this is identical).
    any_true = next(iter(results_by_model.values()))[0]
    prevalence = float(np.mean(any_true))
    ax.axhline(prevalence, linestyle='--', color='grey', linewidth=1,
               label=f'Baseline (prevalence = {prevalence:.2f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — recording-level')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved PR curve → %s", out.name)
    return out


def plot_metric_comparison_bar(
    metrics_by_model: Dict[str, Dict[str, float]],
    output_dir: Path = C.PLOTS_DIR,
    filename: str = 'metric_comparison.png',
) -> Path:
    """Grouped bar chart comparing scalar metrics across models.

    Parameters
    ----------
    metrics_by_model : Dict[str, Dict[str, float]]
        Maps a model display name to a dict of metric name -> value, e.g.
        ``{'CNN': {'accuracy': 0.81, 'f1': 0.82, ...}, 'CNN-LSTM': {...}}``.
        All models should share the same metric keys.
    output_dir : Path, optional
        Destination directory.  Defaults to ``config.PLOTS_DIR``.
    filename : str, optional
        Output filename.

    Returns
    -------
    Path
        The saved figure path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(metrics_by_model)
    # Metric order taken from the first model; assumed consistent across all.
    metric_names = list(next(iter(metrics_by_model.values())))

    n_models = len(model_names)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, model in enumerate(model_names):
        values = [metrics_by_model[model][m] for m in metric_names]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Model Comparison — recording-level metrics')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved metric comparison → %s", out.name)
    return out