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