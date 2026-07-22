# src/evaluation/evaluate.py
"""
Test-set evaluation for the Parkinson's FYP pipeline.

Evaluates one or both trained models (CNN, CNN-LSTM) on the held-out TEST
split, at both window level and recording level, and produces a full set of
metrics, result files, and comparison plots.

Recording-level is the PRIMARY metric: the model scores individual 2-second
windows, but a diagnosis is per recording, so window probabilities are
averaged within each recording and thresholded once.  Window-level metrics
are reported alongside for reference (they are what training optimised and
run optimistically high relative to deployment).

The test set is read ONLY here — never during training or validation.

Outputs
-------
  config.RESULTS_DIR / {run_name}_test_results.json      (per model)
  config.RESULTS_DIR / model_comparison.csv              (when >1 model)
  config.PLOTS_DIR   / {run_name}_confusion_matrix.png
  config.PLOTS_DIR   / roc_comparison.png
  config.PLOTS_DIR   / pr_comparison.png
  config.PLOTS_DIR   / metric_comparison.png

Usage
-----
    python src/evaluation/evaluate.py
    # or, choosing runs:
    #   from evaluation.evaluate import main
    #   main(cnn_run_name="cnn_v2", cnn_lstm_run_name="cnn_lstm_v3")
    #   main(cnn_run_name="cnn_v2", cnn_lstm_run_name=None)  # one model only
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

# ── Path bootstrap ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config as C
from dataset_building.dataset import PDWindowDataset
from models.cnn import ParkinsonCNN
from models.cnn_lstm import ParkinsonCNNLSTM
from utils.plotting import (
    plot_confusion_matrix,
    plot_metric_comparison_bar,
    plot_precision_recall_curve,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)

_DECISION_THRESHOLD: float = 0.5


# =============================================================================
# Model loading
# =============================================================================

def load_checkpoint(run_name: str, model_class: type) -> nn.Module:
    """Load a trained model from its best checkpoint.

    Parameters
    ----------
    run_name : str
        The run whose checkpoint to load, from
        ``config.CHECKPOINTS_DIR / f"{run_name}_best.pt"``.
    model_class : type
        The nn.Module subclass to instantiate (ParkinsonCNN or
        ParkinsonCNNLSTM).

    Returns
    -------
    nn.Module
        The model with restored weights, on the active device, in eval mode.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist.
    """
    ckpt_path = C.CHECKPOINTS_DIR / f"{run_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Available checkpoints: "
            f"{sorted(p.name for p in C.CHECKPOINTS_DIR.glob('*_best.pt'))}"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class().to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(
        "Loaded %s from %s (epoch %s, val loss %.4f).",
        model_class.__name__, ckpt_path.name,
        checkpoint.get('epoch', '?'), checkpoint.get('val_loss', float('nan')),
    )
    return model


# =============================================================================
# Inference over the test set
# =============================================================================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataset: PDWindowDataset,
    batch_size: int = 32,
    num_workers: int = 2,
) -> pd.DataFrame:
    """Run a model over the whole test set, collecting per-window predictions.

    Parameters
    ----------
    model : nn.Module
        A model in eval mode.
    dataset : PDWindowDataset
        The test dataset.  Its per-item metadata lists (recording_ids,
        datasets, labels) are used to tag each prediction without reloading
        audio.
    batch_size, num_workers : int
        DataLoader settings.  shuffle is always False so predictions align
        positionally with the dataset's metadata lists.

    Returns
    -------
    pd.DataFrame
        One row per window: recording_id, dataset, label, probability.
    """
    device = next(model.parameters()).device
    # shuffle=False is essential: it keeps batch order aligned with the
    # dataset's positional metadata lists (recording_ids, datasets, labels).
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    all_probs: List[np.ndarray] = []
    for spectrograms, _labels in loader:
        spectrograms = spectrograms.to(device, non_blocking=True)
        logits = model(spectrograms)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    probs_arr = np.concatenate(all_probs)

    # Metadata lists are positional and 1:1 with the non-shuffled loader order.
    return pd.DataFrame({
        'recording_id': dataset.recording_ids,
        'dataset':      dataset.datasets,
        'label':        dataset.labels,
        'probability':  probs_arr,
    })


# =============================================================================
# Recording-level aggregation
# =============================================================================

def aggregate_to_recording_level(window_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window-level predictions into recording-level predictions.

    Groups by recording_id and averages the window probabilities (a soft
    vote), then thresholds at 0.5.  Soft voting is preferred over hard
    majority voting because it preserves confidence — a recording whose
    windows hover near 0.5 will not swing on a single knife-edge window flip.

    The recording's true label is taken as the (consistent) label shared by
    all its windows; the dataset likewise.

    Parameters
    ----------
    window_df : pd.DataFrame
        Output of ``run_inference`` — columns recording_id, dataset, label,
        probability.

    Returns
    -------
    pd.DataFrame
        One row per recording: recording_id, dataset, label (true),
        mean_probability, prediction, n_windows.
    """
    grouped = window_df.groupby('recording_id', sort=False)
    records = grouped.agg(
        dataset=('dataset', 'first'),
        label=('label', 'first'),
        mean_probability=('probability', 'mean'),
        n_windows=('probability', 'size'),
    ).reset_index()

    records['prediction'] = (
        records['mean_probability'] >= _DECISION_THRESHOLD
    ).astype(int)
    return records


# =============================================================================
# Metric computation
# =============================================================================

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute the full metric suite for one set of predictions.

    Specificity (true-negative rate) is derived from the confusion matrix,
    since sklearn does not expose it directly — it is a key screening metric
    (fraction of healthy subjects correctly cleared).

    ROC-AUC is skipped (reported as NaN) if only one class is present, which
    roc_auc_score cannot handle.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float('nan')   # only one class present

    return {
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'precision':   float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':      float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(specificity),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc':     roc_auc,
        'n':           int(len(y_true)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def evaluate_model(
    run_name: str,
    model_class: type,
    test_dataset: PDWindowDataset,
) -> dict:
    """Full evaluation of one model: inference, aggregation, metrics.

    Returns
    -------
    dict
        Nested results: window-level metrics, recording-level metrics,
        per-dataset recording-level breakdown, and the raw recording-level
        arrays (labels / probabilities / predictions) for plotting.
    """
    model = load_checkpoint(run_name, model_class)
    window_df = run_inference(model, test_dataset)
    record_df = aggregate_to_recording_level(window_df)

    # Window-level (reference).
    window_metrics = _compute_metrics(
        window_df['label'].to_numpy(),
        (window_df['probability'] >= _DECISION_THRESHOLD).astype(int).to_numpy(),
        window_df['probability'].to_numpy(),
    )

    # Recording-level (primary).
    rec_true = record_df['label'].to_numpy()
    rec_pred = record_df['prediction'].to_numpy()
    rec_prob = record_df['mean_probability'].to_numpy()
    recording_metrics = _compute_metrics(rec_true, rec_pred, rec_prob)

    # Per-dataset recording-level breakdown.
    per_dataset: Dict[str, dict] = {}
    for ds_name, grp in record_df.groupby('dataset'):
        per_dataset[ds_name] = _compute_metrics(
            grp['label'].to_numpy(),
            grp['prediction'].to_numpy(),
            grp['mean_probability'].to_numpy(),
        )

    return {
        'run_name':          run_name,
        'model':             model_class.__name__,
        'window_level':      window_metrics,
        'recording_level':   recording_metrics,
        'per_dataset':       per_dataset,
        # Raw arrays for plotting (not JSON-serialised in the summary file).
        '_rec_true':         rec_true,
        '_rec_pred':         rec_pred,
        '_rec_prob':         rec_prob,
    }


# =============================================================================
# Persistence & reporting
# =============================================================================

def _save_results_json(result: dict) -> Path:
    """Write one model's metrics to RESULTS_DIR as JSON (arrays stripped)."""
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / f"{result['run_name']}_test_results.json"

    serialisable = {k: v for k, v in result.items() if not k.startswith('_')}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Saved results → %s", out.name)
    return out


def _save_comparison_csv(results: List[dict]) -> Path:
    """Write a side-by-side recording-level comparison table across models."""
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / "model_comparison.csv"

    rows = []
    for r in results:
        row = {'run_name': r['run_name'], 'model': r['model']}
        row.update({f"rec_{k}": v for k, v in r['recording_level'].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Saved comparison table → %s", out.name)
    return out


def _print_summary(results: List[dict]) -> None:
    """Print a side-by-side recording-level summary to the console."""
    sep = '=' * 72
    print(f"\n{sep}")
    print("TEST-SET EVALUATION — recording-level (primary)")
    print(sep)

    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc']
    name_w = max(len(r['run_name']) for r in results)

    header = f"{'metric':<13}" + "".join(f"{r['run_name']:>{name_w+3}}" for r in results)
    print(header)
    print('-' * len(header))
    for m in metrics:
        line = f"{m:<13}"
        for r in results:
            v = r['recording_level'][m]
            line += f"{v:>{name_w+3}.4f}" if not np.isnan(v) else f"{'n/a':>{name_w+3}}"
        print(line)

    # Per-dataset breakdown.
    print(f"\n{'-'*72}\nPer-dataset (recording-level F1):")
    for r in results:
        parts = [f"{ds}={m['f1']:.3f}" for ds, m in r['per_dataset'].items()]
        print(f"  {r['run_name']:<{name_w}} : {'  '.join(parts)}")

    # Window-level reference.
    print(f"\n{'-'*72}\nWindow-level (reference — optimistic):")
    for r in results:
        w = r['window_level']
        print(f"  {r['run_name']:<{name_w}} : "
              f"acc={w['accuracy']:.3f}  f1={w['f1']:.3f}  auc={w['roc_auc']:.3f}")
    print(f"{sep}\n")


# =============================================================================
# Main
# =============================================================================

def main(
    cnn_run_name: Optional[str] = "cnn_v1",
    cnn_lstm_run_name: Optional[str] = "cnn_lstm_v1",
) -> List[dict]:
    """Evaluate one or both models on the test set and produce all artefacts.

    Pass ``None`` for either run name to skip that model — useful while
    iterating on one architecture before the other is ready.

    Parameters
    ----------
    cnn_run_name : str or None
        Run name of the CNN checkpoint, or None to skip.
    cnn_lstm_run_name : str or None
        Run name of the CNN-LSTM checkpoint, or None to skip.

    Returns
    -------
    List[dict]
        The per-model result dicts.

    Raises
    ------
    ValueError
        If both run names are None.
    """
    jobs: List[Tuple[str, str, type]] = []
    if cnn_run_name is not None:
        jobs.append(('CNN', cnn_run_name, ParkinsonCNN))
    if cnn_lstm_run_name is not None:
        jobs.append(('CNN-LSTM', cnn_lstm_run_name, ParkinsonCNNLSTM))

    if not jobs:
        raise ValueError("Provide at least one of cnn_run_name / cnn_lstm_run_name.")

    logger.info("Loading test dataset …")
    test_dataset = PDWindowDataset(split='test')

    results: List[dict] = []
    display_names: List[str] = []
    for display_name, run_name, model_class in jobs:
        logger.info("Evaluating %s (%s) …", display_name, run_name)
        result = evaluate_model(run_name, model_class, test_dataset)
        result['display_name'] = display_name
        results.append(result)
        display_names.append(display_name)

        _save_results_json(result)
        # Per-model confusion matrix (recording-level).
        plot_confusion_matrix(
            result['_rec_true'], result['_rec_pred'], run_name=run_name,
        )

    # Cross-model artefacts.
    roc_inputs = {
        r['display_name']: (r['_rec_true'], r['_rec_prob']) for r in results
    }
    plot_roc_curve(roc_inputs)
    plot_precision_recall_curve(roc_inputs)

    metric_inputs = {
        r['display_name']: {
            k: r['recording_level'][k]
            for k in ['accuracy', 'precision', 'recall', 'specificity', 'f1']
        }
        for r in results
    }
    plot_metric_comparison_bar(metric_inputs)

    if len(results) > 1:
        _save_comparison_csv(results)

    _print_summary(results)
    return results


# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    main()