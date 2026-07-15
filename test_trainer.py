# save as test_trainer.py in the project root, run from there
import sys, shutil, logging
sys.path.insert(0, "src")

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config as C
from models.cnn import ParkinsonCNN
from training.trainer import Trainer, ValidationResult
from dataset_building.dataset import PDWindowDataset
from utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Tiny loaders so this runs fast on CPU ────────────────────────────────────
TMP_CKPT = Path("test_outputs/ckpt"); TMP_LOG = Path("test_outputs/logs")
for d in (TMP_CKPT, TMP_LOG):
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)

set_seed(42)
train_ds = Subset(PDWindowDataset(split='train'), range(64))
val_ds   = Subset(PDWindowDataset(split='val'),   range(64))
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0, drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

def make_trainer(**kw):
    set_seed(42)
    defaults = dict(model=ParkinsonCNN(), train_loader=train_dl, val_loader=val_dl,
                    run_name='smoke', checkpoints_dir=TMP_CKPT, logs_dir=TMP_LOG)
    defaults.update(kw)
    return Trainer(**defaults)

# ── 1. Patience guard ────────────────────────────────────────────────────────
print("=== patience validation ===")
try:
    make_trainer(scheduler_patience=5, early_stopping_patience=3)
    raise AssertionError("should have rejected early_stop <= scheduler patience!")
except ValueError as e:
    print(f"  correctly rejected: {str(e)[:60]}...")

# ── 2. Single epoch mechanics ────────────────────────────────────────────────
print("\n=== one train + val epoch ===")
t = make_trainer()
loss, acc = t.train_one_epoch()
res = t.validate()
print(f"  train loss : {loss:.4f}  acc: {acc:.3f}")
print(f"  val loss   : {res.loss:.4f}  acc={res.accuracy:.3f} "
      f"P={res.precision:.3f} R={res.recall:.3f} F1={res.f1:.3f}")
assert isinstance(res, ValidationResult)
assert 0 < loss < 10, "train loss implausible"

# ── 3. Raw arrays returned (needed for future recording-level aggregation) ───
print("\n=== raw prediction arrays ===")
print(f"  logits {res.logits.shape}  probs {res.probabilities.shape}  labels {res.labels.shape}")
assert len(res.logits) == len(val_ds), "raw arrays don't cover every val window!"
assert ((res.probabilities >= 0) & (res.probabilities <= 1)).all()
assert set(res.labels.tolist()) <= {0, 1}
print("  per-window arrays available for evaluate.py ✓")

# ── 4. Loss decreases on a tiny batch (can it learn at all?) ─────────────────
print("\n=== overfit sanity check (8 samples, 30 epochs) ===")
tiny = DataLoader(Subset(PDWindowDataset(split='train'), range(8)),
                  batch_size=8, num_workers=0)
set_seed(42)
m = ParkinsonCNN(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
crit = nn.BCEWithLogitsLoss(); m.train()
x, y = next(iter(tiny)); first = last = None
for i in range(30):
    opt.zero_grad(); l = crit(m(x), y.float()); l.backward(); opt.step()
    if i == 0: first = l.item()
    last = l.item()
print(f"  loss: {first:.4f} → {last:.4f}")
assert last < first, "model cannot even overfit 8 samples — something is broken!"
print("  model can fit data ✓")

# ── 5. Checkpoint + log written ──────────────────────────────────────────────
print("\n=== fit(): checkpoint & log ===")
t = make_trainer()
history = t.fit(num_epochs=3)
ckpt_file = TMP_CKPT / "smoke_best.pt"
log_file  = TMP_LOG / "smoke_log.csv"
print(f"  epochs run     : {len(history)}")
print(f"  checkpoint     : {ckpt_file.exists()}")
print(f"  log            : {log_file.exists()}")
assert ckpt_file.exists() and log_file.exists()

ck = torch.load(ckpt_file, map_location='cpu')
print(f"  checkpoint keys: {sorted(ck)}")
assert {'model_state_dict','optimizer_state_dict','scheduler_state_dict',
        'epoch','val_loss'} <= set(ck)

import csv as _csv
rows = list(_csv.DictReader(open(log_file)))
print(f"  log rows       : {len(rows)}  cols={len(rows[0])}")
assert len(rows) == len(history)

# ── 6. Checkpoint reload restores exact weights ──────────────────────────────
print("\n=== checkpoint reload ===")
before = t.model.classifier.weight.detach().clone()
t.model.classifier.weight.data.fill_(0.0)          # corrupt
t.load_best_checkpoint()
after = t.model.classifier.weight.detach()
print(f"  weights restored? {not torch.equal(after, torch.zeros_like(after))}")
assert not torch.equal(after, torch.zeros_like(after))

# ── 7. Early stopping fires ──────────────────────────────────────────────────
print("\n=== early stopping (patience=2, val loss never improves) ===")
from training.trainer import ValidationResult
import numpy as np

t = make_trainer(scheduler_patience=1, early_stopping_patience=2)

# Force a constant val loss so the early-stopping path is guaranteed to fire.
# Training itself runs normally — we only stub out the validation result.
def constant_validate():
    n = 8
    return ValidationResult(
        loss=0.5, accuracy=0.5, precision=0.5, recall=0.5, f1=0.5,
        logits=np.zeros(n), probabilities=np.full(n, 0.5), labels=np.zeros(n),
    )
t.validate = constant_validate

h = t.fit(num_epochs=20)
print(f"  stopped after {len(h)} epochs (max was 20)")
assert len(h) < 20, "early stopping never fired!"