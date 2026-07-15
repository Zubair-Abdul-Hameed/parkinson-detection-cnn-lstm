# save as test_cnn.py in the project root, run from there
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn

import config as C
from models.cnn import ParkinsonCNN
from utils.seed import set_seed

# ── 1. Shape contract ────────────────────────────────────────────────────────
print("=== forward pass shapes ===")
model = ParkinsonCNN()
for B in [1, 8, 32]:
    x = torch.randn(B, 1, C.N_MELS, 63)
    y = model(x)
    print(f"  input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (B,), f"expected ({B},), got {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "non-finite logits!"
print("  output is (B,) — matches BCEWithLogitsLoss target shape ✓")

# ── 2. Parameter count ───────────────────────────────────────────────────────
print("\n=== model summary ===")
print(model.summary())
n = model.count_parameters()
assert n < 500_000, f"model is larger than expected for this dataset ({n:,})"

# ── 3. Loss compatibility — the thing most likely to bite ────────────────────
print("\n=== BCEWithLogitsLoss compatibility ===")
criterion = nn.BCEWithLogitsLoss()
x = torch.randn(8, 1, C.N_MELS, 63)
labels = torch.randint(0, 2, (8,))          # as DataLoader yields: long, (B,)
logits = model(x)
loss = criterion(logits, labels.float())     # note the .float() cast
print(f"  logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
print(f"  loss   : {loss.item():.4f}  shape={tuple(loss.shape)} (expect scalar)")
assert loss.ndim == 0, "loss is not a scalar — shape mismatch broadcast!"

# ── 4. Gradients flow ────────────────────────────────────────────────────────
print("\n=== backward pass ===")
model.zero_grad()
loss.backward()
no_grad = [n_ for n_, p in model.named_parameters()
           if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
print(f"  params with no gradient: {no_grad if no_grad else 'none'}")
assert not no_grad, "some parameters received no gradient!"
print("  gradients reach every layer ✓")

# ── 5. Reproducible init (uses seed.py) ──────────────────────────────────────
print("\n=== reproducible init ===")
set_seed(42); w1 = ParkinsonCNN().classifier.weight.detach().clone()
set_seed(42); w2 = ParkinsonCNN().classifier.weight.detach().clone()
print(f"  identical weights with same seed? {torch.equal(w1, w2)}")
assert torch.equal(w1, w2)

# ── 6. train() vs eval() — dropout/BN actually switch ────────────────────────
print("\n=== train/eval mode ===")
set_seed(0)
x = torch.randn(4, 1, C.N_MELS, 63)
model.train(); a, b = model(x), model(x)
model.eval()
with torch.no_grad():
    c, d = model(x), model(x)
print(f"  train mode: two passes differ? {not torch.allclose(a, b)} (expect True — dropout active)")
print(f"  eval  mode: two passes differ? {not torch.allclose(c, d)} (expect False — deterministic)")
assert torch.allclose(c, d), "eval mode is not deterministic!"

# ── 7. Real batch from the DataLoader ────────────────────────────────────────
print("\n=== real batch end-to-end ===")
from dataset_building.data_loader import build_dataloader
dl = build_dataloader('val', batch_size=8, num_workers=0)
spec, label = next(iter(dl))
model.eval()
with torch.no_grad():
    logits = model(spec)
    probs = torch.sigmoid(logits)
print(f"  batch  : {tuple(spec.shape)} -> logits {tuple(logits.shape)}")
print(f"  probs  : {[f'{p:.3f}' for p in probs.tolist()]}")
print(f"  labels : {label.tolist()}")
assert logits.shape == (8,)
assert ((probs >= 0) & (probs <= 1)).all()

print("\nAll assertions passed.")