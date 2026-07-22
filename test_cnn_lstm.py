# save as test_cnn_lstm.py in the project root (replaces the previous one)
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn

import config as C
from models.cnn import ParkinsonCNN
from models.cnn_lstm import ParkinsonCNNLSTM
from utils.seed import set_seed

# ── 1. Shape contract ────────────────────────────────────────────────────────
print("=== forward pass shapes ===")
model = ParkinsonCNNLSTM()
for B in [1, 8, 32]:
    x = torch.randn(B, 1, C.N_MELS, 63)
    y = model(x)
    print(f"  input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (B,), f"expected ({B},), got {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "non-finite logits!"
print("  output is (B,) — matches BCEWithLogitsLoss target shape ✓")

# ── 2. Per-block shape trace — verify asymmetric pooling ─────────────────────
print("\n=== per-block shape trace ===")
x = torch.randn(4, 1, C.N_MELS, 63)
print(f"  input            : {tuple(x.shape)}   (B, C, freq, time)")
h = x
with torch.no_grad():
    for i, block in enumerate(model.features, start=1):
        h = block(h)
        print(f"  after block {i}    : {tuple(h.shape)}   "
              f"freq={h.shape[2]:>3}  time={h.shape[3]:>2}")
assert h.shape[2] == 16, f"expected freq 16, got {h.shape[2]}"
assert h.shape[3] == 31, f"expected time 31, got {h.shape[3]}"
print("  freq 128→64→32→16 ✓   time 63→31→31→31 ✓")

# ── 3. Conv -> LSTM reshaping ────────────────────────────────────────────────
print("\n=== conv -> LSTM reshaping ===")
with torch.no_grad():
    m = h.mean(dim=2)
    print(f"  after freq mean  : {tuple(m.shape)}        (B, C, time)")
    p = m.permute(0, 2, 1)
    print(f"  after permute    : {tuple(p.shape)}        (B, seq_len, features)")
    out, (h_n, c_n) = model.lstm(p)
    print(f"  LSTM outputs     : {tuple(out.shape)}         h_n: {tuple(h_n.shape)}")
assert p.shape[1] == 31, f"expected seq_len 31, got {p.shape[1]}"
assert p.shape[2] == 128, "per-timestep feature size should be channel count"
print(f"  sequence length is 31 timesteps ✓  (was 7 in the previous design)")

# ── 4. Parameter counts ──────────────────────────────────────────────────────
print("\n=== parameter counts ===")
cnn = ParkinsonCNN()
n_cnn, n_lstm = cnn.count_parameters(), model.count_parameters()
print(model.summary())
n_backbone = sum(p.numel() for p in model.features.parameters())
n_lstm_only = sum(p.numel() for p in model.lstm.parameters())
n_head = sum(p.numel() for p in model.classifier.parameters())
print(f"\n  backbone   : {n_backbone:,}")
print(f"  LSTM       : {n_lstm_only:,}")
print(f"  classifier : {n_head:,}")
print(f"  total      : {n_backbone + n_lstm_only + n_head:,}")
print(f"\n  ParkinsonCNN        : {n_cnn:,}")
print(f"  ParkinsonCNNLSTM    : {n_lstm:,}  ({n_lstm/n_cnn:.2f}x)")
print(f"  previous CNN-LSTM   : 142,625  (unchanged — pooling has no params)")
assert n_lstm == 142_625, f"expected 142,625, got {n_lstm:,}"

# ── 5. Loss compatibility ────────────────────────────────────────────────────
print("\n=== BCEWithLogitsLoss compatibility ===")
criterion = nn.BCEWithLogitsLoss()
x = torch.randn(8, 1, C.N_MELS, 63)
labels = torch.randint(0, 2, (8,))
logits = model(x)
loss = criterion(logits, labels.float())
print(f"  loss: {loss.item():.4f}  shape={tuple(loss.shape)} (expect scalar)")
assert loss.ndim == 0, "loss is not a scalar — shape mismatch broadcast!"

# ── 6. Gradients flow ────────────────────────────────────────────────────────
print("\n=== backward pass ===")
model.zero_grad(); loss.backward()
no_grad = [n for n, p in model.named_parameters()
           if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
print(f"  params with no gradient: {no_grad if no_grad else 'none'}")
assert not no_grad, "some parameters received no gradient!"
lstm_grads = [n for n, p in model.lstm.named_parameters() if p.grad.abs().sum() > 0]
print(f"  LSTM params receiving gradient: {len(lstm_grads)}/4")
print("  gradients reach every layer ✓")

# ── 7. Reproducible init ─────────────────────────────────────────────────────
print("\n=== reproducible init ===")
set_seed(42); w1 = ParkinsonCNNLSTM().classifier.weight.detach().clone()
set_seed(42); w2 = ParkinsonCNNLSTM().classifier.weight.detach().clone()
print(f"  identical weights with same seed? {torch.equal(w1, w2)}")
assert torch.equal(w1, w2)

# ── 8. train() vs eval() ─────────────────────────────────────────────────────
print("\n=== train/eval mode ===")
set_seed(0)
x = torch.randn(4, 1, C.N_MELS, 63)
model.train(); a, b = model(x), model(x)
model.eval()
with torch.no_grad():
    c, d = model(x), model(x)
print(f"  train mode: passes differ? {not torch.allclose(a, b)} (expect True — dropout)")
print(f"  eval  mode: passes differ? {not torch.allclose(c, d)} (expect False)")
assert torch.allclose(c, d), "eval mode is not deterministic!"

# ── 9. Speed check — this design costs compute ───────────────────────────────
print("\n=== relative forward-pass cost ===")
import time
x = torch.randn(32, 1, C.N_MELS, 63)
for name, m in [("ParkinsonCNN", cnn), ("ParkinsonCNNLSTM", model)]:
    m.eval()
    with torch.no_grad():
        m(x)                                   # warm up
        t0 = time.time()
        for _ in range(10): m(x)
        print(f"  {name:<18}: {(time.time()-t0)/10*1000:6.1f} ms / batch of 32")

# ── 10. Real batch end-to-end ────────────────────────────────────────────────
print("\n=== real batch end-to-end ===")
from dataset_building.data_loader import build_dataloader
dl = build_dataloader('val', batch_size=8, num_workers=0)
spec, label = next(iter(dl))
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(spec))
print(f"  batch {tuple(spec.shape)} -> probs {tuple(probs.shape)}")
print(f"  probs : {[f'{p:.3f}' for p in probs.tolist()]}")
assert ((probs >= 0) & (probs <= 1)).all()

print("\nAll assertions passed.")