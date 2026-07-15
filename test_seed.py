# save as test_seed.py in the project root, run from there
import sys
sys.path.insert(0, "src")

import random

import numpy as np
import torch

import config as C
from utils.seed import set_seed


def draw():
    """Pull one sample from each RNG."""
    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )


# ── 1. Same seed → same values ───────────────────────────────────────────────
print("=== same seed reproducibility ===")
set_seed(42)
a = draw()
set_seed(42)
b = draw()
print(f"  run A: {a}")
print(f"  run B: {b}")
assert a == b, "same seed produced different values!"
print("  identical ✓")

# ── 2. Different seed → different values ─────────────────────────────────────
print("\n=== different seed ===")
set_seed(42)
a = draw()
set_seed(1234)
c = draw()
print(f"  seed 42  : {a}")
print(f"  seed 1234: {c}")
assert a != c, "different seeds produced identical values — seeding is broken!"
print("  differ ✓")

# ── 3. Default uses config.RANDOM_SEED ───────────────────────────────────────
print("\n=== default seed ===")
returned = set_seed()
d = draw()
set_seed(C.RANDOM_SEED)
e = draw()
print(f"  returned seed  : {returned} (config.RANDOM_SEED = {C.RANDOM_SEED})")
assert returned == C.RANDOM_SEED, "default is not config.RANDOM_SEED!"
assert d == e, "default seed path differs from explicit path!"
print("  default matches config ✓")

# ── 4. CUDA guard doesn't crash on CPU-only ──────────────────────────────────
print("\n=== environment ===")
print(f"  CUDA available    : {torch.cuda.is_available()}")
print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
print(f"  cudnn.benchmark    : {torch.backends.cudnn.benchmark}")
if torch.cuda.is_available():
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    print("  cuDNN flags set for determinism ✓")
else:
    print("  (CPU-only — cuDNN flags not applicable, no crash ✓)")

# ── 5. Reproducible model init (the case that actually matters) ──────────────
print("\n=== model weight init ===")
set_seed(42)
w1 = torch.nn.Linear(10, 5).weight.detach().clone()
set_seed(42)
w2 = torch.nn.Linear(10, 5).weight.detach().clone()
print(f"  identical init weights? {torch.equal(w1, w2)}")
assert torch.equal(w1, w2), "model init is not reproducible!"

print("\nAll assertions passed.")