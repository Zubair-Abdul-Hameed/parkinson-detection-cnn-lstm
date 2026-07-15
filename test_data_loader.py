# save as test_data_loader.py in the project root, run from there
import sys
sys.path.insert(0, "src")

from collections import Counter

import numpy as np
import pandas as pd
import torch

import config as C
from dataset_building.dataset import PDWindowDataset
from dataset_building.data_loader import (
    build_dataloader, compute_subject_balanced_weights,
)

# ── 1. Weight correctness: every subject's weights must sum to 1.0 ───────────
print("=== weight computation ===")
train_ds = PDWindowDataset(split='train')
weights  = compute_subject_balanced_weights(train_ds.subject_ids)

per_subject_mass = {}
for sid, w in zip(train_ds.subject_ids, weights.tolist()):
    per_subject_mass[sid] = per_subject_mass.get(sid, 0.0) + w

masses = np.array(list(per_subject_mass.values()))
print(f"  subjects              : {len(masses)}")
print(f"  per-subject weight sum: min={masses.min():.6f}, max={masses.max():.6f}")
assert np.allclose(masses, 1.0), "subject weight masses are not equal!"
print("  every subject carries equal probability mass ✓")

raw = pd.Series(train_ds.subject_ids).value_counts()
print(f"  raw windows/subject   : min={raw.min()}, max={raw.max()}, "
      f"skew={raw.max()/raw.min():.1f}x")

# ── 2. Empirical: does sampling actually flatten the skew? ───────────────────
print("\n=== empirical sampling distribution (one epoch of indices) ===")
from torch.utils.data import WeightedRandomSampler
g = torch.Generator(); g.manual_seed(C.RANDOM_SEED)
sampler = WeightedRandomSampler(weights, num_samples=len(train_ds),
                                replacement=True, generator=g)

drawn = [train_ds.subject_ids[i] for i in sampler]
drawn_counts = pd.Series(Counter(drawn))
print(f"  draws                 : {len(drawn)}")
print(f"  sampled/subject       : min={drawn_counts.min()}, "
      f"max={drawn_counts.max()}, skew={drawn_counts.max()/drawn_counts.min():.1f}x")
print(f"  expected per subject  : {len(train_ds)/len(masses):.1f}")
print(f"  (was {raw.max()/raw.min():.1f}x before weighting — should now be near 1x)")

# ── 3. Class ratio under weighting (subject-balance != class-balance) ────────
lab_of = dict(zip(train_ds.subject_ids, train_ds.labels))
drawn_labels = [lab_of[s] for s in drawn]
n_pd = sum(drawn_labels); n_hc = len(drawn_labels) - n_pd
print(f"\n  sampled class ratio   : HC={n_hc/len(drawn)*100:.1f}%  "
      f"PD={n_pd/len(drawn)*100:.1f}%")
subj_lab = pd.Series({s: lab_of[s] for s in set(train_ds.subject_ids)})
print(f"  (train subjects: HC={(subj_lab==0).sum()}, PD={(subj_lab==1).sum()} "
      f"— the sampled ratio tracks this, as designed)")

# ── 4. Loader configuration per split ────────────────────────────────────────
print("\n=== loader configuration ===")
for split in ['train', 'val', 'test']:
    dl = build_dataloader(split, batch_size=8, num_workers=0)
    is_weighted = isinstance(dl.sampler, WeightedRandomSampler)
    print(f"  {split:<6}: weighted={is_weighted!s:<5} "
          f"drop_last={dl.drop_last!s:<5} batches={len(dl)}")
    if split == 'train':
        assert is_weighted, "train must use weighted sampling!"
        assert dl.drop_last, "train should drop_last!"
    else:
        assert not is_weighted, f"{split} must NOT be weighted!"
        assert not dl.drop_last, f"{split} should keep all samples!"

# ── 5. Batch shape smoke test ────────────────────────────────────────────────
print("\n=== batch shapes ===")
dl = build_dataloader('train', batch_size=8, num_workers=0)
spec, label = next(iter(dl))
print(f"  spec  : {tuple(spec.shape)}  dtype={spec.dtype}")
print(f"  label : {tuple(label.shape)} dtype={label.dtype} values={label.tolist()}")
assert spec.shape[:2] == (8, 1) and spec.shape[2] == C.N_MELS
assert torch.isfinite(spec).all(), "non-finite values in batch!"

# ── 6. Reproducibility ───────────────────────────────────────────────────────
print("\n=== reproducibility ===")

def first_n_labels(n=3):
    dl = build_dataloader('train', batch_size=8, num_workers=0)
    out = []
    for i, (spec, label) in enumerate(dl):
        if i >= n:
            break
        out.append(label.tolist())
    return out

a = first_n_labels()
b = first_n_labels()
print(f"  same seed → same first 3 batches? {a == b}")
print(f"  batch labels: {a}")
assert a == b, "sampler is not reproducible!"

print("\nAll assertions passed.")