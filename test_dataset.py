# save as test_dataset.py in the project root, run from there
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch
import pandas as pd

import config as C
from dataset_building.dataset import (
    PDWindowDataset, reconstruct_processed_path, LABEL_MAP,
)

# ── 1. Path reconstruction (the Colab-portability guard) ─────────────────────
print("=== path reconstruction ===")
win = pd.read_csv(C.WINDOWS_MANIFEST_CSV)
stored = win.iloc[0]['processed_file_path']
rebuilt = reconstruct_processed_path(stored)
print(f"  stored      : {stored}")
print(f"  rebuilt     : {rebuilt}")
print(f"  exists?     : {rebuilt.exists()}")
print(f"  under PROCESSED_DIR? {C.PROCESSED_DIR in rebuilt.parents}")
assert rebuilt.exists(), "reconstructed path does not resolve!"

# Simulate a POSIX-style stored path (as if the manifest came from Linux)
posix_style = stored.replace('\\', '/')
assert reconstruct_processed_path(posix_style) == rebuilt, "separator handling broken!"
print("  both separator styles resolve identically ✓\n")

# ── 2. Build all three splits ────────────────────────────────────────────────
print("=== split construction ===")
datasets = {}
for split in ['train', 'val', 'test']:
    ds = PDWindowDataset(split=split)
    datasets[split] = ds
    print(f"  {split:<6}: {len(ds):>6} windows, "
          f"{len(set(ds.subject_ids)):>3} subjects")

total = sum(len(d) for d in datasets.values())
print(f"  total : {total} (windows_manifest has {len(win)})")
assert total == len(win), "splits don't cover every window!"

# No subject appears in two splits
s_tr = set(datasets['train'].subject_ids)
s_va = set(datasets['val'].subject_ids)
s_te = set(datasets['test'].subject_ids)
assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "leakage!"
print("  no subject leakage across dataset splits ✓\n")

# ── 3. Fetch items ───────────────────────────────────────────────────────────
print("=== __getitem__ ===")
ds = datasets['train']
spec, label = ds[0]
print(f"  spec shape  : {tuple(spec.shape)}  (expect (1, {C.N_MELS}, n_frames))")
print(f"  spec dtype  : {spec.dtype}")
print(f"  label       : {label.item()}  dtype={label.dtype}")
print(f"  all finite? : {torch.isfinite(spec).all().item()}")
assert spec.ndim == 3 and spec.shape[0] == 1, "missing channel dimension!"
assert spec.shape[1] == C.N_MELS
assert spec.dtype == torch.float32
assert label.dtype == torch.long
assert torch.isfinite(spec).all(), "non-finite values in spectrogram!"

# All items must share one shape (fixed CNN input)
shapes = set()
rng = np.random.default_rng(0)
for i in rng.integers(0, len(ds), size=25):
    s, _ = ds[int(i)]
    shapes.add(tuple(s.shape))
print(f"  25 random items, unique shapes: {shapes}")
assert len(shapes) == 1, "inconsistent spectrogram shapes!"

# ── 4. Cheap metadata access (needed by data_loader.py's sampler) ────────────
print("\n=== metadata accessors ===")
print(f"  subject_ids[0] : {ds.subject_ids[0]}")
print(f"  get_subject_id(0): {ds.get_subject_id(0)}")
print(f"  labels[:10]    : {ds.labels[:10]}")
print(f"  label map      : {LABEL_MAP}")
assert len(ds.subject_ids) == len(ds), "subject_ids length mismatch!"
assert len(ds.labels) == len(ds), "labels length mismatch!"
assert set(ds.labels) <= {0, 1}

# Label encoding matches the manifest
manifest_labels = ds.windows['label'].tolist()
assert all(LABEL_MAP[m] == e for m, e in zip(manifest_labels, ds.labels))
print("  encoded labels match manifest ✓")

# Confirm the imbalance we'll fix in data_loader.py is visible from here
counts = pd.Series(ds.subject_ids).value_counts()
print(f"\n  windows per subject: min={counts.min()}, max={counts.max()}, "
      f"skew={counts.max()/counts.min():.1f}x")

# ── 5. DataLoader smoke test ─────────────────────────────────────────────────
print("\n=== DataLoader batch ===")
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
batch_spec, batch_label = next(iter(dl))
print(f"  batch spec  : {tuple(batch_spec.shape)}  (expect (8, 1, {C.N_MELS}, n_frames))")
print(f"  batch label : {tuple(batch_label.shape)}  values={batch_label.tolist()}")
assert batch_spec.ndim == 4 and batch_spec.shape[:2] == (8, 1)

print("\nAll assertions passed.")