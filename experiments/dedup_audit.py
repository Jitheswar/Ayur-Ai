"""Leakage / near-duplicate audit for the leaf dataset.

Computes a perceptual hash (pHash, 8-bit) for every image, then finds pairs
whose Hamming distance is ≤ THRESHOLD (by default 8 out of 64 bits ≈ 87.5%
similar). Reports:

    - duplicate pairs where both images are in the SAME split  (intra-split dups)
    - duplicate pairs that SPAN train↔test or train↔val       (cross-split leakage)

Writes a JSON summary to experiments/dedup_results.json.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import imagehash
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent
DATA_RAW = REPO / "data" / "raw"
THRESHOLD = 8          # max Hamming distance to call two images "near-duplicates"
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
# ──────────────────────────────────────────────────────────────────────────────


def _stratified_split(samples, val_split, test_split, seed):
    """Mirror of src.data._stratified_split so we can label every path."""
    import random, collections
    by_class: dict = collections.defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        by_class[label].append(idx)
    train_idx, val_idx, test_idx = [], [], []
    rng = random.Random(seed)
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_split))
        n_val  = int(round(n * val_split))
        n_val  = min(n_val, max(0, n - n_test - 1))
        test_idx  += idxs[:n_test]
        val_idx   += idxs[n_test:n_test + n_val]
        train_idx += idxs[n_test + n_val:]
    return train_idx, val_idx, test_idx


def main():
    from torchvision.datasets import ImageFolder
    base = ImageFolder(str(DATA_RAW))
    samples = base.samples          # list of (path_str, label_int)

    train_idx, val_idx, test_idx = _stratified_split(
        samples, VAL_SPLIT, TEST_SPLIT, SEED
    )
    split_of = {}
    for i in train_idx: split_of[i] = "train"
    for i in val_idx:   split_of[i] = "val"
    for i in test_idx:  split_of[i] = "test"

    print(f"Images: {len(samples)} | train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")

    # Compute pHash for all images
    print("Computing perceptual hashes … (this takes ~30 s for 1835 images)")
    hashes = {}
    for i, (path, label) in enumerate(samples):
        if i % 200 == 0:
            print(f"  {i}/{len(samples)}", flush=True)
        try:
            img = Image.open(path).convert("RGB")
            hashes[i] = imagehash.phash(img, hash_size=8)
        except Exception as e:
            print(f"  WARN: could not hash {path}: {e}", file=sys.stderr)

    # Build per-class index to avoid O(n²) global search
    by_class: dict[int, list[int]] = collections.defaultdict(list)
    for idx in hashes:
        by_class[samples[idx][1]].append(idx)

    intra_same_split: list[dict] = []
    cross_split_leakage: list[dict] = []

    print("Finding near-duplicate pairs (within same class) …")
    for label, idxs in by_class.items():
        class_name = base.classes[label]
        for ai in range(len(idxs)):
            for bi in range(ai + 1, len(idxs)):
                ia, ib = idxs[ai], idxs[bi]
                dist = hashes[ia] - hashes[ib]
                if dist <= THRESHOLD:
                    sa, sb = split_of[ia], split_of[ib]
                    rec = {
                        "class": class_name,
                        "path_a": samples[ia][0],
                        "path_b": samples[ib][0],
                        "split_a": sa,
                        "split_b": sb,
                        "hamming": int(dist),
                    }
                    if sa == sb:
                        intra_same_split.append(rec)
                    else:
                        cross_split_leakage.append(rec)

    # Summarise cross-split leakage
    train_test_leak = [r for r in cross_split_leakage if set([r["split_a"], r["split_b"]]) == {"train", "test"}]
    train_val_leak  = [r for r in cross_split_leakage if set([r["split_a"], r["split_b"]]) == {"train", "val"}]
    val_test_leak   = [r for r in cross_split_leakage if set([r["split_a"], r["split_b"]]) == {"val", "test"}]

    result = {
        "threshold_hamming": THRESHOLD,
        "total_images": len(samples),
        "intra_split_dup_pairs": len(intra_same_split),
        "cross_split_leak_pairs": len(cross_split_leakage),
        "train_test_leak_pairs": len(train_test_leak),
        "train_val_leak_pairs": len(train_val_leak),
        "val_test_leak_pairs": len(val_test_leak),
        "train_test_examples": train_test_leak[:10],
        "train_val_examples": train_val_leak[:10],
    }

    out = Path(__file__).parent / "dedup_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out}")

    print("\n=== SUMMARY ===")
    print(f"Intra-split dup pairs (same split, same class):  {len(intra_same_split)}")
    print(f"Cross-split leak pairs (any pair, diff split):   {len(cross_split_leakage)}")
    print(f"  train↔test leakage pairs:                      {len(train_test_leak)}")
    print(f"  train↔val  leakage pairs:                      {len(train_val_leak)}")
    print(f"  val↔test   leakage pairs:                      {len(val_test_leak)}")
    if train_test_leak:
        print("\nSample train↔test leakage pairs:")
        for r in train_test_leak[:5]:
            print(f"  [{r['class']}] hamming={r['hamming']}  {Path(r['path_a']).name} ↔ {Path(r['path_b']).name}")
    else:
        print("\nNo train↔test leakage detected at this threshold.")


if __name__ == "__main__":
    main()
