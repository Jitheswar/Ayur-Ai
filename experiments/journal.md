# Experiment Journal — Ayurvedic Leaf Classifier

Automated optimization loop. Goal: **maximize held-out test metric** while
running **fully on-device** (local only) with **zero stability issues**
(no OOM, no crash, reproducible). One experiment per iteration.

## Fixed protocol (so iterations are comparable)
- **Seed:** 42 (pins Python/NumPy/torch/CUDA via `set_seed`; split is seeded too).
- **Split:** stratified, seed 42 → train 1287 / val 274 / **test 274** (held-out, never trained on).
- **Metric:** test-set accuracy (primary) + macro-F1, via `experiments/profile_run.py`
  which runs the *real* `src.train` + a test-split evaluation.
- **Model selection during training:** best **val** accuracy.
- **Guardrails (all must hold to KEEP):** train completes; eval runs; predictor/app
  serving path loads the checkpoint; peak GPU ≤ 6 GB; host RAM ≤ ~14 GB; latency
  acceptable; seed pinned.

## Hardware budget (detected iter 1)
- GPU: **NVIDIA RTX 3050 Laptop, 6 GB VRAM** (CUDA available, torch 2.12.0+cu130)
- CPU: 16 cores · RAM: 15 GiB (+19 GiB swap)
- venv: `.venv/bin/python` = **Python 3.11.15** (global python is 3.14, unusable — see memory)
- Dataset: 30 classes, 1835 images, imbalanced (34–122 / class).

---

## Current best
| metric | value | from |
|---|---|---|
| **test_acc** | **0.9964** | BASELINE (iter 1) |
| test_macro_f1 | 0.9960 | BASELINE |
| checkpoint | resnet18 @224, val-selected | — |

---

## Lever status
1. **Data** (leakage/dedup, label-noise, normalization, splits, balancing, aug) — **NEXT (iter 2)**. ⚠ top priority, see flag below.
2. Training (LR sched/warmup, optimizer, wd, grad clip, early stop, longer, EMA) — untried
3. Regularization (dropout, label smoothing, mixup; CV) — untried
4. Capacity (bigger backbone if it fits 6 GB) — untried
5. On-device efficiency (AMP, batch size, grad accum) — untried
6. Compression LAST (PTQ → QAT) — untried

### ⚠ FLAG for iter 2 — metric is near ceiling, suspect leakage
Baseline test_acc = **0.9964 = 273/274 correct (1 error)**. That is suspiciously
high for a 30-class leaf task. The "Indian Medicinal Leaves" dataset is known to
contain **near-duplicate / augmented copies of the same physical leaf**. If
duplicates straddle the train/test split, the test metric is **inflated** and the
"ceiling" is an illusion. **Before declaring the loop complete or chasing the last
0.4%, iteration 2 must run a leakage/dedup check** (perceptual-hash / embedding
near-duplicate detection across splits). Only a leak-free split gives a trustworthy
metric — and may reveal real headroom. Do NOT trigger the "near ceiling" stopping
condition until leakage is ruled out.

---

## Iterations

### Iteration 1 — BASELINE
- **Hypothesis:** establish a reproducible reference (metric + resource profile) from the current pipeline.
- **Change:** none to `src/`. Added `experiments/profile_run.py` (reusable train+eval+profile harness) and this journal.
- **Seed:** 42.
- **Procedure:** from-scratch retrain via real `src.train` (config defaults: resnet18, pretrained, 20 epochs cap, lr 3e-4, wd 1e-4, label-smoothing 0.1, dropout 0.2, batch 32, img 224, cosine LR, early-stop patience 5), then evaluate on the held-out test split. Chose a clean retrain over the pre-existing checkpoint (unknown provenance) so future experiments compare like-for-like.
- **Results:**
  | field | value |
  |---|---|
  | test_acc | **0.9964** (273/274) |
  | test_macro_f1 | 0.9960 |
  | train_time | 74.6 s (early-stopped epoch 13; best val @ epoch 8) |
  | peak GPU alloc / reserved | 0.902 / 1.313 GB (budget 6 GB) ✅ |
  | host RAM peak | 3.11 GB (budget ~14 GB) ✅ |
  | latency (1 img, warm) | 1.91 ms GPU · 14.53 ms CPU ✅ |
  | serving path | predictor loads ckpt + KB, predicts ✅ |
- **DECISION:** **BASELINE recorded.** All guardrails pass; huge memory/latency headroom.
- **Reason / next:** metric near ceiling but leakage unverified → iter 2 = data leakage/dedup audit (lever 1) before any capacity/compression work.

<!-- next-iteration: 2 -->
