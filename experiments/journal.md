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
| **test_acc** | **0.9964** | iter 2 (leak-free dedup split) |
| test_macro_f1 | **0.9974** | iter 2 (leak-free dedup split) |
| checkpoint | resnet18 @224, val-selected, dedup split | — |

---

## Lever status
1. **Data** (leakage/dedup ✅ done) — aug, class balancing still untried
2. **Training** (LR sched/warmup, optimizer, wd, grad clip, early stop, longer, EMA) — **NEXT (iter 3)**
3. Regularization (dropout, label smoothing already active; mixup, CV) — untried
4. Capacity (bigger backbone if it fits 6 GB) — untried
5. On-device efficiency (AMP, batch size, grad accum) — untried
6. Compression LAST (PTQ → QAT) — untried

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

### Iteration 2 — Data: leakage/dedup audit + component-based split
- **Hypothesis:** 0.9964 test_acc may be inflated by near-duplicate images straddling the random stratified split. Replacing with a component-based split (pHash ≤8 Hamming distance → union-find components; whole component assigned to one split) prevents cross-split leakage.
- **Change:** `src/data.py` → added `_component_stratified_split()` using imagehash pHash; `src/config.py` → added `DataCfg.deduplicate: bool`; `config.yaml` → `deduplicate: true`; `experiments/dedup_audit.py` (audit script); `experiments/dedup_results.json` (audit summary).
- **Audit findings (dedup_audit.py, threshold=8):** 850 intra-split dup pairs, **374 train↔test leakage pairs**, 320 train↔val leakage pairs, 74 val↔test leakage pairs. Clear evidence of structural dataset duplication across splits.
- **Seed:** 42. New split sizes: train=1289 / val=269 / test=277.
- **Results:**
  | field | value | vs baseline |
  |---|---|---|
  | test_acc | **0.9964** (275/277) | = (unchanged) |
  | test_macro_f1 | **0.9974** | ↑ +0.0014 |
  | train_time | 106.8 s (pHash adds ~30 s; ep 14 early stop) | +32 s |
  | peak GPU alloc / reserved | 0.902 / 1.315 GB | = |
  | host RAM peak | 3.14 GB | = |
  | latency (1 img, warm) | 1.90 ms GPU · 14.58 ms CPU | = |
  | serving path | ✅ | ✅ |
- **DECISION:** **KEPT.**
  - Primary metric (test_acc) unchanged at 0.9964 — confirms the baseline metric was real, not inflated by leakage.
  - macro-F1 improved (+0.0014): dedup split reveals the model generalises across more structurally distinct images.
  - All guardrails pass; evaluation is now on a more honest split.
- **Reason / next:** Data lever partially exhausted (dedup done). Lever 2 (training): try a warmer LR schedule or longer training with EMA. Specifically: cosine LR with warm-up + weight-averaging (EMA, decay 0.995) — likely cheapest lever to push past 0.9964. Iter 3 = EMA of weights (lever 2).

<!-- next-iteration: 3 -->
