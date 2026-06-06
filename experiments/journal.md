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
| **test_acc** | **0.978 – 1.000** (single-run; ⚠ NOT reproducible) | iters 1–3 |
| test_macro_f1 | 0.975 – 1.000 | iters 1–3 |
| checkpoint | resnet18 @224, val-acc-selected, dedup split | — |
| ⚠ status | single-split metric is **noise-limited** (1 test error = 0.0036); identical config+seed gave 0.9964 then 0.9783. Honest mean±std pending **iter 4**. | — |

---

## Lever status
0. **Evaluation rigor** — ⚠ single-split metric found NON-reproducible (iter 3). **NEXT (iter 4): determinism + multi-seed mean±std + fix saturated-val selection.** Must precede any further lever claims.
1. **Data** (leakage/dedup ✅ done) — aug, class balancing still untried
2. **Training** — EMA ✗ (iter 3: hurt, 0.9964→0.9892, reverted). LR sched/warmup, optimizer, wd, grad clip, selection-on-val-loss, longer — untried
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
  | test_acc | **0.9964** (276/277) | = (unchanged) |
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

### Iteration 3 — Training: EMA of weights → REVERTED (and a bigger discovery)
- **Hypothesis:** an exponential moving average of weights (decay 0.995, ~200-step window for this ~570-step run) would calibrate/generalise better and nudge the metric past the iter-2 result. Implemented as an opt-in `train.ema` flag using `torch.optim.swa_utils.AveragedModel(use_buffers=True)`; the EMA model is selected on val and served.
- **Error analysis first (eval-only on iter-2 checkpoint, `experiments/error_analysis.py`):** the *entire* residual error is **one** test image — `Mangifera Indica (Mango) → Nerium Oleander (Oleander)` (model 0.55 confident wrong; both long lance-shaped leaves). Only Mango (recall 0.917) & Oleander (precision 0.933) sit below F1=1.0; macro-F1 is dragged down by that single error. ~8 fragile-but-correct test preds at conf_true≈0.44–0.47. ⇒ test_acc has **1 error of headroom**; do NOT engineer a fix for that specific image (test-set overfitting).
- **EMA result (seed 42):** test_acc **0.9892** (274/277), macro_f1 **0.9902** — *worse* than iter 2. Val saturated at **1.000** by epoch 9, so val-acc selection can't discriminate; EMA'd BN stats on a short run generalise slightly worse. **DECISION: REVERT** (`ema: false`; code kept as opt-in for retest under determinism).
- **⚠ CRITICAL DISCOVERY (the real result of this iteration):** re-running the **non-EMA, dedup config with the SAME seed 42** to restore the checkpoint gave test_acc **0.9783** (271/277, 6 errors) — vs **0.9964** for the "identical" iter-2 run. Early-stop fired at epoch 10 vs 14. **The single-split, val-acc-selected metric is not reproducible**: it ranges ~0.978–1.000 (0–6 errors) due to CUDA/cuDNN nondeterminism feeding a *saturated* val set (any of epochs 9–14 hit val=1.000, and which one is checkpointed is near-random, each generalising differently to the tiny 277-image test).
  - Consequence: the iter-1/iter-2 "0.9964 ceiling" was **one lucky draw**, not a stable measurement. Every lever comparison so far is within the noise band.
- **Guardrails:** all green throughout (EMA peak GPU 0.95 GB; latency 1.9 ms; serving loads fine). Stability ✅ *except reproducibility*, now the top priority.
- **Reason / next → iter 4 (lever 0, evaluation rigor):** (a) add determinism (cudnn deterministic + seeded DataLoader workers) so a seed is reproducible; (b) **multi-seed eval → report mean±std** test_acc/macro_f1 as the honest metric; (c) fix selection on the saturated-val plateau (select on val *loss*, which keeps decreasing). Only then resume levers with a metric that can tell signal from noise.

<!-- next-iteration: 4 -->
