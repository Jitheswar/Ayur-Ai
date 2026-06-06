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

## Current best — CANONICAL METRIC = component 5-fold CV (iter 5)
| metric | value | from |
|---|---|---|
| **CV pooled test_acc** | **0.9864** (per-fold std 0.0093) | iter 5 CV, select_on=val_acc_loss |
| **CV pooled macro_f1** | **0.9849** | iter 5 CV |
| served checkpoint (single dedup split, seed 42) | test_acc 0.9928 / macro_f1 0.9912 (one draw) | iter 5 |
| recipe | resnet18 @224, dedup (pHash≤8) split, **val_acc_loss** selection, deterministic | — |
| how to reproduce the metric | `.venv/bin/python -m experiments.cv_eval --k 5 --seed 42` | — |
| ⚠ noise context | single-split test_acc is noise-limited (multi-seed 0.968±0.054, range 0.872–1.0). **Judge all future levers on the CV pooled number, not a single split.** | — |

---

## Lever status
0. **Evaluation rigor** — ✅ DONE. determinism (iter 4), multi-seed band (iter 4), **component 5-fold CV canonical metric** (iter 5, pooled 0.9864±0.009), selection fixed → `val_acc_loss` (iter 5). All future levers judged on CV.
1. **Data** (leakage/dedup ✅) — aug **← NEXT (iter 6)**; class balancing untried. dedup *threshold* = eval-definition knob, hold at 8 for metric stability (don't tune as a lever).
2. **Training** — EMA ✗ (iter 3, reverted); `select_on=val_acc_loss` ✅ (iter 5). LR sched/warmup, optimizer, wd, grad clip, longer — untried
3. Regularization (dropout, label smoothing already active; mixup, TTA) — untried
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

### Iteration 4 — Evaluation rigor: determinism + honest multi-seed metric
- **Hypothesis:** the single-split metric is noise; a reproducible setup + multi-seed mean±std will give the honest number.
- **Changes (all reversible / additive):**
  - `src/train.py set_seed`: cuDNN deterministic + `use_deterministic_algorithms(warn_only)` + CUBLAS env → **reproducibility restored** (two identical-seed runs now produce **byte-identical** checkpoints, sha `c4ea38…`). This was a *failing guardrail*.
  - Refactored `main()` → reusable **`train_model(cfg, device)`**; `main` now thin.
  - `src/data.py`: seeded DataLoader workers (`worker_init_fn`) + **pHash disk cache** (`experiments/.phash_cache.json`) so repeated splits skip the ~30 s hash pass.
  - Added config-driven **`select_on`** (`val_acc` default = original behaviour; `val_acc_loss` = loss tie-break, to test in iter 5).
  - New `experiments/multiseed_eval.py` (trains+evals K seeds → mean±std; appends `experiments/multiseed.jsonl`).
- **Result — 5 seeds [42,0,1,2,3], resnet18 / dedup / val_acc / deterministic:**
  | metric | mean | std | min | max |
  |---|---|---|---|---|
  | test_acc | **0.9676** | 0.0539 | 0.8716 (seed 3) | 1.0000 (seed 0) |
  | macro_f1 | **0.9682** | 0.0362 | 0.9069 | 1.0000 |
  - Per-seed `n_test` ranged **233–323** (uneven — components, not images, are split).
  - Determinism cost: negligible (~75–110 s/run); guardrails all green.
- **Diagnosis of the variance (key):** *not* class starvation — every seed keeps ≥3 train imgs/class, no empty classes. The spread comes from **near-duplicate components**: kept-together duplicates make test errors **correlated** (a hard component fails as a block), so the effective number of independent test units ≈ component count, not 1835. Seed 3's `macro_f1 (0.907) > acc (0.872)` confirms errors concentrate in a few large-support (large-component) classes.
- **DECISION: KEEP** all infra. The honest current metric is **test_acc ≈ 0.97 ± 0.05**, *not* the 0.9964 the loop chased through iter 3. The iter-2 "metric is real, not inflated" note is hereby **corrected** — it was a single noisy draw.
- **Reason / next → iter 5:** adopt **StratifiedGroupKFold CV** over components (every component tested once; pooled acc/F1 + per-fold mean±std) as the canonical metric, and within it A/B the `select_on=val_acc_loss` selection fix. Only then resume model levers, judged against a metric that can see past the noise.

### Iteration 5 — Evaluation rigor: component K-fold CV (canonical metric) + selection A/B
- **Hypothesis:** a component-grouped K-fold CV — every near-duplicate component held out exactly once — yields a stable, low-variance metric that replaces the noisy single split, and lets `val_acc` vs `val_acc_loss` be compared cleanly.
- **Changes (all additive / reversible):**
  - `src/data.py`: new `component_kfold(samples, k, seed, threshold=8)` — class-stratified, **size-balanced** (largest-component-first greedy with a global-load tie-break), leak-free folds. Validated: disjoint+complete partition, **0 components span folds**, every class in every fold, deterministic, balanced sizes `[366,388,366,358,357]`. Also `build_dataloaders(cfg, splits=...)` to inject explicit indices through the real loader/transform path.
  - `experiments/cv_eval.py`: K=5 rotation (`test=fold i, val=fold i+1, train=rest`). Trains each fold the FULL 20-epoch budget while recording per-epoch (val_acc, val_loss) + the test-fold predictions, then **offline-replays the production selection+early-stop logic** (`src.train._selection_metric`) per strategy and pools predictions → one set of training runs gives both strategies, with **no test peeking ever feeding training**.
  - `config.yaml`: `select_on: val_acc → val_acc_loss` (kept; see result).
- **Result — k=5, seed 42, resnet18/dedup/deterministic (every one of 1835 imgs tested once):**
  | selection | **pooled acc** | pooled macro_f1 | per-fold std | selected epochs |
  |---|---|---|---|---|
  | val_acc | 0.9858 | 0.9844 | 0.0093 | [5,9,10,9,6] |
  | **val_acc_loss** | **0.9864** | **0.9849** | 0.0095 | [5,**14**,10,9,6] |
  - **CV variance is ~6× tighter** than the single split (per-fold std 0.009 vs multi-seed 0.054). Pooled **0.986** is the honest central estimate — between the lucky single draw (0.9964) and the noisy multi-seed mean (0.9676).
  - **Fold 0 is the hard fold** (acc 0.970): it holds the giant near-dup components (Alpinia 46-img, etc.) tested against few distinct train examples; folds 1–4 are 0.987–0.994. Confirms the iter-4 correlated-error diagnosis exactly.
  - **val_acc_loss weakly dominates** val_acc (paired: identical pick on 4/5 folds, better on fold 1 — epoch 14 vs 9, the lower-val-loss epoch generalised better; pooled +0.0006 ≈ 1 image). Free, never worse, principled on the saturated val plateau.
- **Served checkpoint:** retrained on the production single split with val_acc_loss → test_acc **0.9928** (275/277), macro_f1 0.9912. Guardrails: train 124 s (full 20 ep — val_acc_loss keeps improving, no early stop), peak GPU **0.95/1.53 GB** (≤6), host RAM 3.1 GB, latency **1.96 ms** GPU / 13.9 ms CPU, serving path loads+predicts ✅, reproducible ✅.
- **DECISION: KEEP** all CV infra + `select_on=val_acc_loss`. Canonical metric is now **CV pooled acc 0.9864 / macro_f1 0.9849**. This is a *measurement* iteration — it doesn't raise model quality, it measures it honestly (~0.986) so future levers are judged past the noise.
- **Reason / next → iter 6 (lever 1, Data: augmentation):** the residual error concentrates in hard near-dup components (fold 0). Stronger train-time augmentation is the cheapest lever to improve generalisation there; A/B it on the fixed CV metric (threshold held at 8 so the metric definition is stable).

<!-- next-iteration: 6 -->
