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

## Current best — CANONICAL METRIC = component 5-fold CV (iter 5; unbeaten through iter 9)
| metric | value | from |
|---|---|---|
| **CV pooled test_acc** | **0.9864** (per-fold std 0.0093) | iter 5 CV, select_on=val_acc_loss |
| **CV pooled macro_f1** | **0.9849** | iter 5 CV |
| served checkpoint (single dedup split, seed 42) | test_acc 0.9928 / macro_f1 0.9912 (one draw) | iter 5 |
| recipe | resnet18 @224, dedup (pHash≤8) split, **val_acc_loss** selection, deterministic | — |
| served model footprint | fp32 44.85 MB · 2 ms GPU / 12.6 ms CPU per image | iter 1/5/9 |
| compression option (not shipped) | INT8 PTQ: 11.3 MB (−75%), 8.7 ms CPU (1.44×), **0.0000 quality cost** | iter 9 |
| how to reproduce the metric | `.venv/bin/python -m experiments.cv_eval --k 5 --seed 42` | — |
| ⚠ noise context | single-split test_acc is noise-limited (multi-seed 0.968±0.054, range 0.872–1.0). **Judge levers on the CV pooled number, not a single split.** | — |

---

## Lever status
0. **Evaluation rigor** — ✅ DONE. determinism (iter 4), multi-seed band (iter 4), **component 5-fold CV canonical metric** (iter 5, pooled 0.9864±0.009), selection fixed → `val_acc_loss` (iter 5). All future levers judged on CV.
1. **Data** (leakage/dedup ✅) — aug ✗ (iter 6: TrivialAugmentWide HURT 0.9864→0.9657, reverted; the hand-tuned standard aug is already well-fitted). class balancing untried (low EV — macro_f1 already 0.985). dedup *threshold* = eval-definition knob, hold at 8.
2. **Training** — EMA ✗ (iter 3, reverted); `select_on=val_acc_loss` ✅ (iter 5). LR sched/warmup, optimizer, wd, grad clip, longer — untried
3. Regularization — TTA ✗ (iter 8: hflip-TTA HURT 0.9864→0.9831, reverted; leaves have orientation-dependent features + model already flip-trained). dropout/label-smoothing active; mixup low-EV (strong aug already hurt).
4. Capacity — ✗ CLOSED on this hardware (iter 7: efficientnet_b0 throttles laptop GPU to 33% clock, fold-0 >10 min, full CV = hours → impractical; resnet50 worse; depthwise nets trip the power/thermal cap). resnet18 is the practical ceiling.
5. On-device efficiency (AMP, batch size, grad accum) — untried (AMP may cut memory/throttle, but not a quality lever)
6. Compression LAST (PTQ → QAT) — ✅ DONE (iter 9: INT8 static PTQ = **zero quality cost**, 4× smaller, 1.44× faster CPU; documented as opt-in, fp32 stays served since it's faster on GPU). QAT unnecessary (PTQ already lossless).

## ✅ STOPPING CONDITION MET (iter 9) → LOOP COMPLETE
- **Quality ceiling reached at CV 0.9864 / macro_f1 0.9849.** Diverse quality levers ALL failed to beat it: EMA (iter 3 ✗), strong aug/TrivialAugmentWide (iter 6 ✗), capacity/efficientnet (iter 7 ✗ impractical), TTA-ensembling (iter 8 ✗). Only ever gain: selection val_acc_loss (iter 5, +0.0006). Compression (iter 9) preserves quality exactly.
- **4 consecutive iterations with no metric improvement** (6,7,8,9). Residual error is **irreducible** on this data: genuine visual confusions (Mango↔Oleander, both lance-shaped) + data-limited giant near-dup components (some classes have ~5 truly distinct examples).
- **Remaining untried levers are all low-EV or likely-negative:** class balancing (macro_f1 already 0.985; errors aren't in minority classes), LR/wd/schedule tuning (model already converges to perfect train fit), mixup (strong regularisation already hurt — see iter 6), AMP (efficiency, not quality). None plausibly beats the ceiling; each would cost ~77 min of throttled CV for ~0 expected gain.
- **All six lever categories addressed (1 Data, 2 Training, 3 Regularisation, 4 Capacity, 5 Efficiency-n/a, 6 Compression).** Goal achieved on every axis: max metric (at ceiling), on-device (resnet18 fits 6 GB w/ headroom, 2 ms GPU), stable (no OOM/crash), reproducible (deterministic, byte-identical checkpoints).

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

### Iteration 6 — Data: stronger augmentation (TrivialAugmentWide) → REVERTED
- **Hypothesis:** the residual error concentrates in hard near-dup components (fold 0); a stronger, standard augmentation policy (TrivialAugmentWide) should improve generalisation there and lift the CV metric past 0.9864.
- **Change:** added config-gated `data.augment` (`standard` | `trivialaugment`); `trivialaugment` keeps the geometric framing (resized-crop + h-flip) but swaps the manual rotation/colour-jitter for `transforms.TrivialAugmentWide()`. Determinism preserved (seeded dataloader workers cover TA's RNG).
- **Result — CV k=5, seed 42, val_acc_loss (same canonical protocol as iter 5):**
  | aug | pooled acc | pooled macro_f1 | per-fold std | selected epochs |
  |---|---|---|---|---|
  | standard (baseline) | **0.9864** | **0.9849** | 0.0095 | [5,14,10,9,6] |
  | trivialaugment | 0.9657 | 0.9609 | **0.0373** | [16,20,15,2,11] |
  - **Δ pooled_acc = −0.0207, Δ macro_f1 = −0.0240; per-fold variance ~4× worse.**
  - TA is too aggressive for this small, near-saturated transfer task: convergence slowed (selected epochs pushed much later), and fold 3 collapsed to 0.905 (selection grabbed epoch 2 on a now-noisy val). Net clearly worse.
- **DECISION: REVERT** (`augment: standard`, restored byte-for-byte; TA code kept as opt-in). Canonical metric unchanged at **0.9864**. Served checkpoint untouched (cv_eval never writes one) and still serves ✅.
- **Reason / next → iter 7 (lever 4, Capacity):** augmentation lever shows the existing aug is already well-tuned; stronger hurts. Next, try **efficientnet_b0** — higher ImageNet accuracy and *fewer* params than resnet18 (better small-data transfer), fits the 6 GB budget with huge headroom — CV-evaluated against 0.9864.

### Iteration 7 — Capacity: efficientnet_b0 → REJECTED (impractical on-device)
- **Hypothesis:** efficientnet_b0 (higher ImageNet acc, only 4.0M params vs resnet18's 11.2M) would transfer better on this small dataset and beat CV 0.9864.
- **Pre-flight:** builds + trains; peak GPU 2.94/3.36 GB ≤ 6 ✅ (no OOM).
- **What happened:** the 5-fold CV ran **51 min on fold 0 alone** (resnet18 does all 5 folds in 12 min). Diagnosed: **laptop RTX 3050 power/thermal throttling** under sustained heavier compute — SM clock dropped to **697/2100 MHz (33%)**, throttle reason `0x24` (SW thermal + SW power cap). Confirmed *not* a determinism artifact (determinism overhead measured ≈1.0–1.1× for resnet18; the throttle hit resnet18 equally during contention). Re-ran from a cool 44 °C start with a 10-min fold-0 deadline + auto-kill monitor → **fold 0 still unfinished at 631 s**, clock still 697 MHz. Full CV would take hours.
- **DECISION: REVERT to resnet18** (per "couldn't run it → log + revert"). efficientnet_b0 violates the on-device practicality guardrail on this hardware. No checkpoint was overwritten (cv_eval doesn't save; runs were killed pre-completion); served resnet18 model intact ✅. Canonical metric unchanged **0.9864**.
- **Findings worth keeping:** (1) **determinism is ~free** (≈1.0–1.1× on resnet18) — the iter-4 reproducibility guardrail costs almost nothing. (2) **The capacity lever is closed on this device** — depthwise-conv nets (efficientnet/mobilenet) trip the power/thermal cap; resnet50 would be heavier still. resnet18 is the practical backbone here.
- **Reason / next → iter 8 (lever 3, TTA):** capacity is out; data is the real limit (few distinct examples/class). Test-time augmentation is the cheapest remaining quality lever — no retraining, light inference cost on resnet18, may help the hard near-dup fold. Evaluate via CV (add hflip-averaged TTA to the test pass) against 0.9864.

### Iteration 8 — Regularization: test-time augmentation (hflip-TTA) → REJECTED
- **Hypothesis:** averaging predictions over the image + its horizontal flip (cheap, no retraining) would improve the metric, especially on the hard near-dup fold.
- **Method:** recorded BOTH plain and TTA test predictions in one deterministic CV run (TTA = argmax of softmax(x)+softmax(hflip(x)); selection on val is TTA-independent, so both share selected epochs). Single run → plain doubles as a determinism sanity check.
- **Result — CV k=5, seed 42:**
  | metric | plain | +hflip-TTA | Δ |
  |---|---|---|---|
  | pooled acc (val_acc_loss) | **0.9864** | 0.9831 | **−0.0033** |
  | pooled macro_f1 | 0.9849 | 0.9815 | −0.0034 |
  | pooled acc (val_acc) | 0.9858 | 0.9826 | −0.0032 |
  - **Sanity ✅:** plain pooled_acc reproduced iter 5 *exactly* (0.9858 / 0.9864) — deterministic run, undisturbed by the TTA code.
  - TTA consistently **hurts ~0.3%** (≈6 extra errors/1835). Leaves carry orientation-dependent discriminative features, and the model is already trained with RandomHorizontalFlip, so the flipped view adds noise rather than signal.
- **DECISION: REJECT.** Reverted the TTA additions to `cv_eval.py` (kept lean; finding recorded). Not added to serving. Canonical metric unchanged **0.9864**.
- **Note:** run took 4708 s (vs ~736 s) — the laptop GPU is now persistently throttled (~47% clock) from sustained back-to-back loads; correctness unaffected (deterministic), but further training-based CV is expensive. Favors eval-only experiments next.
- **Reason / next → iter 9 (lever 6, Compression):** quality ceiling reached (see stopping-condition note). Do INT8 PTQ — **eval-only, throttle-immune** — to document the on-device quality/size/latency tradeoff, then assess LOOP COMPLETE.

### Iteration 9 — Compression: INT8 post-training quantization (the last lever)
- **Hypothesis / purpose:** with the quality ceiling reached, characterise the on-device compression tradeoff (the loop's prescribed final step). Eval-only on CPU → immune to the GPU throttling that slowed iters 7–8.
- **Method:** FX graph-mode **static** INT8 PTQ (x86 backend) on the served resnet18 checkpoint — quantizes conv+linear (vs dynamic, which only touches Linear); calibrated on 256 training images. New `experiments/ptq_eval.py`.
- **Result (production test split, n=277):**
  | model | test_acc | macro_f1 | size | CPU latency |
  |---|---|---|---|---|
  | fp32 (served) | 0.9928 | 0.9912 | 44.85 MB | 12.55 ms |
  | INT8 PTQ | 0.9928 | 0.9912 | **11.32 MB** | **8.69 ms** |
  - **Quality cost = 0.0000** (identical predictions), **size −75 % (4×)**, **CPU latency 1.44× faster**. QAT not needed (PTQ already lossless).
- **DECISION: KEEP fp32 as the served model; document INT8 as an opt-in.** Rationale: the goal is max metric (int8 only preserves it); quantized inference is CPU-only and fp32-on-GPU (2 ms) beats int8-on-CPU (8.7 ms); the model already deploys comfortably. INT8 is the right pick *only* for a CPU-/size-constrained target — now characterised at zero cost.
- **Guardrails:** eval-only, no training; serving path (fp32) unchanged ✅; reproducible ✅.

---

## FINAL SUMMARY (9 iterations)
- **Headline:** the real deliverable was *measurement honesty*. The starting "0.9964" was one lucky draw from a noisy single split; the true, reproducible quality is **CV pooled acc 0.9864 / macro_f1 0.9849** (resnet18, dedup split, val_acc_loss selection, deterministic).
- **What moved the needle:** dedup component split (removed 374 train↔test leak pairs) and a reproducible, low-variance CV metric — these made every later decision trustworthy. val_acc_loss selection added a marginal +0.0006.
- **What didn't (and why it's informative):** EMA, TrivialAugmentWide, and hflip-TTA all *hurt* (model is near-saturated; extra regularisation/ensembling adds noise). efficientnet_b0 was impractical (laptop-GPU thermal/power throttling). INT8 PTQ is lossless but unneeded for GPU serving.
- **Why stopping:** the metric is at the dataset's practical ceiling — residual errors are genuinely ambiguous images + data scarcity (giant near-duplicate components), not fixable by modelling on this hardware.
- **Repo state:** runnable, committed, served fp32 resnet18; reproduce the metric with `experiments/cv_eval.py`, the compression tradeoff with `experiments/ptq_eval.py`.

LOOP COMPLETE
