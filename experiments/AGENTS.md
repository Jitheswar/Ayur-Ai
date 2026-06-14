# experiments — optimization loop

Parent: [../AGENTS.md](../AGENTS.md) (ayur project). Inherits all ayur contracts (venv, config-driven, no-LLM, reproducibility, leakage control).

## Purpose

Automated, **fully on-device** optimization loop. Goal: maximize held-out quality with zero stability issues (no OOM, no crash, reproducible). One experiment per iteration; the running record lives in `journal.md`.

## Ownership

This doc owns `ayur/experiments/`: the evaluation harnesses (`cv_eval.py`, `snapshot_eval.py`, `multiseed_eval.py`, `profile_run.py`, `ptq_eval.py`, `dedup_audit.py`, `error_analysis.py`), their result artifacts (`*.jsonl`, `*.json`, `*.log`), the pHash cache, and `journal.md`.

## Local Contracts

- **Canonical metric = component 5-fold CV** (`python -m experiments.cv_eval --k 5 --seed 42`). Every image is predicted once by a model that never saw its near-dup component. Pooled CV acc/macro-F1 is the number that decides KEEP/REJECT.
- **Do not judge on a single split.** Single-split test acc is noise-limited (multi-seed ≈ 0.968 ± 0.054, range 0.872–1.0). A single-split swing is not evidence.
- **Fixed protocol** (keep iterations comparable): seed 42 pinned across Python/NumPy/torch/CUDA and the split; stratified component split; deterministic execution.
- **KEEP guardrails — all must hold:** train completes; eval runs; the predictor/app serving path loads the checkpoint; peak GPU ≤ 6 GB; host RAM ≤ ~14 GB; latency acceptable; seed pinned.
- **One experiment per iteration**, logged in `journal.md` with its CV result and a KEEP/REJECT decision and reason.
- Result artifacts are records — append/extend, don't silently rewrite history.

## Work Guidance — settled findings (do not silently re-try; revisit only with new evidence)

- **EMA hurts** (iter3: test_acc 0.9964→0.9892) — left off; code kept opt-in (`train.ema`).
- **TrivialAugmentWide hurts** (iter6: CV 0.9864→0.9657, ~4× worse variance) — reverted; opt-in via `data.augment`.
- **efficientnet_b0 throttles this laptop GPU** (iter7: 33% clock, fold > 10 min) — impractical here; resnet18 stays the served backbone.
- **Checkpoint selection = `val_acc_loss`** (iter5: weakly dominates `val_acc`, loss tie-break on the saturated val plateau).
- **INT8 PTQ** available (iter9): 11.3 MB (−75%), 1.44× CPU latency, ~0 quality cost — **not shipped**; enable deliberately, re-verify serving + guardrails first.
- **Snapshot ensembling** (iter10, `snapshot_eval.py`): averaging softmax over the 2 best val-plateau epochs is the highest pooled-**acc** measured (0.9875) but **flat on macro-F1**, raises cross-fold variance, regresses the hard fold, 2× serving cost — **marginal, not shipped**. N≥3 hurts. NOT a re-try of EMA (iter3 averaged BN stats; this averages outputs).
- **Regularisation is a two-sided optimum on 3 mechanisms** — adding hurts (EMA iter3, TrivialAug iter6, TTA iter8, snapshot-N≥3 iter10, **WD 5e-4 iter12** CV 0.9864→0.9826) AND removing hurts (**LS 0.05 iter11** 0.9864→0.9673). LS stays 0.1, WD stays 1e-4. mixup shares this axis → low-EV.
- **LR is seed-fragile, not a lever** (iter13): lr 1e-4 beat baseline by +0.0011 at seed 42 but *lost* by −0.0022 at seed 1. The residual cross-seed CV noise (~0.003; baseline 0.9864@s42 vs 0.9837@s1) exceeds any lever delta found after iter 5 — judge candidate gains across ≥2 seeds, never one. `lr` stays 3e-4.
- **Capacity ceiling = resnet18, by two direct tests** — effnet_b0 power-throttles (iter7), resnet50 is deterministic-cuDNN-bound (iter14: 26 min < 1 fold at full clock, *not* a throttle). The `use_deterministic_algorithms` guardrail makes large-conv backbones impractical here.
- `cv_eval.py` takes no-op-by-default `--label_smoothing`/`--weight_decay`/`--lr`/`--backbone` overrides for sweeps (records them in the summary jsonl).

## Verification

- Canonical: `python -m experiments.cv_eval --k 5 --seed 42` (add `--label_smoothing`/`--weight_decay` to sweep training hyperparameters).
- Snapshot-ensemble option: `python -m experiments.snapshot_eval --k 5 --seed 42` (N=1 reproduces the canonical 0.9864 as a determinism check).
- Noise band / determinism: `python -m experiments.multiseed_eval`.
- Footprint/latency + serving-load check: `python -m experiments.profile_run`.
- A change is only KEEP when it beats the current best on pooled CV **and** all guardrails hold; record the outcome in `journal.md`.

## Child DOX Index

_No children._
