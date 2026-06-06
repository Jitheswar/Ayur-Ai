"""Error analysis for the current best checkpoint.

Eval-only (no training). Loads models/best_model.pt, rebuilds the *same* dedup
split (seed 42) the model was trained on, and reports:

    * every misclassified val / test image: path, true vs predicted class,
      the model's confidence in each, and the margin
    * per-class precision / recall / F1 on the test split (so we can see which
      classes drag macro-F1 below 1.0)
    * the most-confused class pairs across val+test combined (more signal than
      the tiny test split alone)

Run:
    .venv/bin/python -m experiments.error_analysis
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import ROOT, Config
from src.data import build_dataloaders
from src.model import load_checkpoint


@torch.no_grad()
def _collect(model, loader, base_samples, indices, class_names, device):
    """Run the loader in order and return a list of per-sample records."""
    records = []
    cursor = 0
    for images, labels in loader:
        logits = model(images.to(device))
        probs = F.softmax(logits, dim=1).cpu()
        conf, pred = probs.max(1)
        # top-2 to get the runner-up (margin)
        top2 = probs.topk(2, dim=1)
        for b in range(images.size(0)):
            global_idx = indices[cursor]
            path = base_samples[global_idx][0]
            true_lbl = labels[b].item()
            pred_lbl = pred[b].item()
            runner_up = top2.indices[b, 1].item()
            records.append(
                {
                    "path": str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
                    "true": class_names[true_lbl],
                    "pred": class_names[pred_lbl],
                    "correct": true_lbl == pred_lbl,
                    "conf_pred": round(conf[b].item(), 4),
                    "conf_true": round(probs[b, true_lbl].item(), 4),
                    "runner_up": class_names[runner_up],
                    "margin": round((top2.values[b, 0] - top2.values[b, 1]).item(), 4),
                }
            )
            cursor += 1
    return records


def main() -> None:
    cfg = Config.load()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names, meta = load_checkpoint(cfg.checkpoint_path, device)
    cfg.data.image_size = meta["image_size"]

    loaders, loader_class_names = build_dataloaders(cfg)
    assert loader_class_names == class_names, "class name mismatch checkpoint vs loader"

    results = {}
    for split in ("val", "test"):
        ds = loaders[split].dataset            # _TransformedSubset
        recs = _collect(
            model, loaders[split], ds.dataset.samples, ds.indices, class_names, device
        )
        results[split] = recs
        n = len(recs)
        errs = [r for r in recs if not r["correct"]]
        acc = (n - len(errs)) / max(n, 1)
        print(f"\n===== {split.upper()} =====  n={n}  acc={acc:.4f}  errors={len(errs)}")
        for r in errs:
            print(
                f"  ✗ {r['true']}  ->  {r['pred']}  "
                f"(conf_pred={r['conf_pred']}, conf_true={r['conf_true']}, "
                f"runner_up={r['runner_up']}, margin={r['margin']})\n"
                f"      {r['path']}"
            )

    # ---- per-class F1 on test --------------------------------------------
    from sklearn.metrics import classification_report

    y_true = [r["true"] for r in results["test"]]
    y_pred = [r["pred"] for r in results["test"]]
    print("\n===== TEST per-class report (classes with support) =====")
    report = classification_report(y_true, y_pred, zero_division=0, digits=4,
                                   labels=sorted(set(y_true)))
    # Only print rows where f1 < 1.0 to keep it readable
    print(report)

    # ---- most-confused pairs across val+test -----------------------------
    pair_counter = Counter()
    for split in ("val", "test"):
        for r in results[split]:
            if not r["correct"]:
                pair_counter[(r["true"], r["pred"])] += 1
    print("\n===== Confused (true -> pred) pairs across val+test =====")
    for (t, p), c in pair_counter.most_common(15):
        print(f"  {c:>2}x  {t}  ->  {p}")

    # ---- low-confidence correct predictions (fragile) on test ------------
    fragile = sorted(
        (r for r in results["test"] if r["correct"]),
        key=lambda r: r["margin"],
    )[:8]
    print("\n===== Lowest-margin CORRECT test predictions (fragile) =====")
    for r in fragile:
        print(f"  margin={r['margin']:.3f}  {r['true']}  (runner_up={r['runner_up']}, conf_true={r['conf_true']})")

    out = ROOT / "experiments" / "error_analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote per-sample records to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
