"""Sanity-check the dataset layout and its overlap with the knowledge base.

    python scripts/check_data.py

Reports the number of classes and images, flags tiny/empty classes, and tells
you which class folders do NOT yet have a matching knowledge-base entry (so the
app would show "no entry matched" for those predictions).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.nlp.knowledge_base import KnowledgeBase  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    cfg = Config.load()
    raw = cfg.raw_dir
    print(f"Data folder : {raw}")
    print(f"Knowledge base: {cfg.knowledge_base_path}\n")

    if not raw.exists():
        print("✗ data/raw does not exist. Create it and add one folder per class.")
        return 1

    class_dirs = sorted([d for d in raw.iterdir() if d.is_dir()])
    if not class_dirs:
        print("✗ No class sub-folders found in data/raw.")
        print("  Expected: data/raw/<Plant Name>/<images...>")
        return 1

    kb = KnowledgeBase.load(cfg.knowledge_base_path)

    counts: Counter[str] = Counter()
    unmatched = []
    for d in class_dirs:
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        counts[d.name] = n
        # Exact-alias match only: fuzzy matching would mask genuinely missing
        # entries, so the coverage warning below would never fire.
        if kb.lookup(d.name, fuzzy=False) is None:
            unmatched.append(d.name)

    total = sum(counts.values())
    print(f"Classes: {len(class_dirs)} | Total images: {total}\n")
    for name, n in counts.items():
        flag = "  ⚠ few images" if n < 10 else ""
        matched = "" if kb.lookup(name, fuzzy=False) else "  ⚠ no KB entry"
        print(f"  {n:4d}  {name}{flag}{matched}")

    empty = [n for n, c in counts.items() if c == 0]
    if empty:
        print(f"\n✗ {len(empty)} empty class folder(s): {empty}")
    if unmatched:
        print(
            f"\n⚠ {len(unmatched)} class(es) have no knowledge-base match: "
            f"{unmatched}\n  Add them to data/knowledge_base/plants.json "
            "(or fix the folder name) so properties show up."
        )
    if not empty and not unmatched:
        print("\n✓ All class folders have images and a knowledge-base entry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
