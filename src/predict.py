"""End-to-end inference: leaf image -> plant -> medicinal properties.

This ties the two halves of the project together:
  * the CNN predicts the plant class from the image, and
  * the knowledge base maps that class to its medicinal properties.

CLI::

    python -m src.predict path/to/leaf.jpg
    python -m src.predict path/to/leaf.jpg --topk 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .config import Config
from .data import inference_transform
from .model import load_checkpoint
from .nlp.knowledge_base import KnowledgeBase, Plant


@dataclass
class Prediction:
    label: str
    confidence: float
    plant: Optional[Plant]


class LeafPredictor:
    """Loads a checkpoint + knowledge base once and reuses them per image."""

    def __init__(self, cfg: Optional[Config] = None, device: Optional[str] = None,
                 checkpoint: Optional[str] = None):
        self.cfg = cfg or Config.load()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(checkpoint) if checkpoint else self.cfg.checkpoint_path
        self.model, self.class_names, self.meta = load_checkpoint(ckpt, self.device)
        self.transform = inference_transform(self.meta["image_size"])
        self.kb = KnowledgeBase.load(self.cfg.knowledge_base_path)

    @torch.no_grad()
    def predict(self, image_path: str | Path, topk: int = 3) -> list[Prediction]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1).squeeze(0)
        k = min(topk, len(self.class_names))
        confidences, indices = probs.topk(k)
        results = []
        for conf, idx in zip(confidences.tolist(), indices.tolist()):
            label = self.class_names[idx]
            results.append(
                Prediction(label=label, confidence=conf, plant=self.kb.lookup(label))
            )
        return results


def format_prediction(pred: Prediction) -> str:
    lines = [f"  {pred.confidence*100:5.1f}%  {pred.label}"]
    if pred.plant:
        p = pred.plant
        lines.append(f"          Botanical : {p.botanical_name}")
        lines.append(f"          Ayurvedic : {p.ayurvedic_name}")
        lines.append(f"          Properties: {', '.join(p.medicinal_properties)}")
        lines.append(f"          Uses      : {', '.join(p.uses)}")
        if p.precautions:
            lines.append(f"          Caution   : {p.precautions}")
    else:
        lines.append("          (no knowledge-base entry matched this class)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify a leaf and list its uses.")
    parser.add_argument("image", help="path to a leaf image")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    predictor = LeafPredictor(checkpoint=args.checkpoint)
    preds = predictor.predict(args.image, topk=args.topk)

    print(f"\nImage: {args.image}")
    print("Top predictions:\n")
    for pred in preds:
        print(format_prediction(pred))
        print()

    top = preds[0]
    print("=" * 60)
    if top.plant:
        print(f"Most likely: {top.plant.display_name}")
        print(top.plant.description)
    print("Disclaimer: educational use only -- not medical advice.")


if __name__ == "__main__":
    main()
