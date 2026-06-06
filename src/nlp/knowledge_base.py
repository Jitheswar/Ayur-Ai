"""Load the curated Ayurvedic plant knowledge base and look plants up by the
label produced by the image classifier.

The classifier's class names are the dataset's folder names (e.g.
``"Ocimum Tenuiflorum (Tulsi)"``).  We index every plant under all of its
aliases (folder names, botanical name, common names, id) after normalisation,
and fall back to fuzzy matching so small naming differences still resolve.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Optional


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    ``"Ocimum Tenuiflorum (Tulsi)"`` -> ``"ocimum tenuiflorum tulsi"``.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Plant:
    id: str
    botanical_name: str
    common_names: list[str]
    ayurvedic_name: str
    family: str
    parts_used: list[str]
    medicinal_properties: list[str]
    uses: list[str]
    description: str
    precautions: str
    folder_aliases: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        primary = self.common_names[0] if self.common_names else self.botanical_name
        return f"{primary} ({self.botanical_name})"

    def search_text(self) -> str:
        """Concatenated text used to build the retrieval corpus."""
        parts = [
            " ".join(self.common_names),
            self.ayurvedic_name,
            self.botanical_name,
            " ".join(self.medicinal_properties),
            " ".join(self.uses),
            self.description,
        ]
        return " ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "botanical_name": self.botanical_name,
            "common_names": self.common_names,
            "ayurvedic_name": self.ayurvedic_name,
            "family": self.family,
            "parts_used": self.parts_used,
            "medicinal_properties": self.medicinal_properties,
            "uses": self.uses,
            "description": self.description,
            "precautions": self.precautions,
        }


class KnowledgeBase:
    """In-memory index of plants keyed by normalised aliases."""

    def __init__(self, plants: list[Plant]):
        self.plants = plants
        self._by_id: dict[str, Plant] = {p.id: p for p in plants}
        self._alias_index: dict[str, Plant] = {}
        for plant in plants:
            for alias in self._aliases_for(plant):
                self._alias_index.setdefault(normalize(alias), plant)

    @staticmethod
    def _aliases_for(plant: Plant) -> list[str]:
        aliases = [plant.id, plant.botanical_name, plant.ayurvedic_name]
        aliases += plant.common_names
        aliases += plant.folder_aliases
        # Botanical genus + species without any parenthetical common name.
        return [a for a in aliases if a and a != "—"]

    # -- loading ---------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        plants = [
            Plant(
                id=p["id"],
                botanical_name=p.get("botanical_name", ""),
                common_names=p.get("common_names", []),
                ayurvedic_name=p.get("ayurvedic_name", ""),
                family=p.get("family", ""),
                parts_used=p.get("parts_used", []),
                medicinal_properties=p.get("medicinal_properties", []),
                uses=p.get("uses", []),
                description=p.get("description", ""),
                precautions=p.get("precautions", ""),
                folder_aliases=p.get("folder_aliases", []),
            )
            for p in data["plants"]
        ]
        return cls(plants)

    # -- lookup ----------------------------------------------------------
    def get(self, plant_id: str) -> Optional[Plant]:
        return self._by_id.get(plant_id)

    def lookup(self, label: str, fuzzy: bool = True) -> Optional[Plant]:
        """Resolve a classifier class name (or any name) to a Plant.

        Tries exact normalised match first, then a fuzzy match against all
        known aliases.  Returns ``None`` if nothing reasonable is found.
        """
        key = normalize(label)
        if key in self._alias_index:
            return self._alias_index[key]

        # Try the parenthetical-stripped form, e.g. drop "(Tulsi)".
        stripped = normalize(re.sub(r"\(.*?\)", " ", label))
        if stripped and stripped in self._alias_index:
            return self._alias_index[stripped]

        if fuzzy and self._alias_index:
            match = get_close_matches(key, self._alias_index.keys(), n=1, cutoff=0.6)
            if match:
                return self._alias_index[match[0]]
        return None

    def __len__(self) -> int:
        return len(self.plants)

    def __iter__(self):
        return iter(self.plants)
