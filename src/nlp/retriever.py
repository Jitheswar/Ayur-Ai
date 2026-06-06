"""TF-IDF semantic retrieval over the plant knowledge base.

This is the "NLP" half of the project and is fully offline -- no LLM and no
network calls.  It supports natural-language queries such as
``"which leaf helps with cough and cold?"`` by ranking plants on the cosine
similarity between the query and each plant's textual profile.

It also powers the reverse direction of the app: given a predicted plant, you
can surface other plants with similar medicinal use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .knowledge_base import KnowledgeBase, Plant


@dataclass
class RetrievalHit:
    plant: Plant
    score: float


class PlantRetriever:
    """Rank plants by relevance to a free-text medicinal query."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        # Imported lazily so the knowledge base can be used without sklearn.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.plants: list[Plant] = list(kb.plants)
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        corpus = [p.search_text() for p in self.plants]
        self._matrix = self._vectorizer.fit_transform(corpus)

    @classmethod
    def from_path(cls, kb_path: str | Path) -> "PlantRetriever":
        return cls(KnowledgeBase.load(kb_path))

    def query(self, text: str, top_k: int = 5, min_score: float = 0.0) -> list[RetrievalHit]:
        """Return the ``top_k`` plants most relevant to ``text``."""
        from sklearn.metrics.pairwise import cosine_similarity

        if not text.strip():
            return []
        q_vec = self._vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self._matrix).ravel()
        ranked = sims.argsort()[::-1]
        hits: list[RetrievalHit] = []
        for idx in ranked[:top_k]:
            score = float(sims[idx])
            if score <= min_score:
                continue
            hits.append(RetrievalHit(plant=self.plants[idx], score=score))
        return hits

    def similar_to(self, plant: Plant, top_k: int = 3) -> list[RetrievalHit]:
        """Plants with a medicinal profile similar to ``plant`` (excludes itself)."""
        hits = self.query(plant.search_text(), top_k=top_k + 1)
        return [h for h in hits if h.plant.id != plant.id][:top_k]


def _demo() -> None:
    """Quick manual check: ``python -m src.nlp.retriever``."""
    from ..config import Config

    cfg = Config.load()
    retriever = PlantRetriever.from_path(cfg.knowledge_base_path)
    print(f"Loaded {len(retriever.plants)} plants.\n")
    for q in ["cough and cold relief", "lower blood sugar diabetes", "skin acne and wounds"]:
        print(f"Query: {q!r}")
        for hit in retriever.query(q, top_k=3):
            print(f"   {hit.score:0.3f}  {hit.plant.display_name}")
        print()


if __name__ == "__main__":
    _demo()
