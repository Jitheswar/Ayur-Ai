"""Offline NLP layer: knowledge-base lookup + TF-IDF semantic retrieval."""

from .knowledge_base import KnowledgeBase, Plant
from .retriever import PlantRetriever

__all__ = ["KnowledgeBase", "Plant", "PlantRetriever"]
