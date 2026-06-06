"""Ayurvedic Leaf Detection and Medicinal Property Retrieval.

A two-stage system:
  1. A CNN (transfer learning) classifies a plant from a leaf image.
  2. An offline NLP layer (knowledge base + TF-IDF retrieval) returns the
     plant's medicinal properties -- no LLM dependency.
"""

__version__ = "1.0.0"
