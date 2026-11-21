"""Do the two models agree? Measured with sentence embeddings + cosine similarity.

An embedding turns a text into a list of numbers (a vector) so that texts
with similar meaning get vectors pointing in similar directions. Cosine
similarity measures that direction: 1.0 = same meaning, ~0 = unrelated.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# Multilingual (50+ languages, including Georgian), ~470 MB, free and local.
# The English-only all-MiniLM-L6-v2 scored a Georgian sentence and its English
# translation at only 0.21 (it saw letters, not meaning); this model gives 0.59
# while unrelated pairs stay around 0.2, and English results are unchanged.
# Texts longer than ~128 tokens are cut off.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Agreement:
    score: float  # cosine similarity between the two answers
    agree: bool  # score >= threshold


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once; loading takes seconds, using it is fast.

    The first run downloads the model from Hugging Face; later runs use the
    local copy. Retrieval (Phase 5) reuses this same model.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Return one embedding vector per text (one row per text)."""
    return get_embedder().encode(texts, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors: dot(a, b) / (|a| * |b|)."""
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms == 0:  # an all-zero vector has no direction to compare
        return 0.0
    return float(np.dot(a, b) / norms)


def compare_answers(answer_a: str, answer_b: str, threshold: float) -> Agreement:
    """Embed both answers and decide whether they agree."""
    vectors = embed([answer_a, answer_b])
    score = cosine_similarity(vectors[0], vectors[1])
    return Agreement(score=score, agree=score >= threshold)
