"""Cosine similarity and the agree/disagree decision."""

import numpy as np
import pytest

import agreement
from agreement import compare_answers, cosine_similarity


def test_same_direction_is_one():
    # b is twice as long as a but points the same way: same "meaning".
    assert cosine_similarity(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.0)


def test_orthogonal_is_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_opposite_is_minus_one():
    assert cosine_similarity(np.array([1.0, 1.0]), np.array([-1.0, -1.0])) == pytest.approx(-1.0)


def test_zero_vector_does_not_divide_by_zero():
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


def fake_embed(vectors):
    """Replace the real embedding model with fixed vectors (fast, no download)."""
    return lambda texts: np.array(vectors, dtype=float)


def test_agree_at_exactly_the_threshold(monkeypatch):
    # cos between [1, 0] and [0.8, 0.6] is exactly 0.8 (both have length 1).
    monkeypatch.setattr(agreement, "embed", fake_embed([[1, 0], [0.8, 0.6]]))
    result = compare_answers("a", "b", threshold=0.8)
    assert result.score == pytest.approx(0.8)
    assert result.agree  # "at or above the threshold" counts as agreeing


def test_disagree_below_the_threshold(monkeypatch):
    monkeypatch.setattr(agreement, "embed", fake_embed([[1, 0], [0, 1]]))
    assert not compare_answers("a", "b", threshold=0.8).agree
