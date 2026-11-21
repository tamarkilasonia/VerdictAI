"""Splitting pages into passages and re-ranking them (no network needed)."""

import numpy as np

import retrieval
from retrieval import Passage, rerank, split_passages


def test_split_groups_sentences():
    text = "One. Two! Three? Four. Five."
    assert split_passages(text, sentences_per_passage=3) == ["One. Two! Three?", "Four. Five."]


def test_split_empty_text():
    assert split_passages("") == []


def test_rerank_orders_by_similarity_and_keeps_top_k(monkeypatch):
    # Vector 0 is the answer; passages b, a, c are increasingly far from it.
    vectors = np.array([[1, 0], [0.6, 0.8], [1, 0.1], [0, 1]], dtype=float)
    monkeypatch.setattr(retrieval, "embed", lambda texts: vectors)
    passages = [Passage("A", "url-a", "a"), Passage("B", "url-b", "b"), Passage("C", "url-c", "c")]

    best = rerank("answer", passages, top_k=2)

    assert [p.title for p in best] == ["B", "A"]
    assert best[0].score > best[1].score


def test_rerank_with_no_passages():
    assert rerank("answer", []) == []


def test_rerank_drops_off_topic_passage(monkeypatch):
    # Cosine to the answer [1, 0]: "Relevant" = 0.8, "Song" = 0.1 (below MIN_RELEVANCE).
    vectors = np.array([[1, 0], [0.8, 0.6], [0.1, 0.995]], dtype=float)
    monkeypatch.setattr(retrieval, "embed", lambda texts: vectors)
    passages = [Passage("Relevant", "url-r", "r"), Passage("Song", "url-s", "s")]

    kept = rerank("answer", passages)

    assert [p.title for p in kept] == ["Relevant"]


def test_rerank_returns_nothing_when_all_passages_are_off_topic(monkeypatch):
    # Small talk: Wikipedia finds only unrelated pages, so there is no evidence.
    vectors = np.array([[1, 0], [0.2, 0.98], [0.0, 1.0]], dtype=float)
    monkeypatch.setattr(retrieval, "embed", lambda texts: vectors)
    passages = [Passage("Hello (song)", "url-1", "a"), Passage("Hello (film)", "url-2", "b")]

    assert rerank("hello, how are you?", passages) == []
