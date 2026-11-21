"""Evidence for the hallucination check: Wikipedia search + embedding re-ranking.

A small two-stage RAG pipeline:
  1. Retrieve: ask Wikipedia's search for pages about the QUESTION (cheap, broad).
  2. Re-rank:  split those pages into short passages and keep the ones whose
               embeddings are closest to the ANSWER (precise, local).
We search with the question because it names the topic; we re-rank with the
answer because the answer's claims are what the auditor must verify.
Georgian questions are searched with an English translation (English Wikipedia
is far richer); the multilingual embeddings can still compare English
passages with a Georgian answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

from agreement import cosine_similarity, embed

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
SEARCH_PAGES = 3  # how many pages to fetch from Wikipedia search
TOP_PASSAGES = 3  # how many passages to give the auditor
SENTENCES_PER_PASSAGE = 3  # short passages keep the auditor's prompt small and focused
TIMEOUT_SECONDS = 10
# Wikipedia search always returns *something*: for "hello how are you" it finds
# songs called "Hello". Passages below this cosine score are dropped, because
# judging an answer "against" unrelated text makes the auditor flag harmless
# answers as hallucinations. Measured with the multilingual embedding model:
# English answer vs. real evidence 0.64-0.72, English small talk vs. songs < 0.35;
# Georgian answer vs. English evidence 0.40-0.49 (cross-language scores are lower,
# so an off-topic passage can occasionally slip through for Georgian).
MIN_RELEVANCE = 0.35


def is_georgian(text: str) -> bool:
    """True if the text contains Georgian letters (Unicode block U+10A0-U+10FF).

    Checking the alphabet is enough here: English questions never contain
    Georgian letters, and a Georgian question with one English word in it
    ("რა არის Python?") still counts as Georgian.
    """
    return any("Ⴀ" <= char <= "ჿ" for char in text)


@dataclass
class Passage:
    title: str  # Wikipedia page title
    url: str  # link shown in the UI
    text: str  # a few sentences from the page
    score: float = 0.0  # cosine similarity to the answer (set by rerank)


def search_wikipedia(query: str, user_agent: str) -> list[tuple[str, str, str]]:
    """Return (title, url, intro text) for the top Wikipedia search results.

    One API call does both the search (generator=search) and fetches each
    page's plain-text introduction (prop=extracts). Raises on network errors.
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": SEARCH_PAGES,
        "prop": "extracts|info",
        "exintro": 1,  # only the introduction: the densest summary of a page
        "explaintext": 1,  # plain text instead of HTML
        "exlimit": SEARCH_PAGES,
        "inprop": "url",
        "redirects": 1,
    }
    # Wikimedia asks every client to identify itself with a descriptive User-Agent.
    response = requests.get(
        WIKIPEDIA_API, params=params, headers={"User-Agent": user_agent}, timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    pages.sort(key=lambda page: page.get("index", 0))  # keep Wikipedia's ranking
    return [
        (page["title"], page.get("fullurl", ""), page.get("extract", ""))
        for page in pages
        if page.get("extract")
    ]


def split_passages(text: str, sentences_per_passage: int = SENTENCES_PER_PASSAGE) -> list[str]:
    """Cut a text into chunks of a few sentences each."""
    # Split after ., ! or ? followed by whitespace. Simple, and good enough
    # for encyclopedia prose (it may split at "e.g. " now and then).
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [
        " ".join(sentences[i : i + sentences_per_passage])
        for i in range(0, len(sentences), sentences_per_passage)
    ]


def rerank(
    answer: str,
    passages: list[Passage],
    top_k: int = TOP_PASSAGES,
    min_score: float = MIN_RELEVANCE,
) -> list[Passage]:
    """Keep the `top_k` passages whose meaning is closest to the answer.

    Passages scoring below `min_score` are dropped first. If none are left
    (e.g. small talk, where Wikipedia has nothing relevant), the result is []
    and the audit runs without evidence instead of with misleading evidence.
    """
    if not passages:
        return []
    vectors = embed([answer] + [passage.text for passage in passages])
    answer_vector = vectors[0]
    for passage, vector in zip(passages, vectors[1:]):
        passage.score = cosine_similarity(answer_vector, vector)
    relevant = [passage for passage in passages if passage.score >= min_score]
    return sorted(relevant, key=lambda passage: passage.score, reverse=True)[:top_k]


def get_evidence(query: str, answer: str, user_agent: str) -> list[Passage]:
    """Search, split and re-rank. Returns [] if Wikipedia has nothing relevant.

    query: what to search for (the question, in English).
    answer: what the passages are ranked against (may be in any language).
    """
    passages = [
        Passage(title=title, url=url, text=chunk)
        for title, url, intro in search_wikipedia(query, user_agent)
        for chunk in split_passages(intro)
    ]
    return rerank(answer, passages)


##cd /Users/tamarkilasonia/modelproof-py
##source .venv/bin/activate
##pytest -q
##streamlit run app.py