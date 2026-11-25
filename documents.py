"""Answering from an attached PDF: reading, splitting into chunks, searching.

When the user attaches a PDF, its text is read page by page (pypdf), cut into
short chunks that remember their page, and embedded once with the app's
existing embedding model (agreement.embed). For each question, the chunks
closest to the QUESTION are given to both models, which are told to answer
only from them and to cite the page; the auditor checks the answers against
the same chunks instead of Wikipedia. If no chunk is close enough, the models
are not asked at all and the reply says the document does not cover it.

The document lives only in the browser session's memory (st.session_state):
nothing is written to disk or to a database.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from agreement import embed
from retrieval import Passage

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PAGES = 50
# The embedding model reads only the first ~128 tokens (~500 characters) of a
# text, so longer chunks would be searched by their beginning only.
CHUNK_CHARS = 500
TOP_CHUNKS = 4  # how many chunks the models and the auditor get
# Chunks less similar to the question than this are not used. Measured on a
# real paper ("Attention Is All You Need", 15 pages, 97 chunks): the best chunk
# for 6 questions it answers scored 0.61-0.72; for 6 unrelated questions
# (football, cooking, Japan's population) 0.06-0.27. Questions close to the
# topic that the paper does not answer ("training cost in dollars") scored
# 0.49-0.57: no threshold can tell those apart, so the models are told to say
# when the excerpts do not contain the answer.
MIN_RELEVANCE = 0.35

NOT_FOUND = "I could not find information about this in {name}, so I can't answer it from the document."
NO_ANSWER_PASSED = (
    "The AI answers did not pass the check against {name}, "
    "so I can't give a reliable answer from the document (see the analysis on the right)."
)


class DocumentError(Exception):
    """A PDF that cannot be used; the message is shown to the user as it is."""


@dataclass
class Chunk:
    page: int  # 1-based page number
    text: str


@dataclass
class Document:
    name: str
    pages: int
    chunks: list[Chunk]
    vectors: np.ndarray  # one embedding per chunk, rows normalised to length 1


# ------------------------------------------------------------------ reading


def read_pages(data: bytes) -> list[str]:
    """The text of each page. Raises DocumentError with a readable message."""
    if len(data) > MAX_BYTES:
        raise DocumentError(f"The file is too large ({len(data) / 1e6:.0f} MB); up to 10 MB is supported.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError("The PDF is password-protected. Please attach an unprotected PDF.")
        if len(reader.pages) > MAX_PAGES:
            raise DocumentError(
                f"The PDF has {len(reader.pages)} pages; up to {MAX_PAGES} pages are supported."
            )
        return [page.extract_text() or "" for page in reader.pages]
    except DocumentError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError, TypeError) as error:
        raise DocumentError(f"The file could not be read as a PDF ({type(error).__name__}).") from None


def split_page(text: str, page: int, max_chars: int = CHUNK_CHARS) -> list[Chunk]:
    """Cut one page into chunks of whole sentences, each up to ~max_chars.

    PDF text has line breaks in the middle of sentences, so whitespace is
    joined first. A single sentence longer than max_chars (e.g. a table row)
    is cut at word boundaries.
    """
    text = " ".join(text.split())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    pieces: list[str] = []
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    chunks: list[Chunk] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(Chunk(page, current))
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(Chunk(page, current))
    return chunks


def load_document(data: bytes, name: str) -> Document:
    """Read, split and embed a PDF. Raises DocumentError if it cannot be used."""
    pages = read_pages(data)
    chunks = [chunk for number, text in enumerate(pages, start=1) for chunk in split_page(text, number)]
    if not chunks:
        raise DocumentError(
            "No readable text was found in this PDF. Scanned PDFs (images of pages) are not supported."
        )
    return Document(name=name, pages=len(pages), chunks=chunks, vectors=_normalised(embed([c.text for c in chunks])))


def _normalised(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


# ------------------------------------------------------------------ searching


def source_label(name: str, page: int) -> str:
    """How a chunk is cited, e.g. "report.pdf, page 4"."""
    return f"{name}, page {page}"


def find_passages(
    document: Document, question: str, top_k: int = TOP_CHUNKS, min_score: float = MIN_RELEVANCE
) -> list[Passage]:
    """The chunks closest to the question, best first, as Passages ([] if none is close).

    Each passage's title is its citation ("report.pdf, page 4"); it has no URL.
    """
    query = _normalised(embed([question]))[0]
    scores = document.vectors @ query  # cosine similarity: all rows have length 1
    best = np.argsort(scores)[::-1][:top_k]
    return [
        Passage(
            title=source_label(document.name, document.chunks[i].page),
            url="",
            text=document.chunks[i].text,
            score=float(scores[i]),
        )
        for i in best
        if scores[i] >= min_score
    ]


def document_question(question: str, passages: list[Passage]) -> str:
    """The message the models get instead of the bare question.

    The original question is what the conversation remembers; this longer
    text is sent only for the current answer.
    """
    excerpts = "\n\n".join(f"[{p.title}]\n{p.text}" for p in passages)
    return (
        "Answer the QUESTION using ONLY the EXCERPTS from the user's document below. "
        "After every claim, cite its source in parentheses exactly as written in the "
        "brackets, for example (report.pdf, page 4). If the excerpts do not contain the "
        "answer, say that the document does not contain this information; do not guess "
        "and do not use outside knowledge. Begin your answer with \"The document says\" "
        "(never with Yes or No), because you report what the document says, not whether "
        "it is true.\n\n"
        f"EXCERPTS:\n{excerpts}\n\nQUESTION: {question}"
    )
