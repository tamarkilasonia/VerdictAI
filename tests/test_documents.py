"""Attached PDFs: reading, limits, chunks with page numbers, search, and the
graph's document mode. Small PDFs are built in code; the embedding model is
replaced with a word-count fake, so no download or internet is needed."""

import io

import numpy as np
import pytest
from pypdf import PdfWriter

import documents
import graph
from answers import ModelAnswer
from audit_log import build_entry
from auditor import CheckOutcome, CheckResult
from config import ModelSpec, Settings
from documents import DocumentError, find_passages, load_document, split_page
from retrieval import Passage

SETTINGS = Settings(
    primary=ModelSpec("p", "https://p.invalid", "k"),
    secondary=ModelSpec("s", "https://s.invalid", "k"),
    auditor=ModelSpec("a", "https://a.invalid", "k"),
    agreement_threshold=0.8,
    history_messages=6,
    wiki_user_agent="test",
)


# ------------------------------------------------------------------ helpers


def make_pdf(pages: list[str]) -> bytes:
    """A minimal, valid PDF with one line of Helvetica text per sentence."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        lines = [s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for s in text.split(". ") if s]
        stream = "BT /F1 11 Tf 14 TL 72 760 Td " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content = len(objects)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content} 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def blank_pdf(pages: int, password: str | None = None) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    if password:
        writer.encrypt(password, algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


VOCAB = ["revenue", "million", "employees", "office", "tbilisi", "football", "cake", "profit"]


def fake_embed(texts):
    """One dimension per vocabulary word: texts sharing words point the same way."""
    vectors = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=float)
    vectors[:, 0] += 1e-9  # no all-zero rows
    return vectors


@pytest.fixture(autouse=True)
def no_real_embeddings(monkeypatch):
    monkeypatch.setattr(documents, "embed", fake_embed)


REPORT = [
    "Annual report 2025. The company had revenue of 12 million dollars. Profit was 2 million dollars.",
    "The company has 85 employees. Its main office is in Tbilisi.",
]


# ------------------------------------------------------------------ reading


def test_pdf_text_is_read_page_by_page():
    pages = documents.read_pages(make_pdf(REPORT))
    assert len(pages) == 2
    assert "12 million" in pages[0] and "Tbilisi" in pages[1]


def test_chunks_remember_their_page():
    document = load_document(make_pdf(REPORT), "report.pdf")
    assert document.name == "report.pdf" and document.pages == 2
    assert {chunk.page for chunk in document.chunks} == {1, 2}
    assert any("Tbilisi" in c.text and c.page == 2 for c in document.chunks)


def test_too_large_file_is_refused():
    with pytest.raises(DocumentError, match="too large"):
        documents.read_pages(b"0" * (documents.MAX_BYTES + 1))


def test_too_many_pages_are_refused():
    with pytest.raises(DocumentError, match="51 pages; up to 50"):
        documents.read_pages(blank_pdf(51))


def test_pdf_without_text_is_refused_as_scanned():
    with pytest.raises(DocumentError, match="No readable text"):
        load_document(blank_pdf(3), "scan.pdf")


def test_broken_file_is_refused():
    with pytest.raises(DocumentError, match="could not be read as a PDF"):
        documents.read_pages(b"this is not a pdf at all")


def test_password_protected_pdf_is_refused():
    with pytest.raises(DocumentError, match="password-protected"):
        documents.read_pages(blank_pdf(1, password="secret"))


# ------------------------------------------------------------------ chunks


def test_chunks_join_broken_lines_and_keep_whole_sentences():
    chunks = split_page("First sentence\nis here. Second one.", page=3, max_chars=500)
    assert [(c.page, c.text) for c in chunks] == [(3, "First sentence is here. Second one.")]


def test_chunks_stay_short():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = split_page(text, page=1, max_chars=100)
    assert len(chunks) > 1 and all(len(c.text) <= 100 for c in chunks)


def test_a_very_long_sentence_is_cut_at_words():
    chunks = split_page("word " * 300, page=1, max_chars=100)
    assert all(len(c.text) <= 100 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == ["word"] * 300


def test_empty_page_gives_no_chunks():
    assert split_page("   \n ", page=1) == []


# ------------------------------------------------------------------ search


def test_closest_chunk_is_found_with_its_page():
    document = load_document(make_pdf(REPORT), "report.pdf")
    best = find_passages(document, "How many employees work there?")[0]
    assert best.title == "report.pdf, page 2"
    assert "85 employees" in best.text and best.url == ""


def test_unrelated_question_finds_nothing():
    document = load_document(make_pdf(REPORT), "report.pdf")
    assert find_passages(document, "Who won the football match?") == []


def test_at_most_top_k_passages():
    document = load_document(make_pdf(REPORT * 5), "report.pdf")
    assert len(find_passages(document, "revenue profit employees", top_k=2, min_score=0.0)) == 2


def test_models_are_told_to_use_only_the_excerpts_and_cite_pages():
    passages = [Passage("report.pdf, page 2", "", "The company has 85 employees.")]
    text = documents.document_question("How many employees?", passages)
    assert "[report.pdf, page 2]\nThe company has 85 employees." in text
    assert "ONLY the EXCERPTS" in text and "(report.pdf, page 4)" in text
    assert "does not contain this information" in text
    assert text.endswith("QUESTION: How many employees?")


# ------------------------------------------------------------ graph: document mode

PASSAGES = [Passage("report.pdf, page 2", "", "The company has 85 employees.", score=0.7)]


def test_models_get_the_document_parts_with_the_question(monkeypatch):
    sent = []

    def fake_ask_both(primary, secondary, messages):
        sent.append(messages[-1].content)
        return ModelAnswer("p", text="85 (report.pdf, page 2)"), ModelAnswer("s", text="85.")

    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "ask_both", fake_ask_both)
    state = {**graph.new_question_input("How many employees?", "report.pdf", PASSAGES), "messages": []}
    graph.generate_answers(state)
    assert "[report.pdf, page 2]" in sent[0] and sent[0].endswith("QUESTION: How many employees?")


def test_document_parts_are_the_evidence_and_wikipedia_is_not_used(monkeypatch):
    def no_wikipedia(*args, **kwargs):
        raise AssertionError("Wikipedia must not be searched when a document is attached")

    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "get_evidence", no_wikipedia)
    monkeypatch.setattr(graph, "make_search_query", no_wikipedia)
    monkeypatch.setattr(graph, "wikipedia_answer", no_wikipedia)
    state = {
        **graph.new_question_input("How many employees?", "report.pdf", PASSAGES),
        "primary": ModelAnswer("p", text="85."),
        "secondary": ModelAnswer("s", text="85."),
    }
    update = graph.retrieve_evidence(state)
    assert update["evidence"] == PASSAGES
    assert graph._evidence_texts({**state, **update}) == ["[report.pdf, page 2] The company has 85 employees."]


def test_nothing_found_skips_the_models_and_says_so(monkeypatch):
    def no_models(*args):
        raise AssertionError("the models must not be asked")

    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "ask_both", no_models)
    state = {**graph.new_question_input("Who won the football match?", "report.pdf", []), "messages": []}
    update = graph.generate_answers(state)
    assert update["not_in_document"] is True
    assert graph.route_after_answers({**state, **update}) == "finalize"
    turn = graph.turn_from_state({**state, **update})
    assert turn.reply == (
        "I could not find information about this in report.pdf, so I can't answer it from the document."
    )
    assert turn.shown is None


def test_failed_document_answers_do_not_fall_back_to_wikipedia():
    failed = {"hallucination": CheckOutcome("hallucination", CheckResult(score=0.9))}
    turn = graph.Turn(
        "How many employees?", ModelAnswer("p", text="500."), ModelAnswer("s", text="600."),
        audit=failed, secondary_audit=failed, document_name="report.pdf",
        wiki_answer=Passage("Employees", "https://en.wikipedia.org/x", "Unrelated."),
    )
    assert "report.pdf" in turn.reply and "Wikipedia" not in turn.reply


def test_passing_document_answer_is_shown():
    passed = {"hallucination": CheckOutcome("hallucination", CheckResult(score=0.0))}
    turn = graph.Turn(
        "How many employees?", ModelAnswer("p", text="85 (report.pdf, page 2)."),
        ModelAnswer("s", error="down"), audit=passed, document_name="report.pdf",
    )
    assert turn.reply == "85 (report.pdf, page 2)."


def test_without_a_document_nothing_changes():
    state = graph.new_question_input("Who painted The Night Watch?")
    assert state["document_name"] == "" and state["document_passages"] == []
    assert graph.route_after_answers({**state, "error": None}) == "compare"


def test_log_records_the_document_and_cited_pages():
    passed = {"hallucination": CheckOutcome("hallucination", CheckResult(score=0.0))}
    turn = graph.Turn(
        "How many employees?", ModelAnswer("p", text="85."), ModelAnswer("s", error="down"),
        audit=passed, evidence=PASSAGES, document_name="report.pdf",
    )
    entry = build_entry(turn, SETTINGS)
    assert entry["document"] == "report.pdf"
    assert entry["document_sources"] == ["report.pdf, page 2"]
    assert entry["evidence_urls"] == []


def test_log_without_a_document():
    turn = graph.Turn("Q?", ModelAnswer("p", text="A."), ModelAnswer("s", error="down"))
    entry = build_entry(turn, SETTINGS)
    assert entry["document"] is None and entry["document_sources"] is None
