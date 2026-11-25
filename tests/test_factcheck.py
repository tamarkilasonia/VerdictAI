"""Fact-checking an attached PDF against Wikipedia (factcheck.py).

No AI service or Wikipedia is called: the extraction reply, the Wikipedia
search, the auditor and the embeddings are all faked.
"""

import numpy as np
import pytest

import documents
import factcheck
from auditor import CheckOutcome, CheckResult, Issue
from config import ModelSpec, Settings
from documents import Chunk, Document
from factcheck import (
    CONFIRMED,
    CONTRADICTED,
    UNCHECKED,
    Claim,
    ClaimResult,
    FactCheck,
    choose_claims,
    match_issue,
    parse_claims,
)
from retrieval import Passage

SETTINGS = Settings(
    primary=ModelSpec("openai/gpt-oss-20b", "https://p.invalid", "k"),
    secondary=ModelSpec("s", "https://s.invalid", "k"),
    auditor=ModelSpec("a", "https://a.invalid", "k"),
    agreement_threshold=0.8,
    history_messages=6,
    wiki_user_agent="test",
)

CAESAR = Claim(1, "Julius Caesar was assassinated in 44 BC.", "Julius Caesar")
MAGNA = Claim(1, "King John agreed to the Magna Carta in 1215.", "Magna Carta")
TAMAR = Claim(2, "Queen Tamar ruled Georgia from 1184 to 1213.", "Tamar of Georgia")
WIKI = {
    CAESAR.text: [Passage("Julius Caesar", "https://en.wikipedia.org/wiki/Julius_Caesar", "He was killed in 44 BC.")],
    MAGNA.text: [Passage("Magna Carta", "https://en.wikipedia.org/wiki/Magna_Carta", "Sealed by King John in 1215.")],
    TAMAR.text: [],
}


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    monkeypatch.setattr(factcheck, "get_settings", lambda: SETTINGS)


def fake_audit(issues=(), error=None, calls=None):
    """A run_audit replacement returning these hallucination issues."""

    def run_audit(question, answer, conversation, evidence):
        if calls is not None:
            calls.append((answer, evidence))
        if error:
            return {"hallucination": CheckOutcome("hallucination", error=error)}
        result = CheckResult(score=0.9 if issues else 0.0, issues=list(issues))
        return {"hallucination": CheckOutcome("hallucination", result)}

    return run_audit


# ------------------------------------------------------------------ extraction


def test_claims_are_parsed_with_page_and_query():
    reply = '{"claims": [{"page": 2, "claim": "Queen  Tamar ruled Georgia.", "query": "Tamar of Georgia"}]}'
    assert parse_claims(reply, {1, 2}) == [Claim(2, "Queen Tamar ruled Georgia.", "Tamar of Georgia")]


def test_reply_cut_off_by_the_length_limit_keeps_the_complete_claims():
    reply = '{"claims":[{"page":1,"claim":"A was built in 2560 BC.","query":"A"},{"page":1,"claim":"B was'
    assert parse_claims(reply, {1}) == [Claim(1, "A was built in 2560 BC.", "A")]


def test_bad_entries_are_skipped_and_a_wrong_page_is_corrected():
    reply = (
        '```json\n{"claims": [{"page": 9, "claim": "C happened.", "query": ""}, '
        '{"claim": ""}, "junk", {"page": "x", "claim": "D happened."}]}\n```'
    )
    claims = parse_claims(reply, {3, 4})
    assert [(c.page, c.text, c.query) for c in claims] == [(3, "C happened.", "C happened."), (3, "D happened.", "D happened.")]


def test_no_json_gives_no_claims():
    assert parse_claims("Sorry, I cannot help.", {1}) == []


def test_page_texts_are_rebuilt_from_the_chunks():
    document = Document("d.pdf", 2, [Chunk(1, "A."), Chunk(1, "B."), Chunk(2, "C.")], np.zeros((3, 2)))
    assert factcheck.page_texts(document) == [(1, "A. B."), (2, "C.")]


def test_claims_are_extracted_a_few_pages_at_a_time_without_duplicates(monkeypatch):
    document = Document("d.pdf", 4, [Chunk(p, f"Page {p} text.") for p in (1, 2, 3, 4)], np.zeros((4, 2)))
    replies = iter([
        '{"claims": [{"page": 1, "claim": "Same fact.", "query": "q"}, {"page": 3, "claim": "Third.", "query": "q"}]}',
        '{"claims": [{"page": 4, "claim": "same fact.", "query": "q"}]}',
    ])
    sent = []

    class FakeModel:
        def invoke(self, messages):
            sent.append(messages[1].content)

            class Reply:
                content = next(replies)

            return Reply()

    options = {}

    def fake_chat_model(spec, **kwargs):
        options.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(factcheck, "get_chat_model", fake_chat_model)
    claims = factcheck.extract_claims(document)
    assert [c.text for c in claims] == ["Same fact.", "Third."]
    assert len(sent) == 2 and "[Page 1]" in sent[0] and "[Page 4]" in sent[1]
    assert options == {"temperature": 0.0, "reasoning_effort": "low"}  # no JSON mode


def test_up_to_the_limit_all_claims_are_kept():
    claims = [Claim(1, f"c{i}", "q") for i in range(5)]
    assert choose_claims(claims, 20) == claims


def test_over_the_limit_claims_come_from_every_page():
    claims = [Claim(1, f"p1-{i}", "q") for i in range(10)] + [Claim(3, f"p3-{i}", "q") for i in range(10)]
    chosen = choose_claims(claims, 6)
    assert [c.text for c in chosen] == ["p1-0", "p1-1", "p1-2", "p3-0", "p3-1", "p3-2"]


# ------------------------------------------------------------------ matching


@pytest.mark.parametrize(
    "quote, expected",
    [
        ("assassinated in 44 BC", 0),  # part of a claim
        ("2. King John agreed to the Magna Carta in 1215.", 1),  # with its list number
        ("Queen Tamar ruled Georgia from 1184 till 1213", 2),  # slightly reworded
        ("Something else entirely", None),
        ("", None),
    ],
)
def test_auditor_issues_are_matched_to_their_claim(quote, expected):
    assert match_issue(Issue(quote=quote, reason="x"), [CAESAR, MAGNA, TAMAR]) == expected


# ------------------------------------------------------------------ checking


def test_one_request_checks_several_claims(monkeypatch):
    calls = []
    issues = [Issue(quote="King John agreed", reason="CONTRADICTED: the evidence says King John sealed it.")]
    monkeypatch.setattr(factcheck, "run_audit", fake_audit(issues, calls=calls))
    claims = [CAESAR, MAGNA, TAMAR]
    results = factcheck.check_batch(claims, [WIKI[c.text] for c in claims])

    assert len(calls) == 1
    assert calls[0][0] == "1. Julius Caesar was assassinated in 44 BC.\n2. King John agreed to the Magna Carta in 1215.\n3. Queen Tamar ruled Georgia from 1184 to 1213."
    assert [r.status for r in results] == [CONFIRMED, CONTRADICTED, UNCHECKED]
    assert results[0].source_url.endswith("Julius_Caesar")
    assert results[1].correction == "the evidence says King John sealed it."
    assert results[1].source_title == "Magna Carta"
    assert results[2].note == "No relevant Wikipedia passage found."  # no passage of its own


def test_unverified_is_cant_be_checked_not_contradicted(monkeypatch):
    issue = Issue(quote="assassinated in 44 BC", reason="UNVERIFIED: the evidence does not mention the year.")
    monkeypatch.setattr(factcheck, "run_audit", fake_audit([issue]))
    [result] = factcheck.check_batch([CAESAR], [WIKI[CAESAR.text]])
    assert result.status == UNCHECKED and result.correction == ""
    assert result.note == "Wikipedia neither confirms nor denies it."


def test_mislabelled_not_mentioned_is_not_a_contradiction(monkeypatch):
    issue = Issue(quote="44 BC", reason="CONTRADICTED: the evidence does not mention the year.")
    monkeypatch.setattr(factcheck, "run_audit", fake_audit([issue]))
    [result] = factcheck.check_batch([CAESAR], [WIKI[CAESAR.text]])
    assert result.status == UNCHECKED


def test_auditor_unavailable_makes_claims_cant_be_checked(monkeypatch):
    monkeypatch.setattr(factcheck, "run_audit", fake_audit(error="Audit unavailable: limit"))
    results = factcheck.check_batch([CAESAR, MAGNA], [WIKI[CAESAR.text], WIKI[MAGNA.text]])
    assert [r.status for r in results] == [UNCHECKED, UNCHECKED]
    assert results[0].note == "Audit unavailable: limit"


def test_no_wikipedia_passages_at_all_spends_no_auditor_request(monkeypatch):
    calls = []
    monkeypatch.setattr(factcheck, "run_audit", fake_audit(calls=calls))
    results = factcheck.check_batch([TAMAR], [[]])
    assert calls == [] and results[0].status == UNCHECKED


def test_fact_check_counts_all_checks_twenty_in_batches_and_reports_progress(monkeypatch):
    claims = [Claim(1 + i // 10, f"Fact number {i}.", f"q{i}") for i in range(23)]
    calls, progress = [], []
    monkeypatch.setattr(factcheck, "extract_claims", lambda document: claims)
    monkeypatch.setattr(factcheck, "get_evidence", lambda q, a, ua: [Passage(q, f"https://w/{q}", a)])
    monkeypatch.setattr(factcheck, "run_audit", fake_audit(calls=calls))
    check = factcheck.fact_check(Document("d.pdf", 3, [], np.zeros((0, 2))), lambda *args: progress.append(args))

    assert check.total_claims == 23 and len(check.results) == 20
    assert len(calls) == 4  # 20 claims, 5 per auditor request
    assert check.count(CONFIRMED) == 20
    assert ("Searching Wikipedia…", 20, 20) in progress
    assert progress[-1] == ("Checking facts…", 20, 20)


def test_failed_wikipedia_search_does_not_break_the_check(monkeypatch):
    def broken(q, a, ua):
        raise ConnectionError("no internet")

    monkeypatch.setattr(factcheck, "extract_claims", lambda document: [CAESAR])
    monkeypatch.setattr(factcheck, "get_evidence", broken)
    monkeypatch.setattr(factcheck, "run_audit", fake_audit())
    check = factcheck.fact_check(Document("d.pdf", 1, [], np.zeros((0, 2))))
    assert check.error == "" and check.results[0].status == UNCHECKED


def test_failed_extraction_is_reported_not_raised(monkeypatch):
    def broken(document):
        raise RuntimeError("boom")

    monkeypatch.setattr(factcheck, "extract_claims", broken)
    check = factcheck.fact_check(Document("d.pdf", 1, [], np.zeros((0, 2))))
    assert check.error.startswith("The facts could not be extracted") and check.results == []


# ------------------------------------------------------------------ warnings

WRONG = ClaimResult(
    Claim(1, "Julius Caesar was assassinated in 34 BC.", "Julius Caesar"),
    CONTRADICTED,
    correction="the evidence says he was assassinated on 15 March 44 BC.",
    source_title="Assassination of Julius Caesar",
    source_url="https://en.wikipedia.org/wiki/Assassination_of_Julius_Caesar",
)
RIGHT = ClaimResult(Claim(1, "Queen Tamar ruled Georgia.", "Tamar"), CONFIRMED)
CHECK = FactCheck("w.pdf", total_claims=2, results=[WRONG, RIGHT])


def fake_embed(texts):
    vocab = ["caesar", "assassinated", "tamar", "pyramid"]
    return np.array([[t.lower().count(w) + 1e-9 for w in vocab] for t in texts])


def test_answer_repeating_a_wrong_claim_gets_a_warning(monkeypatch):
    monkeypatch.setattr(factcheck, "embed", fake_embed)
    answer = "The document says Julius Caesar was assassinated in 34 BC (w.pdf, page 1)."
    assert factcheck.warnings_for(answer, CHECK, pages={1}) == [WRONG]


def test_unrelated_answer_or_other_page_gets_no_warning(monkeypatch):
    monkeypatch.setattr(factcheck, "embed", fake_embed)
    assert factcheck.warnings_for("The document says the pyramid is old.", CHECK, pages={1}) == []
    assert factcheck.warnings_for("Caesar was assassinated.", CHECK, pages={2}) == []
    assert factcheck.warnings_for("Caesar was assassinated.", None, pages={1}) == []


def test_warning_text_has_the_claim_the_correct_fact_and_the_source():
    assert factcheck.warning_text(WRONG) == (
        'The document says "Julius Caesar was assassinated in 34 BC.", but this is wrong: '
        "the evidence says he was assassinated on 15 March 44 BC. (source: [Assassination of "
        "Julius Caesar](https://en.wikipedia.org/wiki/Assassination_of_Julius_Caesar))"
    )


def test_pages_are_read_from_document_passage_titles():
    passages = [Passage("w.pdf, page 1", "", "t"), Passage("w.pdf, page 3", "", "t"), Passage("Berlin Wall", "u", "t")]
    assert factcheck.pages_of(passages) == {1, 3}


def test_document_answers_start_with_the_document_says():
    text = documents.document_question("Was Caesar killed in 34 BC?", [Passage("w.pdf, page 1", "", "t")])
    assert 'Begin your answer with "The document says"' in text and "never with Yes or No" in text


def test_per_minute_limit_waits_and_tries_the_batch_again(monkeypatch):
    calls, waits = [], []
    limited = "Audit unavailable: Free usage limit reached for 'a'. Wait a minute and try again."
    outcomes = iter([fake_audit(error=limited), fake_audit()])

    def run_audit(*args):
        calls.append(1)
        return next(outcomes)(*args)

    monkeypatch.setattr(factcheck, "extract_claims", lambda document: [CAESAR])
    monkeypatch.setattr(factcheck, "get_evidence", lambda q, a, ua: WIKI[CAESAR.text])
    monkeypatch.setattr(factcheck, "run_audit", run_audit)
    monkeypatch.setattr(factcheck.time, "sleep", waits.append)
    check = factcheck.fact_check(Document("d.pdf", 1, [], np.zeros((0, 2))))
    assert len(calls) == 2 and waits == [factcheck.PER_MINUTE_LIMIT_WAIT]
    assert check.results[0].status == CONFIRMED


def test_daily_limit_is_not_waited_for(monkeypatch):
    waits = []
    daily = "Audit unavailable: Today's free limit for 'a' is used up."
    monkeypatch.setattr(factcheck, "extract_claims", lambda document: [CAESAR])
    monkeypatch.setattr(factcheck, "get_evidence", lambda q, a, ua: WIKI[CAESAR.text])
    monkeypatch.setattr(factcheck, "run_audit", fake_audit(error=daily))
    monkeypatch.setattr(factcheck.time, "sleep", waits.append)
    check = factcheck.fact_check(Document("d.pdf", 1, [], np.zeros((0, 2))))
    assert waits == [] and check.results[0].status == UNCHECKED


# ------------------------------------------- "is this document correct?" summary

import graph  # noqa: E402
from answers import ModelAnswer  # noqa: E402
from audit_log import build_entry  # noqa: E402

UNSURE = ClaimResult(Claim(2, "Columbus reached the Americas in 1498.", "Columbus"), UNCHECKED, note="x")
FULL = FactCheck("w.pdf", total_claims=3, results=[WRONG, RIGHT, UNSURE])


def test_summary_counts_lists_the_wrong_claims_and_points_to_the_table():
    text = factcheck.summary_reply(FULL, "w.pdf")
    assert text.startswith("I checked all 3 factual claims in **w.pdf** against Wikipedia: "
                           "**1 ✅ confirmed**, **1 ❌ contradicted**, **1 ⚠️ can't be checked**.")
    assert ('- **Page 1:** "Julius Caesar was assassinated in 34 BC." → the evidence says he was '
            "assassinated on 15 March 44 BC. ([Assassination of Julius Caesar]"
            "(https://en.wikipedia.org/wiki/Assassination_of_Julius_Caesar))") in text
    assert "it is not the same as wrong" in text  # can't be checked != contradicted
    assert "table" in text and "above the input field" in text


def test_summary_says_when_only_some_claims_were_checked():
    text = factcheck.summary_reply(FactCheck("w.pdf", total_claims=34, results=[RIGHT]), "w.pdf")
    assert "I checked 1 of 34 factual claims" in text
    assert "No claim was contradicted by Wikipedia." in text


def test_summary_without_a_fact_check_or_without_claims():
    assert "not available" in factcheck.summary_reply(None, "w.pdf")
    assert factcheck.summary_reply(FactCheck("w.pdf"), "w.pdf") == "I found no factual claims to check in w.pdf."


def test_document_check_questions_are_recognised(monkeypatch):
    # A fake embedding: questions about the whole document's accuracy point one way.
    def embed(texts):
        words = ("correct", "check", "accurate", "mistakes", "errors", "true", "verify", "reliable", "wrong", "caesar")
        return np.array([[t.lower().count(w) for w in words] + [1e-6] for t in texts], dtype=float)

    monkeypatch.setattr(factcheck, "embed", embed)
    factcheck._check_request_vectors.cache_clear()
    try:
        assert factcheck.asks_to_check_document("Is this information correct?")
        assert factcheck.asks_to_check_document("check this file")
        assert not factcheck.asks_to_check_document("Who was Caesar?")
    finally:
        factcheck._check_request_vectors.cache_clear()


def test_prepared_reply_skips_the_models_and_the_audit(monkeypatch):
    def no_models(*args):
        raise AssertionError("the models must not be asked")

    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "ask_both", no_models)
    summary = factcheck.summary_reply(FULL, "w.pdf")
    state = {**graph.new_question_input("Is this information correct?", "w.pdf", fixed_reply=summary), "messages": []}
    update = graph.generate_answers(state)
    assert "not_in_document" not in update  # no "Not in file"
    assert graph.route_after_answers({**state, **update}) == "finalize"
    turn = graph.turn_from_state({**state, **update})
    assert turn.reply == summary
    assert build_entry(turn, SETTINGS)["fact_check_summary"] is True


def test_without_a_prepared_reply_nothing_changes():
    state = graph.new_question_input("Who was Caesar?", "w.pdf", [Passage("w.pdf, page 1", "", "t")])
    assert state["fixed_reply"] == ""
    turn = graph.Turn("Q?", ModelAnswer("p", text="A."), ModelAnswer("s", error="e"))
    assert build_entry(turn, SETTINGS)["fact_check_summary"] is None
