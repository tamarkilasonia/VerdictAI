"""The audit log file, the graph's routing decisions, and settings safety."""

import json

import config
from agreement import Agreement
from answers import ModelAnswer
from audit_log import append_entry, build_entry
from auditor import CheckOutcome, CheckResult
from config import MAX_REVISIONS, ModelSpec, Settings
from graph import Turn, route_after_audit, route_after_review, route_after_revise
from retrieval import Passage

SETTINGS = Settings(
    primary=ModelSpec("p", "https://gemini.invalid", "super-secret-key"),
    secondary=ModelSpec("s", "https://groq.invalid", "other-secret-key"),
    auditor=ModelSpec("a", "https://groq.invalid", "other-secret-key"),
    agreement_threshold=0.8,
    history_messages=6,
    wiki_user_agent="test",
)


def failing_audit():
    return {"hallucination": CheckOutcome("hallucination", CheckResult(score=1.0))}


def passing_audit():
    return {"hallucination": CheckOutcome("hallucination", CheckResult(score=0.0))}


# ------------------------------------------------------------------ audit log


def make_turn():
    return Turn(
        question="When did the Berlin Wall fall?",
        primary=ModelAnswer("p", text="In 1975."),
        secondary=ModelAnswer("s", text="In 1989."),
        agreement=Agreement(score=0.7, agree=False),
        evidence=[Passage("Berlin Wall", "https://en.wikipedia.org/wiki/Berlin_Wall", "...")],
        audit=passing_audit(),
        revisions=["The Berlin Wall fell in 1989."],
        original_audit=failing_audit(),
        latency={"answers": 1.234},
    )


def test_log_writes_one_valid_json_line_per_entry(tmp_path):
    path = tmp_path / "logs" / "audit_log.jsonl"  # folder does not exist yet
    append_entry(build_entry(make_turn(), SETTINGS, "thread-1"), path)
    append_entry(build_entry(make_turn(), SETTINGS, "thread-1"), path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["revision_count"] == 1
    assert entry["final_answer"] == "The Berlin Wall fell in 1989."
    assert entry["audit_scores"] == {"hallucination": 0.0}
    assert entry["original_audit_scores"] == {"hallucination": 1.0}
    assert entry["evidence_urls"] == ["https://en.wikipedia.org/wiki/Berlin_Wall"]


def test_log_never_contains_the_api_key(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    append_entry(build_entry(make_turn(), SETTINGS), path)
    assert "super-secret-key" not in path.read_text(encoding="utf-8")


def test_settings_repr_hides_the_api_keys():
    text = repr(SETTINGS)
    assert "super-secret-key" not in text and "other-secret-key" not in text


def test_placeholder_user_agent_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PRIMARY_API_KEY", "k1")
    monkeypatch.setenv("SECONDARY_API_KEY", "k2")
    monkeypatch.setenv("WIKI_USER_AGENT", "ModelProof (contact: <your contact>)")
    config.get_settings.cache_clear()
    try:
        assert config.get_settings().wiki_user_agent == config.DEFAULT_USER_AGENT
    finally:
        config.get_settings.cache_clear()


# ------------------------------------------------------------------ routing


def routing_state(primary_audit, secondary_audit, revisions=()):
    return {
        "question": "When did the Berlin Wall fall?",
        "primary": ModelAnswer("p", text="In 1975."),
        "secondary": ModelAnswer("s", text="In 1989."),
        "audit": primary_audit,
        "secondary_audit": secondary_audit,
        "revisions": list(revisions),
    }


def test_review_is_offered_only_when_no_answer_passed():
    state = routing_state(failing_audit(), failing_audit())
    assert route_after_audit(state) == "human_review"


def test_a_passing_answer_is_shown_without_review():
    # PRIMARY failed but SECONDARY passed: show SECONDARY, nothing to revise.
    assert route_after_audit(routing_state(failing_audit(), passing_audit())) == "finalize"
    assert route_after_audit(routing_state(passing_audit(), failing_audit())) == "finalize"


def test_no_review_when_the_auditor_could_not_run():
    unchecked = {"hallucination": CheckOutcome("hallucination", None, error="daily limit")}
    assert route_after_audit(routing_state(unchecked, unchecked)) == "finalize"


def test_no_more_review_after_max_revisions():
    state = routing_state(failing_audit(), failing_audit(), ["r"] * MAX_REVISIONS)
    assert route_after_audit(state) == "finalize"


# ---------------------------------------------- which ONE answer is shown


def test_both_pass_shows_the_primary_answer():
    turn = Turn("When did the Berlin Wall fall?", ModelAnswer("p", text="1989."),
                ModelAnswer("s", text="In 1989."), audit=passing_audit(),
                secondary_audit=passing_audit())
    assert turn.shown == ("p", "1989.")


def test_failing_answer_is_hidden_and_the_other_shown():
    turn = Turn("When did the Berlin Wall fall?", ModelAnswer("p", text="In 1975."),
                ModelAnswer("s", text="In 1989."), audit=failing_audit(),
                secondary_audit=passing_audit())
    assert turn.shown == ("s", "In 1989.")
    assert turn.reply == "In 1989."


def test_both_failing_gives_an_honest_dont_know_in_the_question_language():
    english = Turn("When did the Berlin Wall fall?", ModelAnswer("p", text="1975."),
                   ModelAnswer("s", text="1980."), audit=failing_audit(),
                   secondary_audit=failing_audit())
    georgian = Turn("როდის დაეცა ბერლინის კედელი?", ModelAnswer("p", text="1975."),
                    ModelAnswer("s", text="1980."), audit=failing_audit(),
                    secondary_audit=failing_audit())
    assert english.shown is None
    assert english.reply == (
        "The AI answers did not pass the fact check. So I don't know a reliable answer "
        "to this (see the analysis on the right)."
    )
    assert georgian.shown is None and "სანდო პასუხი არ ვიცი" in georgian.reply


def test_wikipedia_text_is_shown_when_no_ai_answer_passed():
    wiki = Passage("ბედნიერი ერი", "https://ka.wikipedia.org/wiki/x",
                   "სიმღერის ტექსტი შექმნილია ილია ჭავჭავაძის ამავე სახელწოდების ლექსის მიხედვით.")
    turn = Turn("„ბედნიერი ერი“ ვინაა ავტორი?", ModelAnswer("p", text="აკაკი წერეთელი."),
                ModelAnswer("s", text="არ ვიცი."), audit=failing_audit(),
                secondary_audit=failing_audit(), wiki_answer=wiki)
    assert turn.shown is None
    assert turn.reply == (
        "AI-ების პასუხებმა ფაქტების შემოწმება ვერ გაიარა. აი, რა წერია Wikipedia-ში:\n\n"
        "> სიმღერის ტექსტი შექმნილია ილია ჭავჭავაძის ამავე სახელწოდების ლექსის მიხედვით.\n\n"
        "— [ბედნიერი ერი](https://ka.wikipedia.org/wiki/x)"
    )


def test_unchecked_answers_are_replaced_by_wikipedia_with_the_reason():
    unchecked = {"hallucination": CheckOutcome("hallucination", None, error="rate limit")}
    wiki = Passage("Berlin Wall", "https://en.wikipedia.org/wiki/Berlin_Wall", "It fell in 1989.")
    turn = Turn("When did the Berlin Wall fall?", ModelAnswer("p", text="1975."),
                ModelAnswer("s", text="1989."), audit=unchecked,
                secondary_audit=unchecked, wiki_answer=wiki)
    assert turn.reply.startswith("The AI answers could not be fact-checked right now")
    assert "> It fell in 1989." in turn.reply


def test_review_decision_routes():
    assert route_after_review({"review_decision": "revise"}) == "revise"
    assert route_after_review({"review_decision": "accept"}) == "finalize"


def test_failed_revision_goes_back_to_review():
    assert route_after_revise({"revise_error": "Revision failed"}) == "human_review"
    assert route_after_revise({"revise_error": None}) == "audit"
