"""The audit log file, the graph's routing decisions, and settings safety."""

import json

import config
from agreement import Agreement
from answers import ModelAnswer
from audit_log import append_entry, build_entry
from auditor import CheckOutcome, CheckResult
from config import MAX_REVISIONS, Settings
from graph import Turn, route_after_audit, route_after_review, route_after_revise
from retrieval import Passage

SETTINGS = Settings(
    model_provider="ollama",
    model_base_url="http://localhost:11434",
    model_api_key="super-secret-key",
    primary_model="p",
    secondary_model="s",
    auditor_model="a",
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


def test_settings_repr_hides_the_api_key():
    assert "super-secret-key" not in repr(SETTINGS)


def test_placeholder_user_agent_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WIKI_USER_AGENT", "ModelProof (contact: <your contact>)")
    config.get_settings.cache_clear()
    try:
        assert config.get_settings().wiki_user_agent == config.DEFAULT_USER_AGENT
    finally:
        config.get_settings.cache_clear()


# ------------------------------------------------------------------ routing


def test_failed_audit_goes_to_human_review():
    assert route_after_audit({"audit": failing_audit(), "revisions": []}) == "human_review"


def test_passed_audit_is_finalized():
    assert route_after_audit({"audit": passing_audit(), "revisions": []}) == "finalize"


def test_no_more_review_after_max_revisions():
    state = {"audit": failing_audit(), "revisions": ["r"] * MAX_REVISIONS}
    assert route_after_audit(state) == "finalize"


def test_review_decision_routes():
    assert route_after_review({"review_decision": "revise"}) == "revise"
    assert route_after_review({"review_decision": "accept"}) == "finalize"


def test_failed_revision_goes_back_to_review():
    assert route_after_revise({"revise_error": "Revision failed"}) == "human_review"
    assert route_after_revise({"revise_error": None}) == "audit"
