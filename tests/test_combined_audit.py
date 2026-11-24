"""The combined audit (four checks in one request) and its fallback to four
separate checks, plus the faster retry settings. No AI service is called."""

import json
from dataclasses import replace

import httpx
import openai

import auditor
import models
from auditor import CHECKS, CheckOutcome, CheckResult, Issue, parse_combined_output
from config import ModelSpec, Settings

SETTINGS = Settings(
    primary=ModelSpec("p", "https://p.invalid", "k"),
    secondary=ModelSpec("s", "https://s.invalid", "k"),
    auditor=ModelSpec("a", "https://a.invalid", "k"),
    agreement_threshold=0.8,
    history_messages=6,
    wiki_user_agent="test",
)

GOOD_REPLY = json.dumps({
    "hallucination": {"score": 0.0, "issues": [{"quote": "in 1928", "reason": "CONTRADICTED: the evidence does not mention the year."}]},
    "bias": {"score": 0.0, "issues": []},
    "toxicity": {"score": 0.0, "issues": []},
    "intent_alignment": {"score": 1.0, "issues": []},
})


class FakeModel:
    """Stands in for the auditor model: returns `reply`, or raises it."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply

        class Reply:
            content = self.reply

        return Reply()


def use_model(monkeypatch, reply):
    model = FakeModel(reply)
    monkeypatch.setattr(auditor, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(auditor, "get_chat_model", lambda spec, **kwargs: model)
    return model


# ------------------------------------------------------------------ parsing


def test_combined_reply_gives_four_results_in_display_order():
    results = parse_combined_output(GOOD_REPLY)
    assert list(results) == list(CHECKS)
    assert results["intent_alignment"].score == 1.0


def test_combined_reply_gets_the_same_label_fix_as_the_separate_check():
    # "CONTRADICTED: ... does not mention" is relabelled UNVERIFIED, score capped.
    hallucination = parse_combined_output(GOOD_REPLY)["hallucination"]
    assert hallucination.issues[0].reason.startswith("UNVERIFIED")
    assert hallucination.score <= 0.2


def test_combined_reply_inside_code_fence_and_chatter_is_parsed():
    assert parse_combined_output(f"Here you go:\n```json\n{GOOD_REPLY}\n```") is not None


def test_combined_reply_with_a_missing_check_is_rejected():
    data = json.loads(GOOD_REPLY)
    del data["toxicity"]
    assert parse_combined_output(json.dumps(data)) is None


def test_combined_reply_with_a_malformed_check_is_rejected():
    data = json.loads(GOOD_REPLY)
    data["bias"] = {"score": 7}  # out of range
    assert parse_combined_output(json.dumps(data)) is None
    data["bias"] = "fine"  # not an object
    assert parse_combined_output(json.dumps(data)) is None


def test_non_json_reply_is_rejected():
    assert parse_combined_output("I cannot do that.") is None
    assert parse_combined_output('{"hallucination": ') is None


# ------------------------------------------------------- running + fallback


def test_good_reply_needs_only_one_request(monkeypatch):
    model = use_model(monkeypatch, GOOD_REPLY)
    separate = []
    monkeypatch.setattr(auditor, "run_separate_checks", lambda *args: separate.append(args))

    audit = auditor.run_audit("When?", "In 1928.", [], ["[Penicillin] discovered in 1928"])

    assert model.calls == 1
    assert separate == []
    assert list(audit) == list(CHECKS)
    assert all(outcome.mode == "combined" for outcome in audit.values())
    assert audit["intent_alignment"].status == "pass"


def test_unparsable_reply_falls_back_to_separate_checks(monkeypatch):
    use_model(monkeypatch, "not json at all")
    fallback = {check: CheckOutcome(check, CheckResult(score=0.0)) for check in CHECKS}
    seen = []

    def fake_separate(question, answer, conversation, evidence):
        seen.append((question, answer, evidence))
        return fallback

    monkeypatch.setattr(auditor, "run_separate_checks", fake_separate)

    audit = auditor.run_audit("When?", "In 1928.", [], ["ev"])

    assert audit is fallback
    assert seen == [("When?", "In 1928.", ["ev"])]


def test_fallback_really_runs_four_separate_requests(monkeypatch):
    # Every reply is the old single-check format, which the combined parser
    # rejects: one combined request, then the four separate checks.
    model = use_model(monkeypatch, '{"score": 0.1, "issues": []}')
    audit = auditor.run_audit("Q?", "A.", [])
    assert model.calls == 1 + len(CHECKS)
    assert all(outcome.mode == "separate" and outcome.result is not None for outcome in audit.values())


def test_service_error_does_not_fall_back(monkeypatch):
    # Four more requests to a service that just failed would fail too.
    request = httpx.Request("POST", "https://a.invalid")
    error = openai.RateLimitError("limit", response=httpx.Response(429, request=request), body=None)
    model = use_model(monkeypatch, error)

    audit = auditor.run_audit("Q?", "A.", [])

    assert model.calls == 1
    assert all(outcome.status == "unavailable" for outcome in audit.values())
    assert "limit" in audit["hallucination"].error.lower()


def test_combined_input_has_question_history_evidence_then_answer():
    from langchain_core.messages import AIMessage, HumanMessage

    text = auditor.build_combined_input(
        "When did he die?", "In 1955.", [HumanMessage("Who was Einstein?"), AIMessage("A physicist.")], ["[Einstein] died 1955"]
    )
    assert text == (
        "CONVERSATION:\nUser: Who was Einstein?\nAssistant: A physicist.\nUser: When did he die?\n"
        "EVIDENCE:\n- [Einstein] died 1955\nANSWER: In 1955."
    )


def test_combined_input_without_evidence_has_no_evidence_section():
    text = auditor.build_combined_input("Hi!", "Hello!", [], None)
    assert "EVIDENCE" not in text
    assert text == "CONVERSATION:\nUser: Hi!\nANSWER: Hello!"


def test_reviser_still_gets_readable_problems():
    result = CheckResult(score=0.9, issues=[Issue(quote="1991", reason="CONTRADICTED: 1989")])
    audit = {"hallucination": CheckOutcome("hallucination", result, mode="combined")}
    assert auditor.describe_problems(audit) == ['hallucination: "1991": CONTRADICTED: 1989']


# ------------------------------------------------------------- fail fast


def test_a_stuck_service_is_given_up_on_quickly():
    assert models.REQUEST_TIMEOUT_SECONDS <= 15
    assert models.MAX_ATTEMPTS == 2


# ------------------------------------------------------- low reasoning effort


def test_gpt_oss_auditor_reasons_briefly():
    assert auditor.auditor_reasoning_effort("openai/gpt-oss-120b") == "low"


def test_other_auditor_models_keep_their_default():
    # e.g. Qwen or Llama may reject the setting
    assert auditor.auditor_reasoning_effort("qwen/qwen3.8-27b") is None


def test_combined_audit_asks_for_low_reasoning(monkeypatch):
    seen = {}

    def fake_chat_model(spec, **kwargs):
        seen.update(kwargs)
        return FakeModel(GOOD_REPLY)

    gpt_oss = replace(SETTINGS, auditor=ModelSpec("openai/gpt-oss-120b", "https://a.invalid", "k"))
    monkeypatch.setattr(auditor, "get_settings", lambda: gpt_oss)
    monkeypatch.setattr(auditor, "get_chat_model", fake_chat_model)
    auditor.run_audit("Q?", "A.", [])
    assert seen == {"temperature": 0.0, "json_mode": True, "reasoning_effort": "low"}


def test_reasoning_effort_reaches_the_client():
    spec = ModelSpec("openai/gpt-oss-120b", "https://a.invalid", "k")
    model = models.get_chat_model(spec, temperature=0.0, reasoning_effort="low")
    assert model.bound.reasoning_effort == "low"
    assert models.get_chat_model(spec).bound.reasoning_effort is None
