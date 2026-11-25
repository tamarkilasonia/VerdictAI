"""The two-service settings, friendly error messages and the graph's steps.

No real AI service is called: settings come from fake environment values and
the model / Wikipedia calls are replaced with small fakes.
"""

import httpx
import openai
import pytest

import answers
import config
import graph
from answers import ModelAnswer
from auditor import CheckOutcome, CheckResult
from config import ModelSpec, Settings
from models import friendly_error
from retrieval import Passage

SPEC = ModelSpec("gemini-test", "https://gemini.invalid", "secret")
SETTINGS = Settings(
    primary=SPEC,
    secondary=ModelSpec("s", "https://groq.invalid", "k"),
    auditor=ModelSpec("a", "https://groq.invalid", "k"),
    agreement_threshold=0.8,
    history_messages=6,
    wiki_user_agent="test",
)

@pytest.fixture(autouse=True)
def no_real_wikipedia(monkeypatch):
    """No test may reach the internet: Wikipedia's own answer is faked as "none"."""
    monkeypatch.setattr(graph, "wikipedia_answer", lambda *args, **kwargs: None)


# ------------------------------------------------------------------ settings


@pytest.fixture
def env(monkeypatch):
    """Fresh settings from fake .env values (the real .env is not overridden)."""
    monkeypatch.setenv("PRIMARY_API_KEY", "gemini-key")
    monkeypatch.setenv("SECONDARY_API_KEY", "groq-key")
    config.get_settings.cache_clear()
    yield monkeypatch
    config.get_settings.cache_clear()


def test_auditor_uses_the_second_service(env):
    settings = config.get_settings()
    assert settings.auditor.base_url == settings.secondary.base_url
    assert settings.auditor.api_key == settings.secondary.api_key == "groq-key"
    assert settings.primary.api_key == "gemini-key"


def test_missing_groq_key_gives_a_helpful_message(env):
    env.setenv("SECONDARY_API_KEY", " ")  # blank, like an unfilled .env line
    with pytest.raises(ValueError, match="console.groq.com"):
        config.get_settings()


def test_one_groq_key_is_enough_by_default(monkeypatch):
    # No .env and no PRIMARY_ settings: all three models run on Groq with one key.
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for name in ("PRIMARY_API_KEY", "PRIMARY_MODEL", "PRIMARY_BASE_URL", "SECONDARY_MODEL",
                 "AUDITOR_MODEL", "SECONDARY_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SECONDARY_API_KEY", "groq-key")
    config.get_settings.cache_clear()
    try:
        s = config.get_settings()
        assert (s.primary.name, s.secondary.name, s.auditor.name) == (
            "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b")
        assert s.primary.base_url == s.secondary.base_url == config.GROQ_BASE_URL
        assert s.primary.api_key == s.secondary.api_key == "groq-key"
    finally:
        config.get_settings.cache_clear()


# ------------------------------------------------------------ error messages


def api_error(cls, status: int):
    request = httpx.Request("POST", "https://service.invalid/v1/chat/completions")
    return cls("error", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    "error, expected",
    [
        (api_error(openai.RateLimitError, 429), "Free usage limit"),
        (api_error(openai.AuthenticationError, 401), "API key"),
        (api_error(openai.NotFoundError, 404), "not found"),
        (api_error(openai.InternalServerError, 503), "busy"),
    ],
)
def test_friendly_error_messages(error, expected):
    message = friendly_error(error, SPEC)
    assert expected in message
    assert "secret" not in message  # the key never appears in a message


# ------------------------------------------------ Wikipedia search query


def run_retrieve(monkeypatch, question, make_query):
    searched = []
    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "make_search_query", make_query)
    monkeypatch.setattr(graph, "get_evidence", lambda query, answer, ua: searched.append((query, answer)) or [])
    state = {
        "question": question,
        "primary": ModelAnswer("p", text="Shota Rustaveli wrote it."),
        "secondary": ModelAnswer("s", text="answer"),
        "latency": {},
    }
    return graph.retrieve_evidence(state), searched


def test_wikipedia_is_searched_with_short_keywords(monkeypatch):
    # A whole sentence found no Wikipedia page; keywords do.
    question = "As far as I know, The Knight in the Panther's Skin was written by Galaktion."
    update, searched = run_retrieve(
        monkeypatch, question, lambda spec, q: "The Knight in the Panther's Skin"
    )
    # Ranked against the answer itself (no translation step any more).
    assert searched == [("The Knight in the Panther's Skin", "Shota Rustaveli wrote it.")]
    assert update["search_query"] == "The Knight in the Panther's Skin"


def test_failed_search_query_falls_back_to_the_question(monkeypatch):
    def broken(spec, q):
        raise api_error(openai.RateLimitError, 429)

    update, searched = run_retrieve(monkeypatch, "Who discovered penicillin?", broken)
    assert searched[0][0] == "Who discovered penicillin?"
    assert "evidence_error" not in update  # Wikipedia itself did not fail


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('"Penicillin"', "Penicillin"),  # wrapped in quotes: unwrap
        ('"Mona Lisa" painting', '"Mona Lisa" painting'),
    ],
)
def test_search_query_keeps_quotes_that_belong_to_the_text(monkeypatch, reply, expected):
    monkeypatch.setattr(answers, "ask_model", lambda spec, messages, temperature: reply)
    assert answers.make_search_query(SPEC, "Who discovered penicillin?") == expected


# ------------------------------------------------------------ English only


def test_models_are_told_to_answer_in_english():
    messages = answers.build_messages([], "Wer hat Penicillin entdeckt?", max_history=6)
    assert "Always answer in English, even when the user writes in another language" in messages[0].content
    assert "Write in English." in answers.REVISE_PROMPT


def test_answer_is_audited_as_it_is(monkeypatch):
    audited = {}

    def fake_run_audit(question, answer, conversation, evidence):
        audited.update(question=question, answer=answer)
        return {}

    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "run_audit", fake_run_audit)
    state = {
        "question": "Who discovered penicillin?",
        "primary": ModelAnswer("p", text="Fleming, in 1928."),
        "secondary": ModelAnswer("s", text="Fleming."),
        "secondary_audit": {"hallucination": CheckOutcome("hallucination")},
        "latency": {},
    }
    update = graph.audit(state)
    assert audited == {"question": "Who discovered penicillin?", "answer": "Fleming, in 1928."}
    assert set(update) == {"audit", "latency"}


def test_both_models_get_the_same_question(monkeypatch):
    sent = []

    def fake_ask_both(primary, secondary, messages):
        sent.append(messages[-1].content)
        return ModelAnswer("p", text="a"), ModelAnswer("s", text="b")

    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "ask_both", fake_ask_both)
    update = graph.generate_answers({"question": "Who wrote it?", "messages": [], "latency": {}})
    assert sent == ["Who wrote it?"]
    assert update["secondary"].text == "b"


# ----------------------------------------- auditing BOTH answers to compare them


def run_both_audits(monkeypatch, secondary_audit=None, primary_ok=True):
    audited = []

    def fake_run_audit(question, answer, conversation, evidence):
        audited.append(answer)
        return {"hallucination": CheckOutcome("hallucination", CheckResult(score=0.9, issues=[]))}

    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "run_audit", fake_run_audit)
    state = {
        "question": "Who wrote The Knight in the Panther's Skin?",
        "primary": (
            ModelAnswer("p", text="Shota Rustaveli wrote it.")
            if primary_ok
            else ModelAnswer("p", error="service down")
        ),
        "secondary": ModelAnswer("s", text="Supran Makvsadze wrote it."),
        "secondary_audit": secondary_audit or {},
        "latency": {},
    }
    return graph.audit(state), audited


def test_both_answers_are_audited(monkeypatch):
    update, audited = run_both_audits(monkeypatch)
    assert sorted(audited) == ["Shota Rustaveli wrote it.", "Supran Makvsadze wrote it."]
    assert update["secondary_audit"]["hallucination"].status == "fail"


def test_secondary_is_not_audited_again_after_a_revision(monkeypatch):
    earlier = {"hallucination": CheckOutcome("hallucination", CheckResult(score=0.0, issues=[]))}
    update, audited = run_both_audits(monkeypatch, secondary_audit=earlier)
    assert audited == ["Shota Rustaveli wrote it."]  # only PRIMARY's (revised) answer
    assert "secondary_audit" not in update


def test_only_answer_is_audited_once_when_primary_failed(monkeypatch):
    # The shown answer IS the secondary one, so auditing it twice would be wasted.
    update, audited = run_both_audits(monkeypatch, primary_ok=False)
    assert audited == ["Supran Makvsadze wrote it."]
    assert "secondary_audit" not in update


# ----------------------------------- Wikipedia's own answer (no AI answer passed)


def test_wikipedia_answer_is_kept_ready(monkeypatch):
    shrosha = Passage("Shrosha", "en-url", "Shrosha is a village in Imereti.")
    monkeypatch.setattr(graph, "wikipedia_answer", lambda question, passages, ua: shrosha)
    update, _ = run_retrieve(monkeypatch, "Where is Shrosha?", lambda spec, q: "Shrosha")
    assert update["wiki_answer"] is shrosha


def test_reply_quotes_wikipedia_when_no_answer_passed():
    from graph import Turn

    failed = {"hallucination": CheckOutcome("hallucination", CheckResult(score=0.9))}
    turn = Turn(
        question="Where is Shrosha?",
        primary=ModelAnswer("p", text="In Spain."),
        secondary=ModelAnswer("s", error="down"),
        audit=failed,
        wiki_answer=Passage("Shrosha", "en-url", "Shrosha is a village in Imereti."),
    )
    assert turn.reply == (
        "The AI answers did not pass the fact check. Here is what Wikipedia says:\n\n"
        "> Shrosha is a village in Imereti.\n\n— [Shrosha](en-url)"
    )


def test_reply_says_i_dont_know_without_wikipedia():
    from graph import Turn

    turn = Turn(
        question="Q?",
        primary=ModelAnswer("p", text="A."),
        secondary=ModelAnswer("s", error="down"),
        audit={"hallucination": CheckOutcome("hallucination", CheckResult(score=0.9))},
    )
    assert turn.reply.endswith("So I don't know a reliable answer to this (see the analysis on the right).")


def test_daily_limit_is_not_retried_and_explained():
    from models import should_retry

    request = httpx.Request("POST", "https://service.invalid/v1/chat/completions")
    daily = openai.RateLimitError(
        "Rate limit reached on tokens per day (TPD): Limit 200000, Used 199598",
        response=httpx.Response(429, request=request),
        body=None,
    )
    minute = api_error(openai.RateLimitError, 429)
    assert should_retry(minute) and not should_retry(daily)
    assert "Today's free limit" in friendly_error(daily, SPEC)


# ------------------------------------------------------------ empty answers


def fake_replies(monkeypatch, replies):
    """ask_model returns these replies one after another; returns the call log."""
    calls = []

    def fake_ask_model(spec, messages, temperature=0.7):
        calls.append(spec.name)
        return replies[len(calls) - 1]

    monkeypatch.setattr(answers, "ask_model", fake_ask_model)
    return calls


def test_empty_answer_is_asked_again(monkeypatch):
    calls = fake_replies(monkeypatch, ["", "Ilia Chavchavadze wrote it."])
    answer = answers._ask_safely(SPEC, [])
    assert answer.ok and answer.text == "Ilia Chavchavadze wrote it."
    assert len(calls) == 2


def test_empty_answer_twice_is_a_failure_not_an_answer(monkeypatch):
    calls = fake_replies(monkeypatch, ["", ""])
    answer = answers._ask_safely(SPEC, [])
    assert not answer.ok
    assert answer.error == "'gemini-test' returned an empty answer."
    assert len(calls) == 2  # not asked forever


def test_normal_answer_is_asked_once(monkeypatch):
    calls = fake_replies(monkeypatch, ["Canberra."])
    assert answers._ask_safely(SPEC, []).text == "Canberra."
    assert len(calls) == 1


def test_empty_primary_answer_leaves_the_other_one(monkeypatch):
    # PRIMARY fails with an empty answer: SECONDARY's answer is the one checked and shown.
    def fake_ask_model(spec, messages, temperature=0.7):
        return "" if spec is SPEC else "Ilia Chavchavadze."

    monkeypatch.setattr(answers, "ask_model", fake_ask_model)
    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    update = graph.generate_answers({"question": "Who wrote A Happy Nation?", "messages": [], "latency": {}})
    assert not update["primary"].ok and update["secondary"].text == "Ilia Chavchavadze."
    assert "error" not in update  # one answer is enough to go on
    from graph import Turn

    turn = Turn("Who wrote A Happy Nation?", update["primary"], update["secondary"])
    assert turn.original_answer == "Ilia Chavchavadze."
