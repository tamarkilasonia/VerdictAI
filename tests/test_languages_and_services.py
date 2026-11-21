"""Georgian/English handling, the two-service settings and friendly error messages.

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
from config import ModelSpec, Settings
from models import friendly_error
from retrieval import is_georgian

SPEC = ModelSpec("gemini-test", "https://gemini.invalid", "secret")
SETTINGS = Settings(
    primary=SPEC,
    secondary=ModelSpec("s", "https://groq.invalid", "k"),
    auditor=ModelSpec("a", "https://groq.invalid", "k"),
    agreement_threshold=0.8,
    history_messages=6,
    wiki_user_agent="test",
)

# ------------------------------------------------------------ language check


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ვინ აღმოაჩინა პენიცილინი?", True),
        ("რა არის Python?", True),  # one English word does not make it English
        ("Who discovered penicillin?", False),
        ("", False),
    ],
)
def test_is_georgian(text, expected):
    assert is_georgian(text) is expected


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


def test_missing_key_gives_a_helpful_message(env):
    env.setenv("PRIMARY_API_KEY", " ")  # blank, like an unfilled .env line
    with pytest.raises(ValueError, match="aistudio.google.com"):
        config.get_settings()


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


# --------------------------------------------- Wikipedia search translation


def run_retrieve(monkeypatch, question, translate):
    searched = []
    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "translate_to_english", translate)
    monkeypatch.setattr(graph, "get_evidence", lambda query, answer, ua: searched.append(query) or [])
    state = {
        "question": question,
        "primary": ModelAnswer("p", text="answer"),
        "secondary": ModelAnswer("s", text="answer"),
        "latency": {},
    }
    return graph.retrieve_evidence(state), searched


def test_georgian_question_is_searched_in_english(monkeypatch):
    update, searched = run_retrieve(
        monkeypatch, "ვინ აღმოაჩინა პენიცილინი?", lambda spec, q: "Who discovered penicillin?"
    )
    assert searched == ["Who discovered penicillin?"]
    assert update["search_query"] == "Who discovered penicillin?"


def test_english_question_is_not_translated(monkeypatch):
    def must_not_translate(spec, q):
        raise AssertionError("English questions must not be translated")

    _, searched = run_retrieve(monkeypatch, "Who discovered penicillin?", must_not_translate)
    assert searched == ["Who discovered penicillin?"]


def test_failed_translation_falls_back_to_the_original_question(monkeypatch):
    def broken(spec, q):
        raise api_error(openai.RateLimitError, 429)

    update, searched = run_retrieve(monkeypatch, "ვინ აღმოაჩინა პენიცილინი?", broken)
    assert searched == ["ვინ აღმოაჩინა პენიცილინი?"]
    assert "evidence_error" not in update  # Wikipedia itself did not fail


# ------------------------- ranking and auditing a Georgian answer in English

GEORGIAN_QUESTION = "ვინ აღმოაჩინა პენიცილინი?"
GEORGIAN_ANSWER = "პენიცილინი 1928 წელს ალექსანდერ ფლემინგმა აღმოაჩინა."
ENGLISH = {
    GEORGIAN_QUESTION: "Who discovered penicillin?",
    GEORGIAN_ANSWER: "Alexander Fleming discovered penicillin in 1928.",
}


def run_audit_node(monkeypatch, answer, translate, translation=""):
    """Run the graph's audit step with fakes; return its update and what was audited."""
    audited = {}

    def fake_run_audit(question, answer, conversation, evidence):
        audited.update(question=question, answer=answer)
        return {}

    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "translate_to_english", translate)
    monkeypatch.setattr(graph, "run_audit", fake_run_audit)
    state = {
        "question": GEORGIAN_QUESTION,
        "search_query": "Who discovered penicillin?",
        "answer_translation": translation,
        "primary": ModelAnswer("p", text=answer),
        "secondary": ModelAnswer("s", text=answer),
        "latency": {},
    }
    return graph.audit(state), audited


def test_georgian_answer_is_ranked_and_audited_in_english(monkeypatch):
    ranked = []
    monkeypatch.setattr(graph, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(graph, "translate_to_english", lambda spec, text: ENGLISH[text])
    monkeypatch.setattr(
        graph, "get_evidence", lambda query, answer, ua: ranked.append((query, answer)) or []
    )
    state = {
        "question": GEORGIAN_QUESTION,
        "primary": ModelAnswer("p", text=GEORGIAN_ANSWER),
        "secondary": ModelAnswer("s", text=GEORGIAN_ANSWER),
        "latency": {},
    }
    update = graph.retrieve_evidence(state)
    assert ranked == [("Who discovered penicillin?", ENGLISH[GEORGIAN_ANSWER])]

    # The audit reuses the same translation instead of translating again.
    def must_not_translate(spec, t):
        raise AssertionError("the answer was already translated")

    _, audited = run_audit_node(
        monkeypatch, GEORGIAN_ANSWER, must_not_translate, update["answer_translation"]
    )
    assert audited == {"question": "Who discovered penicillin?", "answer": ENGLISH[GEORGIAN_ANSWER]}


def test_revised_georgian_answer_is_translated_again(monkeypatch):
    # revise() clears the translation, so the audit must translate the new answer.
    update, audited = run_audit_node(
        monkeypatch, GEORGIAN_ANSWER, lambda spec, text: ENGLISH[text], translation=""
    )
    assert audited["answer"] == ENGLISH[GEORGIAN_ANSWER]
    assert update["answer_translation"] == ENGLISH[GEORGIAN_ANSWER]  # shown in the UI


def test_english_answer_is_audited_as_it_is(monkeypatch):
    def must_not_translate(spec, t):
        raise AssertionError("English answers must not be translated")

    update, audited = run_audit_node(monkeypatch, "Fleming, in 1928.", must_not_translate)
    assert audited["answer"] == "Fleming, in 1928."
    assert update["answer_translation"] == ""


def test_failed_answer_translation_audits_the_georgian_text(monkeypatch):
    def broken(spec, t):
        raise api_error(openai.RateLimitError, 429)

    georgian = "პენიცილინი ფლემინგმა აღმოაჩინა."
    update, audited = run_audit_node(monkeypatch, georgian, broken)
    assert audited["answer"] == georgian
    assert update["answer_translation"] == ""


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('"Who discovered penicillin?"', "Who discovered penicillin?"),  # wrapped: unwrap
        ('"Mona Lisa" was painted by Leonardo.', '"Mona Lisa" was painted by Leonardo.'),
    ],
)
def test_translation_keeps_quotes_that_belong_to_the_text(monkeypatch, reply, expected):
    monkeypatch.setattr(answers, "ask_model", lambda spec, messages, temperature: reply)
    assert answers.translate_to_english(SPEC, "ტექსტი") == expected
