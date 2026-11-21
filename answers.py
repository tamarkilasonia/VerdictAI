"""Asking the answering models (PRIMARY and SECONDARY) a question."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from config import ModelSpec
from models import friendly_error, get_chat_model, reply_text

# Telling the model it is fine to say "I'm not sure" reduces made-up details,
# which makes the later audit more meaningful. The language rule keeps English
# questions in English and lets Georgian questions get Georgian answers.
SYSTEM_PROMPT = (
    "You are a helpful and factual assistant. "
    "Always answer in the same language as the user's last message "
    "(for example English or Georgian). "
    "Answer the user's question clearly and concisely, in a few sentences "
    "unless more detail is asked for. "
    "If you are not sure about a fact, say so instead of guessing. "
    "Never invent sources, quotes, numbers or names."
)


@dataclass
class ModelAnswer:
    """What one model returned. Exactly one of `text` / `error` is set."""

    model: str
    text: str | None = None
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.text is not None


def build_messages(
    history: list[BaseMessage], question: str, max_history: int
) -> list[BaseMessage]:
    """System prompt + the last `max_history` chat messages + the new question.

    Only recent history is sent so the prompt cannot grow without limit.
    """
    recent = history[-max_history:] if max_history > 0 else []
    return [SystemMessage(SYSTEM_PROMPT), *recent, HumanMessage(question)]


def ask_model(spec: ModelSpec, messages: list[BaseMessage], temperature: float = 0.7) -> str:
    """Send the messages to one model and return its answer as plain text."""
    return reply_text(get_chat_model(spec, temperature=temperature).invoke(messages))


def _ask_safely(spec: ModelSpec, messages: list[BaseMessage]) -> ModelAnswer:
    """Like ask_model, but never raises: a failure becomes an error message.

    This is what lets one model fail without taking the other one down.
    """
    start = time.perf_counter()
    try:
        text = ask_model(spec, messages)
        return ModelAnswer(spec.name, text=text, seconds=time.perf_counter() - start)
    except Exception as error:  # noqa: BLE001 - any failure is reported, not raised
        return ModelAnswer(
            spec.name,
            error=friendly_error(error, spec),
            seconds=time.perf_counter() - start,
        )


def ask_both(
    primary: ModelSpec, secondary: ModelSpec, messages: list[BaseMessage]
) -> tuple[ModelAnswer, ModelAnswer]:
    """Ask both models the same messages at the same time.

    Each call mostly waits for the AI service, so two threads finish in
    roughly the time of the slower model instead of the sum of both.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_ask_safely, primary, messages)
        second = pool.submit(_ask_safely, secondary, messages)
        return first.result(), second.result()


# The reviser gets the failed answer plus the audit's findings and the evidence.
# Saying "only the corrected answer" stops models from writing
# "Here is the revised answer:" or talking about the audit itself.
REVISE_PROMPT = (
    "You are revising an answer that failed an automatic quality audit. "
    "Rewrite the ANSWER so that it fixes every PROBLEM listed. "
    "Use the EVIDENCE for facts. If the evidence does not support a claim, "
    "remove it or say that you are not sure. Keep the parts that were correct. "
    "Write in the same language as the QUESTION. "
    "Reply with the corrected answer only: no introduction, and do not mention the audit."
)


def revise_answer(
    spec: ModelSpec,
    history: list[BaseMessage],
    question: str,
    answer: str,
    problems: list[str],
    evidence: list[str],
) -> ModelAnswer:
    """Ask a model to fix `answer` using the audit's problems and the evidence.

    Recent history is included so follow-up questions ("when was he born?")
    still make sense to the reviser.
    """
    evidence_text = "\n".join(f"- {e}" for e in evidence) or "(no evidence available)"
    problems_text = "\n".join(f"- {p}" for p in problems)
    request = (
        f"QUESTION: {question}\n\nANSWER:\n{answer}\n\n"
        f"PROBLEMS:\n{problems_text}\n\nEVIDENCE:\n{evidence_text}"
    )
    messages = [SystemMessage(REVISE_PROMPT), *history, HumanMessage(request)]
    return _ask_safely(spec, messages)


# English Wikipedia is far richer than Georgian Wikipedia (a test question
# like "who discovered penicillin" found nothing there), so Georgian questions
# are translated into an English search query first.
TRANSLATE_PROMPT = (
    "Translate the user's question into English so it can be used as a "
    "Wikipedia search query. Reply with the English translation only."
)


def translate_for_search(spec: ModelSpec, question: str) -> str:
    """English version of a (Georgian) question, for the Wikipedia search."""
    messages = [SystemMessage(TRANSLATE_PROMPT), HumanMessage(question)]
    # temperature 0: a translation should not be creative.
    return ask_model(spec, messages, temperature=0.0).strip().strip('"')
