"""Asking the answering models (PRIMARY, later also SECONDARY) a question."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from models import get_chat_model

# Small models invent facts easily. Telling them it is fine to say "I'm not
# sure" reduces made-up details, which makes the later audit more meaningful.
SYSTEM_PROMPT = (
    "You are a helpful and factual assistant. "
    "Answer the user's question clearly and concisely, in a few sentences "
    "unless more detail is asked for. "
    "If you are not sure about a fact, say so instead of guessing. "
    "Never invent sources, quotes, numbers or names."
)


def build_messages(
    history: list[BaseMessage], question: str, max_history: int
) -> list[BaseMessage]:
    """System prompt + the last `max_history` chat messages + the new question.

    Only recent history is sent so the prompt cannot grow without limit
    (small models also get confused by very long prompts).
    """
    recent = history[-max_history:] if max_history > 0 else []
    return [SystemMessage(SYSTEM_PROMPT), *recent, HumanMessage(question)]


def ask_model(model_name: str, messages: list[BaseMessage]) -> str:
    """Send the messages to one model and return its answer as plain text."""
    reply = get_chat_model(model_name).invoke(messages)
    content = reply.content
    # Most providers return a string; some return a list of content parts.
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content.strip()
