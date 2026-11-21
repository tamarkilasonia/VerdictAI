"""Settings for ModelProof, read once from the .env file.

ModelProof uses two AI services (both free tiers, both reached through the
OpenAI-compatible API format):
  AI 1 - Google Gemini: the primary answer (and revisions)
  AI 2 - Groq:          the second opinion and the auditor
Everything that can differ between machines or providers (model names, URLs,
keys, thresholds) lives in .env, so no other module hard-codes those values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# These are design decisions of the app, not machine settings, so they stay
# in code instead of .env (keeps .env short).
MAX_REVISIONS = 2
LOG_PATH = Path("logs") / "audit_log.jsonl"

DEFAULT_USER_AGENT = "ModelProof-py/0.1 (AI course project)"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class ModelSpec:
    """One model and how to reach it."""

    name: str
    base_url: str
    # repr=False keeps the key out of printed settings, error messages and logs.
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class Settings:
    """All runtime settings. frozen=True so no code can change them by accident."""

    primary: ModelSpec  # AI 1 (Gemini)
    secondary: ModelSpec  # AI 2 (Groq)
    auditor: ModelSpec  # AI 2 (Groq), same connection as secondary
    agreement_threshold: float
    history_messages: int
    wiki_user_agent: str


def _text(name: str, default: str) -> str:
    """Read a text setting; an empty value in .env means "use the default"."""
    value = os.getenv(name, "").strip()
    return value or default


def _number(name: str, default: float, cast: type) -> float | int:
    """Read a numeric setting and give a clear message if it is not a number."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{name} in .env must be a number, got {raw!r}") from None


def _user_agent() -> str:
    """WIKI_USER_AGENT, unless it still holds the "<your contact>" placeholder.

    Sending the placeholder text to Wikipedia would be meaningless, so we use
    the generic default until the user fills in real contact details.
    """
    value = _text("WIKI_USER_AGENT", DEFAULT_USER_AGENT)
    return DEFAULT_USER_AGENT if "<" in value else value


def _api_key(name: str, where: str) -> str:
    """Read an API key; a missing key gets a message that says where to get one."""
    key = os.getenv(name, "").strip()
    if not key:
        raise ValueError(f"{name} is empty in .env. Get a free key at {where}")
    return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load .env once and return the settings.

    lru_cache means the file is read only on the first call; restart the app
    after editing .env.
    """
    load_dotenv()  # silently does nothing if .env is missing

    # AI 1: Gemini answers the question.
    gemini_url = _text("PRIMARY_BASE_URL", GEMINI_BASE_URL)
    gemini_key = _api_key("PRIMARY_API_KEY", "https://aistudio.google.com")
    # AI 2: Groq gives the second opinion and runs the audit.
    groq_url = _text("SECONDARY_BASE_URL", GROQ_BASE_URL)
    groq_key = _api_key("SECONDARY_API_KEY", "https://console.groq.com")

    settings = Settings(
        primary=ModelSpec(_text("PRIMARY_MODEL", "gemini-3.1-flash-lite"), gemini_url, gemini_key),
        secondary=ModelSpec(_text("SECONDARY_MODEL", "qwen/qwen3.8-27b"), groq_url, groq_key),
        auditor=ModelSpec(_text("AUDITOR_MODEL", "openai/gpt-oss-120b"), groq_url, groq_key),
        agreement_threshold=_number("AGREEMENT_THRESHOLD", 0.80, float),
        history_messages=_number("HISTORY_MESSAGES", 6, int),
        wiki_user_agent=_user_agent(),
    )

    # Catch impossible values early, with a message a human can act on.
    if not 0.0 <= settings.agreement_threshold <= 1.0:
        raise ValueError("AGREEMENT_THRESHOLD in .env must be between 0 and 1")
    if settings.history_messages < 0:
        raise ValueError("HISTORY_MESSAGES in .env must be 0 or more")
    return settings
