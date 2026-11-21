"""Settings for ModelProof, read once from the .env file.

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


@dataclass(frozen=True)
class Settings:
    """All runtime settings. frozen=True so no code can change them by accident."""

    model_provider: str
    model_base_url: str
    # repr=False keeps the key out of printed settings, error messages and logs.
    model_api_key: str = field(repr=False)
    primary_model: str
    secondary_model: str
    auditor_model: str
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load .env once and return the settings.

    lru_cache means the file is read only on the first call; restart the app
    after editing .env.
    """
    load_dotenv()  # silently does nothing if .env is missing: defaults still work
    primary = _text("PRIMARY_MODEL", "llama3.2")

    settings = Settings(
        model_provider=_text("MODEL_PROVIDER", "ollama").lower(),
        model_base_url=_text("MODEL_BASE_URL", "http://localhost:11434"),
        model_api_key=os.getenv("MODEL_API_KEY", "").strip(),
        primary_model=primary,
        secondary_model=_text("SECONDARY_MODEL", "qwen2.5:3b"),
        auditor_model=_text("AUDITOR_MODEL", primary),  # empty -> same as primary
        agreement_threshold=_number("AGREEMENT_THRESHOLD", 0.80, float),
        history_messages=_number("HISTORY_MESSAGES", 6, int),
        wiki_user_agent=_text("WIKI_USER_AGENT", DEFAULT_USER_AGENT),
    )

    # Catch impossible values early, with a message a human can act on.
    if not 0.0 <= settings.agreement_threshold <= 1.0:
        raise ValueError("AGREEMENT_THRESHOLD in .env must be between 0 and 1")
    if settings.history_messages < 0:
        raise ValueError("HISTORY_MESSAGES in .env must be 0 or more")
    return settings
