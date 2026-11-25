"""Audit trail: one JSON line per interaction in logs/audit_log.jsonl.

JSONL ("JSON Lines") means one complete JSON object per line. Appending a
line never rewrites earlier entries, and each line can be read on its own,
which makes the file easy to grep, stream or load into pandas later.

Never logged: API keys or any other secret. Only the fields built below.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import LOG_PATH, Settings

if TYPE_CHECKING:  # only for type hints; avoids a circular import with graph.py
    from graph import Turn

logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _models(settings: Settings) -> dict[str, str]:
    return {
        "primary": settings.primary.name,
        "secondary": settings.secondary.name,
        "auditor": settings.auditor.name,
    }


def _scores(audit: dict) -> dict[str, float | None]:
    """{check: score}, with None where the audit was unavailable."""
    return {
        check: (outcome.result.score if outcome.result else None)
        for check, outcome in audit.items()
    }


def build_entry(turn: Turn, settings: Settings, thread_id: str = "") -> dict[str, Any]:
    """Everything worth keeping about one finished question, as plain JSON data."""
    errors = [
        message
        for message in (
            turn.primary.error,
            turn.secondary.error,
            turn.agreement_error,
            turn.evidence_error,
            turn.revise_error,
            *(outcome.error for outcome in turn.audit.values()),
            *(outcome.error for outcome in turn.secondary_audit.values()),
        )
        if message
    ]
    return {
        "timestamp": _timestamp(),
        "thread_id": thread_id,
        "question": turn.question,
        "models": _models(settings),
        "answers": {"primary": turn.primary.text, "secondary": turn.secondary.text},
        "final_answer": turn.final_answer,
        # Which answer the user saw; None = "I don't know" (no answer passed).
        "shown_answer_model": turn.shown[0] if turn.shown else None,
        # Wikipedia's own answer, shown when no AI answer passed.
        "wikipedia_answer_url": turn.wiki_answer.url if turn.wiki_answer and not turn.shown else None,
        "agreement_score": turn.agreement.score if turn.agreement else None,
        "search_query": turn.search_query,
        "evidence_urls": sorted({passage.url for passage in turn.evidence if passage.url}),
        # Attached PDF: its name and the cited parts used as evidence ("report.pdf, page 4").
        "document": turn.document_name or None,
        "document_sources": [p.title for p in turn.evidence] if turn.document_name else None,
        "not_in_document": turn.not_in_document or None,
        # True when the reply was the fact check's summary (no AI answer).
        "fact_check_summary": bool(turn.fixed_reply) or None,
        "audit_scores": _scores(turn.audit),
        "secondary_audit_scores": _scores(turn.secondary_audit) or None,
        "original_audit_scores": _scores(turn.original_audit) if turn.revisions else None,
        "revision_count": len(turn.revisions),
        "latency_seconds": {step: round(sec, 2) for step, sec in turn.latency.items()},
        "errors": errors,
    }


def build_failure_entry(
    question: str, settings: Settings, errors: list[str], thread_id: str = ""
) -> dict[str, Any]:
    """Entry for a question that no model could answer."""
    return {
        "timestamp": _timestamp(),
        "thread_id": thread_id,
        "question": question,
        "models": _models(settings),
        "answers": None,
        "errors": errors,
    }


def append_entry(entry: dict[str, Any], path: Path = LOG_PATH) -> None:
    """Append one line to the log. A logging problem must never break the chat."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            # default=str: anything json can't handle (rare) is written as text
            # instead of crashing; ensure_ascii=False keeps non-English text readable.
            file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as error:
        logger.warning("Could not write audit log %s: %s", path, error)


def read_log(path: Path = LOG_PATH) -> bytes:
    """The whole log file, for the download button (empty if nothing logged yet)."""
    return path.read_bytes() if path.exists() else b""
