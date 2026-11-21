"""Look and feel: CSS, hand-drawn SVG icons and small HTML building blocks.

Kept separate from app.py so the app logic is not buried in markup.
Every piece of text that comes from a model or from Wikipedia goes through
html.escape(), so it can never inject HTML into the page.
"""

from __future__ import annotations

import base64
from html import escape

# ------------------------------------------------------------------ colours

STATUS_COLOR = {
    "pass": "#1f9d55",
    "warn": "#c77c02",
    "fail": "#d93636",
    "unavailable": "#8a94a6",
}
STATUS_WORD = {"pass": "Good", "warn": "Moderate", "fail": "Serious", "unavailable": "Unavailable"}

CSS = """
<style>
/* ---------- page ---------- */
.block-container { padding-top: 1.6rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; }

/* ---------- header ---------- */
.mp-brand { display: flex; align-items: center; gap: 12px; }
.mp-brand-name { font-size: 1.45rem; font-weight: 700; color: #1c2333; line-height: 1.1; }
.mp-brand-tag { font-size: 0.85rem; color: #6b7385; }

/* ---------- chat bubbles (styled Streamlit containers) ---------- */
/* Streamlit gives text blocks a negative bottom margin; inside a padded
   bubble that would cut the last line in half, so it is reset here. */
[class*="st-key-msg-"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
[class*="st-key-msg-user"] {
  margin-left: auto; width: fit-content; max-width: 72%;
  background: #3b6fe0; border-radius: 18px 18px 4px 18px; padding: 10px 16px;
}
[class*="st-key-msg-user"] p { color: #fff; margin: 0; }
[class*="st-key-msg-bot"] {
  width: fit-content; max-width: 88%;
  background: #ffffff; border: 1px solid #e3e7ee;
  border-radius: 18px 18px 18px 4px; padding: 12px 18px;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.mp-bubble-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  font-size: 0.78rem; color: #6b7385; margin-bottom: 2px; }
.mp-label { font-size: 0.78rem; font-weight: 600; color: #6b7385; text-transform: uppercase;
  letter-spacing: 0.04em; }

/* ---------- pills ---------- */
.mp-pill { display: inline-flex; align-items: center; gap: 5px; border-radius: 999px;
  padding: 2px 10px; font-size: 0.76rem; font-weight: 600; }
.mp-pill-good { background: #e7f6ee; color: #1f7a45; }
.mp-pill-warn { background: #fdf3e1; color: #9a5f00; }
.mp-pill-info { background: #e8eefc; color: #2f58b8; }

/* ---------- audit strip under each answer ---------- */
.mp-strip { display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.78rem; color: #6b7385;
  margin-top: 6px; }
.mp-strip span { display: inline-flex; align-items: center; gap: 4px; }

/* ---------- audit panel ---------- */
.st-key-audit-panel { background: #ffffff; border: 1px solid #e3e7ee; border-radius: 16px;
  padding: 18px 20px; }
.mp-panel-title { font-size: 1.2rem; font-weight: 700; color: #1c2333; margin-bottom: 2px; }
.mp-muted { color: #6b7385; font-size: 0.82rem; }
.mp-section { font-size: 0.8rem; font-weight: 700; color: #4a5468; text-transform: uppercase;
  letter-spacing: 0.05em; margin: 18px 0 8px; }

.mp-intent { background: #f1f5fe; border-radius: 12px; padding: 14px 16px; margin-top: 12px; }
.mp-intent-text { color: #3f485a; font-size: 0.88rem; margin-top: 8px; line-height: 1.45; }

.mp-row { display: flex; align-items: center; gap: 10px; }
.mp-row-name { font-weight: 600; color: #1c2333; flex: 1; font-size: 0.93rem; }
.mp-row-score { font-weight: 700; font-size: 0.93rem; }
.mp-row-word { color: #6b7385; font-size: 0.8rem; min-width: 74px; text-align: right; }
.mp-metric { padding: 9px 0; border-bottom: 1px solid #f0f2f6; }
.mp-metric:last-child { border-bottom: none; }
.mp-bar { height: 4px; background: #eef1f5; border-radius: 4px; margin: 7px 0 0 30px; }
.mp-bar > div { height: 4px; border-radius: 4px; }

.mp-issue { border-left: 4px solid; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }
.mp-issue-title { font-weight: 700; font-size: 0.88rem; }
.mp-issue-quote { font-style: italic; font-size: 0.86rem; margin-top: 3px; color: #3f485a; }
.mp-issue-reason { font-size: 0.82rem; color: #5b6476; margin-top: 3px; }

.mp-evidence { border: 1px solid #e3e7ee; border-radius: 10px; margin-bottom: 8px; background: #fbfcfd; }
.mp-evidence summary { cursor: pointer; padding: 9px 12px; display: flex; justify-content: space-between;
  gap: 8px; font-weight: 600; font-size: 0.88rem; color: #1c2333; list-style: none; }
.mp-evidence summary::-webkit-details-marker { display: none; }
.mp-evidence[open] summary { border-bottom: 1px solid #eef1f5; }
.mp-evidence p { font-size: 0.85rem; color: #3f485a; margin: 10px 12px 6px; line-height: 1.45; }
.mp-evidence a { display: inline-block; font-size: 0.82rem; margin: 0 12px 10px; color: #3b6fe0;
  text-decoration: none; font-weight: 600; }
.mp-rel { color: #6b7385; font-weight: 500; font-size: 0.78rem; white-space: nowrap; }

.mp-meta { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 0.82rem; }
.mp-meta dt { color: #6b7385; }
.mp-meta dd { margin: 0; color: #1c2333; }

.mp-callout { border-radius: 12px; padding: 12px 16px; font-size: 0.9rem; margin: 6px 0 8px;
  display: flex; gap: 10px; align-items: flex-start; }
.mp-callout-warn { background: #fdf3e1; color: #7a4b00; }
.mp-callout-fail { background: #fdeaea; color: #8f1f1f; }

.mp-empty { text-align: center; padding: 48px 16px; color: #6b7385; }
.mp-empty-title { font-size: 1.15rem; font-weight: 700; color: #1c2333; margin: 12px 0 6px; }
</style>
"""

# ------------------------------------------------------------------- icons
# Simple line icons drawn here (no emoji), coloured by status.
# Streamlit's HTML sanitizer removes inline <svg> tags, so each icon is
# turned into an <img> whose source is the SVG itself (a "data URI").


def _svg_img(svg_body: str, view_box: str, size: int, alt: str) -> str:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" fill="none">{svg_body}</svg>'
    data = base64.b64encode(svg.encode()).decode()
    return (
        f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" '
        f'alt="{alt}" style="flex-shrink:0;vertical-align:middle">'
    )

_ICON_PATHS = {
    "pass": '<circle cx="12" cy="12" r="9"/><path d="M8 12.4l2.7 2.7L16.2 9.4"/>',
    "warn": '<path d="M12 3.6L21.4 20H2.6z"/><path d="M12 10v4.4"/><path d="M12 17.2v.2"/>',
    "fail": '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
    "unavailable": '<circle cx="12" cy="12" r="9"/><path d="M8.5 12h7"/>',
}


def status_icon(status: str, size: int = 20) -> str:
    color = STATUS_COLOR[status]
    body = (
        f'<g stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{_ICON_PATHS[status]}</g>"
    )
    return _svg_img(body, "0 0 24 24", size, STATUS_WORD[status])


def logo(size: int = 34) -> str:
    """Two overlapping circles: two models whose answers should overlap."""
    body = (
        '<circle cx="15" cy="20" r="11" stroke="#3b6fe0" stroke-width="3"/>'
        '<circle cx="25" cy="20" r="11" stroke="#1c2333" stroke-width="3"/>'
        '<path d="M20 10.2a11 11 0 0 1 0 19.6a11 11 0 0 1 0-19.6z" fill="#3b6fe0" opacity="0.3"/>'
    )
    return _svg_img(body, "0 0 40 40", size, "ModelProof logo")


# ------------------------------------------------------------ html pieces


def header() -> str:
    return (
        f'<div class="mp-brand">{logo()}<div>'
        '<div class="mp-brand-name">ModelProof</div>'
        '<div class="mp-brand-tag">A chat that does not blindly trust one LLM</div>'
        "</div></div>"
    )


def empty_chat() -> str:
    return (
        f'<div class="mp-empty">{logo(56)}'
        '<div class="mp-empty-title">Ask anything</div>'
        "<div>Two models answer, their agreement is measured, and an auditor checks "
        "the answer against Wikipedia.</div></div>"
    )


def pill(text: str, kind: str) -> str:
    return f'<span class="mp-pill mp-pill-{kind}">{escape(text)}</span>'


def bubble_head(models: str, *pills: str) -> str:
    return f'<div class="mp-bubble-head"><span>{escape(models)}</span>{"".join(pills)}</div>'


def audit_strip(items: list[tuple[str, str]]) -> str:
    """Small icon + name per check, shown under each answer. items: (label, status)."""
    spans = "".join(
        f"<span>{status_icon(status, 14)}{escape(label)}</span>" for label, status in items
    )
    return f'<div class="mp-strip">{spans}</div>'


def metric_row(label: str, status: str, score: float | None) -> str:
    """One check as a row: icon, name, percentage, word and a thin bar."""
    color = STATUS_COLOR[status]
    percent = "n/a" if score is None else f"{round(score * 100)}%"
    width = 0 if score is None else round(score * 100)
    return (
        '<div class="mp-metric"><div class="mp-row">'
        f"{status_icon(status)}"
        f'<span class="mp-row-name">{escape(label)}</span>'
        f'<span class="mp-row-score" style="color:{color}">{percent}</span>'
        f'<span class="mp-row-word">{STATUS_WORD[status]}</span></div>'
        f'<div class="mp-bar"><div style="width:{width}%;background:{color}"></div></div></div>'
    )


def intent_card(status: str, score: float | None, explanation: str) -> str:
    color = STATUS_COLOR[status]
    percent = "n/a" if score is None else f"{round(score * 100)}%"
    return (
        '<div class="mp-intent"><div class="mp-row">'
        f"{status_icon(status, 22)}"
        '<span class="mp-row-name">Intent alignment</span>'
        f'<span class="mp-row-score" style="color:{color}">{percent}</span>'
        f'<span class="mp-row-word">{STATUS_WORD[status]}</span></div>'
        f'<div class="mp-intent-text">{escape(explanation)}</div></div>'
    )


def issue_card(check_label: str, status: str, quote: str, reason: str) -> str:
    color = STATUS_COLOR[status]
    background = "#fdf6e8" if status == "warn" else "#fdeeee"
    title = "Potential issue" if status == "warn" else "Serious issue"
    quote_html = f'<div class="mp-issue-quote">“{escape(quote)}”</div>' if quote else ""
    return (
        f'<div class="mp-issue" style="border-color:{color};background:{background}">'
        f'<div class="mp-issue-title" style="color:{color}">{title} · {escape(check_label)}</div>'
        f'{quote_html}<div class="mp-issue-reason">{escape(reason)}</div></div>'
    )


def evidence_card(title: str, url: str, text: str, relevance: float) -> str:
    return (
        '<details class="mp-evidence">'
        f'<summary><span>{escape(title)}</span><span class="mp-rel">relevance {relevance:.2f}</span></summary>'
        f"<p>{escape(text)}</p>"
        f'<a href="{escape(url)}" target="_blank" rel="noopener">Open on Wikipedia →</a>'
        "</details>"
    )


def meta_list(rows: list[tuple[str, str]]) -> str:
    items = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows)
    return f'<dl class="mp-meta">{items}</dl>'


def callout(text: str, kind: str = "warn") -> str:
    status = "warn" if kind == "warn" else "fail"
    return (
        f'<div class="mp-callout mp-callout-{kind}">{status_icon(status, 20)}'
        f"<div>{escape(text)}</div></div>"
    )
