"""Look and feel: CSS, hand-drawn SVG icons and small HTML building blocks.

Kept separate from app.py so the app logic is not buried in markup.
Every piece of text that comes from a model or from Wikipedia goes through
html.escape(), so it can never inject HTML into the page.
"""

from __future__ import annotations

import base64
import re
from functools import lru_cache
from html import escape
from pathlib import Path

# ------------------------------------------------------------------ formulas

# Models write formulas in LaTeX as \( ... \) (inline) or \[ ... \] (display),
# but st.markdown only renders $...$ and $$...$$. In Markdown "\(" just means
# "(", so the answer showed "(PE = 12 \times 9.8 \times 2.5 = 294\ \text{J})".
_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def latex_for_streamlit(text: str) -> str:
    """Rewrite \\( \\) and \\[ \\] formulas as $ $ and $$ $$ so Streamlit renders them.

    Code blocks (``` ... ```) are left exactly as they are.
    """
    parts = text.split("```")
    for i in range(0, len(parts), 2):  # even parts are outside code blocks
        part = _DISPLAY_MATH.sub(lambda m: f"$${m.group(1).strip()}$$", parts[i])
        parts[i] = _INLINE_MATH.sub(lambda m: f"${m.group(1).strip()}$", part)
    return "```".join(parts)

# ------------------------------------------------------------------ colours
# Taken from the VerdictAi logo: a very dark violet background and the violet
# of its "V" and "Ai". Status colours are brightened to read well on dark.

STATUS_COLOR = {
    "pass": "#34D399",
    "warn": "#FBBF24",
    "fail": "#F87171",
    "unavailable": "#8B84A8",
}
STATUS_TINT = {  # translucent card backgrounds for the same statuses
    "pass": "rgba(52, 211, 153, 0.10)",
    "warn": "rgba(251, 191, 36, 0.10)",
    "fail": "rgba(248, 113, 113, 0.10)",
    "unavailable": "rgba(139, 132, 168, 0.10)",
}
STATUS_WORD = {"pass": "Good", "warn": "Moderate", "fail": "Serious", "unavailable": "Unavailable"}

CSS = """
<style>
:root {
  --vd-bg: #0D0121;
  --vd-surface: #150A33;
  --vd-surface-2: #1C1043;
  --vd-border: #2E2163;
  --vd-violet: #7C5CFF;
  --vd-violet-deep: #5B3FD9;
  --vd-violet-light: #B9A6FF;
  --vd-text: #ECE8FF;
  --vd-muted: #A39BC9;
  --vd-faint: #6F6699;
}

/* ---------- page ---------- */
.stApp {
  background:
    radial-gradient(1100px 480px at 50% -12%, rgba(124, 92, 255, 0.20), transparent 62%),
    var(--vd-bg);
}
.block-container { padding-top: 1.4rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; }

/* ---------- header ---------- */
.mp-brand { display: flex; align-items: center; gap: 12px; }
.mp-brand-name { font-size: 1.55rem; font-weight: 800; color: var(--vd-text); line-height: 1.05;
  letter-spacing: -0.02em; }
.mp-brand-name span { color: var(--vd-violet); }
.mp-brand-tag { font-size: 0.84rem; color: var(--vd-muted); margin-top: 2px; }

/* ---------- chat bubbles (styled Streamlit containers) ---------- */
/* Streamlit gives text blocks a negative bottom margin; inside a padded
   bubble that would cut the last line in half, so it is reset here. */
[class*="st-key-msg-"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
[class*="st-key-msg-user"] {
  margin-left: auto; width: fit-content; max-width: 72%;
  background: linear-gradient(135deg, var(--vd-violet), var(--vd-violet-deep));
  border-radius: 18px 18px 4px 18px; padding: 10px 16px;
  box-shadow: 0 6px 20px rgba(124, 92, 255, 0.28);
}
[class*="st-key-msg-user"] p { color: #fff; margin: 0; }
[class*="st-key-msg-bot"] {
  width: fit-content; max-width: 88%;
  background: var(--vd-surface); border: 1px solid var(--vd-border);
  border-radius: 18px 18px 18px 4px; padding: 12px 18px;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}
.mp-bubble-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  font-size: 0.78rem; color: var(--vd-muted); margin-bottom: 4px; }
.mp-label { font-size: 0.78rem; font-weight: 600; color: var(--vd-muted); text-transform: uppercase;
  letter-spacing: 0.04em; }

/* ---------- pills ---------- */
.mp-pill { display: inline-flex; align-items: center; gap: 5px; border-radius: 999px;
  padding: 2px 10px; font-size: 0.76rem; font-weight: 600; border: 1px solid transparent; }
.mp-pill-good { background: rgba(52, 211, 153, 0.12); color: #6EE7B7; border-color: rgba(52, 211, 153, 0.25); }
.mp-pill-warn { background: rgba(251, 191, 36, 0.12); color: #FCD34D; border-color: rgba(251, 191, 36, 0.25); }
.mp-pill-info { background: rgba(124, 92, 255, 0.16); color: var(--vd-violet-light); border-color: rgba(124, 92, 255, 0.35); }
.mp-pill-fail { background: rgba(248, 113, 113, 0.12); color: #FCA5A5; border-color: rgba(248, 113, 113, 0.25); }

/* ---------- audit strip under each answer ---------- */
.mp-strip { display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.78rem; color: var(--vd-muted);
  margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--vd-border); }
.mp-strip span { display: inline-flex; align-items: center; gap: 4px; }

/* ---------- audit panel ---------- */
.st-key-audit-panel {
  background: linear-gradient(180deg, var(--vd-surface-2), var(--vd-surface));
  border: 1px solid var(--vd-border); border-radius: 18px; padding: 18px 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.30);
}
.mp-panel-title { font-size: 1.2rem; font-weight: 700; color: var(--vd-text); margin-bottom: 2px; }
.mp-muted { color: var(--vd-muted); font-size: 0.82rem; }
.mp-section { font-size: 0.74rem; font-weight: 700; color: var(--vd-violet-light); text-transform: uppercase;
  letter-spacing: 0.08em; margin: 20px 0 8px; }

.mp-compare { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.mp-compare th { text-align: left; font-size: 0.74rem; color: var(--vd-muted); font-weight: 600;
  padding: 0 6px 6px; border-bottom: 1px solid var(--vd-border); word-break: break-word; }
.mp-compare td { padding: 8px 6px; border-bottom: 1px solid rgba(46, 33, 99, 0.6); color: var(--vd-text); }
.mp-compare td:first-child { font-weight: 600; padding-left: 0; }
.mp-compare tr:last-child td { border-bottom: none; }
.mp-cell { display: inline-flex; align-items: center; gap: 6px; font-weight: 700; }
.mp-note { color: var(--vd-muted); font-size: 0.76rem; margin-top: 6px; }

.mp-issue { border-left: 4px solid; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }
.mp-issue-title { font-weight: 700; font-size: 0.88rem; }
.mp-issue-quote { font-style: italic; font-size: 0.86rem; margin-top: 3px; color: var(--vd-text); }
.mp-issue-reason { font-size: 0.82rem; color: var(--vd-muted); margin-top: 3px; }

.mp-evidence { border: 1px solid var(--vd-border); border-radius: 12px; margin-bottom: 8px;
  background: rgba(13, 1, 33, 0.55); transition: border-color 0.15s ease; }
.mp-evidence:hover { border-color: var(--vd-violet); }
.mp-evidence summary { cursor: pointer; padding: 9px 12px; display: flex; justify-content: space-between;
  gap: 8px; font-weight: 600; font-size: 0.88rem; color: var(--vd-text); list-style: none; }
.mp-evidence summary::-webkit-details-marker { display: none; }
.mp-evidence[open] summary { border-bottom: 1px solid var(--vd-border); }
.mp-evidence p { font-size: 0.85rem; color: var(--vd-muted); margin: 10px 12px 6px; line-height: 1.5; }
.mp-evidence a { display: inline-block; font-size: 0.82rem; margin: 0 12px 10px; color: var(--vd-violet-light);
  text-decoration: none; font-weight: 600; }
.mp-rel { color: var(--vd-faint); font-weight: 500; font-size: 0.78rem; white-space: nowrap; }

.mp-meta { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 0.82rem; }
.mp-meta dt { color: var(--vd-muted); }
.mp-meta dd { margin: 0; color: var(--vd-text); }

.mp-callout { border-radius: 12px; padding: 12px 16px; font-size: 0.9rem; margin: 6px 0 8px;
  display: flex; gap: 10px; align-items: flex-start; }
.mp-callout-warn { background: rgba(251, 191, 36, 0.10); color: #FDE68A; border: 1px solid rgba(251, 191, 36, 0.25); }
.mp-callout-fail { background: rgba(248, 113, 113, 0.10); color: #FECACA; border: 1px solid rgba(248, 113, 113, 0.25); }

/* ---------- welcome screen (before the first question) ---------- */
.mp-empty { text-align: center; padding: 18px 16px 8px; color: var(--vd-muted); }
.mp-empty-logo { width: min(340px, 70%); height: auto; margin: 0 auto;
  filter: drop-shadow(0 18px 40px rgba(124, 92, 255, 0.30)); }
.mp-empty-title { font-size: 1.25rem; font-weight: 700; color: var(--vd-text); margin: 14px 0 6px; }
.mp-features { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;
  margin: 22px auto 0; max-width: 760px; text-align: left; }
.mp-feature { background: var(--vd-surface); border: 1px solid var(--vd-border); border-radius: 14px;
  padding: 14px 16px; }
.mp-feature b { display: block; color: var(--vd-text); font-size: 0.92rem; margin-bottom: 4px; }
.mp-feature span { font-size: 0.82rem; color: var(--vd-muted); line-height: 1.45; }
.mp-feature i { font-style: normal; color: var(--vd-violet-light); font-weight: 700; margin-right: 6px; }
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


ASSETS = Path(__file__).parent / "assets"
LOGO_FILE = ASSETS / "verdictai-logo.png"  # full logo: the "V" and the word mark
MARK_FILE = ASSETS / "verdictai-mark.png"  # the "V" alone, transparent background


@lru_cache(maxsize=4)
def _png_data_uri(path: Path) -> str:
    """A PNG as a data URI, so st.html can show it (read once, then cached)."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def logo(size: int = 38) -> str:
    """The VerdictAi "V" mark."""
    return (
        f'<img src="{_png_data_uri(MARK_FILE)}" width="{size}" height="{size}" '
        'alt="VerdictAi logo" style="flex-shrink:0">'
    )


# ------------------------------------------------------------ html pieces


def header() -> str:
    return (
        f'<div class="mp-brand">{logo()}<div>'
        '<div class="mp-brand-name">Verdict<span>Ai</span></div>'
        '<div class="mp-brand-tag">Two models answer. An auditor gives the verdict.</div>'
        "</div></div>"
    )


FEATURES = [
    ("1", "Two models answer", "Both answer your question at the same time, and their agreement is measured."),
    ("2", "An auditor checks", "Hallucination, bias, toxicity and intent, against real Wikipedia evidence."),
    ("3", "One verdict", "Only an answer that passed is shown, or an honest \u201cI don\u2019t know\u201d."),
]


def empty_chat() -> str:
    """The welcome screen before the first question: the full logo and how it works."""
    cards = "".join(
        f'<div class="mp-feature"><b><i>{n}</i>{escape(title)}</b><span>{escape(text)}</span></div>'
        for n, title, text in FEATURES
    )
    return (
        f'<div class="mp-empty"><img class="mp-empty-logo" src="{_png_data_uri(LOGO_FILE)}" alt="VerdictAi">'
        '<div class="mp-empty-title">Ask anything</div>'
        "<div>Ask in English, or attach a PDF to get answers from your own document.</div>"
        f'<div class="mp-features">{cards}</div></div>'
    )


def pill(text: str, kind: str) -> str:
    return f'<span class="mp-pill mp-pill-{kind}">{escape(text)}</span>'


def bubble_head(*pills: str) -> str:
    return f'<div class="mp-bubble-head">{"".join(pills)}</div>'


def audit_strip(items: list[tuple[str, str]]) -> str:
    """Small icon + name per check, shown under each answer. items: (label, status)."""
    spans = "".join(
        f"<span>{status_icon(status, 14)}{escape(label)}</span>" for label, status in items
    )
    return f'<div class="mp-strip">{spans}</div>'


def comparison_table(models: list[str], rows: list[tuple[str, list[tuple[str, float | None]]]]) -> str:
    """Checks as rows, models as columns: icon + score in every cell.

    rows: (check label, [(status, score) per model]).
    """
    head = "".join(f"<th>{escape(model)}</th>" for model in models)
    body = ""
    for label, cells in rows:
        tds = ""
        for status, score in cells:
            percent = "n/a" if score is None else f"{round(score * 100)}%"
            tds += (
                f'<td><span class="mp-cell" style="color:{STATUS_COLOR[status]}">'
                f"{status_icon(status, 16)}{percent}</span></td>"
            )
        body += f"<tr><td>{escape(label)}</td>{tds}</tr>"
    return f'<table class="mp-compare"><tr><th></th>{head}</tr>{body}</table>'


def issue_card(check_label: str, status: str, quote: str, reason: str) -> str:
    color = STATUS_COLOR[status]
    background = STATUS_TINT["warn" if status == "warn" else "fail"]
    title = "Potential issue" if status == "warn" else "Serious issue"
    quote_html = f'<div class="mp-issue-quote">“{escape(quote)}”</div>' if quote else ""
    return (
        f'<div class="mp-issue" style="border-color:{color};background:{background}">'
        f'<div class="mp-issue-title" style="color:{color}">{title} · {escape(check_label)}</div>'
        f'{quote_html}<div class="mp-issue-reason">{escape(reason)}</div></div>'
    )


def evidence_card(
    title: str, url: str, text: str, relevance: float, link_text: str = "Open on Wikipedia →"
) -> str:
    # A part of an attached PDF has no URL, so it gets no link.
    link = f'<a href="{escape(url)}" target="_blank" rel="noopener">{escape(link_text)}</a>' if url else ""
    return (
        '<details class="mp-evidence">'
        f'<summary><span>{escape(title)}</span><span class="mp-rel">relevance {relevance:.2f}</span></summary>'
        f"<p>{escape(text)}</p>{link}"
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
