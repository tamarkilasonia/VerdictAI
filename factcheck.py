"""Fact-checking an attached PDF against Wikipedia, right after it is attached.

Here the source of truth is Wikipedia, not the document:
  1. Extract: the primary model lists the document's factual claims (one per
     sentence, with its page and a Wikipedia search query), reading it a few
     pages at a time. All claims are counted; if there are more than
     MAX_CLAIMS, they are picked from all pages in turn (choose_claims), so
     the end of a long document is checked too.
  2. Evidence: Wikipedia is searched for each claim (retrieval.get_evidence,
     plain HTTP, no AI service).
  3. Audit: the existing auditor (auditor.run_audit, unchanged) checks
     CLAIMS_PER_AUDIT claims at a time, given as a numbered list with their
     Wikipedia passages. Each claim it marks CONTRADICTED is wrong; each it
     marks UNVERIFIED, or that has no Wikipedia passage of its own, "can't be
     checked"; the rest are confirmed (the auditor lists only problems).
There is no web search: what Wikipedia cannot settle stays "can't be checked".

The results also warn about answers built on a wrong claim (warnings_for),
using the app's local embedding model, so no AI service is asked again.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from agreement import cosine_similarity, embed
from auditor import Issue, auditor_reasoning_effort, is_contradiction, run_audit
from config import get_settings
from documents import Document
from models import friendly_error, get_chat_model, reply_text
from retrieval import Passage, get_evidence

MAX_CLAIMS = 20
CLAIMS_PER_AUDIT = 5  # one auditor request checks this many claims (saves Groq's limit)
PAGES_PER_EXTRACTION = 3  # pages read per extraction request (keeps each reply short)
WIKIPEDIA_WORKERS = 4  # parallel Wikipedia searches; few, to be polite to Wikimedia
# Groq allows 8,000 tokens per minute per model and one batch uses ~3,000, so
# the 4th batch in a row hit the limit (tested). Then the check waits this
# long and tries that batch once more. A used-up DAILY limit is not retried.
PER_MINUTE_LIMIT_WAIT = 30
PER_MINUTE_LIMIT_MESSAGE = "Free usage limit reached"  # see models.friendly_error

CONFIRMED, CONTRADICTED, UNCHECKED = "confirmed", "contradicted", "unchecked"
RESULT_LABELS = {
    CONFIRMED: "✅ confirmed",
    CONTRADICTED: "❌ contradicted",
    UNCHECKED: "⚠️ can't be checked",
}

EXTRACT_PROMPT = (
    "List the factual claims in the user's text: statements about the world that can be "
    "checked, such as who did what, when, where, or how many. Skip opinions, advice, "
    "questions, headings and statements about the document itself. Give one claim per "
    "sentence of the text: do not split a sentence into several claims. Each claim must make "
    "sense on its own, so replace every 'he', 'she', 'it', 'this' or 'the document' with the "
    "name it refers to (e.g. 'She won it twice.' after a sentence about Marie Curie and the "
    "Nobel Prize becomes 'Marie Curie won the Nobel Prize twice.'), "
    "and keep every name, date and number exactly as the text gives it, even if you think "
    "it is wrong. For each claim, give the page number from the [Page N] marker before it, "
    "and a short English Wikipedia search query (1 to 5 words: the article it belongs to). "
    "Reply with ONE JSON object and nothing else:\n"
    '{"claims": [{"page": <number>, "claim": "<sentence>", "query": "<search query>"}]}\n'
    'If there are no factual claims, reply {"claims": []}.'
)


@dataclass
class Claim:
    page: int
    text: str
    query: str  # Wikipedia search query


@dataclass
class ClaimResult:
    claim: Claim
    status: str  # CONFIRMED, CONTRADICTED or UNCHECKED
    correction: str = ""  # what Wikipedia says instead (contradicted claims)
    note: str = ""  # why it could not be checked
    source_title: str = ""
    source_url: str = ""


@dataclass
class FactCheck:
    document_name: str
    total_claims: int = 0  # all claims found in the document
    results: list[ClaimResult] = field(default_factory=list)  # the checked ones
    error: str = ""  # set if the check could not run at all

    def count(self, status: str) -> int:
        return sum(result.status == status for result in self.results)


# ------------------------------------------------------------------ extraction


def page_texts(document: Document) -> list[tuple[int, str]]:
    """(page number, text) for each page that has text, rebuilt from the chunks."""
    pages: dict[int, list[str]] = {}
    for chunk in document.chunks:
        pages.setdefault(chunk.page, []).append(chunk.text)
    return [(page, " ".join(parts)) for page, parts in sorted(pages.items())]


_CLAIM_OBJECT = re.compile(r"\{[^{}]*\}")


def parse_claims(reply: str, valid_pages: set[int]) -> list[Claim]:
    """Claims from the model's JSON reply; malformed entries are skipped.

    A reply cut off by the length limit is not valid JSON as a whole, so then
    every complete {...} claim object in it is used.
    """
    reply = re.sub(r"```(?:json)?", "", reply)
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(reply[start : end + 1]).get("claims", [])
    except (ValueError, AttributeError):
        items = []
        for match in _CLAIM_OBJECT.finditer(reply[start + 1 :]):
            try:
                items.append(json.loads(match.group(0)))
            except ValueError:
                continue
    claims = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not str(item.get("claim", "")).strip():
            continue
        try:
            page = int(item.get("page", 0))
        except (TypeError, ValueError):
            page = 0
        page = page if page in valid_pages else min(valid_pages, default=0)
        text = " ".join(str(item["claim"]).split())
        claims.append(Claim(page=page, text=text, query=str(item.get("query", "")).strip() or text))
    return claims


def extract_claims(document: Document) -> list[Claim]:
    """All factual claims in the document, in page order (duplicates removed)."""
    pages = page_texts(document)
    spec = get_settings().primary
    # No JSON mode: Groq rejects a JSON-mode reply that is cut off, and gpt-oss
    # spent 1,694 of its 2,048 output tokens thinking, which cut the list off.
    # Low reasoning effort leaves room for the claims.
    model = get_chat_model(spec, temperature=0.0, reasoning_effort=auditor_reasoning_effort(spec.name))
    claims: list[Claim] = []
    seen: set[str] = set()
    for i in range(0, len(pages), PAGES_PER_EXTRACTION):
        group = pages[i : i + PAGES_PER_EXTRACTION]
        text = "\n\n".join(f"[Page {page}]\n{content}" for page, content in group)
        reply = reply_text(model.invoke([SystemMessage(EXTRACT_PROMPT), HumanMessage(text)]))
        for claim in parse_claims(reply, {page for page, _ in group}):
            key = claim.text.lower()
            if key not in seen:
                seen.add(key)
                claims.append(claim)
    return sorted(claims, key=lambda claim: claim.page)  # stable: keeps order within a page


def choose_claims(claims: list[Claim], limit: int) -> list[Claim]:
    """Up to `limit` claims, taken from every page in turn, returned in page order.

    Taking the first `limit` would leave the last pages of a long document
    unchecked; one claim per page per round spreads the check over all of it.
    """
    if len(claims) <= limit:
        return claims
    by_page: dict[int, list[Claim]] = {}
    for claim in claims:
        by_page.setdefault(claim.page, []).append(claim)
    chosen: list[Claim] = []
    while len(chosen) < limit:
        for page_claims in by_page.values():
            if page_claims and len(chosen) < limit:
                chosen.append(page_claims.pop(0))
    order = {id(claim): index for index, claim in enumerate(claims)}
    return sorted(chosen, key=lambda claim: order[id(claim)])


# ------------------------------------------------------------------ checking


def _evidence_for(claim: Claim, user_agent: str) -> list[Passage]:
    """Wikipedia passages for one claim; [] if Wikipedia has nothing or fails."""
    try:
        return get_evidence(claim.query, claim.text, user_agent)
    except Exception:  # noqa: BLE001 - no internet etc.: the claim can't be checked
        return []


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def match_issue(issue: Issue, claims: list[Claim]) -> int | None:
    """Index of the claim an auditor issue is about, or None.

    The issue quotes "exact words from the answer"; the answer is the
    numbered list of claims, so the quote is part of one claim (a leading
    "3." is ignored). Small rewordings are matched by similarity.
    """
    quote = _normalise(re.sub(r"^\s*\d+[.)]\s*", "", issue.quote))
    if not quote:
        return None
    for index, claim in enumerate(claims):
        if quote in _normalise(claim.text):
            return index
    scores = [SequenceMatcher(None, quote, _normalise(claim.text)).ratio() for claim in claims]
    best = max(range(len(claims)), key=scores.__getitem__)
    return best if scores[best] >= 0.6 else None


def _correction(issue: Issue) -> str:
    """The auditor's explanation without its "CONTRADICTED:" label."""
    return re.sub(r"^\s*CONTRADICTED\s*:?\s*", "", issue.reason, flags=re.IGNORECASE).strip()


def check_batch(claims: list[Claim], evidence: list[list[Passage]]) -> list[ClaimResult]:
    """Audit a few claims in one request, each with its own Wikipedia passages."""
    numbered = "\n".join(f"{i}. {claim.text}" for i, claim in enumerate(claims, start=1))
    passages: list[str] = []
    for found in evidence:
        for passage in found:
            line = f"[{passage.title}] {passage.text}"
            if line not in passages:
                passages.append(line)
    if not passages:  # nothing to check against: don't spend an auditor request
        return [ClaimResult(claim, UNCHECKED, note="No relevant Wikipedia passage found.") for claim in claims]
    audit = run_audit("Are these statements from a document true?", numbered, [], passages)
    hallucination = audit.get("hallucination")
    if hallucination is None or hallucination.result is None:
        note = hallucination.error if hallucination and hallucination.error else "The auditor was unavailable."
        return [ClaimResult(claim, UNCHECKED, note=note) for claim in claims]

    contradicted: dict[int, Issue] = {}
    unverified: set[int] = set()
    for issue in hallucination.result.issues:
        index = match_issue(issue, claims)
        if index is None:
            continue
        if is_contradiction(issue):
            contradicted[index] = issue
        else:
            unverified.add(index)

    results = []
    for index, (claim, found) in enumerate(zip(claims, evidence)):
        top = found[0] if found else None
        source = {"source_title": top.title, "source_url": top.url} if top else {}
        if not found:
            # Without a Wikipedia passage of its own, nothing can back a verdict.
            results.append(ClaimResult(claim, UNCHECKED, note="No relevant Wikipedia passage found."))
        elif index in contradicted:
            results.append(ClaimResult(claim, CONTRADICTED, correction=_correction(contradicted[index]), **source))
        elif index in unverified:
            results.append(
                ClaimResult(claim, UNCHECKED, note="Wikipedia neither confirms nor denies it.", **source)
            )
        else:
            results.append(ClaimResult(claim, CONFIRMED, **source))
    return results


def fact_check(
    document: Document, progress: Callable[[str, int, int], None] | None = None
) -> FactCheck:
    """Extract and check the document's claims. Never raises.

    progress(label, done, total) is called as the work advances, for the UI.
    """
    report = progress or (lambda label, done, total: None)
    check = FactCheck(document.name)
    try:
        report("Finding the document's facts…", 0, 0)
        claims = extract_claims(document)
    except Exception as error:  # noqa: BLE001 - e.g. rate limit: shown, never raised
        check.error = f"The facts could not be extracted: {friendly_error(error, get_settings().primary)}"
        return check
    check.total_claims = len(claims)
    claims = choose_claims(claims, MAX_CLAIMS)
    total = len(claims)

    user_agent = get_settings().wiki_user_agent
    evidence: list[list[Passage]] = []
    with ThreadPoolExecutor(max_workers=WIKIPEDIA_WORKERS) as pool:
        for done, found in enumerate(pool.map(lambda c: _evidence_for(c, user_agent), claims), start=1):
            evidence.append(found)
            report("Searching Wikipedia…", done, total)

    report("Checking facts…", 0, total)
    for start in range(0, total, CLAIMS_PER_AUDIT):
        end = start + CLAIMS_PER_AUDIT
        results = check_batch(claims[start:end], evidence[start:end])
        if hit_per_minute_limit(results):
            report("Waiting for the service's per-minute limit…", len(check.results), total)
            time.sleep(PER_MINUTE_LIMIT_WAIT)
            results = check_batch(claims[start:end], evidence[start:end])
        check.results += results
        report("Checking facts…", len(check.results), total)
    return check


def hit_per_minute_limit(results: list[ClaimResult]) -> bool:
    """True if a whole batch failed because of the per-minute (not daily) limit."""
    return bool(results) and all(
        r.status == UNCHECKED and PER_MINUTE_LIMIT_MESSAGE in r.note for r in results
    )


# ------------------------------------------------------------------ warnings

# An answer "built on" a wrong claim restates it, so they are close in meaning.
# Measured with the app's embedding model on real answers (see the README).
WARNING_SIMILARITY = 0.6


def warnings_for(answer: str, check: FactCheck | None, pages: set[int]) -> list[ClaimResult]:
    """Contradicted claims that this answer repeats.

    Only claims from the pages the answer was given (`pages`) are considered,
    and only those whose meaning is close to the answer. Uses the local
    embedding model; no AI service is asked.
    """
    if check is None:
        return []
    wrong = [r for r in check.results if r.status == CONTRADICTED and r.claim.page in pages]
    if not wrong or not answer.strip():
        return []
    vectors = embed([answer] + [result.claim.text for result in wrong])
    return [
        result
        for result, vector in zip(wrong, vectors[1:])
        if cosine_similarity(vectors[0], vector) >= WARNING_SIMILARITY
    ]


def warning_text(result: ClaimResult) -> str:
    """E.g. 'The document says "...", but this is wrong: ... (source: [Title](url)).'"""
    source = f" (source: [{result.source_title}]({result.source_url}))" if result.source_url else ""
    return f'The document says "{result.claim.text}", but this is wrong: {result.correction}{source}'


def pages_of(passages: list[Passage]) -> set[int]:
    """Page numbers of document passages titled like "report.pdf, page 4"."""
    pages = set()
    for passage in passages:
        match = re.search(r", page (\d+)$", passage.title)
        if match:
            pages.add(int(match.group(1)))
    return pages


# ------------------------------------------------- "is this document correct?"

# Questions about the whole document's accuracy ("is this information
# correct?") are answered with the fact check's summary instead of searching
# the document for an answer (which found nothing and said "Not in file").
# They are recognised by meaning, with the local embedding model: measured on
# 20 questions, such requests scored 0.67-1.00 against these examples, and
# ordinary questions about the document at most 0.54 (a question about one
# fact, "Was Julius Caesar assassinated in 34 BC?", scored 0.21).
CHECK_REQUEST_EXAMPLES = [
    "Is this information correct?",
    "Is this document accurate?",
    "Check this file.",
    "Fact-check this document.",
    "Are there any mistakes in this document?",
    "Are there any errors or wrong dates in this file?",
    "Is everything in this file true?",
    "Verify the facts in this PDF.",
    "How reliable is this document?",
    "Which facts in the file are wrong?",
]
CHECK_REQUEST_SIMILARITY = 0.62


@lru_cache(maxsize=1)
def _check_request_vectors() -> np.ndarray:
    vectors = embed(CHECK_REQUEST_EXAMPLES)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def asks_to_check_document(question: str) -> bool:
    """True if the question asks whether the whole document is correct."""
    vector = embed([question])[0]
    vector = vector / (np.linalg.norm(vector) or 1)
    return float((_check_request_vectors() @ vector).max()) >= CHECK_REQUEST_SIMILARITY


def summary_reply(check: FactCheck | None, document_name: str) -> str:
    """The chat reply for "is this document correct?": the fact check in short.

    Lists every contradicted claim with what Wikipedia says instead, and
    points to the full table under the file name.
    """
    if check is None:
        return (
            f"The fact check of {document_name} is not available (it may have failed "
            "because of the service's limit). Attach the file again to run it."
        )
    checked = len(check.results)
    if not checked:
        return f"I found no factual claims to check in {document_name}."
    scope = f"{checked} of {check.total_claims}" if check.total_claims > checked else f"all {checked}"
    lines = [
        f"I checked {scope} factual claims in **{document_name}** against Wikipedia: "
        f"**{check.count(CONFIRMED)} ✅ confirmed**, **{check.count(CONTRADICTED)} ❌ contradicted**, "
        f"**{check.count(UNCHECKED)} ⚠️ can't be checked**."
    ]
    wrong = [r for r in check.results if r.status == CONTRADICTED]
    if wrong:
        lines.append("\n**These claims are wrong:**")
        for r in wrong:
            source = f" ([{r.source_title}]({r.source_url}))" if r.source_url else ""
            lines.append(f'- **Page {r.claim.page}:** "{r.claim.text}" → {r.correction}{source}')
    else:
        lines.append("\nNo claim was contradicted by Wikipedia.")
    if check.count(UNCHECKED):
        lines.append(
            "\n⚠️ *Can't be checked* means Wikipedia neither confirms nor denies the claim; "
            "it is not the same as wrong."
        )
    lines.append("\nThe full table with every claim, its page and its source is under the file name, above the input field.")
    return "\n".join(lines)
