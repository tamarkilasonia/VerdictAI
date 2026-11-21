"""Streamlit UI for ModelProof. Only UI code lives here; the logic is in other modules.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import uuid

import streamlit as st
from langgraph.types import Command

import ui
from audit_log import read_log
from auditor import CheckOutcome, needs_revision
from config import MAX_REVISIONS, Settings, get_settings
from graph import Turn, build_graph, new_question_input, turn_from_state

# A Material icon (not an emoji) as the browser-tab icon: two joined circles.
st.set_page_config(page_title="ModelProof", page_icon=":material/join_inner:", layout="wide")

CHECK_LABELS = {
    "hallucination": "Hallucination",
    "bias": "Bias",
    "toxicity": "Toxicity",
    "intent_alignment": "Intent alignment",
}
SAFETY_CHECKS = ("hallucination", "bias", "toxicity")  # lower score = better


def load_settings_or_stop() -> Settings:
    """Show a readable message instead of a crash if .env has a bad value."""
    try:
        return get_settings()
    except ValueError as error:
        st.error(f"Configuration problem: {error}")
        st.stop()


def score_of(outcome: CheckOutcome) -> float | None:
    return outcome.result.score if outcome.result else None


# ---------------------------------------------------------------- header


def render_header() -> None:
    """Logo + name on the left, conversation tools on the right."""
    brand, new_chat, download = st.columns([6, 1.3, 1.3], vertical_alignment="center")
    brand.html(ui.header())
    # A new thread_id = a new, empty conversation in the checkpointer.
    if new_chat.button("New conversation", icon=":material/add_comment:", width="stretch"):
        st.session_state.thread_id = uuid.uuid4().hex
        st.rerun()
    log = read_log()
    download.download_button(
        "Download log",
        data=log,
        file_name="audit_log.jsonl",
        mime="application/jsonl",
        icon=":material/download:",
        width="stretch",
        disabled=not log,  # nothing to download before the first answer
        help="One JSON line per question: answers, scores, evidence, latency, errors.",
    )


# ------------------------------------------------------------ chat column


def render_user_bubble(text: str, key: str) -> None:
    # The key gives the container a CSS class (st-key-...), which ui.CSS styles as a bubble.
    with st.container(key=f"msg-user-{key}"):
        st.markdown(text)


def agreement_pill(turn: Turn) -> str:
    if turn.agreement is None:
        return ui.pill("Agreement n/a", "info")
    score = f"{turn.agreement.score:.2f}"
    if turn.agreement.agree:
        return ui.pill(f"Models agree · {score}", "good")
    return ui.pill(f"Models disagree · {score}", "warn")


def render_answer_body(turn: Turn) -> None:
    """Agree -> one answer. Disagree -> both side by side. One failed -> say why."""
    if not (turn.primary.ok and turn.secondary.ok):
        working, failed = (
            (turn.primary, turn.secondary) if turn.primary.ok else (turn.secondary, turn.primary)
        )
        st.markdown(working.text)
        st.caption(f"Only {working.model} answered. {failed.model}: {failed.error}")
        return
    if turn.agreement is None or turn.agreement.agree:
        st.markdown(turn.primary.text)
        if turn.agreement_error:
            st.caption(turn.agreement_error)
        return
    for column, answer in zip(st.columns(2, gap="medium"), (turn.primary, turn.secondary)):
        with column:
            st.html(f'<div class="mp-label">{answer.model}</div>')
            st.markdown(answer.text)
    render_secondary_check(turn)


def render_secondary_check(turn: Turn) -> None:
    """When the models disagree, say whether SECONDARY's answer holds up too.

    The full audit (strip below the answers) is about PRIMARY's answer; this
    line tells the user which of the two disagreeing answers looks wrong.
    """
    check = turn.secondary_check
    if check is None:
        return
    if check.status == "pass":
        text = f"{turn.secondary.model}: no contradiction found either."
    elif check.result is None:
        text = f"{turn.secondary.model}: {check.error}"
    else:
        reasons = "; ".join(issue.reason for issue in check.result.issues[:2])
        text = f"{turn.secondary.model}'s answer: {reasons}"
    st.html(ui.audit_strip([(f"Hallucination ({turn.secondary.model})", check.status)]))
    st.caption(text)


def render_revised_body(turn: Turn) -> None:
    """Original and revised answer side by side, each with its audit verdict."""
    original, revised = st.columns(2, gap="medium")
    with original:
        st.html('<div class="mp-label">Original</div>')
        st.markdown(turn.original_answer)
        st.html(audit_strip(turn.original_audit))
    with revised:
        st.html('<div class="mp-label">Revised</div>')
        st.markdown(turn.final_answer)
        st.html(audit_strip(turn.audit))


def audit_strip(audit: dict[str, CheckOutcome]) -> str:
    return ui.audit_strip([(CHECK_LABELS[c], o.status) for c, o in audit.items()])


def render_turn(turn: Turn, index: int) -> None:
    render_user_bubble(turn.question, str(index))
    with st.container(key=f"msg-bot-{index}"):
        models = f"{turn.primary.model} + {turn.secondary.model}"
        if turn.revisions:
            revised = ui.pill(f"Revised after audit · {len(turn.revisions)} of {MAX_REVISIONS}", "info")
            st.html(ui.bubble_head(models, agreement_pill(turn), revised))
            render_revised_body(turn)
            return
        st.html(ui.bubble_head(models, agreement_pill(turn)))
        render_answer_body(turn)
        if turn.audit:
            st.html(audit_strip(turn.audit))


def render_review_controls(turn: Turn) -> str | None:
    """Human-in-the-loop: ask whether to revise. Returns "revise", "accept" or None."""
    if turn.revise_error:
        st.html(ui.callout(turn.revise_error, "fail"))
    remaining = MAX_REVISIONS - len(turn.revisions)
    st.html(
        ui.callout(
            "The audit found a serious problem with this answer. Ask the primary model "
            f"to fix it using the audit and the evidence? ({remaining} revision(s) left)"
        )
    )
    revise_column, keep_column, _ = st.columns([1.2, 1.2, 2])
    if revise_column.button("Revise answer", type="primary", icon=":material/edit:", width="stretch"):
        return "revise"
    if keep_column.button("Keep this answer", width="stretch"):
        return "accept"
    return None


# ----------------------------------------------------------- audit column


def intent_explanation(outcome: CheckOutcome) -> str:
    if outcome.result is None:
        return outcome.error or "Audit unavailable."
    if outcome.result.issues:
        return outcome.result.issues[0].reason
    return "The answer addresses what was asked."


def render_issues(audit: dict[str, CheckOutcome]) -> None:
    cards = [
        ui.issue_card(CHECK_LABELS[check], outcome.status, issue.quote, issue.reason)
        for check, outcome in audit.items()
        if outcome.result and outcome.status in ("warn", "fail")
        for issue in outcome.result.issues
    ]
    if cards:
        st.html('<div class="mp-section">Issues found</div>' + "".join(cards))


def render_evidence(turn: Turn) -> None:
    """The Wikipedia passages the hallucination check was judged against."""
    html = '<div class="mp-section">Evidence · Wikipedia</div>'
    if turn.search_query and turn.search_query != turn.question:
        # A Georgian question was translated: show what was actually searched.
        html += f'<div class="mp-muted">Searched for: “{ui.escape(turn.search_query)}”</div>'
    if turn.evidence_error:
        html += f'<div class="mp-muted">{ui.escape(turn.evidence_error)}</div>'
    elif not turn.evidence:
        html += '<div class="mp-muted">No relevant passages found; checked without evidence.</div>'
    else:
        # Relevance = cosine similarity to the answer (the re-ranking score).
        html += "".join(
            ui.evidence_card(p.title, p.url, p.text, p.score) for p in turn.evidence
        )
    st.html(html)


def render_details(turn: Turn, settings: Settings) -> None:
    rows = [
        ("Agreement", f"{turn.agreement.score:.2f}" if turn.agreement else "n/a"),
        ("Models", f"{turn.primary.model}, {turn.secondary.model}"),
        ("Auditor", settings.auditor.name),
        ("Latency", " · ".join(f"{step} {sec:.1f}s" for step, sec in turn.latency.items())),
    ]
    st.html('<div class="mp-section">Details</div>' + ui.meta_list(rows))


def render_audit_panel(turn: Turn | None, settings: Settings) -> None:
    """Details for the latest answer, laid out like a report card."""
    with st.container(key="audit-panel"):
        st.html('<div class="mp-panel-title">Answer audit</div>')
        if turn is None or not turn.audit:
            st.html('<div class="mp-muted">The audit of the latest answer appears here.</div>')
            return

        audit = turn.audit
        st.html(f'<div class="mp-muted">Audited answer by {turn.final_model}</div>')
        if turn.answer_translation:
            # The evidence was ranked against this translation, and the quotes
            # in "Issues found" come from it.
            with st.expander("Georgian answer: checked in English translation"):
                st.markdown(turn.answer_translation)
        if "intent_alignment" in audit:
            intent = audit["intent_alignment"]
            st.html(ui.intent_card(intent.status, score_of(intent), intent_explanation(intent)))

        rows = "".join(
            ui.metric_row(CHECK_LABELS[c], audit[c].status, score_of(audit[c]))
            for c in SAFETY_CHECKS
            if c in audit
        )
        st.html('<div class="mp-section">Content safety</div>' + rows)

        if needs_revision(audit) and len(turn.revisions) >= MAX_REVISIONS:
            st.html(ui.callout(f"Still failing after {MAX_REVISIONS} revisions: treat with caution.", "fail"))

        render_issues(audit)
        render_evidence(turn)
        render_details(turn, settings)


# ------------------------------------------------------------------- flow


@st.cache_resource
def get_graph():
    """Build the graph once for the whole server, not on every rerun.

    It holds one in-memory checkpointer; conversations stay separate because
    each browser tab uses its own thread_id.
    """
    return build_graph()


def thread_config() -> dict:
    """Tells LangGraph which conversation (thread) to load and save."""
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# What the status box says while the NEXT step runs, keyed by the step that just finished.
NEXT_STEP_LABEL = {
    "generate_answers": "Comparing answers…",
    "compare": "Checking evidence…",
    "retrieve_evidence": "Auditing…",
    "human_review": "Revising…",
    "revise": "Re-auditing the revised answer…",
    "audit": "Saving…",
    "__interrupt__": "Waiting for your review",
}


def pending_turn() -> Turn | None:
    """The current turn if the graph is paused at human_review, else None."""
    snapshot = get_graph().get_state(thread_config())
    return turn_from_state(snapshot.values) if "human_review" in snapshot.next else None


def run_graph(graph_input, first_label: str) -> str | None:
    """Run (or resume) the graph, showing each step. Returns an error message, or None.

    graph_input is either a new question's input, or Command(resume=...) to
    continue a run that is paused at human_review.
    """
    graph, config = get_graph(), thread_config()
    with st.status(first_label) as status:
        try:
            # stream_mode="updates" yields after every finished node, which
            # lets us show the current step instead of one long spinner.
            for update in graph.stream(graph_input, config, stream_mode="updates"):
                for node in update:
                    status.update(label=NEXT_STEP_LABEL.get(node, "Working…"))
        except Exception as error:  # noqa: BLE001 - never show a stack trace
            status.update(label="Failed", state="error")
            return f"Something went wrong: {type(error).__name__}: {error}"
        error = graph.get_state(config).values.get("error")
        status.update(label="Failed" if error else "Done", state="error" if error else "complete")
    return error


def main() -> None:
    settings = load_settings_or_stop()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex  # one conversation per tab
    st.markdown(ui.CSS, unsafe_allow_html=True)
    render_header()

    # The checkpointer is the single source of truth for the conversation.
    turns: list[Turn] = get_graph().get_state(thread_config()).values.get("turns", [])
    waiting = pending_turn()  # a turn paused at human_review is shown too
    shown = turns + [waiting] if waiting else turns

    chat_column, audit_column = st.columns([3, 2], gap="large")

    # Streamlit re-runs this whole script on every interaction, so the page
    # is redrawn from the saved turns each time.
    with chat_column:
        if not shown:
            st.html(ui.empty_chat())
        for index, turn in enumerate(shown):
            render_turn(turn, index)
        decision = render_review_controls(waiting) if waiting else None
    with audit_column:
        render_audit_panel(shown[-1] if shown else None, settings)

    if decision:
        # Resume the paused run: interrupt() inside human_review returns `decision`.
        with chat_column:
            label = "Revising…" if decision == "revise" else "Saving…"
            error = run_graph(Command(resume=decision), label)
            if error:
                st.error(error)
                return
        st.rerun()

    question = st.chat_input("Ask a question… · დასვით კითხვა…")
    if question:
        if waiting:
            # A new question means "keep the previous answer": close that turn
            # first so it is saved to memory before the new one starts.
            with chat_column:
                if error := run_graph(Command(resume="accept"), "Saving previous answer…"):
                    st.error(error)
                    return
        with chat_column:
            render_user_bubble(question, "pending")
            error = run_graph(new_question_input(question), "Asking models…")
            if error:
                st.error(error)  # failed questions are not saved; just ask again
                return
        st.rerun()  # redraw so the new turn and its audit appear


main()
