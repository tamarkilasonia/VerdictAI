"""Streamlit UI for ModelProof. Only UI code lives here; the logic is in other modules.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import uuid

import streamlit as st
from langgraph.types import Command

import ui
from audit_log import read_log
from auditor import CheckOutcome
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


def audit_strip(audit: dict[str, CheckOutcome]) -> str:
    return ui.audit_strip([(CHECK_LABELS[c], o.status) for c, o in audit.items()])


def shown_audit(turn: Turn) -> dict[str, CheckOutcome]:
    """The audit of the answer shown in the chat ({} if none is shown)."""
    for model, text, audit in turn.candidates:
        if (model, text) == turn.shown:
            return audit
    return {}


def failed_checks(audit: dict[str, CheckOutcome]) -> str:
    return ", ".join(CHECK_LABELS[c] for c, o in audit.items() if o.status == "fail")


def render_turn(turn: Turn, index: int) -> None:
    """ONE answer: the one that passed the audit, or an honest "I don't know".

    Both answers and their analyses are in the audit panel on the right.
    """
    render_user_bubble(turn.question, str(index))
    with st.container(key=f"msg-bot-{index}"):
        if turn.shown is None:
            # No AI answer passed: Wikipedia's own text, or "I don't know".
            pill = ui.pill("Answer from Wikipedia", "info") if turn.wiki_answer else ui.pill("No reliable answer", "fail")
            st.html(ui.bubble_head(pill))
            st.markdown(ui.latex_for_streamlit(turn.reply))
            return
        model, text = turn.shown
        audit = shown_audit(turn)
        # Agreement is part of the analysis (panel), not of the one answer shown.
        pills = [ui.pill(f"Fact-checked · {model}", "good")]
        if turn.revisions and model == turn.primary.model:
            pills.append(ui.pill(f"Revised after audit · {len(turn.revisions)} of {MAX_REVISIONS}", "info"))
        if any(outcome.status == "unavailable" for outcome in audit.values()):
            # Not hidden (a rate limit is not a wrong fact), but said openly.
            pills.append(ui.pill("Not fully checked", "warn"))
        st.html(ui.bubble_head(*pills))
        st.markdown(ui.latex_for_streamlit(text))
        st.html(audit_strip(audit))
        if turn.agreement_error:
            st.caption(turn.agreement_error)


def render_review_controls(turn: Turn) -> str | None:
    """Human-in-the-loop: ask whether to revise. Returns "revise", "accept" or None."""
    if turn.revise_error:
        st.html(ui.callout(turn.revise_error, "fail"))
    remaining = MAX_REVISIONS - len(turn.revisions)
    st.html(
        ui.callout(
            f"No AI answer passed the fact check. Ask {turn.primary.model} to write a new "
            f"answer from the audit and the evidence? ({remaining} revision(s) left)"
        )
    )
    revise_column, keep_column, _ = st.columns([1.2, 1.4, 2])
    if revise_column.button("Revise answer", type="primary", icon=":material/edit:", width="stretch"):
        return "revise"
    keep_label = "Keep the Wikipedia answer" if turn.wiki_answer else "Keep “I don't know”"
    if keep_column.button(keep_label, width="stretch"):
        return "accept"
    return None


# ----------------------------------------------------------- audit column


def render_comparison(turn: Turn) -> None:
    """Every check for every answer, side by side: which answer holds up?"""
    candidates = turn.candidates
    rows = [
        (
            CHECK_LABELS[check],
            [
                (audit[check].status, score_of(audit[check])) if check in audit else ("unavailable", None)
                for _, _, audit in candidates
            ],
        )
        for check in CHECK_LABELS
    ]
    shown = turn.shown[0] if turn.shown else None
    if shown:
        verdict = f"Shown in the chat: <b>{ui.escape(shown)}</b>"
    elif turn.wiki_answer:
        verdict = "<b>No AI answer passed</b>: the chat shows Wikipedia's own text."
    else:
        verdict = "<b>No AI answer passed</b> and Wikipedia had nothing relevant: the chat says “I don't know”."
    st.html(
        '<div class="mp-section">Comparison</div>'
        + ui.comparison_table([model for model, _, _ in candidates], rows)
        + '<div class="mp-note">Hallucination, bias, toxicity: lower is better. '
        "Intent alignment: higher is better. An AI answer is shown only if its facts "
        f"were checked and no check is Serious.</div><div class=\"mp-muted\" style=\"margin-top:8px\">{verdict}</div>"
        f'<div style="margin-top:6px">{agreement_pill(turn)}</div>'
    )


def render_answers(turn: Turn) -> None:
    """Both answers in full, including the ones hidden from the chat."""
    st.html('<div class="mp-section">Both answers</div>')
    for model, text, audit in turn.candidates:
        if turn.shown == (model, text):
            label = f"{model} · shown in the chat"
        else:
            # Hidden because a check failed, or because both passed and the other
            # had fewer warnings / unverified details (auditor.pick_answer).
            if not (audit.get("hallucination") and audit["hallucination"].result):
                reason = "its facts could not be checked"
            else:
                reason = failed_checks(audit) or "both passed, the other was checked better"
            label = f"{model} · hidden: {reason}"
        with st.expander(label):
            st.markdown(ui.latex_for_streamlit(text))
            st.html(audit_strip(audit))
            if model == turn.secondary.model and turn.secondary.english_text:
                # Georgian question: SECONDARY answered in English (checked as is),
                # PRIMARY translated it into Georgian for reading.
                st.caption(f"{model} answered in English; Gemini translated it. Original:")
                st.markdown(ui.latex_for_streamlit(turn.secondary.english_text))
    if turn.revisions:
        with st.expander(f"{turn.primary.model} · original answer, before revising"):
            st.markdown(ui.latex_for_streamlit(turn.original_answer))
            st.html(audit_strip(turn.original_audit))


def render_issues(turn: Turn) -> None:
    """Problems found in either answer, labelled with the model that wrote it."""
    audits = [(model, audit) for model, _, audit in turn.candidates]
    cards = [
        ui.issue_card(
            f"{CHECK_LABELS[check]} · {model}",
            # "Not in the evidence" is not a false fact: show it as a potential issue.
            "warn" if issue.reason.upper().startswith("UNVERIFIED") else outcome.status,
            issue.quote,
            issue.reason,
        )
        for model, audit in audits
        for check, outcome in audit.items()
        if outcome.result and outcome.status in ("warn", "fail")
        for issue in outcome.result.issues
    ]
    errors = [
        f"{model} · {CHECK_LABELS[check]}: {outcome.error}"
        for model, audit in audits
        for check, outcome in audit.items()
        if outcome.error
    ]
    html = ""
    if cards:
        html += '<div class="mp-section">Issues found</div>' + "".join(cards)
    if errors:
        html += "".join(f'<div class="mp-muted">{ui.escape(e)}</div>' for e in errors)
    if html:
        st.html(html)


def render_evidence(turn: Turn) -> None:
    """The Wikipedia passages: Georgian ones to read, English ones the check used."""
    html = ""
    if turn.georgian_sources:
        html += '<div class="mp-section">წყაროები · ქართული Wikipedia</div>'
        for p in turn.georgian_sources:
            # Pages English Wikipedia lacks were translated and checked too.
            title = f"{p.title} · შემოწმებაშიც" if p.english_text else p.title
            html += ui.evidence_card(title, p.url, p.text, p.score, "გახსენი Wikipedia-ზე →")
        html += (
            '<div class="mp-muted">შემოწმება ძირითადად ინგლისურ Wikipedia-ს ეყრდნობა '
            "(ქვემოთ), რადგან იქ მეტი ფაქტია. „შემოწმებაშიც“ ნიშნავს, რომ ეს ქართული "
            "სტატია ინგლისურ Wikipedia-ში არ იყო, ამიტომ ითარგმნა და აუდიტორმაც "
            "გამოიყენა.</div>"
        )
    html += '<div class="mp-section">Evidence used for the check · English Wikipedia</div>'
    if turn.search_query and turn.search_query != turn.question:
        html += f'<div class="mp-muted">Searched for: “{ui.escape(turn.search_query)}”</div>'
    if turn.evidence_error:
        html += f'<div class="mp-muted">{ui.escape(turn.evidence_error)}</div>'
    elif not turn.evidence:
        html += '<div class="mp-muted">No relevant English passages found.</div>'
    else:
        # Relevance = cosine similarity to the answer (the re-ranking score).
        html += "".join(
            ui.evidence_card(p.title, p.url, p.text, p.score) for p in turn.evidence
        )
    if not turn.evidence and not any(p.english_text for p in turn.georgian_sources):
        html += '<div class="mp-muted">Checked without evidence.</div>'
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

        if turn.answer_translation:
            # The evidence was ranked against this translation, and the quotes
            # in "Issues found" come from it.
            with st.expander(f"{turn.primary.model} answer, as checked (English translation)"):
                st.markdown(ui.latex_for_streamlit(turn.answer_translation))
        render_comparison(turn)
        render_answers(turn)
        render_issues(turn)
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
