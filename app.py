"""Streamlit UI for ModelProof. Only UI code lives here; the logic is in other modules.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import uuid

import streamlit as st

from auditor import HIGHER_IS_BETTER, ICONS, CheckOutcome, needs_revision
from config import Settings, get_settings
from graph import Turn, build_graph, new_question_input

st.set_page_config(page_title="ModelProof", page_icon="🛡️", layout="wide")


def load_settings_or_stop() -> Settings:
    """Show a readable message instead of a crash if .env has a bad value."""
    try:
        return get_settings()
    except ValueError as error:
        st.error(f"Configuration problem: {error}")
        st.stop()


def render_sidebar(settings: Settings) -> None:
    """Read-only view of the current configuration."""
    with st.sidebar:
        st.header("ModelProof")
        st.caption("Settings come from .env. Restart the app after editing it.")
        st.markdown(
            f"- **Provider:** `{settings.model_provider}`\n"
            f"- **Primary model:** `{settings.primary_model}`\n"
            f"- **Secondary model:** `{settings.secondary_model}`\n"
            f"- **Auditor model:** `{settings.auditor_model}`\n"
            f"- **Agreement threshold:** `{settings.agreement_threshold:.2f}`\n"
            f"- **History sent to models:** last `{settings.history_messages}` messages"
        )
        st.divider()
        # A new thread_id = a new, empty conversation in the checkpointer.
        if st.button("🗑️ New conversation", width="stretch"):
            st.session_state.thread_id = uuid.uuid4().hex
            st.rerun()


def check_label(check: str) -> str:
    return check.replace("_", " ").capitalize()


# ------------------------------------------------------------ chat column


def render_one_answer_only(turn: Turn) -> None:
    """One model failed: show the answer we have and say why the other is missing."""
    working, failed = (
        (turn.primary, turn.secondary) if turn.primary.ok else (turn.secondary, turn.primary)
    )
    st.markdown(working.text)
    st.warning(
        f"Only `{working.model}` answered, so agreement could not be measured. "
        f"`{failed.model}`: {failed.error}"
    )


def render_answers(turn: Turn, threshold: float) -> None:
    """Agree -> primary answer + badge. Disagree -> both answers side by side."""
    if turn.agreement is None:
        # Both answered but the embedding model could not run.
        st.markdown(turn.primary.text)
        st.warning(turn.agreement_error or "Agreement check unavailable.")
        return

    score = f"similarity **{turn.agreement.score:.2f}** (threshold {threshold:.2f})"
    if turn.agreement.agree:
        st.markdown(turn.primary.text)
        st.success(f"✅ Models agree · {score}")
        return

    st.warning(f"⚠️ Models disagree · {score}. Compare both answers:")
    for column, answer in zip(st.columns(2), (turn.primary, turn.secondary)):
        with column:
            st.markdown(f"**{answer.model}**")
            st.markdown(answer.text)


def render_turn(turn: Turn, threshold: float) -> None:
    with st.chat_message("human"):
        st.markdown(turn.question)
    with st.chat_message("ai"):
        if turn.primary.ok and turn.secondary.ok:
            render_answers(turn, threshold)
        else:
            render_one_answer_only(turn)
        # A one-line audit summary, so older answers keep their verdict visible.
        if turn.audit:
            st.caption(
                "Audit: "
                + " · ".join(
                    f"{ICONS[o.status]} {check_label(c)}" for c, o in turn.audit.items()
                )
            )


# ----------------------------------------------------------- audit column


def render_check(check: str, outcome: CheckOutcome) -> None:
    """One audit check: icon, score, and the issues the auditor quoted."""
    direction = "higher is better" if check in HIGHER_IS_BETTER else "lower is better"
    if outcome.result is None:
        st.markdown(f"{ICONS['unavailable']} **{check_label(check)}**: audit unavailable")
        st.caption(outcome.error)
        return

    st.markdown(
        f"{ICONS[outcome.status]} **{check_label(check)}**: "
        f"`{outcome.result.score:.2f}` ({direction})"
    )
    if outcome.result.issues:
        with st.expander(f"{len(outcome.result.issues)} issue(s)"):
            for issue in outcome.result.issues:
                if issue.quote:
                    st.markdown(f"> {issue.quote}")
                st.markdown(issue.reason)


def render_evidence(turn: Turn) -> None:
    """The Wikipedia passages the hallucination check was judged against."""
    st.markdown("**📚 Evidence (Wikipedia)**")
    if turn.evidence_error:
        st.warning(turn.evidence_error)
        return
    if not turn.evidence:
        st.caption("No relevant Wikipedia passages found; checked without evidence.")
        return
    for passage in turn.evidence:
        # The score shows how close the passage is to the answer (re-ranking).
        with st.expander(f"{passage.title} · relevance {passage.score:.2f}"):
            st.markdown(passage.text)
            st.markdown(f"[Open on Wikipedia ↗]({passage.url})")


def render_audit_panel(turn: Turn | None, settings: Settings) -> None:
    """Details for the latest answer: verdicts, issues, agreement, models, latency."""
    st.subheader("🔍 Audit")
    if turn is None:
        st.info("Ask a question to see how its answer is audited.")
        return

    st.caption(f"Audited answer: `{turn.final_model}` · auditor: `{settings.auditor_model}`")
    for check, outcome in turn.audit.items():
        render_check(check, outcome)
    if needs_revision(turn.audit):
        st.error("❌ At least one check failed: treat this answer with caution.")

    st.divider()
    render_evidence(turn)
    st.divider()
    if turn.agreement is not None:
        st.markdown(f"**Agreement:** `{turn.agreement.score:.2f}`")
    else:
        st.markdown("**Agreement:** n/a")
    st.markdown(
        f"**Models:** `{turn.primary.model}` · `{turn.secondary.model}` · "
        f"auditor `{settings.auditor_model}`"
    )
    st.markdown(
        "**Latency:** "
        + " · ".join(f"{step} {seconds:.1f}s" for step, seconds in turn.latency.items())
    )


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
    "audit": "Saving…",
}


def run_graph(question: str) -> str | None:
    """Run one question through the graph. Returns an error message, or None."""
    graph, config = get_graph(), thread_config()
    with st.status("Asking models…") as status:
        try:
            # stream_mode="updates" yields after every finished node, which
            # lets us show the current step instead of one long spinner.
            for update in graph.stream(new_question_input(question), config, stream_mode="updates"):
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
    render_sidebar(settings)

    # The checkpointer is the single source of truth for the conversation.
    turns: list[Turn] = get_graph().get_state(thread_config()).values.get("turns", [])

    st.title("🛡️ ModelProof")
    st.caption("A chat that does not blindly trust one LLM.")
    chat_column, audit_column = st.columns([3, 2], gap="large")

    # Streamlit re-runs this whole script on every interaction, so the page
    # is redrawn from the saved turns each time.
    with chat_column:
        for turn in turns:
            render_turn(turn, settings.agreement_threshold)
    with audit_column:
        render_audit_panel(turns[-1] if turns else None, settings)

    question = st.chat_input("Ask a question…")
    if question:
        with chat_column:
            with st.chat_message("human"):
                st.markdown(question)
            error = run_graph(question)
            if error:
                st.error(error)  # failed questions are not saved; just ask again
                return
        st.rerun()  # redraw so the new turn and its audit appear


main()
