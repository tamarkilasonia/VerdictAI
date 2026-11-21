"""Streamlit UI for ModelProof. Only UI code lives here; the logic is in other modules.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agreement import Agreement, compare_answers
from answers import ModelAnswer, ask_both, build_messages
from auditor import HIGHER_IS_BETTER, ICONS, CheckOutcome, needs_revision, run_audit
from config import Settings, get_settings

st.set_page_config(page_title="ModelProof", page_icon="🛡️", layout="wide")


@dataclass
class Turn:
    """One question and everything we learned about its answers."""

    question: str
    primary: ModelAnswer
    secondary: ModelAnswer
    agreement: Agreement | None = None
    agreement_error: str | None = None
    audit: dict[str, CheckOutcome] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)  # seconds per step

    @property
    def final_answer(self) -> str:
        # The primary answer is the "official" one; if PRIMARY failed we fall
        # back to SECONDARY so the user still gets an answer.
        return self.primary.text if self.primary.ok else self.secondary.text

    @property
    def final_model(self) -> str:
        return self.primary.model if self.primary.ok else self.secondary.model


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
            f"- **Agreement threshold:** `{settings.agreement_threshold:.2f}`"
        )


def history_messages(turns: list[Turn]) -> list[BaseMessage]:
    """The chat so far, as the models see it: each question + its final answer."""
    messages: list[BaseMessage] = []
    for turn in turns:
        messages += [HumanMessage(turn.question), AIMessage(turn.final_answer)]
    return messages


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


def render_audit_panel(turn: Turn | None, settings: Settings) -> None:
    """Details for the latest answer: verdicts, issues, agreement, models, latency."""
    st.subheader("🔍 Audit")
    if turn is None:
        st.info("Ask a question to see how its answer is audited.")
        return

    st.caption(f"Audited answer: `{turn.final_model}` · auditor: `{settings.auditor_model}`")
    for check, outcome in turn.audit.items():
        render_check(check, outcome)
    if "hallucination" in turn.audit:
        st.caption("Hallucination was checked without evidence (general knowledge only).")
    if needs_revision(turn.audit):
        st.error("❌ At least one check failed: treat this answer with caution.")

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
# Phase 4 moves this flow into a LangGraph graph; for now the UI runs it.


def run_turn(settings: Settings, question: str) -> Turn | None:
    """Ask both models, compare their answers, audit the final one."""
    history = history_messages(st.session_state.turns)
    messages = build_messages(history, question, settings.history_messages)

    start = time.perf_counter()
    with st.spinner("Asking models…"):
        primary, secondary = ask_both(
            settings.primary_model, settings.secondary_model, messages
        )
    if not primary.ok and not secondary.ok:
        with st.chat_message("ai"):
            st.error(f"No model could answer.\n\n- {primary.error}\n- {secondary.error}")
        return None  # nothing to save: the question can simply be asked again
    turn = Turn(question, primary, secondary)
    turn.latency["answers"] = time.perf_counter() - start

    if primary.ok and secondary.ok:
        start = time.perf_counter()
        with st.spinner("Comparing answers…"):
            try:
                turn.agreement = compare_answers(
                    primary.text, secondary.text, settings.agreement_threshold
                )
            # e.g. no internet on the very first run, when the embedding
            # model still has to be downloaded.
            except Exception as error:  # noqa: BLE001
                turn.agreement_error = f"Agreement check unavailable: {error}"
        turn.latency["agreement"] = time.perf_counter() - start

    start = time.perf_counter()
    with st.spinner("Auditing…"):
        recent = history[-settings.history_messages :] if settings.history_messages else []
        turn.audit = run_audit(question, turn.final_answer, recent)
    turn.latency["audit"] = time.perf_counter() - start
    return turn


def main() -> None:
    settings = load_settings_or_stop()
    render_sidebar(settings)

    # One list per browser tab. In Phase 4 this moves into LangGraph memory.
    if "turns" not in st.session_state:
        st.session_state.turns = []
    turns: list[Turn] = st.session_state.turns

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
            turn = run_turn(settings, question)
        if turn is not None:
            turns.append(turn)
            st.rerun()  # redraw so the new turn and its audit appear


main()
