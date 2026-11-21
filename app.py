"""Streamlit UI for ModelProof. Only UI code lives here; the logic is in other modules.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agreement import Agreement, compare_answers
from answers import ModelAnswer, ask_both, build_messages
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

    @property
    def final_answer(self) -> str:
        # The primary answer is the "official" one; if PRIMARY failed we fall
        # back to SECONDARY so the user still gets an answer.
        return self.primary.text if self.primary.ok else self.secondary.text


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


# ---------------------------------------------------------------- rendering


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
        st.caption(
            f"⏱ {turn.primary.model}: {turn.primary.seconds:.1f}s · "
            f"{turn.secondary.model}: {turn.secondary.seconds:.1f}s"
        )


# ------------------------------------------------------------------- flow
# Phase 4 moves this flow into a LangGraph graph; for now the UI runs it.


def run_turn(settings: Settings, question: str) -> Turn | None:
    """Ask both models, compare their answers, and return the finished turn."""
    messages = build_messages(
        history_messages(st.session_state.turns), question, settings.history_messages
    )
    with st.spinner("Asking models…"):
        primary, secondary = ask_both(
            settings.primary_model, settings.secondary_model, messages
        )

    if not primary.ok and not secondary.ok:
        with st.chat_message("ai"):
            st.error(f"No model could answer.\n\n- {primary.error}\n- {secondary.error}")
        return None  # nothing to save: the question can simply be asked again

    turn = Turn(question, primary, secondary)
    if primary.ok and secondary.ok:
        with st.spinner("Comparing answers…"):
            try:
                turn.agreement = compare_answers(
                    primary.text, secondary.text, settings.agreement_threshold
                )
            # e.g. no internet on the very first run, when the embedding
            # model still has to be downloaded.
            except Exception as error:  # noqa: BLE001
                turn.agreement_error = f"Agreement check unavailable: {error}"
    return turn


def main() -> None:
    settings = load_settings_or_stop()
    render_sidebar(settings)

    # One list per browser tab. In Phase 4 this moves into LangGraph memory.
    if "turns" not in st.session_state:
        st.session_state.turns = []

    st.title("🛡️ ModelProof")
    st.caption("A chat that does not blindly trust one LLM.")

    # Streamlit re-runs this whole script on every interaction, so the chat
    # is redrawn from the saved turns each time.
    for turn in st.session_state.turns:
        render_turn(turn, settings.agreement_threshold)

    question = st.chat_input("Ask a question…")
    if question:
        with st.chat_message("human"):
            st.markdown(question)
        turn = run_turn(settings, question)
        if turn is not None:
            st.session_state.turns.append(turn)
            st.rerun()  # redraw so the new turn appears exactly like older ones


main()
