"""Streamlit UI for ModelProof. Only UI code lives here; the logic is in other modules.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from answers import ask_model, build_messages
from config import Settings, get_settings
from models import friendly_error

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
            f"- **Agreement threshold:** `{settings.agreement_threshold:.2f}`"
        )


def render_history(history: list[BaseMessage]) -> None:
    # Streamlit re-runs this whole script on every interaction, so the chat
    # must be redrawn from the saved history each time.
    for message in history:
        with st.chat_message(message.type):  # "human" or "ai"
            st.markdown(message.content)


def answer_question(settings: Settings, question: str) -> None:
    """Ask PRIMARY, show the answer, and save the exchange to the history."""
    history: list[BaseMessage] = st.session_state.history
    with st.chat_message("human"):
        st.markdown(question)

    with st.chat_message("ai"):
        try:
            with st.spinner("Asking model…"):
                messages = build_messages(history, question, settings.history_messages)
                answer = ask_model(settings.primary_model, messages)
        # The UI is the last line of defence: any failure becomes a message,
        # never a stack trace.
        except Exception as error:  # noqa: BLE001
            st.error(friendly_error(error, settings.primary_model))
            return  # failed questions are not saved to the history
        st.markdown(answer)

    history.extend([HumanMessage(question), AIMessage(answer)])


def main() -> None:
    settings = load_settings_or_stop()
    render_sidebar(settings)

    # One list per browser tab. In Phase 4 this moves into LangGraph memory.
    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("🛡️ ModelProof")
    st.caption("A chat that does not blindly trust one LLM.")
    render_history(st.session_state.history)

    question = st.chat_input("Ask a question…")
    if question:
        answer_question(settings, question)


main()
