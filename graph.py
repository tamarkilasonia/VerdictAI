"""The ModelProof workflow as a LangGraph graph.

    START -> generate_answers -> compare -> audit -> finalize -> END
                    |
                    +--(no model answered)--> END

Each node is a small function: it reads the shared state and returns only the
fields it changed. The real work happens in answers.py, agreement.py and
auditor.py; this file only wires the steps together.

Memory: the graph is compiled with a checkpointer (MemorySaver). LangGraph
saves the state after every step under a `thread_id`, so each conversation
(one per browser tab) keeps its own history between questions.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agreement import Agreement, compare_answers
from answers import ModelAnswer, ask_both, build_messages
from auditor import CheckOutcome, CheckResult, Issue, run_audit
from config import get_settings


@dataclass
class Turn:
    """One finished question with everything we learned about its answer."""

    question: str
    primary: ModelAnswer
    secondary: ModelAnswer
    agreement: Agreement | None = None
    agreement_error: str | None = None
    audit: dict[str, CheckOutcome] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)  # seconds per step

    @property
    def final_answer(self) -> str:
        # PRIMARY's answer is the "official" one; if PRIMARY failed we fall
        # back to SECONDARY so the user still gets an answer.
        return self.primary.text if self.primary.ok else self.secondary.text

    @property
    def final_model(self) -> str:
        return self.primary.model if self.primary.ok else self.secondary.model


class ChatState(TypedDict, total=False):
    """Everything the graph remembers.

    `Annotated[..., reducer]` tells LangGraph how to combine a node's update
    with the saved value: add_messages / operator.add APPEND to the list
    instead of replacing it. That is what makes them long-term memory.
    All other fields belong to the current question and are simply replaced.
    """

    # Long-term memory (kept across questions)
    messages: Annotated[list[BaseMessage], add_messages]  # what the models see
    turns: Annotated[list[Turn], operator.add]  # what the UI shows

    # The current question (reset by new_question_input at every question)
    question: str
    primary: ModelAnswer | None
    secondary: ModelAnswer | None
    agreement: Agreement | None
    agreement_error: str | None
    audit: dict[str, CheckOutcome]
    latency: dict[str, float]
    error: str | None


def new_question_input(question: str) -> dict[str, Any]:
    """Input for one graph run: the new question, with last question's fields cleared."""
    return {
        "question": question,
        "primary": None,
        "secondary": None,
        "agreement": None,
        "agreement_error": None,
        "audit": {},
        "latency": {},
        "error": None,
    }


def timed(step: str) -> Callable:
    """Decorator: measure how long a node takes and add it to state["latency"].

    Written once here instead of repeating the timing code in every node.
    """

    def decorator(node: Callable[[ChatState], dict]) -> Callable[[ChatState], dict]:
        @wraps(node)
        def wrapper(state: ChatState) -> dict:
            start = time.perf_counter()
            update = node(state)
            latency = {**state.get("latency", {}), step: time.perf_counter() - start}
            return {**update, "latency": latency}

        return wrapper

    return decorator


# ------------------------------------------------------------------ nodes


@timed("answers")
def generate_answers(state: ChatState) -> dict:
    """Fan out: ask PRIMARY and SECONDARY in parallel (with recent history)."""
    settings = get_settings()
    messages = build_messages(
        state.get("messages", []), state["question"], settings.history_messages
    )
    primary, secondary = ask_both(settings.primary_model, settings.secondary_model, messages)
    update: dict[str, Any] = {"primary": primary, "secondary": secondary}
    if not primary.ok and not secondary.ok:
        update["error"] = f"No model could answer.\n\n- {primary.error}\n- {secondary.error}"
    return update


@timed("agreement")
def compare(state: ChatState) -> dict:
    """Embed both answers and measure how similar they are."""
    primary, secondary = state["primary"], state["secondary"]
    if not (primary.ok and secondary.ok):
        return {}  # only one answer: nothing to compare
    try:
        agreement = compare_answers(
            primary.text, secondary.text, get_settings().agreement_threshold
        )
        return {"agreement": agreement}
    # e.g. no internet on the very first run, when the embedding model
    # still has to be downloaded.
    except Exception as error:  # noqa: BLE001
        return {"agreement_error": f"Agreement check unavailable: {error}"}


def _final_answer(state: ChatState) -> str:
    primary, secondary = state["primary"], state["secondary"]
    return primary.text if primary.ok else secondary.text


@timed("audit")
def audit(state: ChatState) -> dict:
    """Run the four checks on the final answer, in parallel."""
    limit = get_settings().history_messages
    recent = state.get("messages", [])[-limit:] if limit else []
    return {"audit": run_audit(state["question"], _final_answer(state), recent)}


def finalize(state: ChatState) -> dict:
    """The turn is done: save it to memory (for the models) and to the UI list."""
    turn = Turn(
        question=state["question"],
        primary=state["primary"],
        secondary=state["secondary"],
        agreement=state.get("agreement"),
        agreement_error=state.get("agreement_error"),
        audit=state.get("audit", {}),
        latency=state.get("latency", {}),
    )
    return {
        "messages": [HumanMessage(turn.question), AIMessage(turn.final_answer)],
        "turns": [turn],
    }


# ------------------------------------------------------------------ edges


def route_after_answers(state: ChatState) -> str:
    """Conditional edge: stop early if no model answered."""
    return END if state.get("error") else "compare"


def create_checkpointer() -> MemorySaver:
    """In-memory checkpointer that is allowed to restore our own state classes.

    LangGraph only restores types it knows are safe (loading arbitrary types
    from saved data is a security risk), so we list ours explicitly.
    """
    own_types = [ModelAnswer, Agreement, CheckResult, Issue, CheckOutcome, Turn]
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[(cls.__module__, cls.__name__) for cls in own_types]
    )
    return MemorySaver(serde=serializer)


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Wire the nodes together and compile the graph with memory."""
    graph = StateGraph(ChatState)
    graph.add_node("generate_answers", generate_answers)
    graph.add_node("compare", compare)
    graph.add_node("audit", audit)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "generate_answers")
    graph.add_conditional_edges("generate_answers", route_after_answers, ["compare", END])
    graph.add_edge("compare", "audit")
    graph.add_edge("audit", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or create_checkpointer())
