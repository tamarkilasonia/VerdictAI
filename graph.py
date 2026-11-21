"""The ModelProof workflow as a LangGraph graph.

    START -> generate_answers -> compare -> retrieve_evidence -> audit
                  |                                               |
                  +--(no model answered)--> END                   |
                                                                  v
                       any ❌ and revisions < MAX_REVISIONS ?  --no--> finalize -> END
                                       | yes
                                       v
                                 human_review  (pauses: waits for the user)
                                   |        |
                          "revise" |        | "accept"
                                   v        v
                     audit <-- revise    finalize

Each node is a small function: it reads the shared state and returns only the
fields it changed. The real work happens in answers.py, agreement.py,
retrieval.py and auditor.py; this file only wires the steps together.

Memory: the graph is compiled with a checkpointer (MemorySaver). LangGraph
saves the state after every step under a `thread_id`, so each conversation
(one per browser tab) keeps its own history between questions, and a run
paused by interrupt() can be resumed later, even after Streamlit reruns.
"""

from __future__ import annotations

import operator
from concurrent.futures import ThreadPoolExecutor
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from agreement import Agreement, compare_answers
from audit_log import append_entry, build_entry, build_failure_entry
from answers import (
    ModelAnswer,
    ask_both,
    build_messages,
    make_search_query,
    revise_answer,
    translate_to_english,
    translate_to_georgian,
)
from auditor import (
    CheckOutcome,
    CheckResult,
    Issue,
    describe_problems,
    pick_answer,
    run_audit,
)
from config import MAX_REVISIONS, get_settings
from retrieval import (
    GEORGIAN_WIKIPEDIA_API,
    WIKIPEDIA_API,
    Passage,
    get_evidence,
    get_georgian_sources,
    is_georgian,
    wikipedia_answer,
)


# Shown instead of an answer when both answers failed the audit: a wrong
# answer is worse than none.
# When no AI answer passed, the user gets Wikipedia's own text (quoted, not
# rewritten), or, if Wikipedia had nothing relevant, an honest "I don't know".
# Why no AI answer is shown: "failed" (a check found a problem) or "unchecked"
# (the auditor could not run, e.g. a rate limit).
WHY_NO_AI_ANSWER = {
    ("failed", "en"): "The AI answers did not pass the fact check.",
    ("failed", "ka"): "AI-ების პასუხებმა ფაქტების შემოწმება ვერ გაიარა.",
    ("unchecked", "en"): "The AI answers could not be fact-checked right now (service limit).",
    ("unchecked", "ka"): "AI-ების პასუხების შემოწმება ახლა ვერ მოხერხდა (სერვისის ლიმიტი).",
}
FROM_WIKIPEDIA = {
    "en": "Here is what Wikipedia says:",
    "ka": "აი, რა წერია Wikipedia-ში:",
}
TRANSLATED_NOTE = "(ინგლისური Wikipedia-დან; თარგმანი: Gemini)"
DONT_KNOW = {
    "en": "So I don't know a reliable answer to this (see the analysis on the right).",
    "ka": "ამიტომ ამ კითხვაზე სანდო პასუხი არ ვიცი (ანალიზი იხილეთ მარჯვნივ).",
}


@dataclass
class Turn:
    """One question with everything we learned about its answer."""

    question: str
    primary: ModelAnswer
    secondary: ModelAnswer
    agreement: Agreement | None = None
    agreement_error: str | None = None
    evidence: list[Passage] = field(default_factory=list)
    georgian_sources: list[Passage] = field(default_factory=list)  # shown only, not audited
    evidence_error: str | None = None
    search_query: str = ""  # what Wikipedia was searched for (English)
    audit: dict[str, CheckOutcome] = field(default_factory=dict)  # of the final answer
    answer_translation: str = ""  # English version of a Georgian answer (ranked + audited)
    secondary_audit: dict[str, CheckOutcome] = field(default_factory=dict)  # of SECONDARY's answer
    wiki_answer: Passage | None = None  # Wikipedia's own answer, used if no AI answer passed
    revisions: list[str] = field(default_factory=list)  # corrected answers, oldest first
    original_audit: dict[str, CheckOutcome] = field(default_factory=dict)  # before revising
    revise_error: str | None = None
    latency: dict[str, float] = field(default_factory=dict)  # seconds per step

    @property
    def original_answer(self) -> str:
        # PRIMARY's answer is the "official" one; if PRIMARY failed we fall
        # back to SECONDARY so the user still gets an answer.
        return self.primary.text if self.primary.ok else self.secondary.text

    @property
    def final_answer(self) -> str:
        """The latest revision if there is one, otherwise the original answer."""
        return self.revisions[-1] if self.revisions else self.original_answer

    @property
    def final_model(self) -> str:
        return self.primary.model if self.primary.ok else self.secondary.model

    @property
    def candidates(self) -> list[tuple[str, str, dict[str, CheckOutcome]]]:
        """Every answer that exists, as (model, text, audit); PRIMARY first."""
        if not self.primary.ok:  # then `audit` belongs to SECONDARY's answer
            return [(self.secondary.model, self.secondary.text, self.audit)] if self.secondary.ok else []
        answers = [(self.primary.model, self.final_answer, self.audit)]
        if self.secondary.ok:
            answers.append((self.secondary.model, self.secondary.text, self.secondary_audit))
        return answers

    @property
    def shown(self) -> tuple[str, str] | None:
        """(model, text) of the ONE answer shown in the chat, or None.

        Only an answer that passed the audit is shown; see auditor.pick_answer.
        """
        candidates = self.candidates
        index = pick_answer([audit for _, _, audit in candidates])
        return None if index is None else candidates[index][:2]

    @property
    def facts_checked(self) -> bool:
        """True if the hallucination check ran on every AI answer."""
        return all(
            audit.get("hallucination") is not None and audit["hallucination"].result is not None
            for _, _, audit in self.candidates
        )

    @property
    def reply(self) -> str:
        """What the user sees and the models remember, in the question's language:
        the AI answer that passed, else Wikipedia's text, else "I don't know"."""
        if self.shown:
            return self.shown[1]
        language = "ka" if is_georgian(self.question) else "en"
        why = WHY_NO_AI_ANSWER[("failed" if self.facts_checked else "unchecked", language)]
        wiki = self.wiki_answer
        if wiki is None:
            return f"{why} {DONT_KNOW[language]}"
        quote = "\n".join(f"> {line}" if line else ">" for line in wiki.text.splitlines())
        source = f"— [{wiki.title}]({wiki.url})"
        if wiki.english_text:
            source += f" {TRANSLATED_NOTE}"
        return f"{why} {FROM_WIKIPEDIA[language]}\n\n{quote}\n\n{source}"


class ChatState(TypedDict, total=False):
    """Everything the graph remembers.

    `Annotated[..., reducer]` tells LangGraph how to combine a node's update
    with the saved value: add_messages / operator.add APPEND to the list
    instead of replacing it. That is what makes them long-term memory.
    All other fields belong to the current question and are simply replaced.
    """

    # Long-term memory (kept across questions)
    messages: Annotated[list[BaseMessage], add_messages]  # what the models see
    turns: Annotated[list[Turn], operator.add]  # finished turns, for the UI

    # The current question (reset by new_question_input at every question)
    question: str
    primary: ModelAnswer | None
    secondary: ModelAnswer | None
    agreement: Agreement | None
    agreement_error: str | None
    evidence: list[Passage]
    georgian_sources: list[Passage]
    evidence_error: str | None
    search_query: str
    audit: dict[str, CheckOutcome]
    answer_translation: str
    secondary_audit: dict[str, CheckOutcome]
    wiki_answer: Passage | None
    revisions: list[str]
    original_audit: dict[str, CheckOutcome]
    review_decision: str | None  # "revise" or "accept", set by human_review
    revise_error: str | None
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
        "evidence": [],
        "georgian_sources": [],
        "evidence_error": None,
        "search_query": "",
        "audit": {},
        "answer_translation": "",
        "secondary_audit": {},
        "wiki_answer": None,
        "revisions": [],
        "original_audit": {},
        "review_decision": None,
        "revise_error": None,
        "latency": {},
        "error": None,
    }


def turn_from_state(state: ChatState) -> Turn:
    """Collect the current question's fields into a Turn (also used by the UI
    to show a turn that is still waiting for review)."""
    return Turn(
        question=state["question"],
        primary=state["primary"],
        secondary=state["secondary"],
        agreement=state.get("agreement"),
        agreement_error=state.get("agreement_error"),
        evidence=state.get("evidence", []),
        georgian_sources=state.get("georgian_sources", []),
        evidence_error=state.get("evidence_error"),
        search_query=state.get("search_query", ""),
        audit=state.get("audit", {}),
        answer_translation=state.get("answer_translation", ""),
        secondary_audit=state.get("secondary_audit", {}),
        wiki_answer=state.get("wiki_answer"),
        revisions=state.get("revisions", []),
        original_audit=state.get("original_audit", {}),
        revise_error=state.get("revise_error"),
        latency=state.get("latency", {}),
    )


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


def _recent_history(state: ChatState) -> list[BaseMessage]:
    limit = get_settings().history_messages
    return state.get("messages", [])[-limit:] if limit else []


def _final_answer(state: ChatState) -> str:
    return turn_from_state(state).final_answer


def _thread_id() -> str:
    """The conversation id of the current run (written to the audit log)."""
    return get_config().get("configurable", {}).get("thread_id", "")


def _evidence_texts(state: ChatState) -> list[str]:
    """What the auditor and the reviser read: English Wikipedia passages, plus
    English translations of Georgian pages that English Wikipedia lacks."""
    # Each passage keeps its page title, so the model knows the source.
    texts = [f"[{p.title}] {p.text}" for p in state.get("evidence", [])]
    texts += [
        f"[{p.title}, Georgian Wikipedia, translated] {p.english_text}"
        for p in state.get("georgian_sources", [])
        if p.english_text
    ]
    return texts


# ------------------------------------------------------------------ nodes


@timed("answers")
def generate_answers(state: ChatState) -> dict:
    """Fan out: ask PRIMARY and SECONDARY in parallel (with recent history)."""
    settings = get_settings()
    messages = build_messages(
        state.get("messages", []), state["question"], settings.history_messages
    )
    # SECONDARY (Qwen) writes poor Georgian: asked directly, it named a made-up
    # author for "The Knight in the Panther's Skin" and garbled names; asked in
    # English, it answered correctly. So a Georgian question goes to it in
    # English, and PRIMARY (Gemini) translates its answer back into Georgian.
    english_question = _translation(state["question"])
    secondary_messages = (
        [*messages[:-1], HumanMessage(english_question)] if english_question else None
    )
    primary, secondary = ask_both(
        settings.primary, settings.secondary, messages, secondary_messages
    )
    if secondary_messages and secondary.ok:
        georgian = _to_georgian(secondary.text)
        if georgian:  # if translating fails, the English answer is shown as it is
            secondary = replace(secondary, text=georgian, english_text=secondary.text)
    update: dict[str, Any] = {"primary": primary, "secondary": secondary}
    if not primary.ok and not secondary.ok:
        update["error"] = f"No model could answer.\n\n- {primary.error}\n- {secondary.error}"
        # The run ends here (finalize is skipped), so log the failure now.
        append_entry(
            build_failure_entry(
                state["question"], settings, [primary.error, secondary.error], _thread_id()
            )
        )
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


def _translation(text: str) -> str:
    """PRIMARY's English translation of a Georgian text.

    Returns "" if the text is already English, or if the translation failed
    (e.g. rate limit): the caller then simply uses the original text.
    """
    if not is_georgian(text):
        return ""
    try:
        return translate_to_english(get_settings().primary, text)
    except Exception:  # noqa: BLE001 - a failed translation must not stop the turn
        return ""


def _to_georgian(text: str) -> str:
    """PRIMARY's Georgian translation of an English text; "" if that fails."""
    try:
        return translate_to_georgian(get_settings().primary, text)
    except Exception:  # noqa: BLE001
        return ""


def _search_query(question: str, language: str = "English") -> str:
    """A few keywords for Wikipedia; the question itself if that fails."""
    try:
        return make_search_query(get_settings().primary, question, language) or question
    except Exception:  # noqa: BLE001 - e.g. rate limit: search with the question
        return question


def _georgian_sources(state: ChatState, english_evidence: list[Passage]) -> list[Passage]:
    """Georgian Wikipedia passages for a Georgian question (see retrieval.py).

    Passages from pages that English Wikipedia lacks are translated into
    English, so the auditor can use them: it misreads Georgian text.
    """
    sources = get_georgian_sources(
        [p.title for p in english_evidence],
        _search_query(state["question"], "Georgian"),
        _final_answer(state),
        get_settings().wiki_user_agent,
    )
    for passage in sources:
        if passage.only_in_georgian:
            passage.english_text = _translation(passage.text)
    return sources


@timed("evidence")
def retrieve_evidence(state: ChatState) -> dict:
    """Find Wikipedia passages that can confirm or contradict the final answer.

    The search uses a few English keywords, not the whole question. A Georgian
    answer is translated into English, because English passages are ranked more
    precisely against an English answer; the translation is kept for the audit
    too, so it is made only once.
    """
    answer = _final_answer(state)
    query = _search_query(state["question"])
    translation = _translation(answer)
    update: dict[str, Any] = {"search_query": query, "answer_translation": translation}
    try:
        update["evidence"] = get_evidence(query, translation or answer, get_settings().wiki_user_agent)
    # No internet, Wikipedia down, timeout...: the audit simply continues
    # without evidence, and the UI says so.
    except Exception as error:  # noqa: BLE001
        update["evidence_error"] = f"English Wikipedia unavailable ({type(error).__name__})."
    if is_georgian(state["question"]):
        # Georgian Wikipedia too: to read the sources in Georgian, and for
        # Georgian topics that English Wikipedia does not cover. A failure
        # here only means fewer sources.
        try:
            update["georgian_sources"] = _georgian_sources(state, update.get("evidence", []))
        except Exception:  # noqa: BLE001
            pass
    try:
        update["wiki_answer"] = _wikipedia_answer(
            state["question"], update.get("georgian_sources", []), update.get("evidence", [])
        )
    except Exception:  # noqa: BLE001 - then the fallback is "I don't know"
        pass
    return update


def _wikipedia_answer(
    question: str, georgian: list[Passage], english: list[Passage]
) -> Passage | None:
    """Wikipedia's own answer, kept ready in case no AI answer passes.

    A Georgian question is answered from Georgian Wikipedia when it has a
    relevant passage; otherwise from English Wikipedia, translated by PRIMARY
    (the original stays in `english_text`, and the reply says it is a
    translation). An English question is answered from English Wikipedia.
    """
    user_agent = get_settings().wiki_user_agent
    if is_georgian(question):
        answer = wikipedia_answer(question, georgian, user_agent, GEORGIAN_WIKIPEDIA_API)
        if answer is not None:
            return answer
        answer = wikipedia_answer(question, english, user_agent, WIKIPEDIA_API)
        georgian_text = _to_georgian(answer.text) if answer else ""
        if not georgian_text:
            return answer  # better English Wikipedia than nothing
        return replace(answer, text=georgian_text, english_text=answer.text)
    return wikipedia_answer(question, english, user_agent, WIKIPEDIA_API)


def _audit_answer(state: ChatState, text: str, translation: str = "") -> dict[str, CheckOutcome]:
    """The four checks on one answer; a Georgian answer is checked in English."""
    english = translation or _translation(text)
    return run_audit(
        state["question"], english or text, _recent_history(state), _evidence_texts(state)
    )


@timed("audit")
def audit(state: ChatState) -> dict:
    """Run the four checks on BOTH answers, so the user can compare them.

    A Georgian answer is audited in its English translation, because the
    auditor model misreads Georgian: in tests it read "შოტლანდიელმა" (Scottish)
    as "Dutch" and flagged a correct answer as contradicted, 3 times out of 3.
    With the translation the false alarm was gone, and a wrong answer was
    still caught 3 times out of 3.

    SECONDARY is audited only once: a revision changes PRIMARY's answer, not
    SECONDARY's. The audits decide which answer the user sees (Turn.shown).
    """
    answer = _final_answer(state)
    # revise() clears the translation, so a revised answer is translated anew.
    translation = state.get("answer_translation") or _translation(answer)
    secondary = state["secondary"]
    audit_secondary = (
        state["primary"].ok and secondary.ok and not state.get("secondary_audit")
    )
    # Both audits at the same time: the user waits for one, not for two.
    with ThreadPoolExecutor(max_workers=2) as pool:
        primary_job = pool.submit(_audit_answer, state, answer, translation)
        # SECONDARY's original English answer is audited directly when it exists.
        secondary_job = (
            pool.submit(_audit_answer, state, secondary.text, secondary.english_text)
            if audit_secondary
            else None
        )
        update: dict[str, Any] = {
            "audit": primary_job.result(),
            "answer_translation": translation or "",
        }
        if secondary_job is not None:
            update["secondary_audit"] = secondary_job.result()
    return update


def human_review(state: ChatState) -> dict:
    """Pause the graph and wait for the user's decision (human-in-the-loop).

    interrupt() saves the state in the checkpointer and stops the run. The
    UI shows a "Revise answer" button; clicking it resumes the run with
    Command(resume="revise"), and interrupt() then returns "revise".
    Note: when resumed, this node runs again from the top, so it must not do
    anything before interrupt() that should only happen once.
    """
    decision = interrupt(
        {"revisions_done": len(state.get("revisions", [])), "max_revisions": MAX_REVISIONS}
    )
    return {"review_decision": "revise" if decision == "revise" else "accept"}


@timed("revise")
def revise(state: ChatState) -> dict:
    """Ask PRIMARY to fix the answer using the audit's problems and the evidence."""
    revised = revise_answer(
        get_settings().primary,
        _recent_history(state),
        state["question"],
        _final_answer(state),
        describe_problems(state["audit"]),
        _evidence_texts(state),
    )
    if not revised.ok:
        return {"revise_error": f"Revision failed: {revised.error}"}

    update: dict[str, Any] = {
        "revisions": state.get("revisions", []) + [revised.text],
        "revise_error": None,
        "answer_translation": "",  # belonged to the old answer; audit() makes a new one
    }
    if not state.get("revisions"):  # first revision: keep the original verdict
        update["original_audit"] = state["audit"]
    return update


def finalize(state: ChatState) -> dict:
    """The turn is done: write the audit log, save it to memory and to the UI list."""
    turn = turn_from_state(state)
    append_entry(build_entry(turn, get_settings(), _thread_id()))
    return {
        # The models remember what the user saw (the checked answer, or "I
        # don't know"), so follow-up questions never build on a wrong answer.
        "messages": [HumanMessage(turn.question), AIMessage(turn.reply)],
        "turns": [turn],
    }


# ------------------------------------------------------------------ edges


def route_after_answers(state: ChatState) -> str:
    """Stop early if no model answered."""
    return END if state.get("error") else "compare"


def route_after_audit(state: ChatState) -> str:
    """Offer a revision only if NO answer passed the audit.

    If one answer passed, it is simply shown. Only PRIMARY can be revised, so
    there must be a PRIMARY answer and revisions left. If the auditor could
    not run (e.g. its daily limit), a revision could not be checked either,
    so none is offered.
    """
    turn = turn_from_state(state)
    revisions_left = len(turn.revisions) < MAX_REVISIONS
    can_revise = turn.primary.ok and revisions_left and turn.facts_checked
    return "human_review" if turn.shown is None and can_revise else "finalize"


def route_after_review(state: ChatState) -> str:
    return "revise" if state.get("review_decision") == "revise" else "finalize"


def route_after_revise(state: ChatState) -> str:
    """Re-audit the new answer; if revising failed, ask the user again."""
    return "human_review" if state.get("revise_error") else "audit"


def create_checkpointer() -> MemorySaver:
    """In-memory checkpointer that is allowed to restore our own state classes.

    LangGraph only restores types it knows are safe (loading arbitrary types
    from saved data is a security risk), so we list ours explicitly.
    """
    own_types = [ModelAnswer, Agreement, Passage, CheckResult, Issue, CheckOutcome, Turn]
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[(cls.__module__, cls.__name__) for cls in own_types]
    )
    return MemorySaver(serde=serializer)


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Wire the nodes together and compile the graph with memory."""
    graph = StateGraph(ChatState)
    graph.add_node("generate_answers", generate_answers)
    graph.add_node("compare", compare)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("audit", audit)
    graph.add_node("human_review", human_review)
    graph.add_node("revise", revise)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "generate_answers")
    graph.add_conditional_edges("generate_answers", route_after_answers, ["compare", END])
    graph.add_edge("compare", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "audit")
    graph.add_conditional_edges("audit", route_after_audit, ["human_review", "finalize"])
    graph.add_conditional_edges("human_review", route_after_review, ["revise", "finalize"])
    graph.add_conditional_edges("revise", route_after_revise, ["audit", "human_review"])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or create_checkpointer())
