"""The full learner graph: routes each turn by input kind and ends every path at memory_writer."""

import logging
import uuid
from enum import StrEnum
from typing import Final, NotRequired, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from civica.assessment.grounding import QuizGrounding
from civica.domain.question import Question
from civica.explanation.engine import Explanation
from civica.graph.answers import (
    ALREADY_ANSWERED_REPLY,
    EXPIRED_REPLY,
    AnswerResult,
    answer_recap,
    find_question,
    record_mastery,
    score_choice,
)
from civica.graph.nodes import PlannerContext, build_planner_graph
from civica.graph.planner import (
    SUMMARY_KEY,
    HaltReason,
    PlannerDeps,
    PreviousExchange,
    ToolCall,
    default_planner_deps,
)
from civica.llm.usage import TurnUsage
from civica.memory.checkpointer import get_saver
from civica.memory.records import SessionSummary
from civica.memory.writer import put

logger = logging.getLogger(__name__)

PLANNER: Final = "planner"
SCORE_ANSWER: Final = "score_answer"
UPDATE_MASTERY: Final = "update_mastery"
MEMORY_WRITER: Final = "memory_writer"


class AnswerOutcome(StrEnum):
    """Whether an answer counted."""

    RIGHT = "right"
    WRONG = "wrong"
    EXPIRED = "expired"
    ALREADY_ANSWERED = "already_answered"


class InputKind(StrEnum):
    """What a turn carries. Each kind enters the graph on its own route."""

    MESSAGE = "message"
    ANSWER = "answer"


class GraphInput(TypedDict):
    kind: InputKind
    learner_message: str
    # Set only on an answer turn.
    question_id: NotRequired[str]
    chosen_index: NotRequired[int]


class GraphState(GraphInput):
    reply: str
    turn_id: str # set by the node that opens the turn
    lesson: Explanation | None
    quiz_questions: list[Question]
    quiz_grounding: QuizGrounding | None
    tool_calls: list[ToolCall]
    halt_reason: HaltReason | None
    previous_exchange: PreviousExchange | None
    token_usage: TurnUsage | None
    answer_result: AnswerResult | None
    answered: dict[str, AnswerResult]


class Node(Protocol):
    def __call__(self, state: GraphState, runtime: Runtime[PlannerContext]) -> dict[str, object]: ...


_ROUTES: Final[dict[InputKind, str]] = {InputKind.MESSAGE: PLANNER, InputKind.ANSWER: SCORE_ANSWER}


def route_input(state: GraphState) -> str:
    """Send the turn to the entry node for its kind."""
    return _ROUTES[InputKind(state["kind"])]


def _recap(state: GraphState) -> str:
    """What the turn delivered, read from its outcomes because a requested tool call can fail."""
    parts: list[str] = []
    lesson = state["lesson"]
    if lesson is not None and lesson.context_chunks:
        parts.append(f"Lesson on {lesson.context_chunks[0].chunk.theme.slug}.")
    elif lesson is not None:
        parts.append("Lesson had no official material.")
    questions = state["quiz_questions"]
    if questions:
        parts.append(f"Quiz of {len(questions)} on {questions[0].theme.slug}.")
    if not parts:
        parts.append("No lesson or quiz.")
    if state["halt_reason"] != HaltReason.FINAL_MESSAGE:
        parts.append(f"Turn ended early: {state['halt_reason']}.")
    return " ".join(parts)


def _make_score_answer(deps: PlannerDeps) -> Node:
    def score_answer(state: GraphState, runtime: Runtime[PlannerContext]) -> dict[str, object]:
        """Score and log the submitted choice, unless its question expired or was already answered."""
        user_id, question_id = runtime.context.user_id, state["question_id"]
        answered = state.get("answered", {})
        question = find_question(state.get("quiz_questions", []), question_id)
        update: dict[str, object] = {"turn_id": str(uuid.uuid4()), "answer_result": None}
        if question is None:
            outcome, update["reply"] = AnswerOutcome.EXPIRED, EXPIRED_REPLY
        elif question_id in answered:
            outcome, update["reply"] = AnswerOutcome.ALREADY_ANSWERED, ALREADY_ANSWERED_REPLY
        else:
            result = score_choice(deps, user_id, question, state["chosen_index"])
            outcome = AnswerOutcome.RIGHT if result.is_correct else AnswerOutcome.WRONG
            update |= {"answer_result": result, "answered": {**answered, question_id: result}, "reply": ""}
        logger.info("answer_scored user_id=%s question_id=%s outcome=%s", user_id, question_id, outcome.value)
        return update

    return score_answer


def _make_update_mastery(deps: PlannerDeps) -> Node:
    def update_mastery(state: GraphState, runtime: Runtime[PlannerContext]) -> dict[str, object]:
        """Recompute the answered theme's mastery. The only writer of topic_mastery."""
        result = state["answer_result"]
        assert result is not None
        record_mastery(deps, runtime.context.user_id, result.theme)
        return {}

    return update_mastery


def _after_score(state: GraphState) -> str:
    return MEMORY_WRITER if state["answer_result"] is None else UPDATE_MASTERY


def _make_memory_writer(deps: PlannerDeps) -> Node:
    def memory_writer(state: GraphState, runtime: Runtime[PlannerContext]) -> dict[str, object]:
        """End of turn actions.

        Examples: Replaces the session summary with this turn's recap. Saves the turn's token totals."""
        if InputKind(state["kind"]) is InputKind.MESSAGE:
            summary: str | None = _recap(state)
        else:
            # An answer that was not scored leaves the previous summary in place.
            result = state["answer_result"]
            summary = answer_recap(result) if result is not None else None
        if summary is not None:
            put(runtime.context.user_id, SUMMARY_KEY, SessionSummary(summary=summary), deps.store)
        tally = runtime.context.usage
        return {"token_usage": tally.totals() if tally is not None else None}

    return memory_writer


def build_graph(
    deps: PlannerDeps | None = None, checkpointer: BaseCheckpointSaver[str] | None = None
) -> CompiledStateGraph[GraphState, PlannerContext, GraphInput, GraphState]:
    """The compiled graph. Defaults to production wiring; callers pass user_id in `context`."""
    deps = deps or default_planner_deps()

    graph = StateGraph(GraphState, context_schema=PlannerContext, input_schema=GraphInput)

    graph.add_node(PLANNER, build_planner_graph(deps).compile())
    graph.add_node(SCORE_ANSWER, _make_score_answer(deps))
    graph.add_node(UPDATE_MASTERY, _make_update_mastery(deps))
    graph.add_node(MEMORY_WRITER, _make_memory_writer(deps))

    graph.add_conditional_edges(START, route_input, [PLANNER, SCORE_ANSWER])
    graph.add_edge(PLANNER, MEMORY_WRITER)
    graph.add_conditional_edges(SCORE_ANSWER, _after_score, [UPDATE_MASTERY, MEMORY_WRITER])
    graph.add_edge(UPDATE_MASTERY, MEMORY_WRITER)
    graph.add_edge(MEMORY_WRITER, END)

    return graph.compile(checkpointer=checkpointer or get_saver(), store=deps.store)
