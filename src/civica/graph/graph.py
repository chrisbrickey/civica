"""The full learner graph: routes each turn by input kind and ends every path at memory_writer."""

from enum import StrEnum
from typing import Final, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from civica.domain.question import Question
from civica.explanation.engine import Explanation
from civica.graph.nodes import PlannerContext, build_planner_graph
from civica.graph.planner import (
    SUMMARY_KEY,
    HaltReason,
    PlannerDeps,
    PreviousExchange,
    ToolCall,
    default_planner_deps,
)
from civica.memory.checkpointer import get_saver
from civica.memory.records import SessionSummary
from civica.memory.writer import put

PLANNER: Final = "planner"
MEMORY_WRITER: Final = "memory_writer"


class InputKind(StrEnum):
    """What a turn carries. Each kind enters the graph on its own route."""

    MESSAGE = "message"


class GraphInput(TypedDict):
    kind: InputKind
    learner_message: str


class GraphState(GraphInput):
    reply: str
    lesson: Explanation | None
    quiz_questions: list[Question]
    tool_calls: list[ToolCall]
    halt_reason: HaltReason | None
    previous_exchange: PreviousExchange | None


class Node(Protocol):
    def __call__(self, state: GraphState, runtime: Runtime[PlannerContext]) -> dict[str, object]: ...


_ROUTES: Final[dict[InputKind, str]] = {InputKind.MESSAGE: PLANNER}


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


def _make_memory_writer(deps: PlannerDeps) -> Node:
    def memory_writer(state: GraphState, runtime: Runtime[PlannerContext]) -> dict[str, object]:
        """Replace the learner's session summary with this turn's recap."""
        put(runtime.context.user_id, SUMMARY_KEY, SessionSummary(summary=_recap(state)), deps.store)
        return {}

    return memory_writer


def build_graph(
    deps: PlannerDeps | None = None, checkpointer: BaseCheckpointSaver[str] | None = None
) -> CompiledStateGraph[GraphState, PlannerContext, GraphInput, GraphState]:
    """The compiled graph. Defaults to production wiring; callers pass user_id in `context`."""
    deps = deps or default_planner_deps()

    graph = StateGraph(GraphState, context_schema=PlannerContext, input_schema=GraphInput)

    graph.add_node(PLANNER, build_planner_graph(deps).compile())
    graph.add_node(MEMORY_WRITER, _make_memory_writer(deps))

    graph.add_conditional_edges(START, route_input, [PLANNER])
    graph.add_edge(PLANNER, MEMORY_WRITER)
    graph.add_edge(MEMORY_WRITER, END)

    return graph.compile(checkpointer=checkpointer or get_saver(), store=deps.store)
