"""All the nodes of the planner loop

Nodes only orchestrate. Business logic lives in explanation, assessment, and memory.
"""

import uuid
from collections.abc import Callable
from typing import Annotated, Any, Final, Protocol, TypedDict, TypeVar, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode, ToolRuntime, tools_condition
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from civica.domain.question import Question
from civica.explanation.engine import Explanation
from civica.graph.planner import (
    MAX_PLANNER_STEPS,
    LESSON_EXCERPT_LIMIT,
    HaltReason,
    PlannerContext,
    PlannerDeps,
    PreviousExchange,
    ToolCall,
    build_planner_tools,
    teach_theme,
)
from civica.graph.prompts import PLANNER_PROMPTS
from civica.graph.router import suggest_review_theme
from civica.graph.trace import TraceStep
from civica.memory.writer import MemoryNotAllowed

__all__ = ["PlannerContext", "build_planner_graph"]

START_TURN: Final = "start_turn"
PLANNER_MODEL: Final = "planner_model"
PLANNER_TOOLS: Final = "planner_tools"
FALLBACK_LESSON: Final = "fallback_lesson"
FINISH_TURN: Final = "finish_turn"

_T = TypeVar("_T")


def _last_wins(_: _T, new: _T) -> _T:
    """Reducer for fields that concurrent tool calls may each set."""
    return new


class PlannerInput(TypedDict):
    learner_message: str
    previous_exchange: PreviousExchange | None


class PlannerState(PlannerInput):
    messages: Annotated[list[AnyMessage], add_messages]
    model_calls: int
    turn_id: str
    tool_calls: list[ToolCall]
    halt_reason: Annotated[HaltReason | None, _last_wins]
    reply: str
    lesson: Annotated[Explanation | None, _last_wins]
    quiz_questions: Annotated[list[Question], _last_wins]


NodeRuntime = Runtime[PlannerContext]


class Node(Protocol):
    def __call__(self, state: PlannerState, runtime: NodeRuntime) -> dict[str, object]: ...


def _thread_id(config: RunnableConfig) -> str:
    return str(config["configurable"]["thread_id"])


def _node_thread_id(runtime: NodeRuntime) -> str:
    thread_id = runtime.execution_info.thread_id if runtime.execution_info else None
    assert thread_id is not None, "the planner graph needs a checkpointer and a thread_id"
    return thread_id


def _step(
    deps: PlannerDeps,
    state: PlannerState,
    runtime: NodeRuntime,
    step_index: int,
    *,
    call: ToolCall | None = None,
    halt_reason: HaltReason | None = None,
) -> None:
    """Write one planner_trace row for this turn."""

    deps.trace.record_step(TraceStep(
        user_id=runtime.context.user_id, thread_id=_node_thread_id(runtime),
        turn_id=uuid.UUID(state["turn_id"]), step_index=step_index,
        tool_name=call.name if call else None, tool_args=call.args if call else None,
        halt_reason=halt_reason, prompt_version=PLANNER_PROMPTS.version,
    ))


def _last_ai(state: PlannerState) -> AIMessage:
    message = state["messages"][-1]
    assert isinstance(message, AIMessage)
    return message


def start_turn(state: PlannerState) -> dict[str, object]:
    """Reset per-turn state; the checkpointer keeps the thread across turns."""

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            HumanMessage(content=state["learner_message"]),
        ],
        "model_calls": 0, "turn_id": str(uuid.uuid4()), "tool_calls": [],
        "halt_reason": None, "reply": "", "lesson": None, "quiz_questions": [],
    }


def finish_turn(state: PlannerState) -> dict[str, object]:
    """Save this turn as the previous exchange for the next one."""

    lesson = state["lesson"]
    excerpt = lesson.text[:LESSON_EXCERPT_LIMIT] if lesson is not None else None
    # On a fallback the reply is the lesson itself, so keep only the excerpt.
    reply = "" if lesson is not None and state["reply"] == lesson.text else state["reply"]
    stems = tuple(q.text for q in state["quiz_questions"])
    exchange = PreviousExchange(state["learner_message"], reply, excerpt, stems)
    return {"previous_exchange": exchange}


def _coach_text(exchange: PreviousExchange) -> str:
    """The coach's side of the previous exchange: reply, lesson excerpt, numbered quiz stems."""

    numbered = [f"{n}. {stem}" for n, stem in enumerate(exchange.quiz_stems, start=1)]
    parts = [exchange.reply, exchange.lesson_excerpt or "", *numbered]
    return "\n\n".join(part for part in parts if part)


def _previous_messages(state: PlannerState) -> list[AnyMessage]:
    """The previous exchange as a human and AI message pair; empty on a thread's first turn."""

    exchange = state.get("previous_exchange")
    if exchange is None:
        return []
    return [HumanMessage(content=exchange.learner_message), AIMessage(content=_coach_text(exchange))]


def _make_planner_model(deps: PlannerDeps) -> Node:
    model = deps.planner_model.bind_tools(build_planner_tools(deps))

    def planner_model(
        state: PlannerState, runtime: NodeRuntime
    ) -> dict[str, object]:
        """One model call; the system prompt and previous exchange are added here and never stored."""

        system = SystemMessage(content=PLANNER_PROMPTS.system())
        response = model.invoke([system, *_previous_messages(state), *state["messages"]])
        assert isinstance(response, AIMessage)

        model_calls = state["model_calls"] + 1
        update: dict[str, object] = {"messages": [response], "model_calls": model_calls}
        calls = [ToolCall(name=c["name"], args=dict(c["args"])) for c in response.tool_calls]

        if not calls:
            _step(deps, state, runtime, len(state["tool_calls"]), halt_reason=HaltReason.FINAL_MESSAGE)
            update |= {"reply": response.text, "halt_reason": HaltReason.FINAL_MESSAGE}
        elif model_calls >= MAX_PLANNER_STEPS:
            update["halt_reason"] = HaltReason.MAX_STEPS
        else:
            update["tool_calls"] = [*state["tool_calls"], *calls]
        return update

    return planner_model


def _make_tool_node(deps: PlannerDeps) -> ToolNode:
    def trace_and_guard(  # type: ignore[explicit-any]
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Trace the call, then run it; a refused write halts the turn instead of raising."""

        state: PlannerState = request.state
        call = request.tool_call
        calls = _last_ai(state).tool_calls
        position = [c["id"] for c in calls].index(call["id"])
        index = len(state["tool_calls"]) - len(calls) + position
        runtime = cast("ToolRuntime[PlannerContext]", request.runtime)

        deps.trace.record_step(TraceStep(
            user_id=runtime.context.user_id, thread_id=_thread_id(runtime.config),
            turn_id=uuid.UUID(state["turn_id"]), step_index=index, tool_name=call["name"],
            tool_args=dict(call["args"]), halt_reason=None, prompt_version=PLANNER_PROMPTS.version,
        ))

        try:
            return execute(request)
        except MemoryNotAllowed as error:
            refusal = ToolMessage(
                content=f"Error: {error}", tool_call_id=call["id"], name=call["name"], status="error"
            )
            return Command(update={"halt_reason": HaltReason.FORBIDDEN_WRITE, "messages": [refusal]})

    return ToolNode(
        build_planner_tools(deps),
        handle_tool_errors=(ValueError, ToolException),
        wrap_tool_call=trace_and_guard,
    )


def _make_fallback_lesson(deps: PlannerDeps) -> Node:
    def fallback_lesson(
        state: PlannerState, runtime: NodeRuntime
    ) -> dict[str, object]:
        """Serve a plain lesson on the review theme and write the terminal trace row."""

        theme = suggest_review_theme(runtime.context.user_id, store=deps.store)
        lesson = teach_theme(deps, theme, theme.display_name_fr)
        reason = state["halt_reason"]
        assert reason is not None
        _step(deps, state, runtime, len(state["tool_calls"]), halt_reason=reason)
        return {"lesson": lesson, "reply": lesson.text}

    return fallback_lesson


def _after_model(state: PlannerState) -> str:
    if state["halt_reason"] == HaltReason.MAX_STEPS:
        return FALLBACK_LESSON
    wants_tools = tools_condition(state) == "tools"  # type: ignore[arg-type]
    return PLANNER_TOOLS if wants_tools else FINISH_TURN


def _after_tools(state: PlannerState) -> str:
    return FALLBACK_LESSON if state["halt_reason"] == HaltReason.FORBIDDEN_WRITE else PLANNER_MODEL


def build_planner_graph(deps: PlannerDeps) -> StateGraph[PlannerState, PlannerContext, PlannerInput, PlannerState]:
    """The uncompiled planner graph. Caller selects the checkpointer and store."""

    graph = StateGraph(PlannerState, context_schema=PlannerContext, input_schema=PlannerInput)

    graph.add_node(START_TURN, start_turn)
    graph.add_node(PLANNER_MODEL, _make_planner_model(deps))
    graph.add_node(PLANNER_TOOLS, _make_tool_node(deps))
    graph.add_node(FALLBACK_LESSON, _make_fallback_lesson(deps))
    graph.add_node(FINISH_TURN, finish_turn)

    graph.add_edge(START, START_TURN)
    graph.add_edge(START_TURN, PLANNER_MODEL)
    graph.add_conditional_edges(PLANNER_MODEL, _after_model, [PLANNER_TOOLS, FALLBACK_LESSON, FINISH_TURN])
    graph.add_conditional_edges(PLANNER_TOOLS, _after_tools, [PLANNER_MODEL, FALLBACK_LESSON])
    graph.add_edge(FALLBACK_LESSON, FINISH_TURN)
    graph.add_edge(FINISH_TURN, END)

    return graph
