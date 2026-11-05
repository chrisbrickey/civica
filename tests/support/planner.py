"""Scripted tool-calling chat model and message builders for planner tests."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.store.memory import InMemoryStore
from pydantic import PrivateAttr

from civica.assessment.critic import CritiqueFn
from civica.domain.themes import DROITS_ET_DEVOIRS
from civica.graph.planner import PlannerDeps
from civica.graph.trace import RetrievalGap, TraceStep
from civica.memory.records import MistakeEpisode

from tests.support.assessment import (
    FakeAnswerLogger,
    FakeChatModel,
    FakeQuestionStore,
    FakeRejectionStore,
    FakeRetriever,
)
from tests.support.llm import FakeChatModelBase

GOAL = "sample-goal"
SAMPLE_KEY = "sample-key-001"
SAMPLE_QUESTION_ID = "sample-question-id-001"
DEFAULT_THEME = DROITS_ET_DEVOIRS
MISTAKE = MistakeEpisode(theme=DEFAULT_THEME, question_id=SAMPLE_QUESTION_ID)

# Shapes bind_tools accepts, mirroring BaseChatModel.bind_tools.
ToolLike = dict[str, Any] | type | Callable[..., Any] | BaseTool  # type: ignore[explicit-any]


def tool_call(name: str, call_id: str, **args: object) -> AIMessage:
    """A planner-model turn that requests exactly one tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": dict(args), "id": call_id}])


def parallel_tool_calls(*replies: AIMessage) -> AIMessage:
    """One planner-model turn that requests every tool call of `replies`, in listing order."""
    return AIMessage(content="", tool_calls=[call for reply in replies for call in reply.tool_calls])


def final_message(text: str) -> AIMessage:
    """A planner-model turn with no tool call: the loop must stop here."""
    return AIMessage(content=text)


def _with_fresh_id(message: AIMessage, call_index: int) -> AIMessage:
    calls = [{**call, "id": f"{call['id']}-{call_index:03d}"} for call in message.tool_calls]
    return AIMessage(content=message.content, tool_calls=calls)


class FakePlannerModel(FakeChatModelBase):
    """Fake tool-calling planner model: replies with `script` in order, then `repeat` forever.

    bind_tools returns the model itself and remembers the offered tools. Every message list the
    model receives is recorded, so tests can inspect prompts and tool results at the LLM boundary.
    """

    script: tuple[AIMessage, ...] = ()
    # Emitted (with a call id unique to the invocation) once the script runs out.
    repeat: AIMessage | None = None

    _bound_tools: list[ToolLike] = PrivateAttr(default_factory=list)

    def bind_tools(  # type: ignore[explicit-any, override]
        self,
        tools: Sequence[ToolLike],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self._bound_tools = list(tools)
        return self  # type: ignore[return-value]

    def bound_tools(self) -> list[ToolLike]:
        """The tools the planner offered the model."""
        return self._bound_tools

    def offered_tools(self) -> dict[str, dict[str, Any]]:  # type: ignore[explicit-any]
        """Each bound tool's name mapped to its JSON-schema parameters."""
        functions = [convert_to_openai_tool(tool)["function"] for tool in self._bound_tools]
        return {f["name"]: f.get("parameters", {}) for f in functions}

    def tool_output(self, call_id: str) -> str:
        """The tool result the planner sent back for `call_id`."""
        for messages in self.received:
            for message in messages:
                if isinstance(message, ToolMessage) and message.tool_call_id == call_id:
                    return str(message.content)
        raise AssertionError(f"the planner never returned a tool result for call {call_id!r}")

    def _reply_for(self, call_index: int) -> AIMessage:
        if call_index < len(self.script):
            return self.script[call_index]
        assert self.repeat is not None, f"unexpected planner call #{call_index + 1}: script exhausted"
        return _with_fresh_id(self.repeat, call_index)

    @property
    def _llm_type(self) -> str:
        return "sample-fake-planner-model"


@dataclass
class FakeTraceWriter:
    """Keeps every step and retrieval gap in the order they were recorded."""

    steps: list[TraceStep] = field(default_factory=list)
    gaps: list[RetrievalGap] = field(default_factory=list)

    def record_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def record_gap(self, gap: RetrievalGap) -> None:
        self.gaps.append(gap)


@dataclass(frozen=True)
class PlannerFakes:
    """Every planner seam except the planner model; swap one with dataclasses.replace."""

    store: InMemoryStore
    retriever: FakeRetriever
    content_model: FakeChatModel
    question_store: FakeQuestionStore
    answer_logger: FakeAnswerLogger
    rejection_store: FakeRejectionStore
    trace: FakeTraceWriter


def planner_deps(
    fakes: PlannerFakes, planner_model: FakePlannerModel, critic: CritiqueFn | None = None
) -> PlannerDeps:
    """Planner dependencies with every seam faked."""
    return PlannerDeps(
        planner_model=planner_model,
        store=fakes.store,
        content_model=fakes.content_model,
        retriever=fakes.retriever,
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
        critic=critic,
        rejection_store=fakes.rejection_store,
        trace=fakes.trace,
    )
