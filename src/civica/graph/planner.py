"""Bounded tool-calling planner: picks what to teach or quiz next, using tools for every fact.

The user id is bound in a closure, so the model can never choose whose memory it touches.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.store.base import BaseStore

from civica.assessment.critic import CritiqueFn
from civica.assessment.engine import generate_quiz
from civica.assessment.errors import InsufficientMaterialError, QuestionGenerationError
from civica.assessment.scoring import AnswerLogger, Quiz
from civica.assessment.store import (
    PostgresQuestionStore,
    PostgresRejectionStore,
    QuestionStore,
    RejectionStore,
)
from civica.domain.themes import EXAM_QUESTION_COUNTS, Theme
from civica.domain.user import UserId
from civica.explanation.engine import MINIMUM_SIMILARITY, Explanation, explain
from civica.graph.prompts import PLANNER_PROMPTS
from civica.graph.router import suggest_review_theme as review_heuristic
from civica.graph.trace import (
    HaltReason,
    PostgresTraceWriter,
    RetrievalGap,
    TraceStep,
    TraceWriter,
)
from civica.llm.client import PLANNER_PROFILE, get_chat_model
from civica.memory.records import LearnerProfile, MistakeEpisode, TopicMastery
from civica.memory.store import get_store
from civica.memory.writer import MemoryNotAllowed, get, list_records, put_planner
from civica.progress.quiz_log import log_answer
from civica.retrieval.content import ContentChunk, SearchFn, search

MAX_PLANNER_STEPS = 6

# Caps the quiz size the model may request, bounding generation cost per call.
MAX_QUIZ_QUESTIONS = 10

# One LearnerProfile per user lives under this key.
PROFILE_KEY = "current"


class PlannerMode(StrEnum):
    TEACH = "teach"
    QUIZ = "quiz"


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model made, with the arguments it supplied."""

    name: str
    args: Mapping[str, object]


@dataclass(frozen=True)
class PlannerResult:
    reply: str
    halt_reason: HaltReason
    tool_calls: tuple[ToolCall, ...]
    lesson: Explanation | None
    quiz: Quiz | None


@dataclass(frozen=True)
class PlannerDeps:
    """Every seam the planner uses. None for retriever or content_model means each engine's default."""

    planner_model: BaseChatModel
    store: BaseStore
    question_store: QuestionStore
    answer_logger: AnswerLogger
    rejection_store: RejectionStore
    trace: TraceWriter
    content_model: BaseChatModel | None = None
    retriever: SearchFn | None = None
    critic: CritiqueFn | None = None


@dataclass(frozen=True)
class LearnerState:
    mastery: Mapping[Theme, float]
    mistakes: tuple[MistakeEpisode, ...]
    goal: str | None


@dataclass
class _TurnState:
    """What the tools produced this turn, for the caller."""

    lesson: Explanation | None = None
    quiz: Quiz | None = None
    calls: list[ToolCall] = field(default_factory=list)


def default_planner_deps() -> PlannerDeps:
    """Production wiring; each engine keeps its own default retriever and content model."""
    return PlannerDeps(
        planner_model=get_chat_model(PLANNER_PROFILE),
        store=get_store(),
        question_store=PostgresQuestionStore(),
        answer_logger=log_answer,
        rejection_store=PostgresRejectionStore(),
        trace=PostgresTraceWriter(),
    )


def read_learner_state(user_id: UserId, store: BaseStore) -> LearnerState:
    """Mastery for recorded themes, all mistake episodes, and the stated goal."""
    mastery: dict[Theme, float] = {}
    for theme in EXAM_QUESTION_COUNTS:
        record = get(user_id, theme.slug, TopicMastery, store)
        if record is not None:
            mastery[theme] = record.mastery
    profile = get(user_id, PROFILE_KEY, LearnerProfile, store)
    return LearnerState(
        mastery=mastery,
        mistakes=tuple(list_records(user_id, MistakeEpisode, store)),
        goal=profile.goal if profile is not None else None,
    )


def _corpus_search(query: str, theme: Theme | None) -> list[ContentChunk]:
    """Default retrieve_corpus search, with the same similarity floor as teach."""
    return search(query, theme, min_similarity=MINIMUM_SIMILARITY)


def _format_state(state: LearnerState) -> str:
    mastery = ", ".join(f"{t.slug}={m:.2f}" for t, m in state.mastery.items()) or "none recorded"
    mistakes = ", ".join(f"{m.theme.slug}/{m.question_id}" for m in state.mistakes) or "none"
    return f"Goal: {state.goal or 'not stated'}\nMastery: {mastery}\nMistakes: {mistakes}"


def _format_quiz(quiz: Quiz) -> str:
    lines: list[str] = []
    for number, question in enumerate(quiz.questions, start=1):
        options = "; ".join(f"{i}) {option}" for i, option in enumerate(question.options))
        lines.append(f"{number}. {question.text} Options: {options}")
    return "\n".join(lines)


def _teach(deps: PlannerDeps, theme: Theme, focus: str) -> Explanation:
    """Explain a focus; an ungrounded lesson is logged as a retrieval gap."""
    if deps.retriever is None:
        lesson = explain(focus, theme, chat_client=deps.content_model)
    else:
        lesson = explain(focus, theme, retriever=deps.retriever, chat_client=deps.content_model)
    if not lesson.context_chunks:
        deps.trace.record_gap(RetrievalGap(theme=theme, query=focus))
    return lesson


def _make_quiz(deps: PlannerDeps, user_id: UserId, theme: Theme, n: int) -> Quiz:
    if not 1 <= n <= MAX_QUIZ_QUESTIONS:
        raise ValueError(f"n must be between 1 and {MAX_QUIZ_QUESTIONS}")
    if deps.retriever is None:
        return generate_quiz(
            user_id, theme, n, chat_client=deps.content_model,
            question_store=deps.question_store, answer_logger=deps.answer_logger,
            critic=deps.critic, rejection_store=deps.rejection_store,
        )
    return generate_quiz(
        user_id, theme, n, retriever=deps.retriever, chat_client=deps.content_model,
        question_store=deps.question_store, answer_logger=deps.answer_logger,
        critic=deps.critic, rejection_store=deps.rejection_store,
    )


def _build_tools(user_id: UserId, deps: PlannerDeps, turn: _TurnState) -> list[BaseTool]:
    def suggest_review_theme() -> str:
        """Suggest the theme the learner should review next."""
        return review_heuristic(user_id, store=deps.store).slug

    def read_state() -> str:
        """Read the learner's goal, mastery per theme, and past mistakes."""
        return _format_state(read_learner_state(user_id, deps.store))

    def retrieve_corpus(theme: str, query: str) -> str:
        """Retrieve official corpus passages for a theme slug and a query."""
        retriever = deps.retriever or _corpus_search
        parsed = Theme.from_slug(theme)
        chunks = retriever(query, parsed)
        if not chunks:
            deps.trace.record_gap(RetrievalGap(theme=parsed, query=query))
        return "\n\n".join(c.chunk.text for c in chunks) or "No passages found."

    def teach(theme: str, focus: str) -> str:
        """Explain a focus topic within a theme slug, grounded in the corpus."""
        lesson = _teach(deps, Theme.from_slug(theme), focus)
        turn.lesson = lesson
        return lesson.text

    def generate_quiz_tool(theme: str, n: int) -> str:
        """Generate n multiple-choice questions for a theme slug."""
        try:
            quiz = _make_quiz(deps, user_id, Theme.from_slug(theme), n)
        except InsufficientMaterialError as error:
            for query in error.queries:
                deps.trace.record_gap(RetrievalGap(theme=error.theme, query=query))
            return f"No quiz could be generated: {error}"
        except QuestionGenerationError as error:
            return f"No quiz could be generated: {error}"
        turn.quiz = quiz
        return _format_quiz(quiz)

    def record_outcome(kind: str, key: str, value: dict[str, object]) -> str:
        """Record a mistake_episode or session_summary for the learner."""
        put_planner(user_id, kind, key, value, deps.store)
        return f"Recorded {kind}."

    return [
        StructuredTool.from_function(suggest_review_theme),
        StructuredTool.from_function(read_state, name="read_learner_state"),
        StructuredTool.from_function(retrieve_corpus),
        StructuredTool.from_function(teach),
        StructuredTool.from_function(generate_quiz_tool, name="generate_quiz"),
        StructuredTool.from_function(record_outcome),
    ]


def _run_tool(tool: BaseTool, args: dict[str, object]) -> str:
    """Run one tool; bad arguments come back as text for the model. MemoryNotAllowed propagates."""
    try:
        return str(tool.invoke(args))
    except (ValueError, ToolException) as error:
        return f"Error: {error}"


def _fallback_lesson(user_id: UserId, deps: PlannerDeps) -> Explanation:
    theme = review_heuristic(user_id, store=deps.store)
    return _teach(deps, theme, theme.display_name_fr)


def run_planner(
    user_id: UserId, learner_message: str, mode: PlannerMode, deps: PlannerDeps, *, thread_id: str
) -> PlannerResult:
    """Run the tool loop for one learner turn, halting after MAX_PLANNER_STEPS model calls."""
    turn = _TurnState()
    turn_id = uuid.uuid4()

    def write_row(tool: ToolCall | None, halt_reason: HaltReason | None) -> None:
        deps.trace.record_step(TraceStep(
            user_id=user_id, thread_id=thread_id, turn_id=turn_id, step_index=len(turn.calls),
            tool_name=tool.name if tool else None, tool_args=tool.args if tool else None,
            halt_reason=halt_reason, prompt_version=PLANNER_PROMPTS.version,
        ))

    tools = {tool.name: tool for tool in _build_tools(user_id, deps, turn)}
    model = deps.planner_model.bind_tools(list(tools.values()))
    messages: list[BaseMessage] = [
        SystemMessage(content=PLANNER_PROMPTS.system(mode=mode.value)),
        HumanMessage(content=learner_message),
    ]
    for step in range(1, MAX_PLANNER_STEPS + 1):
        response = model.invoke(messages)
        if not isinstance(response, AIMessage) or not response.tool_calls:
            write_row(None, HaltReason.FINAL_MESSAGE)
            return PlannerResult(
                reply=response.text, halt_reason=HaltReason.FINAL_MESSAGE,
                tool_calls=tuple(turn.calls), lesson=turn.lesson, quiz=turn.quiz,
            )
        if step == MAX_PLANNER_STEPS:
            break
        messages.append(response)
        for call in response.tool_calls:
            args: dict[str, object] = dict(call["args"])
            tool_call = ToolCall(name=call["name"], args=args)
            write_row(tool_call, None)
            turn.calls.append(tool_call)
            try:
                output = _run_tool(tools[call["name"]], args) if call["name"] in tools else "Error: unknown tool"
            except MemoryNotAllowed:
                write_row(None, HaltReason.FORBIDDEN_WRITE)
                raise
            messages.append(ToolMessage(content=output, tool_call_id=call["id"] or ""))

    lesson = _fallback_lesson(user_id, deps)
    write_row(None, HaltReason.MAX_STEPS)
    return PlannerResult(
        reply=lesson.text, halt_reason=HaltReason.MAX_STEPS,
        tool_calls=tuple(turn.calls), lesson=lesson, quiz=turn.quiz,
    )
