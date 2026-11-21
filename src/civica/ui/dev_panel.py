"""The developer panel shown beside each coach response."""

from dataclasses import dataclass
from typing import Final

from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot

from civica.assessment.grounding import QuestionSimilarity, QuizGrounding
from civica.domain.question import Question
from civica.explanation.engine import MINIMUM_SIMILARITY as LESSON_MINIMUM_SIMILARITY
from civica.explanation.engine import RETRIEVAL_K as LESSON_RETRIEVAL_K
from civica.explanation.engine import Explanation
from civica.graph.nodes import FALLBACK_LESSON
from civica.graph.planner import HaltReason, LearnerState, ToolCall
from civica.graph.tool_names import PlannerTool
from civica.llm.usage import TokenCounts, TurnUsage
from civica.ui.turn_view import is_fallback_reason

CODE_PATH: Final = "Code path"
TOOLS_CALLED: Final = "Tools called"
GROUNDING: Final = "Grounding"
TURN: Final = "Turn"
TOKEN_COUNTS: Final = "Token Counts"
LEARNER_STATE: Final = "Learner state"
ERROR: Final = "Error"
NONE_TEXT: Final = "none"
NO_PASSAGES_TEXT: Final = "no passages found"
MAX_MISTAKES_SHOWN: Final = 5

_FALLBACK_PRODUCER: Final = f"{FALLBACK_LESSON} node"
_FINAL_MESSAGE: Final = "planner final message"
_TOTAL: Final = "total"


@dataclass(frozen=True)
class PanelLine:
    """One labelled line. Values are never empty: NONE_TEXT stands in when there is nothing to show."""

    label: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class DevPanel:
    lines: tuple[PanelLine, ...]


@dataclass(frozen=True)
class TurnPanels:
    """A panel for each response the page draws; a response the page hides has none."""

    lesson: DevPanel | None
    quiz: DevPanel | None
    reply: DevPanel | None


def _written_by(calls: list[ToolCall], name: PlannerTool) -> str:
    """The code path that triggered the response."""
    return name if any(call.name == name for call in calls) else NONE_TEXT


def _lesson_grounding(lesson: Explanation) -> tuple[str, ...]:
    # Similarities stay in retrieval order, unsorted, so the developer sees what retrieval returned.
    similarities = ", ".join(f"{item.similarity:.2f}" for item in lesson.context_chunks)
    return (
        f"chunks retrieved: {len(lesson.context_chunks)} (max {LESSON_RETRIEVAL_K})",
        f"similarities: {similarities or NO_PASSAGES_TEXT}",
        f"minimum allowed: {LESSON_MINIMUM_SIMILARITY:.2f}",
    )


def _similarity_range(entry: QuestionSimilarity) -> str:
    text = f"{entry.highest:.2f}-{entry.lowest:.2f}"
    if entry.chunks is None: # may be true if loading old versions of quiz turns
        return text
    return f"{text} ({entry.chunks} chunk{'' if entry.chunks == 1 else 's'})"


def _quiz_grounding(questions: list[Question], grounding: QuizGrounding | None) -> tuple[str, ...]:
    sections = {source for question in questions for source in question.sources}
    counts = (f"questions generated: {len(questions)}", f"sections cited: {len(sections)}")
    if grounding is None:
        return counts
    ranges = (f"Q{number}: {_similarity_range(entry)}" for number, entry in enumerate(grounding.questions, start=1))
    return (
        *counts,
        f"chunks retrieved: {grounding.chunks_retrieved} (max {grounding.max_chunks})",
        "similarity ranges:",
        *ranges,
        f"minimum allowed: {grounding.minimum_similarity:.2f}",
    )


def _turn_line(halt_reason: HaltReason, config: RunnableConfig) -> PanelLine:
    fallback = "yes" if is_fallback_reason(halt_reason) else "no"
    thread_id = config["configurable"]["thread_id"]
    return PanelLine(TURN, (f"halt_reason={halt_reason.value}", f"fallback={fallback}", f"thread_id={thread_id}"))


def token_values(usage: TurnUsage) -> tuple[str, ...]:
    """The turn's total, then each caller by name."""

    def value(name: str, counts: TokenCounts) -> str:
        return f"{name}: in={counts.input} out={counts.output} reasoning={counts.reasoning}"

    callers = (value(name, usage.by_caller[name]) for name in sorted(usage.by_caller))
    return (value(_TOTAL, usage.total), *callers)


def _token_line(usage: TurnUsage | None) -> PanelLine:
    """The turn's token values. A turn saved before tokens were counted has none."""
    return PanelLine(TOKEN_COUNTS, (NONE_TEXT,) if usage is None else token_values(usage))


def _learner_line(learner: LearnerState) -> PanelLine:
    mastery = ", ".join(f"{theme.slug}={score:.2f}" for theme, score in learner.mastery.items())
    shown = [f"{mistake.theme.slug}/{mistake.question_id}" for mistake in learner.mistakes[:MAX_MISTAKES_SHOWN]]
    if len(learner.mistakes) > MAX_MISTAKES_SHOWN:
        shown.append(f"+{len(learner.mistakes) - MAX_MISTAKES_SHOWN} more")
    return PanelLine(
        LEARNER_STATE,
        (
            f"goal={learner.goal or NONE_TEXT}",
            f"mastery={mastery or NONE_TEXT}",
            f"mistakes={', '.join(shown) or NONE_TEXT}",
        ),
    )


def build_turn_panels(  # type: ignore[explicit-any]
    snapshot: StateSnapshot, config: RunnableConfig, learner: LearnerState
) -> TurnPanels | None:
    """A panel per drawn response of the saved turn, or None for an empty thread or an unfinished turn."""
    values = snapshot.values
    if not values or snapshot.next:
        return None
    calls: list[ToolCall] = values["tool_calls"]
    lesson: Explanation | None = values["lesson"]
    questions: list[Question] = values["quiz_questions"]
    halt_reason = HaltReason(values["halt_reason"])
    tools = PanelLine(TOOLS_CALLED, tuple(call.name for call in calls) or (NONE_TEXT,))
    shared = (_turn_line(halt_reason, config), _token_line(values.get("token_usage")), _learner_line(learner))

    def panel(producer: str, grounding: tuple[str, ...]) -> DevPanel:
        return DevPanel((PanelLine(CODE_PATH, (producer,)), tools, PanelLine(GROUNDING, grounding), *shared))

    lesson_panel = quiz_panel = reply_panel = None
    if lesson is not None:
        producer = _FALLBACK_PRODUCER if is_fallback_reason(halt_reason) else _written_by(calls, PlannerTool.TEACH)
        lesson_panel = panel(producer, _lesson_grounding(lesson))
    if questions:
        quiz_panel = panel(_written_by(calls, PlannerTool.GENERATE_QUIZ), _quiz_grounding(questions, values.get("quiz_grounding")))
    if lesson is None and not questions:
        reply_panel = panel(_FINAL_MESSAGE, (NONE_TEXT,))
    return TurnPanels(lesson_panel, quiz_panel, reply_panel)


def build_error_panel(error_type: str) -> DevPanel:
    """The panel for a turn that failed: just the error type."""
    return DevPanel((PanelLine(ERROR, (error_type,)),))
