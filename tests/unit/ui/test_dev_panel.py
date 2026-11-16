"""Unit tests for civica.ui.dev_panel. Final states are built by hand: no graph, no database."""

from collections.abc import Mapping, Sequence

import pytest
from langchain_core.runnables import RunnableConfig

from civica.assessment.engine import MINIMUM_SIMILARITY as QUIZ_MINIMUM_SIMILARITY
from civica.assessment.engine import RETRIEVAL_K as QUIZ_RETRIEVAL_K
from civica.assessment.grounding import QuestionSimilarity, QuizGrounding
from civica.domain.question import Question
from civica.domain.source_ref import SourceRef
from civica.domain.themes import HISTOIRE_GEOGRAPHIE_ET_CULTURE
from civica.explanation.engine import MINIMUM_SIMILARITY as LESSON_MINIMUM_SIMILARITY
from civica.explanation.engine import RETRIEVAL_K as LESSON_RETRIEVAL_K
from civica.explanation.engine import Explanation
from civica.graph.graph import PLANNER
from civica.graph.planner import HaltReason, LearnerState, ToolCall
from civica.llm.usage import TokenCounts, TurnUsage
from civica.memory.records import MistakeEpisode
from civica.retrieval.content import ContentChunk
from civica.ui.dev_panel import (
    ERROR,
    GROUNDING,
    LEARNER_STATE,
    MAX_MISTAKES_SHOWN,
    NO_PASSAGES_TEXT,
    NONE_TEXT,
    CODE_PATH,
    TOKEN_COUNTS,
    TOOLS_CALLED,
    TURN,
    DevPanel,
    PanelLine,
    TurnPanels,
    build_error_panel,
    build_turn_panels,
)
from tests.support.assessment import SAMPLE_SOURCE, make_content_chunk, make_question
from tests.support.coach import (
    FALLBACK_NODE,
    FINAL_MESSAGE_TEXT,
    FOCUS,
)
from tests.support.llm import CONTENT_USAGE, EXPLANATION_TAG, PLANNER_TAG, PLANNER_USAGE, QUIZ_TAG
from tests.support.memory import HALF_MASTERY
from tests.support.planner import (
    DEFAULT_THEME,
    GENERATE_QUIZ,
    GOAL,
    READ_STATE,
    RETRIEVE_CORPUS,
    SUGGEST_THEME,
    TEACH,
)
from tests.support.turn_state import REPLY, finished_state, snapshot_of
from tests.support.usage import counts_of, token_line, turn_usage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THREAD_ID = "sample-thread-001"
_CONFIG: RunnableConfig = {"configurable": {"thread_id": _THREAD_ID}}

_LESSON_TEXT = "sample-lesson-text"
_FALLBACK_TEXT = "sample-fallback-lesson-text"
_QUERY = "sample-query"
_QUESTION_COUNT = 2
_ERROR_TYPE = "SampleError"

_OTHER_THEME = HISTOIRE_GEOGRAPHIE_ET_CULTURE
_OTHER_MASTERY = 0.4
# Shares no text with SAMPLE_SOURCE, so one source cannot be mistaken for the other.
_OTHER_SOURCE = SourceRef(page_slug="other-page", section_id="other-part")

# Deliberately unsorted, so a panel that re-ranks the chunks is caught.
_SIMILARITIES = (0.7, 0.9, 0.8)
_SIMILARITY_TEXTS = ("0.70", "0.90", "0.80")
_LESSON_MINIMUM_TEXT = f"minimum allowed: {LESSON_MINIMUM_SIMILARITY:.2f}"

_QUIZ_CHUNKS_RETRIEVED = 23
_RANGE_HIGH = 0.62
_RANGE_LOW = 0.55
_RANGE_HIGH_TEXT = "0.62"
_RANGE_LOW_TEXT = "0.55"
_SINGLE_SIMILARITY = 0.58
_SINGLE_SIMILARITY_TEXT = "0.58"
_MANY_CHUNKS = 3
_RANGE_TEXT = f"{_RANGE_HIGH_TEXT}-{_RANGE_LOW_TEXT}"

_LABELS = (CODE_PATH, TOOLS_CALLED, GROUNDING, TURN, TOKEN_COUNTS, LEARNER_STATE)
_NONE = (NONE_TEXT,)
_FALLBACK_REASONS = (HaltReason.MAX_STEPS, HaltReason.FORBIDDEN_WRITE)
_STEPS_BEFORE_FALLBACK = 5

# Values the Turn line shows: halt_reason, fallback, thread_id.
_TURN_VALUE_COUNT = 3
_TOTAL = "total"
_PLANNER_COUNTS = counts_of(PLANNER_USAGE, calls=3)
_EXPLANATION_COUNTS = counts_of(CONTENT_USAGE)
_QUIZ_COUNTS = counts_of(CONTENT_USAGE, calls=2)
# Deliberately not in name order, so a panel that keeps the saved order is caught.
_TOKEN_USAGE = turn_usage({QUIZ_TAG: _QUIZ_COUNTS, PLANNER_TAG: _PLANNER_COUNTS, EXPLANATION_TAG: _EXPLANATION_COUNTS})

_NO_LEARNER_STATE = LearnerState(mastery={}, mistakes=(), goal=None, last_session=None)


def _call(name: str, **args: object) -> ToolCall:
    return ToolCall(name=name, args=args)


_READ = _call(READ_STATE)
_TEACH = _call(TEACH, theme=DEFAULT_THEME.slug, focus=FOCUS)
_SUGGEST = _call(SUGGEST_THEME)
_QUIZ = _call(GENERATE_QUIZ, theme=DEFAULT_THEME.slug, n=_QUESTION_COUNT)
_RETRIEVE = _call(RETRIEVE_CORPUS, query=_QUERY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks() -> tuple[ContentChunk, ...]:
    return tuple(
        make_content_chunk(DEFAULT_THEME, index).model_copy(update={"similarity": similarity})
        for index, similarity in enumerate(_SIMILARITIES)
    )


def _lesson(chunks: Sequence[ContentChunk] | None = None, text: str = _LESSON_TEXT) -> Explanation:
    return Explanation(text=text, context_chunks=tuple(_chunks() if chunks is None else chunks))


def _lesson_turn(*calls: ToolCall) -> dict[str, object]:
    return {"lesson": _lesson(), "tool_calls": list(calls or (_READ, _TEACH))}


def _two_questions(first: SourceRef, second: SourceRef) -> list[Question]:
    return [
        make_question(text="sample-question-text-001", sources=(first,)),
        make_question(text="sample-question-text-002", sources=(second,)),
    ]


def _quiz_grounding(questions: Sequence[Question], *ranges: tuple[float, float, int | None]) -> QuizGrounding:
    """A grounding record giving each question its (highest, lowest, chunks), in question order."""
    return QuizGrounding(
        chunks_retrieved=_QUIZ_CHUNKS_RETRIEVED,
        max_chunks=QUIZ_RETRIEVAL_K,
        minimum_similarity=QUIZ_MINIMUM_SIMILARITY,
        questions=tuple(
            QuestionSimilarity(question_id=question.question_id, highest=highest, lowest=lowest, chunks=chunks)
            for question, (highest, lowest, chunks) in zip(questions, ranges, strict=True)
        ),
    )


def _quiz_turn(*calls: ToolCall) -> dict[str, object]:
    return {"quiz_questions": [make_question()], "tool_calls": list(calls or (_SUGGEST, _QUIZ))}


def _lesson_and_quiz_turn(*calls: ToolCall) -> dict[str, object]:
    """A lesson with a comprehension check; by default every call has a different name."""
    return _lesson_turn(*(calls or (_READ, _TEACH, _SUGGEST, _QUIZ, _RETRIEVE))) | {"quiz_questions": [make_question()]}


def _reply_turn() -> dict[str, object]:
    return {"tool_calls": [_READ]}


def _fallback_turn(halt_reason: HaltReason = HaltReason.MAX_STEPS, *earlier: ToolCall) -> dict[str, object]:
    """A turn the planner never finished: the fallback lesson is also the reply."""
    return {
        "lesson": _lesson(text=_FALLBACK_TEXT),
        "reply": _FALLBACK_TEXT,
        "tool_calls": [*earlier, *[_SUGGEST] * _STEPS_BEFORE_FALLBACK],
        "halt_reason": halt_reason,
    }


def _panels(turn: Mapping[str, object], learner: LearnerState = _NO_LEARNER_STATE) -> TurnPanels:
    panels = build_turn_panels(snapshot_of(finished_state(**turn)), _CONFIG, learner)
    assert panels is not None
    return panels


def _drawn(panels: TurnPanels) -> list[DevPanel]:
    return [panel for panel in (panels.lesson, panels.quiz, panels.reply) if panel is not None]


def _values(panel: DevPanel | None, label: str) -> tuple[str, ...]:
    """The values of the panel's one line with this label."""
    assert panel is not None, "the response has no panel"
    matching = [line.values for line in panel.lines if line.label == label]
    assert len(matching) == 1, f"expected one {label!r} line, found {len(matching)}"
    return matching[0]


def _value(panel: DevPanel | None, label: str) -> str:
    values = _values(panel, label)
    assert len(values) == 1, f"expected one {label!r} value, found {values}"
    return values[0]


def _token_values(panel: DevPanel | None) -> tuple[str, ...]:
    return _values(panel, TOKEN_COUNTS)


def _range_line(panel: DevPanel | None, number: int) -> str:
    """The Grounding value for question `number`'s similarity range."""
    matching = [value for value in _values(panel, GROUNDING) if value.startswith(f"Q{number}: ")]
    assert len(matching) == 1, f"expected one Q{number} range line, found {matching}"
    return matching[0]


def _keyed(panel: DevPanel | None, label: str, key: str) -> str:
    """The line's one value written as `key=...`."""
    matching = [value for value in _values(panel, label) if value.startswith(f"{key}=")]
    assert len(matching) == 1, f"expected one {key}= value under {label!r}, found {matching}"
    return matching[0]


def _shows(value: str, call: ToolCall) -> bool:
    """Whether the value reads as this call."""
    return value == call.name


def _assert_calls(values: Sequence[str], *calls: ToolCall) -> None:
    """The values are exactly these calls, one value each, in call order."""
    assert len(values) == len(calls), f"expected {len(calls)} calls, found {values}"
    for value, call in zip(values, calls):
        assert _shows(value, call), f"{value!r} does not show {call}"


def _learner_panel(learner: LearnerState) -> DevPanel | None:
    return _panels(_lesson_turn(), learner).lesson


def _mistakes(count: int) -> tuple[MistakeEpisode, ...]:
    return tuple(
        MistakeEpisode(theme=DEFAULT_THEME, question_id=f"sample-question-id-{number:03d}")
        for number in range(1, count + 1)
    )


def _mistakes_shown(value: str, mistakes: Sequence[MistakeEpisode]) -> int:
    return sum(f"{mistake.theme.slug}/{mistake.question_id}" in value for mistake in mistakes)


_EVERY_TURN = [
    pytest.param(_lesson_turn(), id="lesson"),
    pytest.param(_lesson_turn() | {"lesson": _lesson(chunks=())}, id="insufficient-material"),
    pytest.param(_quiz_turn(), id="quiz-only"),
    pytest.param(_lesson_and_quiz_turn(), id="lesson-and-quiz"),
    pytest.param(_reply_turn(), id="reply-only"),
    pytest.param(_fallback_turn(), id="fallback"),
]


# ---------------------------------------------------------------------------
# Which responses get a panel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("turn", "lesson", "quiz", "reply"),
    [
        pytest.param(_lesson_turn(), True, False, False, id="lesson"),
        pytest.param(_quiz_turn(), False, True, False, id="quiz-only"),
        pytest.param(_lesson_and_quiz_turn(), True, True, False, id="lesson-and-quiz"),
        pytest.param(_reply_turn(), False, False, True, id="reply-only"),
        pytest.param(_fallback_turn(), True, False, False, id="fallback"),
    ],
)
def test_a_panel_exists_for_each_response_the_page_draws_and_none_for_a_hidden_reply(
    turn: dict[str, object], lesson: bool, quiz: bool, reply: bool
) -> None:
    panels = _panels(turn)

    assert (panels.lesson is not None, panels.quiz is not None, panels.reply is not None) == (lesson, quiz, reply)


def test_empty_thread_has_no_panels() -> None:
    assert build_turn_panels(snapshot_of({}), _CONFIG, _NO_LEARNER_STATE) is None


def test_unfinished_turn_has_no_panels() -> None:
    # A pending node means the values may still hold the previous turn's output.
    stale = snapshot_of(finished_state(**_lesson_and_quiz_turn()), next_nodes=[PLANNER])

    assert build_turn_panels(stale, _CONFIG, _NO_LEARNER_STATE) is None


@pytest.mark.parametrize("turn", _EVERY_TURN)
def test_every_panel_has_the_same_five_lines_in_order_and_no_line_is_empty(turn: dict[str, object]) -> None:
    drawn = _drawn(_panels(turn))

    assert drawn
    for panel in drawn:
        assert tuple(line.label for line in panel.lines) == _LABELS
        assert all(line.values and all(line.values) for line in panel.lines)


# ---------------------------------------------------------------------------
# Code path
# ---------------------------------------------------------------------------


def test_lesson_panel_names_the_teach_call_as_its_producer() -> None:
    assert _shows(_value(_panels(_lesson_turn()).lesson, CODE_PATH), _TEACH)


def test_quiz_panel_names_the_generate_quiz_call_as_its_producer() -> None:
    assert _shows(_value(_panels(_quiz_turn()).quiz, CODE_PATH), _QUIZ)


def test_reply_panel_names_the_planners_final_message_as_its_producer() -> None:
    assert FINAL_MESSAGE_TEXT in _value(_panels(_reply_turn()).reply, CODE_PATH)


@pytest.mark.parametrize("halt_reason", _FALLBACK_REASONS)
def test_fallback_panel_names_the_fallback_node_and_shows_the_fallback_flag(halt_reason: HaltReason) -> None:
    lesson = _panels(_fallback_turn(halt_reason)).lesson

    producer = _value(lesson, CODE_PATH)
    assert FALLBACK_NODE in producer
    assert TEACH not in producer
    assert _keyed(lesson, TURN, "fallback") == "fallback=yes"


def test_fallback_node_stays_the_producer_when_the_planner_had_already_taught() -> None:
    lesson = _panels(_fallback_turn(HaltReason.MAX_STEPS, _TEACH)).lesson

    assert TEACH not in _value(lesson, CODE_PATH)
    _assert_calls(_values(lesson, TOOLS_CALLED), _TEACH, *[_SUGGEST] * _STEPS_BEFORE_FALLBACK)


# ---------------------------------------------------------------------------
# Tools called
# ---------------------------------------------------------------------------


def test_tool_calls_show_their_names_without_arguments() -> None:
    panels = _panels(_lesson_and_quiz_turn())

    assert _values(panels.lesson, TOOLS_CALLED) == (READ_STATE, TEACH, SUGGEST_THEME, GENERATE_QUIZ, RETRIEVE_CORPUS)
    assert _value(panels.lesson, CODE_PATH) == TEACH
    assert _value(panels.quiz, CODE_PATH) == GENERATE_QUIZ


@pytest.mark.parametrize("turn", _EVERY_TURN)
def test_every_panel_lists_every_tool_call_of_the_turn_in_call_order(turn: dict[str, object]) -> None:
    calls = turn["tool_calls"]
    assert isinstance(calls, list)

    for panel in _drawn(_panels(turn)):
        _assert_calls(_values(panel, TOOLS_CALLED), *calls)


def test_calls_made_after_the_response_was_produced_are_listed_too() -> None:
    quiz = _panels(_lesson_and_quiz_turn()).quiz

    assert _shows(_values(quiz, TOOLS_CALLED)[-1], _RETRIEVE)


def test_turn_with_no_tool_calls_shows_none() -> None:
    assert _values(_panels({}).reply, TOOLS_CALLED) == _NONE


@pytest.mark.parametrize("turn", [pytest.param(_lesson_turn(), id="lesson"), pytest.param(_quiz_turn(), id="quiz")])
def test_hidden_final_message_is_shown_on_no_line(turn: dict[str, object]) -> None:
    for panel in _drawn(_panels(turn)):
        assert not [value for line in panel.lines for value in line.values if REPLY in value]


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def test_lesson_panel_shows_the_chunk_count_each_similarity_in_retrieval_order_and_the_floor() -> None:
    grounding = _values(_panels(_lesson_turn()).lesson, GROUNDING)

    assert grounding == (
        f"chunks retrieved: {len(_SIMILARITIES)} (max {LESSON_RETRIEVAL_K})",
        f"similarities: {', '.join(_SIMILARITY_TEXTS)}",
        _LESSON_MINIMUM_TEXT,
    )


def test_lesson_panel_counts_two_chunks_of_one_section_separately() -> None:
    # The learner's sources merge a section's chunks; the developer sees what retrieval returned.
    chunks = (make_content_chunk(DEFAULT_THEME, 0), make_content_chunk(DEFAULT_THEME, 0, chunk_index=1))

    lesson = _panels(_lesson_turn() | {"lesson": _lesson(chunks)}).lesson

    assert _values(lesson, GROUNDING)[0] == f"chunks retrieved: {len(chunks)} (max {LESSON_RETRIEVAL_K})"


def test_insufficient_material_lesson_shows_zero_chunks_and_no_passages_found() -> None:
    lesson = _panels(_lesson_turn() | {"lesson": _lesson(chunks=())}).lesson

    assert _values(lesson, GROUNDING) == (
        f"chunks retrieved: 0 (max {LESSON_RETRIEVAL_K})",
        f"similarities: {NO_PASSAGES_TEXT}",
        _LESSON_MINIMUM_TEXT,
    )


def test_quiz_panel_shows_counts_each_questions_similarity_range_and_the_floor() -> None:
    questions = _two_questions(SAMPLE_SOURCE, _OTHER_SOURCE)
    turn = _quiz_turn() | {
        "quiz_questions": questions,
        "quiz_grounding": _quiz_grounding(
            questions, (_RANGE_HIGH, _RANGE_LOW, _MANY_CHUNKS), (_SINGLE_SIMILARITY, _SINGLE_SIMILARITY, 1)
        ),
    }

    assert _values(_panels(turn).quiz, GROUNDING) == (
        f"questions generated: {len(questions)}",
        f"sections cited: {len(questions)}",
        f"chunks retrieved: {_QUIZ_CHUNKS_RETRIEVED} (max {QUIZ_RETRIEVAL_K})",
        "similarity ranges:",
        f"Q1: {_RANGE_TEXT} ({_MANY_CHUNKS} chunks)",
        f"Q2: {_SINGLE_SIMILARITY_TEXT}-{_SINGLE_SIMILARITY_TEXT} (1 chunk)",
        f"minimum allowed: {QUIZ_MINIMUM_SIMILARITY:.2f}",
    )


@pytest.mark.parametrize(
    ("highest", "lowest"),
    [
        pytest.param(_SINGLE_SIMILARITY, _SINGLE_SIMILARITY, id="one-chunk"),
        pytest.param(_SINGLE_SIMILARITY + 0.001, _SINGLE_SIMILARITY - 0.001, id="equal-when-rounded"),
    ],
)
def test_quiz_range_with_equal_ends_still_shows_both_ends(highest: float, lowest: float) -> None:
    questions = [make_question()]
    turn = _quiz_turn() | {"quiz_questions": questions, "quiz_grounding": _quiz_grounding(questions, (highest, lowest, 1))}

    expected = f"Q1: {_SINGLE_SIMILARITY_TEXT}-{_SINGLE_SIMILARITY_TEXT} (1 chunk)"
    assert _range_line(_panels(turn).quiz, 1) == expected


@pytest.mark.parametrize(
    ("chunks", "count"),
    [
        pytest.param(1, " (1 chunk)", id="singular"),
        pytest.param(2, " (2 chunks)", id="plural"),
        pytest.param(None, "", id="saved-before-counts"),
    ],
)
def test_quiz_range_line_shows_the_sections_chunk_count_when_recorded(chunks: int | None, count: str) -> None:
    questions = [make_question()]
    grounding = _quiz_grounding(questions, (_RANGE_HIGH, _RANGE_LOW, chunks))
    turn = _quiz_turn() | {"quiz_questions": questions, "quiz_grounding": grounding}

    assert _range_line(_panels(turn).quiz, 1) == f"Q1: {_RANGE_TEXT}{count}"


def test_quiz_sections_cited_counts_a_section_shared_by_two_questions_once() -> None:
    questions = _two_questions(SAMPLE_SOURCE, SAMPLE_SOURCE)
    turn = _quiz_turn() | {"quiz_questions": questions}

    assert "sections cited: 1" in _values(_panels(turn).quiz, GROUNDING)


@pytest.mark.parametrize(
    "grounding", [pytest.param({}, id="key-missing"), pytest.param({"quiz_grounding": None}, id="none")]
)
def test_quiz_turn_saved_without_a_grounding_record_shows_only_the_counts(grounding: dict[str, object]) -> None:
    questions = _two_questions(SAMPLE_SOURCE, _OTHER_SOURCE)
    turn = _quiz_turn() | {"quiz_questions": questions} | grounding

    assert _values(_panels(turn).quiz, GROUNDING) == (
        f"questions generated: {len(questions)}",
        f"sections cited: {len(questions)}",
    )


def test_reply_panel_has_no_grounding() -> None:
    assert _values(_panels(_reply_turn()).reply, GROUNDING) == _NONE


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("turn", "halt_reason", "fallback"),
    [
        pytest.param(_lesson_turn(), HaltReason.FINAL_MESSAGE, "no", id="final-message"),
        pytest.param(_fallback_turn(HaltReason.MAX_STEPS), HaltReason.MAX_STEPS, "yes", id="max-steps"),
        pytest.param(
            _fallback_turn(HaltReason.FORBIDDEN_WRITE), HaltReason.FORBIDDEN_WRITE, "yes", id="forbidden-write"
        ),
    ],
)
def test_turn_line_shows_why_the_turn_ended_the_fallback_flag_and_the_thread_id(
    turn: dict[str, object], halt_reason: HaltReason, fallback: str
) -> None:
    lesson = _panels(turn).lesson

    assert len(_values(lesson, TURN)) == _TURN_VALUE_COUNT
    assert halt_reason.value in _keyed(lesson, TURN, "halt_reason")
    assert _keyed(lesson, TURN, "fallback") == f"fallback={fallback}"
    assert _THREAD_ID in _keyed(lesson, TURN, "thread_id")


@pytest.mark.parametrize(
    ("turn", "halt_reason"),
    [
        pytest.param(_lesson_turn(), HaltReason.FINAL_MESSAGE, id="final-message"),
        pytest.param(_fallback_turn(HaltReason.MAX_STEPS), HaltReason.MAX_STEPS, id="max-steps"),
    ],
)
def test_halt_reason_saved_as_a_plain_string_builds_the_same_panels(
    turn: dict[str, object], halt_reason: HaltReason
) -> None:
    as_saved = _panels(turn | {"halt_reason": halt_reason.value})

    assert as_saved == _panels(turn | {"halt_reason": halt_reason})


def test_token_counts_show_the_total_then_each_caller_in_name_order() -> None:
    lesson = _panels(_lesson_turn() | {"token_usage": _TOKEN_USAGE}).lesson

    assert _token_values(lesson) == (
        token_line(_TOTAL, _TOKEN_USAGE.total),
        token_line(EXPLANATION_TAG, _EXPLANATION_COUNTS),
        token_line(PLANNER_TAG, _PLANNER_COUNTS),
        token_line(QUIZ_TAG, _QUIZ_COUNTS),
    )


def test_turn_line_shows_no_token_counts() -> None:
    lesson = _panels(_lesson_turn() | {"token_usage": _TOKEN_USAGE}).lesson

    keys = [value.partition("=")[0] for value in _values(lesson, TURN)]
    assert keys == ["halt_reason", "fallback", "thread_id"]


def test_turn_that_counted_no_model_call_shows_a_zero_total() -> None:
    usage = TurnUsage(by_caller={}, total=TokenCounts())

    reply = _panels(_reply_turn() | {"token_usage": usage}).reply

    assert _token_values(reply) == (token_line(_TOTAL, TokenCounts()),)


@pytest.mark.parametrize(
    "saved", [pytest.param({}, id="key-missing"), pytest.param({"token_usage": None}, id="none")]
)
def test_turn_saved_without_token_usage_shows_none(saved: dict[str, object]) -> None:
    # A turn saved before tokens were counted has no key; a turn run without a tally saves None.
    lesson = _panels(_lesson_turn() | saved).lesson

    assert _token_values(lesson) == _NONE


def test_every_panel_of_a_turn_shows_the_same_tools_turn_and_learner_state() -> None:
    learner = LearnerState(mastery={DEFAULT_THEME: HALF_MASTERY}, mistakes=(), goal=GOAL, last_session=None)

    panels = _panels(_lesson_and_quiz_turn() | {"token_usage": _TOKEN_USAGE}, learner)

    for label in (TOOLS_CALLED, TURN, TOKEN_COUNTS, LEARNER_STATE):
        assert _values(panels.lesson, label) == _values(panels.quiz, label)


# ---------------------------------------------------------------------------
# Learner state
# ---------------------------------------------------------------------------


def test_empty_learner_state_shows_none_for_goal_mastery_and_mistakes() -> None:
    panel = _learner_panel(_NO_LEARNER_STATE)

    assert len(_values(panel, LEARNER_STATE)) == 3
    for key in ("goal", "mastery", "mistakes"):
        assert _keyed(panel, LEARNER_STATE, key) == f"{key}={NONE_TEXT}"


def test_learner_state_shows_the_goal_each_themes_mastery_and_each_mistake() -> None:
    mistakes = _mistakes(2)
    learner = LearnerState(
        mastery={DEFAULT_THEME: HALF_MASTERY, _OTHER_THEME: _OTHER_MASTERY},
        mistakes=mistakes,
        goal=GOAL,
        last_session=None,
    )

    panel = _learner_panel(learner)

    assert len(_values(panel, LEARNER_STATE)) == 3
    assert _keyed(panel, LEARNER_STATE, "goal") == f"goal={GOAL}"
    mastery = _keyed(panel, LEARNER_STATE, "mastery")
    assert f"{DEFAULT_THEME.slug}=0.50" in mastery
    assert f"{_OTHER_THEME.slug}=0.40" in mastery
    shown = _keyed(panel, LEARNER_STATE, "mistakes")
    assert _mistakes_shown(shown, mistakes) == len(mistakes)
    assert "more" not in shown


def test_learner_state_lists_every_mistake_up_to_the_cap() -> None:
    mistakes = _mistakes(MAX_MISTAKES_SHOWN)

    shown = _keyed(_learner_panel(LearnerState({}, mistakes, None, None)), LEARNER_STATE, "mistakes")

    assert _mistakes_shown(shown, mistakes) == MAX_MISTAKES_SHOWN
    assert "more" not in shown


def test_learner_state_caps_the_mistakes_and_says_how_many_more_there_are() -> None:
    extra = 2
    mistakes = _mistakes(MAX_MISTAKES_SHOWN + extra)

    shown = _keyed(_learner_panel(LearnerState({}, mistakes, None, None)), LEARNER_STATE, "mistakes")

    assert _mistakes_shown(shown, mistakes) == MAX_MISTAKES_SHOWN
    assert f"+{extra} more" in shown


# ---------------------------------------------------------------------------
# Failed turns
# ---------------------------------------------------------------------------


def test_error_panel_holds_only_the_error_type() -> None:
    assert build_error_panel(_ERROR_TYPE) == DevPanel(lines=(PanelLine(ERROR, (_ERROR_TYPE,)),))
