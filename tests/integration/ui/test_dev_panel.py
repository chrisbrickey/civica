"""Integration tests for the developer panel beside each coach response."""

import pytest
from nicegui.testing import User

from civica.domain.user import UserId
from civica.explanation.engine import RETRIEVAL_K
from civica.graph.planner import PROFILE_KEY
from civica.memory.records import LearnerProfile
from civica.memory.writer import put
from civica.ui.dev_panel import GROUNDING, LEARNER_STATE, CODE_PATH, TOKEN_COUNTS, TOOLS_CALLED, TURN
from civica.ui.turns import TurnRunner, coach_config
from tests.support.assessment import FakeChatModel
from tests.support.llm import CONTENT_USAGE, EXPLANATION_TAG, PLANNER_TAG, PLANNER_USAGE
from tests.support.coach import (
    COACH_LOGGER,
    DEV_PANEL_TITLE,
    FAILED_TEXT,
    FALLBACK_NODE,
    FALLBACK_TEXT,
    FINAL_MESSAGE_TEXT,
    FOCUS,
    LESSON_TEXT,
    QUIZ_AFTER_LESSON_CALL,
    QUIZ_ONLY_CALL,
    REPLY,
    UNFINISHED_TEXT,
    dev_panels,
    held_turn_is_shown,
    marked,
    panel_beside,
    panel_text,
    panel_texts,
    question_texts,
    script_failure,
    script_fallback,
    script_lesson,
    script_lesson_with_quiz,
    script_quiz_only,
    script_reply,
    see,
    see_dev_panel,
    submit,
    wait_for_turn_end,
)
from tests.support.memory import HALF_MASTERY, seed_mastery, seed_mistake
from tests.support.planner import (
    DEFAULT_THEME,
    GENERATE_QUIZ,
    GOAL,
    PLANNER_FAILURE_TEXT,
    READ_STATE,
    TEACH,
    ControlledPlannerModel,
    PlannerFakes,
)
from tests.support.ui import DEV_ROW_MARKER, LESSON_MARKER, QUIZ_MARKER, REPLY_MARKER
from tests.support.usage import counts_of, sum_counts, token_line

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LINE_LABELS = (CODE_PATH, TOOLS_CALLED, GROUNDING, TURN, TOKEN_COUNTS, LEARNER_STATE)
_LESSON_QUESTION = question_texts(QUIZ_AFTER_LESSON_CALL, 1)[0]
_QUIZ_QUESTION = question_texts(QUIZ_ONLY_CALL, 1)[0]
_OTHER_GOAL = "sample-goal-set-after-the-turn"
# The type of the error ControlledPlannerModel raises at `fail_at_call`.
_ERROR_TYPE = RuntimeError.__name__


@pytest.fixture()
def dev_mode() -> bool:
    """Every test in this module runs the app in developer mode."""
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _finish_turn(user: User, turn_runner: TurnRunner, learner_id: UserId, *shown: str) -> None:
    """Send the prompt, then wait for the turn's content and a developer panel to be drawn."""
    submit(user)
    wait_for_turn_end(turn_runner, learner_id)
    await see(user, *shown)
    await see_dev_panel(user)


def _two_decimals(number: float) -> str:
    return f"{number:.2f}"


# ---------------------------------------------------------------------------
# One panel per drawn response
# ---------------------------------------------------------------------------


async def test_lesson_with_a_comprehension_check_shows_a_labelled_panel_beside_each(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson_with_quiz(planner_model, content_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id, LESSON_TEXT, _LESSON_QUESTION)

    lesson_panel = panel_beside(signed_in_user, LESSON_MARKER)
    quiz_panel = panel_beside(signed_in_user, QUIZ_MARKER)
    assert dev_panels(signed_in_user) == [lesson_panel, quiz_panel]
    assert len(marked(signed_in_user, DEV_ROW_MARKER)) == 2
    for panel in (lesson_panel, quiz_panel):
        assert panel_text(panel).count(DEV_PANEL_TITLE) == 1
    assert TEACH in panel_text(lesson_panel)
    assert GENERATE_QUIZ in panel_text(quiz_panel)


async def test_lesson_turn_shows_one_panel_and_its_hidden_reply_gets_none(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id, LESSON_TEXT)

    assert dev_panels(signed_in_user) == [panel_beside(signed_in_user, LESSON_MARKER)]
    assert marked(signed_in_user, REPLY_MARKER) == []
    await signed_in_user.should_not_see(REPLY)


async def test_quiz_only_turn_shows_one_panel_beside_the_quiz_card(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_quiz_only(planner_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id, _QUIZ_QUESTION)

    quiz_panel = panel_beside(signed_in_user, QUIZ_MARKER)
    assert dev_panels(signed_in_user) == [quiz_panel]
    assert GENERATE_QUIZ in panel_text(quiz_panel)
    assert marked(signed_in_user, REPLY_MARKER) == []


async def test_fallback_turn_shows_one_panel_naming_the_fallback_node(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_fallback(planner_model, content_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id, FALLBACK_TEXT)

    lesson_panel = panel_beside(signed_in_user, LESSON_MARKER)
    assert dev_panels(signed_in_user) == [lesson_panel]
    text = panel_text(lesson_panel)
    assert FALLBACK_NODE in text
    assert "fallback=yes" in text
    assert marked(signed_in_user, REPLY_MARKER) == []


async def test_reply_only_turn_shows_one_panel_beside_the_reply(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_reply(planner_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id, REPLY)

    reply_panel = panel_beside(signed_in_user, REPLY_MARKER)
    assert dev_panels(signed_in_user) == [reply_panel]
    text = panel_text(reply_panel)
    assert DEV_PANEL_TITLE in text
    assert FINAL_MESSAGE_TEXT in text


# ---------------------------------------------------------------------------
# What a panel shows
# ---------------------------------------------------------------------------


async def test_lesson_panel_shows_the_teach_call_its_grounding_the_thread_and_the_learner_state(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    put(learner_id, PROFILE_KEY, LearnerProfile(goal=GOAL), planner_fakes.store)
    seed_mastery(planner_fakes.store, learner_id, {DEFAULT_THEME: HALF_MASTERY})
    mistake = seed_mistake(planner_fakes.store, learner_id, DEFAULT_THEME)
    script_lesson(planner_model, content_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id, LESSON_TEXT)

    text = panel_text(panel_beside(signed_in_user, LESSON_MARKER))
    for label in _LINE_LABELS:
        assert label in text
    for shown in (READ_STATE, TEACH):
        assert shown in text
    for argument in ("focus=", FOCUS):
        assert argument not in text
    passages = planner_fakes.retriever.chunks(DEFAULT_THEME)
    assert f"chunks retrieved: {len(passages)} (max {RETRIEVAL_K})" in text
    for passage in passages:
        assert _two_decimals(passage.similarity) in text
    assert str(coach_config(learner_id)["configurable"]["thread_id"]) in text
    assert f"goal={GOAL}" in text
    assert f"{DEFAULT_THEME.slug}={_two_decimals(HALF_MASTERY)}" in text
    assert f"{mistake.theme.slug}/{mistake.question_id}" in text


async def test_learner_state_is_read_when_the_panel_is_drawn(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)
    await _finish_turn(signed_in_user, turn_runner, learner_id, LESSON_TEXT)
    put(learner_id, PROFILE_KEY, LearnerProfile(goal=_OTHER_GOAL), planner_fakes.store)

    await signed_in_user.open("/")

    await see(signed_in_user, f"goal={_OTHER_GOAL}")


async def test_token_counts_show_the_total_and_each_caller_and_they_survive_a_reload(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id, LESSON_TEXT)

    assert content_model.invocation_count == 1
    planner = counts_of(PLANNER_USAGE, len(planner_model.script))
    explanation = counts_of(CONTENT_USAGE)
    token_lines = (
        token_line("total", sum_counts(planner, explanation)),
        token_line(EXPLANATION_TAG, explanation),
        token_line(PLANNER_TAG, planner),
    )
    shown = panel_text(panel_beside(signed_in_user, LESSON_MARKER))
    assert TOKEN_COUNTS in shown
    for line in token_lines:
        assert line in shown

    await signed_in_user.open("/")

    await see_dev_panel(signed_in_user)
    reloaded = panel_text(panel_beside(signed_in_user, LESSON_MARKER))
    for line in token_lines:
        assert line in reloaded


# ---------------------------------------------------------------------------
# Returning to a turn
# ---------------------------------------------------------------------------


async def test_panels_are_the_same_after_a_reload(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson_with_quiz(planner_model, content_model)
    await _finish_turn(signed_in_user, turn_runner, learner_id, LESSON_TEXT, _LESSON_QUESTION)
    before_reload = panel_texts(signed_in_user)

    await signed_in_user.open("/")

    await see_dev_panel(signed_in_user)
    assert len(before_reload) == 2
    assert panel_texts(signed_in_user) == before_reload
    assert dev_panels(signed_in_user) == [
        panel_beside(signed_in_user, LESSON_MARKER),
        panel_beside(signed_in_user, QUIZ_MARKER),
    ]


async def test_panels_are_the_same_in_a_second_browser(
    signed_in_user: User,
    second_browser: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    # Held at its last planner call, so the second browser is following the turn when it ends.
    script_lesson_with_quiz(planner_model, content_model)
    planner_model.hold_at_call = len(planner_model.script) - 1
    submit(signed_in_user)
    await held_turn_is_shown(second_browser, planner_model)

    planner_model.release()
    wait_for_turn_end(turn_runner, learner_id)
    for browser in (signed_in_user, second_browser):
        await see(browser, LESSON_TEXT, _LESSON_QUESTION)
        await see_dev_panel(browser)

    first = panel_texts(signed_in_user)
    assert len(first) == 2
    assert panel_texts(second_browser) == first


# ---------------------------------------------------------------------------
# Failed turns
# ---------------------------------------------------------------------------


async def test_failed_turn_shows_the_error_type_in_a_panel(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_failure(planner_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id, FAILED_TEXT)

    texts = panel_texts(signed_in_user)
    assert len(texts) == 1
    assert DEV_PANEL_TITLE in texts[0]
    assert _ERROR_TYPE in texts[0]
    # The panel names the kind of error only; its message stays in the server log.
    assert PLANNER_FAILURE_TEXT not in texts[0]


async def test_reload_after_a_failed_turn_shows_the_notice_with_no_panel(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_failure(planner_model)
    await _finish_turn(signed_in_user, turn_runner, learner_id, FAILED_TEXT)

    await signed_in_user.open("/")

    await see(signed_in_user, UNFINISHED_TEXT)
    assert dev_panels(signed_in_user) == []
    await signed_in_user.should_not_see(content=_ERROR_TYPE)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


async def test_drawing_panels_logs_no_tool_name_or_tool_argument(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
) -> None:
    # Tool arguments often repeat the learner's words, so they belong in the panel only (logging convention).
    script_lesson(planner_model, content_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id, LESSON_TEXT)
    await signed_in_user.open("/")
    await see_dev_panel(signed_in_user)

    messages = [record.getMessage() for record in app_logs.records if record.name == COACH_LOGGER]
    assert messages, "the coach page wrote no log lines, so the scan proves nothing"
    for tool_text in (READ_STATE, TEACH, FOCUS, DEFAULT_THEME.slug):
        assert not [message for message in messages if tool_text in message]
