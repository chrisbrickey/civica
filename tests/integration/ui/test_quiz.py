"""Integration tests for asking the coach for a quiz and checking answers as practice."""

from string import ascii_uppercase

import pytest
from nicegui import ui
from nicegui.testing import User

from civica.domain.user import UserId
from civica.ui.pages.coach import CORRECT_TEXTS, MISSED_TEXT
from civica.ui.turns import TurnRunner
from tests.support.assessment import (
    OPTION_COUNT,
    SAMPLE_CORRECT_INDEX,
    SAMPLE_OPTIONS,
    FakeChatModel,
    generated_item_explanation,
)
from tests.support.coach import (
    COACH_LOGGER,
    FINAL_CALL,
    LESSON_TEXT,
    PRACTICE_LABEL,
    PROMPT,
    QUIZ_AFTER_LESSON_CALL,
    QUIZ_ME_TEXT,
    QUIZ_ONLY_CALL,
    SECOND_PROMPT,
    TEACH_ME_TEXT,
    check_buttons,
    choose_option,
    click_check,
    click_marked,
    coach_lines,
    held_turn_is_shown,
    labelled,
    prompt_is_open,
    question_id_of,
    question_texts,
    quiz_radios,
    quiz_text,
    result_text,
    script_lesson,
    script_lesson_with_quiz,
    script_quiz_only,
    script_quiz_then_lesson,
    see,
    submit,
    the,
    wait_for_turn_end,
    wait_until,
)
from tests.support.logs import events, only
from tests.support.planner import ControlledPlannerModel, PlannerFakes
from tests.support.ui import INTRO_OPEN_MARKER, LOGOUT_MARKER, QUIZ_MARKER, QUIZ_ME_MARKER, TEACH_ME_MARKER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THATS_CORRECT = "That's correct!"
_QUICK_ACTIONS = (TEACH_ME_MARKER, QUIZ_ME_MARKER)

_QUIZ_QUESTION = question_texts(QUIZ_ONLY_CALL, 1)[0]
_QUIZ_EXPLANATION = generated_item_explanation(QUIZ_ONLY_CALL, 0)
_TWO_QUESTIONS = 2

_WRONG_INDEX = (SAMPLE_CORRECT_INDEX + 1) % OPTION_COUNT
_CORRECT_LETTER = ascii_uppercase[SAMPLE_CORRECT_INDEX]
_CORRECT_OPTION = SAMPLE_OPTIONS[SAMPLE_CORRECT_INDEX]

# Log line events and their fixed field values.
_TURN_REQUEST = "turn_request"
_PRACTICE_CHECK = "practice_check"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _finish_quiz_me(user: User, turn_runner: TurnRunner, learner_id: UserId, question_count: int = 1) -> None:
    """Press Quiz me and wait until its questions are on the page and the prompt is open again."""
    click_marked(user, QUIZ_ME_MARKER)
    wait_for_turn_end(turn_runner, learner_id)
    await prompt_is_open(user)
    await see(user, *question_texts(QUIZ_ONLY_CALL, question_count))


async def _check(user: User, question: int, option: int) -> None:
    """Select an option, press Check once it is enabled, and wait for that question to lock."""
    choose_option(user, question, option)
    await wait_until(lambda: check_buttons(user)[question].enabled)
    click_check(user, question)
    await wait_until(lambda: not check_buttons(user)[question].enabled)


def _says_correct(text: str) -> bool:
    return any(verdict in text for verdict in CORRECT_TEXTS)


def _quick_action_enabled(user: User, marker: str) -> bool:
    button = the(user, marker)
    assert isinstance(button, ui.button)
    return bool(button.enabled)


# ---------------------------------------------------------------------------
# Quick actions
# ---------------------------------------------------------------------------


async def test_quick_actions_sit_in_the_header_between_the_intro_button_and_log_out(signed_in_user: User) -> None:
    header = next(iter(signed_in_user.find(kind=ui.header).elements))
    header_order = list(header.descendants())
    markers = (INTRO_OPEN_MARKER, *_QUICK_ACTIONS, LOGOUT_MARKER)

    for marker in markers:
        assert the(signed_in_user, marker) in header_order, f"{marker!r} should sit in the header"
    positions = [header_order.index(the(signed_in_user, marker)) for marker in markers]
    assert positions == sorted(positions), f"header order should be {markers}"


@pytest.mark.parametrize(
    ("marker", "text"),
    [(TEACH_ME_MARKER, TEACH_ME_TEXT), (QUIZ_ME_MARKER, QUIZ_ME_TEXT)],
    ids=["teach-me", "quiz-me"],
)
async def test_quick_action_has_a_tooltip_naming_it(signed_in_user: User, marker: str, text: str) -> None:
    # On narrow screens only the icon shows, so the tooltip carries the name.
    tooltips = [child for child in the(signed_in_user, marker).descendants() if isinstance(child, ui.tooltip)]

    assert [tooltip.text for tooltip in tooltips] == [text]


@pytest.mark.parametrize(
    ("marker", "text"),
    [(TEACH_ME_MARKER, TEACH_ME_TEXT), (QUIZ_ME_MARKER, QUIZ_ME_TEXT)],
    ids=["teach-me", "quiz-me"],
)
async def test_quick_action_sends_its_text_as_the_learner_message(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    marker: str,
    text: str,
) -> None:
    script_lesson(planner_model, content_model)

    click_marked(signed_in_user, marker)
    wait_for_turn_end(turn_runner, learner_id)

    await see(signed_in_user, text, LESSON_TEXT)
    assert set(planner_model.human_texts()) == {text}


async def test_quiz_me_renders_the_questions_with_no_option_selected(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    questions = question_texts(QUIZ_ONLY_CALL, _TWO_QUESTIONS)
    script_quiz_only(planner_model, _TWO_QUESTIONS)

    await _finish_quiz_me(signed_in_user, turn_runner, learner_id, _TWO_QUESTIONS)

    text = quiz_text(signed_in_user)
    for question in questions:
        assert question in text
    for option in SAMPLE_OPTIONS:
        assert option in text
    radios = quiz_radios(signed_in_user)
    assert len(radios) == len(questions)
    assert all(radio.value is None for radio in radios)


async def test_quick_actions_are_disabled_while_a_turn_runs(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)
    planner_model.hold_at_call = FINAL_CALL
    submit(signed_in_user)
    await held_turn_is_shown(signed_in_user, planner_model)

    await wait_until(lambda: not any(_quick_action_enabled(signed_in_user, marker) for marker in _QUICK_ACTIONS))
    for marker in _QUICK_ACTIONS:
        click_marked(signed_in_user, marker)
    planner_model.release()
    wait_for_turn_end(turn_runner, learner_id)

    await prompt_is_open(signed_in_user)
    await wait_until(lambda: all(_quick_action_enabled(signed_in_user, marker) for marker in _QUICK_ACTIONS))
    assert set(planner_model.human_texts()) == {PROMPT}


# ---------------------------------------------------------------------------
# Practice checks
# ---------------------------------------------------------------------------


def test_the_correct_verdicts_include_thats_correct() -> None:
    assert _THATS_CORRECT in CORRECT_TEXTS


async def test_check_is_disabled_until_an_option_is_selected(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id)

    assert not check_buttons(signed_in_user)[0].enabled

    choose_option(signed_in_user, 0, _WRONG_INDEX)

    await wait_until(lambda: check_buttons(signed_in_user)[0].enabled)


async def test_before_a_check_the_card_shows_no_explanation_or_verdict(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id)
    choose_option(signed_in_user, 0, SAMPLE_CORRECT_INDEX)

    text = quiz_text(signed_in_user)
    assert _QUIZ_EXPLANATION not in text
    assert "correct" not in text.lower()
    assert result_text(signed_in_user) == ""


async def test_the_card_labels_checks_as_practice(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id)

    assert PRACTICE_LABEL in quiz_text(signed_in_user)


async def test_checking_the_correct_option_says_it_is_correct(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id)

    await _check(signed_in_user, 0, SAMPLE_CORRECT_INDEX)

    await wait_until(lambda: _says_correct(result_text(signed_in_user)))


async def test_checking_a_wrong_option_shows_the_correct_option_and_why(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id)

    await _check(signed_in_user, 0, _WRONG_INDEX)

    await wait_until(lambda: _QUIZ_EXPLANATION in result_text(signed_in_user))
    result = result_text(signed_in_user)
    assert labelled(_CORRECT_LETTER, _CORRECT_OPTION).search(result), f"no lettered correct option in {result!r}"
    assert not _says_correct(result)


async def test_checking_a_wrong_option_opens_by_saying_it_was_missed(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id)

    await _check(signed_in_user, 0, _WRONG_INDEX)

    await wait_until(lambda: _QUIZ_EXPLANATION in result_text(signed_in_user))
    assert result_text(signed_in_user).startswith(MISSED_TEXT)


async def test_a_checked_question_locks_and_unchecked_questions_stay_usable(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model, _TWO_QUESTIONS)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id, _TWO_QUESTIONS)

    await _check(signed_in_user, 0, SAMPLE_CORRECT_INDEX)

    first, second = quiz_radios(signed_in_user)
    assert not first.enabled
    assert not check_buttons(signed_in_user)[0].enabled
    assert second.enabled
    assert second.value is None
    await _check(signed_in_user, 1, _WRONG_INDEX)
    await wait_until(lambda: generated_item_explanation(QUIZ_ONLY_CALL, 1) in result_text(signed_in_user))


async def test_a_check_records_no_answer(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    planner_fakes: PlannerFakes,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_quiz_only(planner_model, _TWO_QUESTIONS)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id, _TWO_QUESTIONS)

    await _check(signed_in_user, 0, SAMPLE_CORRECT_INDEX)
    await _check(signed_in_user, 1, _WRONG_INDEX)

    assert planner_fakes.answer_logger.calls == []


async def test_a_lesson_comprehension_question_can_be_checked(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson_with_quiz(planner_model, content_model)
    submit(signed_in_user)
    wait_for_turn_end(turn_runner, learner_id)
    await see(signed_in_user, LESSON_TEXT, *question_texts(QUIZ_AFTER_LESSON_CALL, 1))

    await _check(signed_in_user, 0, _WRONG_INDEX)

    explanation = generated_item_explanation(QUIZ_AFTER_LESSON_CALL, 0)
    await wait_until(lambda: explanation in result_text(signed_in_user))


# ---------------------------------------------------------------------------
# Returning and moving on
# ---------------------------------------------------------------------------


async def test_reload_after_a_quiz_shows_the_same_questions_with_checks_cleared(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id)
    await _check(signed_in_user, 0, _WRONG_INDEX)
    await wait_until(lambda: _QUIZ_EXPLANATION in result_text(signed_in_user))

    await signed_in_user.open("/")

    await see(signed_in_user, _QUIZ_QUESTION)
    (radio,) = quiz_radios(signed_in_user)
    assert radio.value is None
    assert radio.enabled
    assert result_text(signed_in_user) == ""
    assert _QUIZ_EXPLANATION not in quiz_text(signed_in_user)


async def test_a_lesson_after_a_quiz_replaces_the_questions(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_quiz_then_lesson(planner_model, content_model)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id)

    submit(signed_in_user, SECOND_PROMPT)
    wait_for_turn_end(turn_runner, learner_id)

    await see(signed_in_user, SECOND_PROMPT, LESSON_TEXT)
    await signed_in_user.should_not_see(content=_QUIZ_QUESTION)
    await signed_in_user.should_not_see(marker=QUIZ_MARKER)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("marker", "via"), [(TEACH_ME_MARKER, "teach_me"), (QUIZ_ME_MARKER, "quiz_me")], ids=["teach-me", "quiz-me"]
)
async def test_quick_action_logs_one_request_line_naming_it(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
    marker: str,
    via: str,
) -> None:
    script_lesson(planner_model, content_model)

    click_marked(signed_in_user, marker)
    wait_for_turn_end(turn_runner, learner_id)
    await see(signed_in_user, LESSON_TEXT)

    assert only(coach_lines(app_logs), _TURN_REQUEST) == {"user_id": str(learner_id), "via": via}


@pytest.mark.parametrize(
    ("choice", "correct"), [(SAMPLE_CORRECT_INDEX, "True"), (_WRONG_INDEX, "False")], ids=["right", "wrong"]
)
async def test_practice_check_logs_one_line_with_its_question_and_result_but_no_option_text(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
    choice: int,
    correct: str,
) -> None:
    script_quiz_only(planner_model)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id)

    await _check(signed_in_user, 0, choice)

    assert only(coach_lines(app_logs), _PRACTICE_CHECK) == {
        "user_id": str(learner_id),
        "question_id": question_id_of(_QUIZ_QUESTION),
        "correct": correct,
    }
    coach_messages = [record.getMessage() for record in app_logs.records if record.name == COACH_LOGGER]
    assert not [message for message in coach_messages if SAMPLE_OPTIONS[choice] in message]


async def test_a_quiz_turn_and_practice_checks_log_no_question_option_or_explanation_text(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
) -> None:
    # Extends the lesson turn's no-learner-text test in test_lessons.py to quizzes and checks.
    script_quiz_only(planner_model, _TWO_QUESTIONS)
    await _finish_quiz_me(signed_in_user, turn_runner, learner_id, _TWO_QUESTIONS)
    await _check(signed_in_user, 0, SAMPLE_CORRECT_INDEX)
    await _check(signed_in_user, 1, _WRONG_INDEX)
    await signed_in_user.open("/")
    await see(signed_in_user, _QUIZ_QUESTION)

    assert _PRACTICE_CHECK in events(coach_lines(app_logs)), "no check was logged, so the scan proves nothing"
    explanations = [generated_item_explanation(QUIZ_ONLY_CALL, index) for index in range(_TWO_QUESTIONS)]
    for learner_text in (*question_texts(QUIZ_ONLY_CALL, _TWO_QUESTIONS), *SAMPLE_OPTIONS, *explanations):
        assert learner_text not in app_logs.text
