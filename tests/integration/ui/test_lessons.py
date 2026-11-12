"""Integration tests for asking the coach for a lesson."""

import logging
from collections.abc import Callable
from string import ascii_uppercase

import pytest
from nicegui import ui
from nicegui.testing import User

from civica.domain.user import UserId
from civica.ui.pages.coach import SPINNER_VERBS
from civica.ui.turns import TurnRunner
from tests.support.assessment import SAMPLE_OPTIONS, FakeChatModel
from tests.support.coach import (
    CALLS_PER_LESSON,
    COACH_LOGGER,
    FAILED_TEXT,
    FALLBACK_TEXT,
    FINAL_CALL,
    LESSON_TEXT,
    PROMPT,
    QUIZ_AFTER_LESSON_CALL,
    QUIZ_ONLY_CALL,
    READ_STATE,
    REPLY,
    SECOND_LESSON_TEXT,
    SECOND_PROMPT,
    TEACH,
    TURN_RETRIES,
    UNFINISHED_TEXT,
    click_submit,
    coach_lines,
    labelled,
    held_turn_is_shown,
    press_enter,
    prompt_input,
    prompt_is_locked,
    prompt_is_open,
    question_texts,
    quiz_radios,
    quiz_text,
    script_failure,
    script_fallback,
    script_lesson,
    script_lesson_with_quiz,
    script_quiz_only,
    script_two_lessons,
    see,
    submit,
    the,
    type_prompt,
    wait_for_turn_end,
)
from tests.support.logs import LogLine, events, lines_from, only
from tests.support.planner import (
    DEFAULT_THEME,
    PLANNER_FAILURE_TEXT,
    ControlledPlannerModel,
    PlannerFakes,
)
from tests.support.ui import (
    CHAT_MARKER,
    LESSON_MARKER,
    QUIZ_HOOK_MARKER,
    QUIZ_MARKER,
    SPINNER_VERB_MARKER,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SOURCES_TITLE = "Sources for this lesson"
_UNTITLED_SOURCE = "Official source"
_ALREADY_RUNNING_TEXT = "A turn is already running"
_FALLBACK_NOTE = "This is a fallback lesson."
_TRACEBACK_HEADER = "Traceback (most recent call last)"
_TURNS_LOGGER = "civica.ui.turns"

# Log line events and their fixed field values.
_PAGE_OPEN = "coach_page_open"
_TURN_REQUEST = "turn_request"
_TURN_REJECTED = "turn_rejected"
_TURN_START = "turn_start"
_TURN_END = "turn_end"
_VIA_TYPED = "typed"
_ALREADY_RUNNING_REASON = "already_running"

# Distinctive, so a scan of the logs cannot match anything else by accident.
_SCANNED_LEARNER_MESSAGE = "scanned-sample-learner-message"

_QUESTION_TEXT = question_texts(QUIZ_AFTER_LESSON_CALL, 1)[0]
_QUIZ_TITLE = "Check your comprehension"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_text(user: User, marker: str) -> str:
    """Every label inside the marked area, joined so assertions do not depend on layout."""
    return "\n".join(
        element.text for element in the(user, marker).descendants() if isinstance(element, ui.label)
    )


async def _see_lesson(user: User, turn_runner: TurnRunner, learner_id: UserId) -> None:
    wait_for_turn_end(turn_runner, learner_id)
    await see(user, LESSON_TEXT)


def _turn_lines(caplog: pytest.LogCaptureFixture) -> list[LogLine]:
    return lines_from(caplog.records, _TURNS_LOGGER)


def _page_open_line(user_id: UserId, found: str) -> str:
    return f"{_PAGE_OPEN} user_id={user_id} found={found}"


# ---------------------------------------------------------------------------
# Asking for a lesson
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("send", [click_submit, press_enter], ids=["submit-button", "enter-key"])
async def test_submitting_a_prompt_renders_the_lesson_once_without_repeating_it_in_a_reply(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    send: Callable[[User], None],
) -> None:
    script_lesson(planner_model, content_model)

    type_prompt(signed_in_user, PROMPT)
    send(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)

    await see(signed_in_user, PROMPT)
    await signed_in_user.should_not_see(content=REPLY)
    assert len(signed_in_user.find(content=LESSON_TEXT).elements) == 1


async def test_the_prompt_input_clears_after_sending(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)

    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)

    assert not prompt_input(signed_in_user).value


async def test_learner_sees_no_slugs_scores_or_tool_names(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)

    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)

    for tool_name in (READ_STATE, TEACH):
        await signed_in_user.should_not_see(content=tool_name)
    for passage in planner_fakes.retriever.chunks(DEFAULT_THEME):
        await signed_in_user.should_not_see(content=passage.chunk.page_slug)
        await signed_in_user.should_not_see(content=str(passage.similarity))


async def test_sources_sit_inside_the_lesson_as_one_closed_section_of_closed_passages(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)

    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)

    lesson = the(signed_in_user, LESSON_MARKER)
    expansions = [element for element in lesson.descendants() if isinstance(element, ui.expansion)]
    sections = [expansion for expansion in expansions if _SOURCES_TITLE in expansion.text]
    assert len(sections) == 1
    inner = [element for element in sections[0].descendants() if isinstance(element, ui.expansion)]
    passages = planner_fakes.retriever.chunks(DEFAULT_THEME)
    assert len(inner) == len(passages)
    assert not any(expansion.value for expansion in expansions)
    for passage in passages:
        assert passage.chunk.text in _label_text(signed_in_user, LESSON_MARKER)


async def test_sources_from_one_page_show_that_page_title_once(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    # The fake passages carry no title, so they all come from one untitled page.
    script_lesson(planner_model, content_model)

    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)

    assert len(planner_fakes.retriever.chunks(DEFAULT_THEME)) > 1
    assert _label_text(signed_in_user, LESSON_MARKER).splitlines().count(_UNTITLED_SOURCE) == 1


async def test_sources_footer_heading_shows_no_count(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)

    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)

    lesson = the(signed_in_user, LESSON_MARKER)
    headings = [element.text for element in lesson.descendants() if isinstance(element, ui.expansion)]
    assert _SOURCES_TITLE in headings


async def test_sources_survive_a_reload(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)
    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)

    await signed_in_user.open("/")

    await see(signed_in_user, _SOURCES_TITLE)
    for passage in planner_fakes.retriever.chunks(DEFAULT_THEME):
        assert passage.chunk.text in _label_text(signed_in_user, LESSON_MARKER)


async def test_a_running_turn_shows_a_spinner_verb_from_the_fixed_list(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    script_lesson(planner_model, content_model)
    planner_model.hold_at_call = FINAL_CALL

    submit(signed_in_user)
    await held_turn_is_shown(signed_in_user, planner_model)

    await signed_in_user.should_see(kind=ui.spinner, retries=TURN_RETRIES)
    verb = the(signed_in_user, SPINNER_VERB_MARKER)
    assert isinstance(verb, ui.label)
    assert verb.text in SPINNER_VERBS


def test_every_spinner_verb_is_one_word() -> None:
    # Generic single words, so no verb suggests the app consults anything beyond its study material.
    assert all(len(verb.removesuffix("…").split()) == 1 for verb in SPINNER_VERBS)


async def test_fallback_lesson_shows_the_fallback_note(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_fallback(planner_model, content_model)

    submit(signed_in_user)
    wait_for_turn_end(turn_runner, learner_id)

    await see(signed_in_user, FALLBACK_TEXT, _FALLBACK_NOTE)
    assert len(signed_in_user.find(content=FALLBACK_TEXT).elements) == 1


# ---------------------------------------------------------------------------
# Quiz card
# ---------------------------------------------------------------------------


async def test_questions_show_their_options_with_none_selected(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson_with_quiz(planner_model, content_model)

    submit(signed_in_user)
    wait_for_turn_end(turn_runner, learner_id)
    await see(signed_in_user, _QUESTION_TEXT)

    for option in SAMPLE_OPTIONS:
        assert option in quiz_text(signed_in_user)
    radios = quiz_radios(signed_in_user)
    assert len(radios) == 1
    assert radios[0].value is None


async def test_all_questions_sit_in_one_quiz_card_titled_with_no_question_count(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    questions = question_texts(QUIZ_AFTER_LESSON_CALL, 2)
    script_lesson_with_quiz(planner_model, content_model, len(questions))

    submit(signed_in_user)
    wait_for_turn_end(turn_runner, learner_id)
    await see(signed_in_user, *questions)

    quiz = the(signed_in_user, QUIZ_MARKER)
    assert isinstance(quiz, ui.card)
    chat = the(signed_in_user, CHAT_MARKER)
    assert [element for element in chat.descendants() if isinstance(element, ui.card)] == [quiz]
    card_text = quiz_text(signed_in_user)
    assert [line for line in card_text.splitlines() if _QUIZ_TITLE in line] == [_QUIZ_TITLE]
    for question in questions:
        assert question in card_text


async def test_quiz_questions_are_numbered_and_options_lettered(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    questions = question_texts(QUIZ_AFTER_LESSON_CALL, 2)
    script_lesson_with_quiz(planner_model, content_model, len(questions))

    submit(signed_in_user)
    wait_for_turn_end(turn_runner, learner_id)
    await see(signed_in_user, *questions)

    quiz = quiz_text(signed_in_user)
    for number, question in enumerate(questions, start=1):
        assert labelled(str(number), question).search(quiz), f"question {number} is not numbered"
    for letter, option in zip(ascii_uppercase, SAMPLE_OPTIONS):
        assert len(labelled(letter, option).findall(quiz)) == len(questions), f"{option} is not lettered {letter}"


async def test_quiz_after_a_lesson_hangs_off_it_by_a_hooked_line(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson_with_quiz(planner_model, content_model)

    submit(signed_in_user)
    wait_for_turn_end(turn_runner, learner_id)
    await see(signed_in_user, _QUESTION_TEXT)

    await signed_in_user.should_see(marker=QUIZ_HOOK_MARKER)


async def test_quiz_only_turn_draws_the_same_card_with_no_hooked_line(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    question = question_texts(QUIZ_ONLY_CALL, 1)[0]
    script_quiz_only(planner_model)

    submit(signed_in_user)
    wait_for_turn_end(turn_runner, learner_id)
    await see(signed_in_user, question)

    assert _QUIZ_TITLE in quiz_text(signed_in_user).splitlines()
    await signed_in_user.should_not_see(marker=QUIZ_HOOK_MARKER)
    await signed_in_user.should_not_see(content=REPLY)


# ---------------------------------------------------------------------------
# Returning to a turn
# ---------------------------------------------------------------------------


async def test_reload_shows_the_same_turn_from_the_checkpointer(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson_with_quiz(planner_model, content_model)
    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)
    calls_before_reload = planner_model.invocation_count

    await signed_in_user.open("/")

    await see(signed_in_user, PROMPT, LESSON_TEXT, _QUESTION_TEXT)
    await signed_in_user.should_not_see(content=REPLY)
    assert planner_model.invocation_count == calls_before_reload


async def test_reload_during_a_turn_shows_it_running_then_finished(
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

    await signed_in_user.open("/")

    await see(signed_in_user, PROMPT)
    await prompt_is_locked(signed_in_user)

    planner_model.release()

    await _see_lesson(signed_in_user, turn_runner, learner_id)
    await prompt_is_open(signed_in_user)


async def test_closed_page_does_not_stop_the_turn(
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
    assert signed_in_user.client is not None

    signed_in_user.client.delete()
    planner_model.release()
    wait_for_turn_end(turn_runner, learner_id)

    await signed_in_user.open("/")
    await see(signed_in_user, LESSON_TEXT)


# ---------------------------------------------------------------------------
# Starting another turn
# ---------------------------------------------------------------------------


async def test_a_second_turn_replaces_the_first(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_two_lessons(planner_model, content_model)
    planner_model.hold_at_call = CALLS_PER_LESSON + FINAL_CALL
    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)
    await prompt_is_open(signed_in_user)

    submit(signed_in_user, SECOND_PROMPT)

    # Gone as soon as the new turn is on screen, not only when it finishes.
    await held_turn_is_shown(signed_in_user, planner_model)
    await see(signed_in_user, SECOND_PROMPT)
    await signed_in_user.should_not_see(content=LESSON_TEXT)

    planner_model.release()
    wait_for_turn_end(turn_runner, learner_id)

    await see(signed_in_user, SECOND_LESSON_TEXT)
    await signed_in_user.should_not_see(content=LESSON_TEXT)


async def test_prompt_is_disabled_while_a_turn_runs_so_turns_cannot_overlap(
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

    submit(signed_in_user, SECOND_PROMPT)
    press_enter(signed_in_user)
    planner_model.release()

    await _see_lesson(signed_in_user, turn_runner, learner_id)
    await prompt_is_open(signed_in_user)
    assert set(planner_model.human_texts()) == {PROMPT}


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "spaces"])
async def test_blank_prompt_starts_nothing(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
    blank: str,
) -> None:
    # Held at the first call, so a turn that did start would still be live below.
    planner_model.hold_at_call = 0
    app_logs.clear()

    submit(signed_in_user, blank)

    assert turn_runner.active(learner_id) is None
    assert planner_model.invocation_count == 0
    assert prompt_input(signed_in_user).enabled
    assert coach_lines(app_logs) == []


async def test_second_browser_submitting_during_a_turn_is_told_logs_it_and_starts_nothing(
    signed_in_user: User,
    second_browser: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
) -> None:
    script_lesson(planner_model, content_model)
    planner_model.hold_at_call = FINAL_CALL
    type_prompt(second_browser, SECOND_PROMPT)

    # No await between the two: the second page's timer has not yet seen the turn start.
    submit(signed_in_user)
    click_submit(second_browser)

    await see(second_browser, _ALREADY_RUNNING_TEXT)

    # That page then follows the running turn through to its result.
    await held_turn_is_shown(second_browser, planner_model)
    planner_model.release()
    await _see_lesson(second_browser, turn_runner, learner_id)
    assert set(planner_model.human_texts()) == {PROMPT}
    rejected = only(coach_lines(app_logs), _TURN_REJECTED)
    assert rejected == {"user_id": str(learner_id), "reason": _ALREADY_RUNNING_REASON}


# ---------------------------------------------------------------------------
# Failed turns
# ---------------------------------------------------------------------------


async def test_graph_that_raises_partway_shows_the_error_logs_it_and_reopens_the_prompt(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_failure(planner_model)

    with caplog.at_level(logging.ERROR):
        submit(signed_in_user)
        wait_for_turn_end(turn_runner, learner_id)
        await see(signed_in_user, FAILED_TEXT, PROMPT)

    await prompt_is_open(signed_in_user)
    assert _TRACEBACK_HEADER in caplog.text
    assert PLANNER_FAILURE_TEXT in caplog.text


async def test_reload_after_a_failed_turn_shows_the_notice_and_not_the_previous_lesson(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_two_lessons(planner_model, content_model)
    planner_model.fail_at_call = CALLS_PER_LESSON + 1
    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)
    await prompt_is_open(signed_in_user)
    submit(signed_in_user, SECOND_PROMPT)
    wait_for_turn_end(turn_runner, learner_id)
    await see(signed_in_user, FAILED_TEXT)

    await signed_in_user.open("/")

    await see(signed_in_user, UNFINISHED_TEXT)
    await signed_in_user.should_not_see(content=LESSON_TEXT)
    await signed_in_user.should_not_see(content=REPLY)
    await prompt_is_open(signed_in_user)


# ---------------------------------------------------------------------------
# Page-open logging
# ---------------------------------------------------------------------------


async def _open_and_read_log(user: User, caplog: pytest.LogCaptureFixture) -> list[str]:
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=COACH_LOGGER):
        await user.open("/")
    return [record.getMessage() for record in caplog.records if record.name == COACH_LOGGER]


async def test_opening_the_page_with_no_turn_logs_that(
    signed_in_user: User, learner_id: UserId, caplog: pytest.LogCaptureFixture
) -> None:
    lines = await _open_and_read_log(signed_in_user, caplog)

    assert lines == [_page_open_line(learner_id, "no_turn")]


async def test_opening_the_page_on_a_finished_turn_logs_that(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_lesson(planner_model, content_model)
    submit(signed_in_user)
    wait_for_turn_end(turn_runner, learner_id)

    lines = await _open_and_read_log(signed_in_user, caplog)

    assert lines == [_page_open_line(learner_id, "saved_turn")]


async def test_opening_the_page_during_a_turn_logs_that(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    learner_id: UserId,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_lesson(planner_model, content_model)
    planner_model.hold_at_call = FINAL_CALL
    submit(signed_in_user)
    await held_turn_is_shown(signed_in_user, planner_model)

    lines = await _open_and_read_log(signed_in_user, caplog)
    planner_model.release()

    assert lines == [_page_open_line(learner_id, "running_turn")]


async def test_opening_the_page_on_an_unfinished_turn_logs_that(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_failure(planner_model)
    submit(signed_in_user)
    wait_for_turn_end(turn_runner, learner_id)

    lines = await _open_and_read_log(signed_in_user, caplog)

    assert lines == [_page_open_line(learner_id, "unfinished_turn")]


# ---------------------------------------------------------------------------
# Turn logging
# ---------------------------------------------------------------------------


async def test_typed_prompt_logs_one_request_line(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
) -> None:
    script_lesson(planner_model, content_model)

    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)

    request = only(coach_lines(app_logs), _TURN_REQUEST)
    assert request == {"user_id": str(learner_id), "via": _VIA_TYPED}


async def test_turn_lines_carry_the_turn_id_of_its_trace_rows(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
) -> None:
    script_lesson(planner_model, content_model)

    submit(signed_in_user)
    await _see_lesson(signed_in_user, turn_runner, learner_id)

    trace_turn_ids = {str(step.turn_id) for step in planner_fakes.trace.steps}
    assert len(trace_turn_ids) == 1
    lines = _turn_lines(app_logs)
    assert {_TURN_START, _TURN_END} <= set(events(lines))
    assert {fields["turn_id"] for _event, fields in lines} == trace_turn_ids


async def test_a_turn_logs_no_learner_message_lesson_text_or_passage(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
) -> None:
    # Later steps extend this test whenever they log a new kind of event (logging convention).
    script_lesson(planner_model, content_model)

    submit(signed_in_user, _SCANNED_LEARNER_MESSAGE)
    await _see_lesson(signed_in_user, turn_runner, learner_id)
    await signed_in_user.open("/")
    await see(signed_in_user, LESSON_TEXT)

    assert _turn_lines(app_logs), "the turn wrote no log lines, so the scan proves nothing"
    passages = [passage.chunk.text for passage in planner_fakes.retriever.chunks(DEFAULT_THEME)]
    for learner_text in (_SCANNED_LEARNER_MESSAGE, LESSON_TEXT, *passages):
        assert learner_text not in app_logs.text
