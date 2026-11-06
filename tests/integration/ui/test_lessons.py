"""
Integration tests for asking the coach for a lesson.

The app runs the real graph from build_graph(...) with only external services faked, so wiring
bugs such as a missing runtime context or unstreamed subgraph steps fail here.
NiceGUI's User fixture simulates the browser in Python (no real browser).

A held planner call keeps a turn running for as long as a test needs. Waits are bounded polls.
"""

import asyncio
import logging
import re
from collections.abc import Callable
from string import ascii_uppercase

import pytest
from langchain_core.messages import AIMessage
from nicegui import ui
from nicegui.elements.mixins.text_element import TextElement
from nicegui.testing import User

from civica.domain.user import UserId
from civica.ui.turns import TurnRunner
from tests.support.assessment import SAMPLE_OPTIONS, FakeChatModel, generated_item_text
from tests.support.planner import (
    DEFAULT_THEME,
    PLANNER_FAILURE_TEXT,
    ControlledPlannerModel,
    PlannerFakes,
    final_message,
    tool_call,
)
from tests.support.ui import (
    CHAT_MARKER,
    PROMPT_INPUT_MARKER,
    QUIZ_MARKER,
    SUBMIT_MARKER,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPT = "sample-prompt-001"
_SECOND_PROMPT = "sample-prompt-002"
_LESSON_TEXT = "sample-lesson-text-001"
_SECOND_LESSON_TEXT = "sample-lesson-text-002"
_REPLY = "sample-final-reply-001"
_SECOND_REPLY = "sample-final-reply-002"
_FOCUS = "sample-focus"

_READ_STATE = "read_learner_state"
_TEACH = "teach"
_GENERATE_QUIZ = "generate_quiz"

_SOURCES_TITLE = "Sources for this lesson"
_UNTITLED_SOURCE = "Official source"

_FAILED_TEXT = "Something went wrong. Please try again."
_UNFINISHED_TEXT = "Your last request didn't finish. Please try again."
_ALREADY_RUNNING_TEXT = "A turn is already running"
_TRACEBACK_HEADER = "Traceback (most recent call last)"

# The first generate_quiz call follows the one teach call on the content model.
_QUIZ_CALL = 1
_QUESTION_TEXT = generated_item_text(_QUIZ_CALL, 0)
_QUIZ_TITLE = "Check your comprehension"
_ANSWER_CONTROLS: tuple[type[ui.element], ...] = (ui.button, ui.radio, ui.checkbox, ui.toggle)

# should_see polls every 0.1 seconds, so a turn has up to 5 seconds to reach the page.
_TURN_RETRIES = 50
_POLL_SECONDS = 0.02
_POLL_LIMIT = 250

# How the planner calls of `_lesson_script` line up: hold the last one to pause mid-turn.
_FINAL_CALL = 2
_CALLS_PER_LESSON = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lesson_script(reply: str = _REPLY, call_suffix: str = "001") -> tuple[AIMessage, ...]:
    return (
        tool_call(_READ_STATE, f"call-read-{call_suffix}"),
        tool_call(_TEACH, f"call-teach-{call_suffix}", theme=DEFAULT_THEME.slug, focus=_FOCUS),
        final_message(reply),
    )


def _script_lesson(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    planner_model.script = _lesson_script()
    content_model.replies = (_LESSON_TEXT,)


def _script_lesson_with_question(
    planner_model: ControlledPlannerModel, content_model: FakeChatModel, question_count: int = 1
) -> None:
    planner_model.script = (
        tool_call(_TEACH, "call-teach", theme=DEFAULT_THEME.slug, focus=_FOCUS),
        tool_call(_GENERATE_QUIZ, "call-quiz", theme=DEFAULT_THEME.slug, n=question_count),
        final_message(_REPLY),
    )
    content_model.replies = (_LESSON_TEXT,)


def _script_two_lessons(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    planner_model.script = (*_lesson_script(), *_lesson_script(_SECOND_REPLY, "002"))
    content_model.replies = (_LESSON_TEXT, _SECOND_LESSON_TEXT)


def _type_prompt(user: User, prompt: str) -> None:
    user.find(marker=PROMPT_INPUT_MARKER).type(prompt)


def _click_submit(user: User) -> None:
    user.find(marker=SUBMIT_MARKER).click()


def _press_enter(user: User) -> None:
    user.find(marker=PROMPT_INPUT_MARKER).trigger("keydown.enter")


def _submit(user: User, prompt: str = _PROMPT) -> None:
    _type_prompt(user, prompt)
    _click_submit(user)


def _prompt_input(user: User) -> ui.input:
    return next(iter(user.find(kind=ui.input, marker=PROMPT_INPUT_MARKER).elements))


def _submit_button(user: User) -> ui.button:
    return next(iter(user.find(kind=ui.button, marker=SUBMIT_MARKER).elements))


def _label_text(user: User, marker: str) -> str:
    """Every label inside the marked area, joined so assertions do not depend on layout."""
    area = next(iter(user.find(marker=marker).elements))
    return "\n".join(element.text for element in area.descendants() if isinstance(element, ui.label))


def _question_texts(question_count: int) -> tuple[str, ...]:
    return tuple(generated_item_text(_QUIZ_CALL, index) for index in range(question_count))


def _quiz_card(user: User) -> ui.element:
    """The one element marked as the quiz panel."""
    marked = list(user.find(marker=QUIZ_MARKER).elements)
    assert len(marked) == 1, f"expected one element marked {QUIZ_MARKER!r}, found {len(marked)}"
    return marked[0]


def _quiz_text(user: User) -> str:
    """Every piece of text inside the quiz panel, in page order."""
    return "\n".join(
        element.text for element in _quiz_card(user).descendants() if isinstance(element, TextElement)
    )


def _labelled(marker: str, text: str) -> re.Pattern[str]:
    """A marker such as "1" or "A" directly before the text, as "1. text", "A) text", or on its own line."""
    return re.compile(rf"(?<!\w){re.escape(marker)}[.)]?\s+{re.escape(text)}")


def _lesson_message(user: User) -> ui.chat_message:
    """The chat message holding the lesson text."""
    chat = next(iter(user.find(marker=CHAT_MARKER).elements))
    return next(
        message
        for message in chat.descendants()
        if isinstance(message, ui.chat_message)
        and any(isinstance(child, ui.markdown) and child.content == _LESSON_TEXT for child in message.descendants())
    )


async def _see(user: User, *contents: str) -> None:
    """Wait until each text is on the page or in a notification, allowing for a turn in flight."""
    for content in contents:
        await user.should_see(content, retries=_TURN_RETRIES)


async def _wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(_POLL_LIMIT):
        if condition():
            return
        await asyncio.sleep(_POLL_SECONDS)
    raise AssertionError("condition not met in time")


async def _prompt_is_locked(user: User) -> None:
    await _wait_until(lambda: not _prompt_input(user).enabled and not _submit_button(user).enabled)


async def _prompt_is_open(user: User) -> None:
    await _wait_until(lambda: _prompt_input(user).enabled and _submit_button(user).enabled)


async def _turn_is_held(user: User, planner_model: ControlledPlannerModel) -> None:
    """Wait until the planner is paused at its held call and the page shows the turn running."""
    hold_at_call = planner_model.hold_at_call
    assert hold_at_call is not None
    await _wait_until(lambda: planner_model.invocation_count > hold_at_call)
    await _prompt_is_locked(user)


async def _turn_ended(turn_runner: TurnRunner, learner_id: UserId) -> None:
    await _wait_until(lambda: turn_runner.active(learner_id) is None)


# ---------------------------------------------------------------------------
# Asking for a lesson
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("send", [_click_submit, _press_enter], ids=["submit-button", "enter-key"])
async def test_submitting_a_prompt_renders_the_lesson_without_repeating_it_in_a_reply(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    send: Callable[[User], None],
) -> None:
    _script_lesson(planner_model, content_model)

    _type_prompt(signed_in_user, _PROMPT)
    send(signed_in_user)

    await _see(signed_in_user, _LESSON_TEXT, _PROMPT)
    await signed_in_user.should_not_see(content=_REPLY)


async def test_learner_sees_no_slugs_scores_or_tool_names(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
) -> None:
    _script_lesson(planner_model, content_model)

    _submit(signed_in_user)
    await _see(signed_in_user, _LESSON_TEXT, _SOURCES_TITLE)

    for tool_name in (_READ_STATE, _TEACH):
        await signed_in_user.should_not_see(content=tool_name)
    for passage in planner_fakes.retriever.chunks(DEFAULT_THEME):
        await signed_in_user.should_not_see(content=passage.chunk.page_slug)
        await signed_in_user.should_not_see(content=str(passage.similarity))


async def test_learner_sees_sources_inside_the_lesson_as_one_closed_section_of_closed_passages(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
) -> None:
    _script_lesson(planner_model, content_model)

    _submit(signed_in_user)
    await _see(signed_in_user, _LESSON_TEXT)

    lesson = _lesson_message(signed_in_user)
    expansions = [element for element in lesson.descendants() if isinstance(element, ui.expansion)]
    sections = [expansion for expansion in expansions if _SOURCES_TITLE in expansion.text]
    assert len(sections) == 1
    section = sections[0]
    inner = [element for element in section.descendants() if isinstance(element, ui.expansion)]
    passages = planner_fakes.retriever.chunks(DEFAULT_THEME)
    assert len(inner) == len(passages)
    assert not section.value
    assert not any(expansion.value for expansion in inner)
    for passage in passages:
        assert passage.chunk.text in _label_text(signed_in_user, CHAT_MARKER)


async def test_sources_from_one_page_show_that_page_once(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
) -> None:
    # The fake passages carry no title, so they all come from one untitled page.
    _script_lesson(planner_model, content_model)

    _submit(signed_in_user)
    await _see(signed_in_user, _LESSON_TEXT)

    assert len(planner_fakes.retriever.chunks(DEFAULT_THEME)) > 1
    page_labels = _label_text(signed_in_user, CHAT_MARKER).splitlines()
    assert page_labels.count(_UNTITLED_SOURCE) == 1


async def test_reloaded_turn_still_shows_its_sources(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    planner_fakes: PlannerFakes,
) -> None:
    _script_lesson(planner_model, content_model)
    _submit(signed_in_user)
    await _see(signed_in_user, _LESSON_TEXT)

    await signed_in_user.open("/")

    await _see(signed_in_user, _SOURCES_TITLE)
    for passage in planner_fakes.retriever.chunks(DEFAULT_THEME):
        assert passage.chunk.text in _label_text(signed_in_user, CHAT_MARKER)


async def test_the_prompt_input_clears_after_sending(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    _script_lesson(planner_model, content_model)

    _submit(signed_in_user)

    await _see(signed_in_user, _LESSON_TEXT)
    assert not _prompt_input(signed_in_user).value


async def test_questions_show_their_options_with_no_way_to_answer(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    _script_lesson_with_question(planner_model, content_model)

    _submit(signed_in_user)

    await _see(signed_in_user, _QUESTION_TEXT, *SAMPLE_OPTIONS)
    await signed_in_user.should_not_see(kind=ui.radio)
    await signed_in_user.should_not_see(kind=ui.checkbox)
    quiz = _quiz_card(signed_in_user)
    assert not [element for element in quiz.descendants() if isinstance(element, _ANSWER_CONTROLS)]


async def test_all_questions_sit_in_one_quiz_card(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    questions = _question_texts(2)
    _script_lesson_with_question(planner_model, content_model, len(questions))

    _submit(signed_in_user)
    await _see(signed_in_user, *questions)

    quiz = _quiz_card(signed_in_user)
    assert isinstance(quiz, ui.card)
    chat = next(iter(signed_in_user.find(marker=CHAT_MARKER).elements))
    assert [element for element in chat.descendants() if isinstance(element, ui.card)] == [quiz]
    for question in questions:
        assert question in _quiz_text(signed_in_user)


async def test_quiz_card_is_titled_with_no_question_count(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    questions = _question_texts(2)
    _script_lesson_with_question(planner_model, content_model, len(questions))

    _submit(signed_in_user)
    await _see(signed_in_user, *questions)

    quiz_lines = _quiz_text(signed_in_user).splitlines()
    assert _QUIZ_TITLE in quiz_lines
    assert not [line for line in quiz_lines if "on this lesson" in line]


async def test_quiz_questions_are_numbered_and_options_lettered(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    questions = _question_texts(2)
    _script_lesson_with_question(planner_model, content_model, len(questions))

    _submit(signed_in_user)
    await _see(signed_in_user, *questions)

    quiz = _quiz_text(signed_in_user)
    for number, question in enumerate(questions, start=1):
        assert _labelled(str(number), question).search(quiz), f"question {number} is not numbered"
    for letter, option in zip(ascii_uppercase, SAMPLE_OPTIONS):
        assert len(_labelled(letter, option).findall(quiz)) == len(questions), f"{option} is not lettered {letter}"


# ---------------------------------------------------------------------------
# Returning to a turn
# ---------------------------------------------------------------------------


async def test_reload_shows_the_same_turn_from_the_checkpointer(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    _script_lesson_with_question(planner_model, content_model)
    _submit(signed_in_user)
    await _see(signed_in_user, _LESSON_TEXT)
    calls_before_reload = planner_model.invocation_count

    await signed_in_user.open("/")

    await _see(signed_in_user, _PROMPT, _LESSON_TEXT, _QUESTION_TEXT)
    await signed_in_user.should_not_see(content=_REPLY)
    assert planner_model.invocation_count == calls_before_reload


async def test_reload_during_a_turn_shows_it_running_then_finished(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    _script_lesson(planner_model, content_model)
    planner_model.hold_at_call = _FINAL_CALL
    _submit(signed_in_user)
    await _turn_is_held(signed_in_user, planner_model)

    await signed_in_user.open("/")

    await _see(signed_in_user, _PROMPT)
    await _prompt_is_locked(signed_in_user)

    planner_model.release()

    await _see(signed_in_user, _LESSON_TEXT)
    await _prompt_is_open(signed_in_user)


async def test_closed_page_does_not_stop_the_turn(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    _script_lesson(planner_model, content_model)
    planner_model.hold_at_call = _FINAL_CALL
    _submit(signed_in_user)
    await _turn_is_held(signed_in_user, planner_model)
    assert signed_in_user.client is not None

    signed_in_user.client.delete()
    planner_model.release()
    await _turn_ended(turn_runner, learner_id)

    await signed_in_user.open("/")
    await _see(signed_in_user, _LESSON_TEXT)


# ---------------------------------------------------------------------------
# Starting another turn
# ---------------------------------------------------------------------------


async def test_a_second_turn_replaces_the_first(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    _script_two_lessons(planner_model, content_model)
    planner_model.hold_at_call = _CALLS_PER_LESSON + _FINAL_CALL
    _submit(signed_in_user)
    await _see(signed_in_user, _LESSON_TEXT)
    await _prompt_is_open(signed_in_user)

    _submit(signed_in_user, _SECOND_PROMPT)

    # Gone as soon as the new turn is on screen, not only when it finishes.
    await _see(signed_in_user, _SECOND_PROMPT)
    await signed_in_user.should_not_see(content=_LESSON_TEXT, retries=_TURN_RETRIES)
    await signed_in_user.should_not_see(content=_REPLY, retries=_TURN_RETRIES)

    planner_model.release()

    await _see(signed_in_user, _SECOND_LESSON_TEXT)
    await signed_in_user.should_not_see(content=_LESSON_TEXT)


async def test_prompt_is_disabled_while_a_turn_runs_so_turns_cannot_overlap(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    _script_lesson(planner_model, content_model)
    planner_model.hold_at_call = _FINAL_CALL
    _submit(signed_in_user)

    await _prompt_is_locked(signed_in_user)
    _submit(signed_in_user, _SECOND_PROMPT)
    planner_model.release()

    await _see(signed_in_user, _LESSON_TEXT)
    await _prompt_is_open(signed_in_user)
    assert set(planner_model.human_texts()) == {_PROMPT}


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "spaces"])
async def test_blank_prompt_starts_nothing(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    blank: str,
) -> None:
    # Held at the first call, so a turn that did start would still be live below.
    planner_model.hold_at_call = 0

    _submit(signed_in_user, blank)
    await asyncio.sleep(0)

    assert turn_runner.active(learner_id) is None
    assert planner_model.invocation_count == 0
    assert _prompt_input(signed_in_user).enabled


async def test_second_browser_submitting_during_a_turn_is_told_and_starts_nothing(
    signed_in_user: User,
    second_browser: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
) -> None:
    _script_lesson(planner_model, content_model)
    planner_model.hold_at_call = _FINAL_CALL
    _type_prompt(second_browser, _SECOND_PROMPT)

    # No await between the two: the second page's timer has not yet seen the turn start.
    _submit(signed_in_user)
    _click_submit(second_browser)

    await _see(second_browser, _ALREADY_RUNNING_TEXT)

    # That page then follows the running turn through to its result.
    await _turn_is_held(second_browser, planner_model)
    planner_model.release()
    await _see(second_browser, _LESSON_TEXT)
    assert set(planner_model.human_texts()) == {_PROMPT}


# ---------------------------------------------------------------------------
# Failed turns
# ---------------------------------------------------------------------------


async def test_graph_that_raises_partway_shows_the_error_logs_it_and_reopens_the_prompt(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    planner_model.script = _lesson_script()
    planner_model.fail_at_call = 1

    with caplog.at_level(logging.ERROR):
        _submit(signed_in_user)
        await _see(signed_in_user, _FAILED_TEXT)

    await _prompt_is_open(signed_in_user)
    assert _TRACEBACK_HEADER in caplog.text
    assert PLANNER_FAILURE_TEXT in caplog.text


async def test_reload_after_a_failed_turn_shows_the_notice_and_not_the_previous_lesson(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    _script_two_lessons(planner_model, content_model)
    planner_model.fail_at_call = _CALLS_PER_LESSON + 1
    _submit(signed_in_user)
    await _see(signed_in_user, _LESSON_TEXT)
    await _prompt_is_open(signed_in_user)
    _submit(signed_in_user, _SECOND_PROMPT)
    await _see(signed_in_user, _FAILED_TEXT)

    await signed_in_user.open("/")

    await _see(signed_in_user, _UNFINISHED_TEXT)
    await signed_in_user.should_not_see(content=_LESSON_TEXT)
    await signed_in_user.should_not_see(content=_REPLY)
    await _prompt_is_open(signed_in_user)


# ---------------------------------------------------------------------------
# Page-open logging
# ---------------------------------------------------------------------------


async def _open_and_read_log(user: User, caplog: pytest.LogCaptureFixture) -> str:
    caplog.clear()
    with caplog.at_level(logging.INFO):
        await user.open("/")
    return caplog.text


async def test_opening_the_page_with_no_turn_logs_that(
    signed_in_user: User, learner_id: UserId, caplog: pytest.LogCaptureFixture
) -> None:
    log = await _open_and_read_log(signed_in_user, caplog)

    assert f"Coach page opened for user {learner_id}: no turn yet" in log


async def test_opening_the_page_on_a_finished_turn_logs_that(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    learner_id: UserId,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _script_lesson(planner_model, content_model)
    _submit(signed_in_user)
    await _see(signed_in_user, _LESSON_TEXT)

    log = await _open_and_read_log(signed_in_user, caplog)

    assert f"Coach page opened for user {learner_id}: saved turn" in log


async def test_opening_the_page_during_a_turn_logs_that(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    learner_id: UserId,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _script_lesson(planner_model, content_model)
    planner_model.hold_at_call = _FINAL_CALL
    _submit(signed_in_user)
    await _turn_is_held(signed_in_user, planner_model)

    log = await _open_and_read_log(signed_in_user, caplog)
    planner_model.release()

    assert f"Coach page opened for user {learner_id}: running turn" in log


async def test_opening_the_page_on_an_unfinished_turn_logs_that(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    learner_id: UserId,
    caplog: pytest.LogCaptureFixture,
) -> None:
    planner_model.script = _lesson_script()
    planner_model.fail_at_call = 1
    _submit(signed_in_user)
    await _see(signed_in_user, _FAILED_TEXT)

    log = await _open_and_read_log(signed_in_user, caplog)

    assert f"Coach page opened for user {learner_id}: unfinished turn" in log
