"""
Integration tests for where the prompt sits and how the page moves when a turn is drawn.

The prompt belongs to the page, right after the turn area, so the learner reads a turn from
its top and finds the prompt where the turn ends. Page order is a depth-first walk of the
element tree. Scripts sent to the browser are read from the client's outbox, the boundary
between the app and the browser.
"""

import asyncio
from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage
from nicegui import ui
from nicegui.testing import User

from tests.support.assessment import FakeChatModel, generated_item_text
from tests.support.planner import DEFAULT_THEME, ControlledPlannerModel, final_message, tool_call
from tests.support.ui import (
    CHAT_MARKER,
    INTRO_CARD_MARKER,
    PROMPT_INPUT_MARKER,
    QUIZ_MARKER,
    SUBMIT_MARKER,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPT = "sample-prompt-001"
_LESSON_TEXT = "sample-lesson-text-001"
_REPLY = "sample-final-reply-001"
_FOCUS = "sample-focus"

# The first generate_quiz call follows the one teach call on the content model.
_QUESTION_TEXT = generated_item_text(1, 0)

_SCROLL_TO_END_SIGN = "document.body.scrollHeight"
_SCRIPT_MESSAGE = "run_javascript"

# should_see polls every 0.1 seconds, so a turn has up to 5 seconds to reach the page.
_TURN_RETRIES = 50
_POLL_SECONDS = 0.02
_POLL_LIMIT = 250
# Scripts are sent from a background task, so give them a few loop turns to reach the outbox.
_SETTLE_SECONDS = 0.3

# The lesson script's last planner call; holding it keeps the turn running.
_FINAL_CALL = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lesson_script() -> tuple[AIMessage, ...]:
    return (
        tool_call("read_learner_state", "call-read-001"),
        tool_call("teach", "call-teach-001", theme=DEFAULT_THEME.slug, focus=_FOCUS),
        final_message(_REPLY),
    )


def _script_lesson(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    planner_model.script = _lesson_script()
    content_model.replies = (_LESSON_TEXT,)


def _script_lesson_with_quiz(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    planner_model.script = (
        tool_call("teach", "call-teach", theme=DEFAULT_THEME.slug, focus=_FOCUS),
        tool_call("generate_quiz", "call-quiz", theme=DEFAULT_THEME.slug, n=1),
        final_message(_REPLY),
    )
    content_model.replies = (_LESSON_TEXT,)


def _submit(user: User) -> None:
    user.find(marker=PROMPT_INPUT_MARKER).type(_PROMPT)
    user.find(marker=SUBMIT_MARKER).click()


def _the(user: User, marker: str) -> ui.element:
    marked = list(user.find(marker=marker).elements)
    assert len(marked) == 1, f"expected one element marked {marker!r}, found {len(marked)}"
    return marked[0]


def _ancestors(element: ui.element) -> list[ui.element]:
    """Every element that holds this one, nearest first."""
    found: list[ui.element] = []
    slot = element.parent_slot
    while slot is not None:
        found.append(slot.parent)
        slot = slot.parent.parent_slot
    return found


async def _wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(_POLL_LIMIT):
        if condition():
            return
        await asyncio.sleep(_POLL_SECONDS)
    raise AssertionError("condition not met in time")


async def _prompt_is_locked(user: User) -> None:
    prompt = _the(user, PROMPT_INPUT_MARKER)
    send = _the(user, SUBMIT_MARKER)
    assert isinstance(prompt, ui.input) and isinstance(send, ui.button)
    await _wait_until(lambda: not prompt.enabled and not send.enabled)


async def _prompt_is_open(user: User) -> None:
    prompt = _the(user, PROMPT_INPUT_MARKER)
    assert isinstance(prompt, ui.input)
    await _wait_until(lambda: prompt.enabled)


def _assert_prompt_follows_the_turn_area(user: User) -> None:
    """The prompt and send button are page content, not footer, and come after the whole turn area."""
    assert user.client is not None
    turn_area = _the(user, CHAT_MARKER)
    page_order = list(user.client.layout.descendants())
    end_of_turn_area = max(page_order.index(element) for element in turn_area.descendants(include_self=True))
    for marker in (PROMPT_INPUT_MARKER, SUBMIT_MARKER):
        control = _the(user, marker)
        holders = _ancestors(control)
        assert not any(isinstance(holder, ui.footer) for holder in holders), f"{marker!r} is in the page footer"
        assert user.client.content in holders, f"{marker!r} is not part of the page content"
        assert turn_area not in holders, f"{marker!r} is inside the turn area"
        assert page_order.index(control) > end_of_turn_area, f"{marker!r} comes before the end of the turn area"


def _record_scripts(user: User, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """From now on, collect every script the page sends to this browser."""
    assert user.client is not None
    outbox = user.client.outbox
    send = outbox.enqueue_message
    scripts: list[str] = []

    def recording_send(message_type: str, data: dict[str, str], target_id: str) -> None:
        if message_type == _SCRIPT_MESSAGE:
            scripts.append(data["code"])
        send(message_type, data, target_id)

    monkeypatch.setattr(outbox, "enqueue_message", recording_send)
    return scripts


def _assert_no_scroll_to_end(scripts: list[str]) -> None:
    scrolls = [script for script in scripts if _SCROLL_TO_END_SIGN in script]
    assert not scrolls, f"the page scrolled to its end: {scrolls}"


async def _see_running_turn(user: User, planner_model: ControlledPlannerModel) -> None:
    """Send a prompt whose turn is held, and wait until its message, spinner, and locked prompt show."""
    planner_model.hold_at_call = _FINAL_CALL
    _submit(user)
    await user.should_see(_PROMPT, retries=_TURN_RETRIES)
    await user.should_see(kind=ui.spinner, retries=_TURN_RETRIES)
    await _prompt_is_locked(user)


async def _see_finished_lesson_and_quiz(user: User) -> None:
    _submit(user)
    await user.should_see(_LESSON_TEXT, retries=_TURN_RETRIES)
    await user.should_see(_QUESTION_TEXT, retries=_TURN_RETRIES)
    await user.should_see(marker=QUIZ_MARKER, retries=_TURN_RETRIES)
    await _prompt_is_open(user)


# ---------------------------------------------------------------------------
# Where the prompt sits
# ---------------------------------------------------------------------------


async def test_with_no_saved_turn_the_prompt_follows_the_welcome_card(signed_in_user: User) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)

    _assert_prompt_follows_the_turn_area(signed_in_user)


async def test_while_a_turn_runs_the_locked_prompt_follows_the_message_and_spinner(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    _script_lesson(planner_model, content_model)

    await _see_running_turn(signed_in_user, planner_model)

    _assert_prompt_follows_the_turn_area(signed_in_user)


async def test_after_a_lesson_and_quiz_the_prompt_follows_them(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    _script_lesson_with_quiz(planner_model, content_model)

    await _see_finished_lesson_and_quiz(signed_in_user)

    _assert_prompt_follows_the_turn_area(signed_in_user)


# ---------------------------------------------------------------------------
# Where the page shows a turn
# ---------------------------------------------------------------------------


async def test_drawing_a_running_turn_does_not_scroll_to_the_end_of_the_page(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_lesson(planner_model, content_model)
    scripts = _record_scripts(signed_in_user, monkeypatch)

    await _see_running_turn(signed_in_user, planner_model)
    await asyncio.sleep(_SETTLE_SECONDS)

    _assert_no_scroll_to_end(scripts)


async def test_drawing_a_finished_turn_does_not_scroll_to_the_end_of_the_page(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script_lesson_with_quiz(planner_model, content_model)
    scripts = _record_scripts(signed_in_user, monkeypatch)

    await _see_finished_lesson_and_quiz(signed_in_user)
    await asyncio.sleep(_SETTLE_SECONDS)

    _assert_no_scroll_to_end(scripts)
