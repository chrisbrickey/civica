"""Integration tests for where the prompt sits and how the page scrolls when a turn is drawn."""

import pytest
from nicegui import ui
from nicegui.testing import User

from civica.domain.user import UserId
from civica.ui.turns import TurnRunner
from tests.support.assessment import FakeChatModel
from tests.support.coach import (
    FINAL_CALL,
    LESSON_TEXT,
    PROMPT,
    QUIZ_AFTER_LESSON_CALL,
    TURN_RETRIES,
    finish_turn,
    held_turn_is_shown,
    question_texts,
    script_lesson,
    script_lesson_with_quiz,
    submit,
    the,
)
from tests.support.planner import ControlledPlannerModel
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

_QUESTION_TEXT = question_texts(QUIZ_AFTER_LESSON_CALL, 1)[0]

_SCROLL_TO_END_SIGN = "scrollHeight"
_SCRIPT_MESSAGE = "run_javascript"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ancestors(element: ui.element) -> list[ui.element]:
    """Every element that holds this one, nearest first."""
    found: list[ui.element] = []
    slot = element.parent_slot
    while slot is not None:
        found.append(slot.parent)
        slot = slot.parent.parent_slot
    return found


def _assert_prompt_follows_the_turn_area(user: User) -> None:
    """The prompt and send button are page content, not footer, and come after the whole turn area."""
    assert user.client is not None
    turn_area = the(user, CHAT_MARKER)
    page_order = list(user.client.layout.descendants())
    end_of_turn_area = max(page_order.index(element) for element in turn_area.descendants(include_self=True))
    for marker in (PROMPT_INPUT_MARKER, SUBMIT_MARKER):
        control = the(user, marker)
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
        send(message_type, data, target_id)  # type: ignore[arg-type]

    monkeypatch.setattr(outbox, "enqueue_message", recording_send)
    return scripts


def _assert_no_scroll_to_end(scripts: list[str]) -> None:
    scrolls = [script for script in scripts if _SCROLL_TO_END_SIGN in script]
    assert not scrolls, f"the page scrolled to its end: {scrolls}"


async def _see_running_turn(user: User, planner_model: ControlledPlannerModel) -> None:
    """Send a prompt whose turn is held, and wait until its message, spinner, and locked prompt show."""
    planner_model.hold_at_call = FINAL_CALL
    submit(user)
    await held_turn_is_shown(user, planner_model)
    await user.should_see(PROMPT, retries=TURN_RETRIES)
    await user.should_see(kind=ui.spinner, retries=TURN_RETRIES)


async def _see_finished_lesson_and_quiz(user: User, turn_runner: TurnRunner, learner_id: UserId) -> None:
    """Send a prompt and wait for the turn's final draw: lesson, quiz, and the reopened prompt."""
    await finish_turn(user, turn_runner, learner_id)
    await user.should_see(LESSON_TEXT, retries=TURN_RETRIES)
    await user.should_see(_QUESTION_TEXT)
    await user.should_see(marker=QUIZ_MARKER)


# ---------------------------------------------------------------------------
# Where the prompt sits
# ---------------------------------------------------------------------------


async def test_with_no_saved_turn_the_prompt_follows_the_welcome_card(signed_in_user: User) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)

    _assert_prompt_follows_the_turn_area(signed_in_user)


async def test_while_a_turn_runs_the_locked_prompt_follows_the_message_and_spinner(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    script_lesson(planner_model, content_model)

    await _see_running_turn(signed_in_user, planner_model)

    _assert_prompt_follows_the_turn_area(signed_in_user)


async def test_after_a_lesson_and_quiz_the_prompt_follows_them(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson_with_quiz(planner_model, content_model)

    await _see_finished_lesson_and_quiz(signed_in_user, turn_runner, learner_id)

    _assert_prompt_follows_the_turn_area(signed_in_user)


# ---------------------------------------------------------------------------
# How the page moves when a turn is drawn
# ---------------------------------------------------------------------------


async def test_drawing_a_running_turn_does_not_scroll_to_the_end_of_the_page(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_lesson(planner_model, content_model)
    scripts = _record_scripts(signed_in_user, monkeypatch)

    await _see_running_turn(signed_in_user, planner_model)

    _assert_no_scroll_to_end(scripts)


async def test_drawing_a_finished_turn_does_not_scroll_to_the_end_of_the_page(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_lesson_with_quiz(planner_model, content_model)
    scripts = _record_scripts(signed_in_user, monkeypatch)

    await _see_finished_lesson_and_quiz(signed_in_user, turn_runner, learner_id)

    _assert_no_scroll_to_end(scripts)
