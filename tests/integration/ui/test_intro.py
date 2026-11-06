"""
Integration tests for the welcome card a new learner sees and the header button that reopens it.

A new learner is one with no saved turn on the coach thread. The card sits in the turn area;
the dialog shows the same text from the header at any time. Both are checked for the five
official theme names, which proves they draw from the same content without pinning the prose.
"""

import asyncio
from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from nicegui import ui
from nicegui.elements.mixins.content_element import ContentElement
from nicegui.elements.mixins.text_element import TextElement
from nicegui.testing import User

from civica.domain.themes import THEMES_BY_SLUG
from civica.domain.user import UserId
from civica.graph.graph import GraphInput, GraphState
from civica.graph.nodes import PlannerContext
from civica.ui.turns import TurnRunner
from tests.support.assessment import FakeChatModel
from tests.support.planner import DEFAULT_THEME, ControlledPlannerModel, final_message, tool_call
from tests.support.ui import (
    CHAT_MARKER,
    INTRO_CARD_MARKER,
    INTRO_CLOSE_MARKER,
    INTRO_DIALOG_MARKER,
    INTRO_OPEN_MARKER,
    PROMPT_INPUT_MARKER,
    SUBMIT_MARKER,
)

_CoachGraph = CompiledStateGraph[GraphState, PlannerContext, GraphInput, GraphState]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPT = "sample-prompt-001"
_LESSON_TEXT = "sample-lesson-text-001"
_REPLY = "sample-final-reply-001"
_FOCUS = "sample-focus"
_READ_FAILURE = "sample-read-failure"

_FAILED_TEXT = "Something went wrong. Please try again."
_UNFINISHED_TEXT = "Your last request didn't finish. Please try again."

_HEADER_COLOR = "primary"

_THEME_NAMES =tuple(theme.display_name_fr for theme in THEMES_BY_SLUG.values())

# should_see polls every 0.1 seconds, so a turn has up to 5 seconds to reach the page.
_TURN_RETRIES = 50
_POLL_SECONDS = 0.02
_POLL_LIMIT = 250

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


def _script_reply(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    planner_model.script = (final_message(_REPLY),)


def _submit(user: User) -> None:
    user.find(marker=PROMPT_INPUT_MARKER).type(_PROMPT)
    user.find(marker=SUBMIT_MARKER).click()


async def _wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(_POLL_LIMIT):
        if condition():
            return
        await asyncio.sleep(_POLL_SECONDS)
    raise AssertionError("condition not met in time")


async def _finish_turn(user: User, turn_runner: TurnRunner, learner_id: UserId) -> None:
    """Send the prompt and wait until the turn has ended and the prompt is usable again."""
    _submit(user)
    await _wait_until(lambda: turn_runner.active(learner_id) is None)
    prompt = next(iter(user.find(kind=ui.input, marker=PROMPT_INPUT_MARKER).elements))
    await _wait_until(lambda: prompt.enabled)


def _the(user: User, marker: str) -> ui.element:
    marked = list(user.find(marker=marker).elements)
    assert len(marked) == 1, f"expected one element marked {marker!r}, found {len(marked)}"
    return marked[0]


def _text_in(element: ui.element) -> str:
    """Every piece of text inside the element, whether label or markdown."""
    parts: list[str] = []
    for child in element.descendants(include_self=True):
        if isinstance(child, TextElement):
            parts.append(child.text)
        elif isinstance(child, ContentElement):
            parts.append(child.content)
    return "\n".join(parts)


def _assert_names_every_theme(element: ui.element) -> None:
    text = _text_in(element)
    missing = [name for name in _THEME_NAMES if name not in text]
    assert not missing, f"themes missing from the intro: {missing}"


async def _no_card(user: User) -> None:
    await user.should_not_see(marker=INTRO_CARD_MARKER)


def _dialog(user: User) -> ui.dialog:
    dialog = _the(user, INTRO_DIALOG_MARKER)
    assert isinstance(dialog, ui.dialog), f"{INTRO_DIALOG_MARKER!r} should mark a dialog"
    return dialog


# ---------------------------------------------------------------------------
# The welcome card
# ---------------------------------------------------------------------------


async def test_new_learner_sees_the_welcome_card_in_the_turn_area_naming_every_theme(
    signed_in_user: User,
) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)

    card = _the(signed_in_user, INTRO_CARD_MARKER)
    turn_area = _the(signed_in_user, CHAT_MARKER)
    assert card in list(turn_area.descendants()), "the welcome card should sit in the turn area"
    _assert_names_every_theme(card)


@pytest.mark.parametrize("script", [_script_lesson, _script_reply], ids=["lesson", "reply"])
async def test_learner_with_a_finished_turn_sees_no_welcome_card_even_after_reload(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    script: Callable[[ControlledPlannerModel, FakeChatModel], None],
) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)
    script(planner_model, content_model)

    await _finish_turn(signed_in_user, turn_runner, learner_id)
    await _no_card(signed_in_user)

    await signed_in_user.open("/")
    await signed_in_user.should_see(_PROMPT)
    await _no_card(signed_in_user)


async def test_welcome_card_is_gone_once_a_turn_starts(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)
    _script_lesson(planner_model, content_model)
    planner_model.hold_at_call = _FINAL_CALL

    _submit(signed_in_user)
    await _wait_until(lambda: planner_model.invocation_count > _FINAL_CALL)
    await signed_in_user.should_see(_PROMPT)
    await _no_card(signed_in_user)

    # A page opened while the turn runs shows no card either.
    await signed_in_user.open("/")
    await signed_in_user.should_see(_PROMPT)
    await _no_card(signed_in_user)
    planner_model.release()


async def test_failed_turn_shows_no_welcome_card_even_after_reload(
    signed_in_user: User, planner_model: ControlledPlannerModel
) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)
    planner_model.script = _lesson_script()
    planner_model.fail_at_call = 1

    _submit(signed_in_user)
    await signed_in_user.should_see(_FAILED_TEXT, retries=_TURN_RETRIES)
    await _no_card(signed_in_user)

    await signed_in_user.open("/")
    await signed_in_user.should_see(_UNFINISHED_TEXT)
    await _no_card(signed_in_user)


async def test_read_failure_shows_no_welcome_card(
    signed_in_user: User, coach_graph: _CoachGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(_READ_FAILURE)

    # The saved thread cannot be read, so the page cannot tell whether the learner is new.
    monkeypatch.setattr(coach_graph, "get_state", unreadable)

    await signed_in_user.open("/")

    await signed_in_user.should_see(_FAILED_TEXT)
    await _no_card(signed_in_user)


# ---------------------------------------------------------------------------
# Reopening the intro from the header
# ---------------------------------------------------------------------------


async def _open_and_close_intro(user: User) -> None:
    dialog = _dialog(user)
    assert not dialog.value, "the intro dialog should start closed"
    header = next(iter(user.find(kind=ui.header).elements))
    assert _the(user, INTRO_OPEN_MARKER) in list(header.descendants()), "the intro button should sit in the header"

    user.find(marker=INTRO_OPEN_MARKER).click()
    await _wait_until(lambda: dialog.value)
    _assert_names_every_theme(dialog)

    user.find(marker=INTRO_CLOSE_MARKER).click()
    await _wait_until(lambda: not dialog.value)


async def test_header_button_opens_the_intro_on_an_empty_thread_and_closes_from_inside(
    signed_in_user: User,
) -> None:
    await _open_and_close_intro(signed_in_user)


async def test_header_button_stands_out_from_the_header(signed_in_user: User) -> None:
    # The header is the theme's primary color, so a flat primary button would be invisible on it.
    button = _the(signed_in_user, INTRO_OPEN_MARKER)
    assert button.props.get("color") != _HEADER_COLOR, "the intro button should not match the header's color"


async def test_header_button_opens_the_intro_after_a_finished_turn(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    _script_lesson(planner_model, content_model)
    await _finish_turn(signed_in_user, turn_runner, learner_id)

    await _open_and_close_intro(signed_in_user)
