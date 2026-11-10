"""Integration tests for the welcome card and the header button that reopens it."""

from collections.abc import Callable

import pytest
from nicegui import ui
from nicegui.elements.mixins.content_element import ContentElement
from nicegui.elements.mixins.text_element import TextElement
from nicegui.testing import User

from civica.domain.themes import THEMES_BY_SLUG
from civica.domain.user import UserId
from civica.ui.turns import CoachGraph, TurnRunner
from tests.support.assessment import FakeChatModel
from tests.support.coach import (
    FAILED_TEXT,
    FINAL_CALL,
    PROMPT,
    UNFINISHED_TEXT,
    finish_turn,
    held_turn_is_shown,
    script_failure,
    script_lesson,
    script_reply,
    submit,
    the,
    wait_until,
)
from tests.support.planner import ControlledPlannerModel
from tests.support.ui import (
    CHAT_MARKER,
    INTRO_CARD_MARKER,
    INTRO_CLOSE_MARKER,
    INTRO_DIALOG_MARKER,
    INTRO_OPEN_MARKER,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_READ_FAILURE = "sample-read-failure"

# The header uses the theme's primary color, so a button of that color would vanish into it.
_HEADER_COLOR = "primary"

_THEME_NAMES = tuple(theme.display_name_fr for theme in THEMES_BY_SLUG.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _script_lesson(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    script_lesson(planner_model, content_model)


def _script_reply(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    script_reply(planner_model)


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
    dialog = the(user, INTRO_DIALOG_MARKER)
    assert isinstance(dialog, ui.dialog), f"{INTRO_DIALOG_MARKER!r} should mark a dialog"
    return dialog


# ---------------------------------------------------------------------------
# The welcome card
# ---------------------------------------------------------------------------


async def test_new_learner_sees_the_welcome_card_in_the_turn_area_naming_every_theme(
    signed_in_user: User,
) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)

    card = the(signed_in_user, INTRO_CARD_MARKER)
    assert card in list(the(signed_in_user, CHAT_MARKER).descendants()), "the welcome card should sit in the turn area"
    _assert_names_every_theme(card)


@pytest.mark.parametrize("script", [_script_lesson, _script_reply], ids=["lesson", "reply"])
async def test_finished_turn_shows_no_welcome_card_even_after_reload(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    script: Callable[[ControlledPlannerModel, FakeChatModel], None],
) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)
    script(planner_model, content_model)

    await finish_turn(signed_in_user, turn_runner, learner_id)
    await _no_card(signed_in_user)

    await signed_in_user.open("/")
    await signed_in_user.should_see(PROMPT)
    await _no_card(signed_in_user)


async def test_welcome_card_is_gone_once_a_turn_starts_even_after_reload(
    signed_in_user: User, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)
    script_lesson(planner_model, content_model)
    planner_model.hold_at_call = FINAL_CALL

    submit(signed_in_user)
    await held_turn_is_shown(signed_in_user, planner_model)
    await signed_in_user.should_see(PROMPT)
    await _no_card(signed_in_user)

    await signed_in_user.open("/")
    await signed_in_user.should_see(PROMPT)
    await _no_card(signed_in_user)
    planner_model.release()


async def test_failed_turn_shows_no_welcome_card_even_after_reload(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    await signed_in_user.should_see(marker=INTRO_CARD_MARKER)
    script_failure(planner_model)

    await finish_turn(signed_in_user, turn_runner, learner_id)
    await signed_in_user.should_see(FAILED_TEXT)
    await _no_card(signed_in_user)

    await signed_in_user.open("/")
    await signed_in_user.should_see(UNFINISHED_TEXT)
    await _no_card(signed_in_user)


async def test_unreadable_thread_shows_the_error_and_no_welcome_card_even_after_reload(
    signed_in_user: User, coach_graph: CoachGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(_READ_FAILURE)

    # The saved thread cannot be read, so the page cannot tell whether the learner is new.
    monkeypatch.setattr(coach_graph, "get_state", unreadable)

    for _visit in range(2):
        await signed_in_user.open("/")
        await signed_in_user.should_see(FAILED_TEXT)
        await _no_card(signed_in_user)


# ---------------------------------------------------------------------------
# Reopening the intro from the header
# ---------------------------------------------------------------------------


async def _open_and_close_intro(user: User) -> None:
    dialog = _dialog(user)
    assert not dialog.value, "the intro dialog should start closed"
    header = next(iter(user.find(kind=ui.header).elements))
    assert the(user, INTRO_OPEN_MARKER) in list(header.descendants()), "the intro button should sit in the header"

    user.find(marker=INTRO_OPEN_MARKER).click()
    await wait_until(lambda: dialog.value)
    _assert_names_every_theme(dialog)

    user.find(marker=INTRO_CLOSE_MARKER).click()
    await wait_until(lambda: not dialog.value)


async def test_header_button_opens_the_intro_on_an_empty_thread_and_closes_from_inside(
    signed_in_user: User,
) -> None:
    await _open_and_close_intro(signed_in_user)


async def test_header_button_opens_the_intro_after_a_finished_turn(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)
    await finish_turn(signed_in_user, turn_runner, learner_id)

    await _open_and_close_intro(signed_in_user)


async def test_header_button_stands_out_from_the_header(signed_in_user: User) -> None:
    button = the(signed_in_user, INTRO_OPEN_MARKER)

    assert button.props.get("color", _HEADER_COLOR) != _HEADER_COLOR, "the intro button should not match the header"
