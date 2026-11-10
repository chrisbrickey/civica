"""External smoke test for UI that spins up browser"""

import os
from pathlib import Path

import psycopg
import psycopg.rows
import pytest
from nicegui.testing import Screen
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement

from civica.domain.user import UserId
from civica.ui.auth import HOME_PATH, LOGIN_PATH
from civica.ui.deps import AppDeps
from civica.ui.pages.account import PASSWORD_LABEL, USERNAME_LABEL
from civica.ui.pages.coach import PROMPT_HINT
from civica.ui.pages import register_pages
from civica.ui.turns import CoachGraph, TurnRunner
from tests.support.assessment import FakeChatModel
from tests.support.coach import LESSON_TEXT, PROMPT, script_lesson
from tests.support.planner import ControlledPlannerModel
from tests.support.ui import PASSWORD, USERNAME
from tests.support.users import TEST_BCRYPT_ROUNDS

# Set before the session-scoped driver starts, so the fetched driver stays inside the project.
os.environ["SE_CACHE_PATH"] = str(Path(__file__).resolve().parents[3] / ".selenium-cache")

pytestmark = pytest.mark.external

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STORAGE_SECRET = "sample-storage-secret"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def nicegui_remove_all_screenshots() -> None:
    """Replaces NiceGUI's fixture, which creates a screenshots folder to prune old screenshots."""


def _no_screenshot(name: str, *, failed: bool) -> None:
    """Stands in for `Screen.shot`, so no screenshot is written after each test."""


@pytest.fixture()
def coach_screen(
    screen: Screen,
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    coach_graph: CoachGraph,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> Screen:
    """A real browser on the app; the server thread shares the per-test connection and fakes."""
    register_pages(
        AppDeps(graph=coach_graph, turn_runner=turn_runner, conn=db_schema, bcrypt_rounds=TEST_BCRYPT_ROUNDS)
    )
    screen.ui_run_kwargs["storage_secret"] = _STORAGE_SECRET
    screen.shot = _no_screenshot  # type: ignore[method-assign]
    return screen


def _prompt_input(screen: Screen) -> WebElement:
    return screen.find_by_css(f'input[placeholder="{PROMPT_HINT}"]')


def _sign_in(screen: Screen) -> None:
    """Sign in through the form, so the browser holds a real session cookie."""
    screen.open(LOGIN_PATH)
    screen.find_by_css(f'input[aria-label="{USERNAME_LABEL}"]').send_keys(USERNAME)
    screen.find_by_css(f'input[aria-label="{PASSWORD_LABEL}"]').send_keys(PASSWORD + Keys.ENTER)
    _prompt_input(screen)


def _shown(screen: Screen, text: str) -> list[WebElement]:
    """Page elements holding the text. Scripts are skipped: the page's startup script also carries it."""
    query = f'//*[not(self::script) and not(self::style) and text()[contains(., "{text}")]]'
    return screen.selenium.find_elements(By.XPATH, query)


def _ask_for_a_lesson(screen: Screen) -> None:
    _prompt_input(screen).send_keys(PROMPT + Keys.ENTER)
    screen.should_contain(LESSON_TEXT)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_signed_in_learner_asks_for_a_lesson_and_reads_it(
    coach_screen: Screen, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    script_lesson(planner_model, content_model)

    _sign_in(coach_screen)
    _ask_for_a_lesson(coach_screen)

    coach_screen.should_contain(PROMPT)
    prompt = _prompt_input(coach_screen)
    assert prompt.get_attribute("value") == ""
    assert prompt.is_enabled()


def test_reload_shows_the_same_turn(
    coach_screen: Screen, planner_model: ControlledPlannerModel, content_model: FakeChatModel
) -> None:
    script_lesson(planner_model, content_model)
    _sign_in(coach_screen)
    _ask_for_a_lesson(coach_screen)

    coach_screen.open(HOME_PATH)

    coach_screen.should_contain(PROMPT)
    coach_screen.should_contain(LESSON_TEXT)
    assert len(_shown(coach_screen, LESSON_TEXT)) == 1, "the lesson should show once"
