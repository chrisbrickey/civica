"""
Integration tests for the web app authentication.

NiceGUI's User fixture simulates the browser in Python (no real browser).
Register and sign-in run through the real users.service against the per-test schema.
"""

import logging
from collections.abc import Generator
from uuid import UUID

import nicegui
import psycopg
import psycopg.rows
import pytest
from nicegui import ui
from nicegui.testing import User

from civica.domain.user import UserId
from civica.users.service import register
from tests.support.ui import (
    LOGOUT_MARKER,
    PASSWORD,
    PASSWORD_MARKER,
    SIGN_IN_MARKER,
    USERNAME,
    USERNAME_MARKER,
    fill_sign_in_form,
    sign_in,
)
from tests.support.users import TEST_BCRYPT_ROUNDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WRONG_PASSWORD = "wrong-password"
_OTHER_PASSWORD = "other-password"
_UNKNOWN_USERNAME = "nobody-registered"

# One under the 8-character minimum, and one byte past bcrypt's 72-byte limit.
_UNDER_LENGTH_PASSWORD = "a" * 7
_OVER_LENGTH_PASSWORD = "a" * 73

_SIGNED_IN_TEXT = f"Signed in as {USERNAME}"
_SIGN_IN_FAILED_TEXT = "Wrong username or password."
_USERNAME_TAKEN_TEXT = "That username is taken."
_PASSWORD_MISMATCH_TEXT = "Passwords do not match."
_PASSWORD_RULES_TEXT = "8 to 72 characters"
_USERNAME_RULES_TEXT = "3 to 32 characters"
_USERNAME_RULES_ERROR_TEXT = "Username must be"
_INVALID_USERNAME = "test user!"
_COMMON_PASSWORD = "password"
_COMMON_PASSWORD_TEXT = "too common"
_PASSWORD_RULES_ERROR_TEXT = "Password must be 8 to 72 characters."
_MISSING_INPUT_TEXT = "Enter a username and a password."

# A stylesheet the login page needs before anyone has signed in.
_NICEGUI_ASSET_PATH = f"/_nicegui/{nicegui.__version__}/static/nicegui.css"

_CONFIRM_PASSWORD_MARKER = "confirm-password"
_REGISTER_LINK_MARKER = "register-link"
_SIGN_IN_LINK_MARKER = "sign-in-link"
_CREATE_ACCOUNT_MARKER = "create-account"
_PASSWORD_RULES_MARKER = "password-rules"
_USERNAME_RULES_MARKER = "username-rules"

# Distinctive values, so a scan of the logs cannot match anything else by accident.
_SCANNED_PASSWORD = "scanned-sample-password"
_SCANNED_WRONG_PASSWORD = "scanned-sample-wrong-entry"
_SCANNED_OVER_LENGTH_PASSWORD = "scanned-sample-long-entry-".ljust(73, "x")
_SCANNED_USERNAME = "scanned-sample-name"
_SCANNED_UNKNOWN_USERNAME = "Scanned-Sample-Typed-Secret"
_SCANNED_INVALID_USERNAME = "Scanned Sample Name!"

_FAILING_PAGE_PATH = "/sample-failing-page"
_FAIL_MARKER = "sample-fail"
_PAGE_ERROR_TEXT = "sample page failure"
_TRACEBACK_HEADER = "Traceback (most recent call last)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_users(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> int:
    result = conn.execute("SELECT COUNT(*) FROM users").fetchone()
    assert result is not None
    return int(result[0])


def _register_directly(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> UserId:
    return register(USERNAME, PASSWORD, conn=conn, rounds=TEST_BCRYPT_ROUNDS)


def _stored_user_id(conn: psycopg.Connection[psycopg.rows.TupleRow], username: str) -> UUID:
    result = conn.execute("SELECT user_id FROM users WHERE username = %s", (username,)).fetchone()
    assert result is not None
    return UUID(str(result[0]))


async def _redirect_location(user: User, path: str) -> str | None:
    """Where a plain GET of path redirects this browser, or None when it is served."""
    response = await user.http_client.get(path, follow_redirects=False)
    return response.headers.get("location") if response.is_redirect else None


async def _fill_account_form(
    user: User, username: str, password: str, confirmation: str | None = None
) -> None:
    await user.open("/register")
    user.find(marker=USERNAME_MARKER).type(username)
    user.find(marker=PASSWORD_MARKER).type(password)
    user.find(marker=_CONFIRM_PASSWORD_MARKER).type(
        password if confirmation is None else confirmation
    )


async def _create_account(
    user: User, username: str, password: str, confirmation: str | None = None
) -> None:
    await _fill_account_form(user, username, password, confirmation)
    user.find(marker=_CREATE_ACCOUNT_MARKER).click()


async def _log_out(user: User) -> None:
    user.find(marker=LOGOUT_MARKER).click()
    await user.should_see(marker=SIGN_IN_MARKER)


# modules that log registration, sign-in, and logout
_AUTH_LOGGERS = {"civica.ui.pages.account", "civica.ui.pages.home"}


def _output_lines(caplog: pytest.LogCaptureFixture, start: int = 0) -> list[logging.LogRecord]:
    """Records at the app's configured level or above, from index start onward."""
    return [record for record in caplog.records[start:] if record.levelno >= logging.INFO]


def _all_auth_lines(caplog: pytest.LogCaptureFixture, start: int = 0) -> list[logging.LogRecord]:
    """Auth event lines only. Other modules may log during the same action for their own reasons."""
    return [line for line in _output_lines(caplog, start) if line.name in _AUTH_LOGGERS]


def _auth_lines(
    caplog: pytest.LogCaptureFixture, user_id: UUID, start: int = 0
) -> list[logging.LogRecord]:
    """Auth event lines about this user."""
    return [line for line in _all_auth_lines(caplog, start) if str(user_id) in line.getMessage()]


def _raise_sample_error() -> None:
    raise RuntimeError(_PAGE_ERROR_TEXT)


@pytest.fixture()
def app_logs(caplog: pytest.LogCaptureFixture) -> Generator[pytest.LogCaptureFixture, None, None]:
    """Everything every logger writes during the test, DEBUG included."""
    with caplog.at_level(logging.DEBUG):
        yield caplog


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


async def test_unauthenticated_visit_redirects_to_login(ui_user: User) -> None:
    assert await _redirect_location(ui_user, "/") == "/login"

    await ui_user.open("/")
    await ui_user.should_see(marker=SIGN_IN_MARKER)


async def test_register_page_is_reachable_without_signing_in(ui_user: User) -> None:
    assert await _redirect_location(ui_user, "/register") is None


async def test_nicegui_assets_are_served_without_signing_in(ui_user: User) -> None:
    response = await ui_user.http_client.get(_NICEGUI_ASSET_PATH, follow_redirects=False)

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Sign-in page
# ---------------------------------------------------------------------------


async def test_sign_in_with_correct_password_signs_in(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    _register_directly(db_schema)

    await sign_in(ui_user, USERNAME, PASSWORD)

    await ui_user.should_see(_SIGNED_IN_TEXT)


async def test_enter_key_signs_in(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    _register_directly(db_schema)

    await fill_sign_in_form(ui_user, USERNAME, PASSWORD)
    ui_user.find(marker=PASSWORD_MARKER).trigger("keydown.enter")

    await ui_user.should_see(_SIGNED_IN_TEXT)


async def test_sign_in_with_wrong_password_shows_error_and_stays_on_login(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    _register_directly(db_schema)

    await sign_in(ui_user, USERNAME, _WRONG_PASSWORD)

    await ui_user.should_see(_SIGN_IN_FAILED_TEXT)
    await ui_user.should_not_see(_SIGNED_IN_TEXT)
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_sign_in_with_unknown_username_shows_error_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await sign_in(ui_user, _UNKNOWN_USERNAME, PASSWORD)

    await ui_user.should_see(_SIGN_IN_FAILED_TEXT)
    assert _count_users(db_schema) == 0
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_sign_in_with_over_length_password_shows_error(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    _register_directly(db_schema)

    await sign_in(ui_user, USERNAME, _OVER_LENGTH_PASSWORD)

    await ui_user.should_see(_SIGN_IN_FAILED_TEXT)
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_sign_in_with_empty_input_shows_error(ui_user: User) -> None:
    await sign_in(ui_user, "", "")

    await ui_user.should_see(_MISSING_INPUT_TEXT)


async def test_register_link_opens_registration_page(ui_user: User) -> None:
    await ui_user.open("/login")

    ui_user.find(marker=_REGISTER_LINK_MARKER).click()

    await ui_user.should_see(marker=_CONFIRM_PASSWORD_MARKER)
    await ui_user.should_see(marker=_CREATE_ACCOUNT_MARKER)


# ---------------------------------------------------------------------------
# Registration page
# ---------------------------------------------------------------------------


async def test_registration_page_shows_password_rules(ui_user: User) -> None:
    await ui_user.open("/register")

    await ui_user.should_see(content=_PASSWORD_RULES_TEXT, marker=_PASSWORD_RULES_MARKER)


async def test_registration_page_shows_username_rules(ui_user: User) -> None:
    await ui_user.open("/register")

    await ui_user.should_see(content=_USERNAME_RULES_TEXT, marker=_USERNAME_RULES_MARKER)


async def test_invalid_username_shows_rules_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, _INVALID_USERNAME, PASSWORD)

    await ui_user.should_see(_USERNAME_RULES_ERROR_TEXT)
    assert _count_users(db_schema) == 0


async def test_common_password_shows_error_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, USERNAME, _COMMON_PASSWORD)

    await ui_user.should_see(_COMMON_PASSWORD_TEXT)
    assert _count_users(db_schema) == 0


async def test_sign_in_link_returns_to_sign_in_page(ui_user: User) -> None:
    await ui_user.open("/register")

    ui_user.find(marker=_SIGN_IN_LINK_MARKER).click()

    await ui_user.should_see(marker=SIGN_IN_MARKER)


async def test_create_account_registers_and_signs_in(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, USERNAME, PASSWORD)

    await ui_user.should_see(_SIGNED_IN_TEXT)
    assert _count_users(db_schema) == 1


async def test_create_account_with_taken_username_shows_error(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    _register_directly(db_schema)

    await _create_account(ui_user, USERNAME, _OTHER_PASSWORD)

    await ui_user.should_see(_USERNAME_TAKEN_TEXT)
    assert _count_users(db_schema) == 1
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_mismatched_confirmation_shows_error_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, USERNAME, PASSWORD, confirmation=_OTHER_PASSWORD)

    await ui_user.should_see(_PASSWORD_MISMATCH_TEXT)
    assert _count_users(db_schema) == 0


async def test_under_length_password_shows_rules_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, USERNAME, _UNDER_LENGTH_PASSWORD)

    await ui_user.should_see(_PASSWORD_RULES_ERROR_TEXT)
    assert _count_users(db_schema) == 0


async def test_over_length_password_shows_rules_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, USERNAME, _OVER_LENGTH_PASSWORD)

    await ui_user.should_see(_PASSWORD_RULES_ERROR_TEXT)
    assert _count_users(db_schema) == 0


async def test_create_account_with_empty_input_shows_error_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, "", "")

    await ui_user.should_see(_MISSING_INPUT_TEXT)
    assert _count_users(db_schema) == 0


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


async def test_refresh_keeps_user_signed_in(ui_user: User) -> None:
    await _create_account(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)

    await ui_user.open("/")

    await ui_user.should_see(_SIGNED_IN_TEXT)
    await ui_user.should_not_see(marker=SIGN_IN_MARKER)


async def test_logout_shows_login_and_blocks_signed_in_page(ui_user: User) -> None:
    await _create_account(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)

    ui_user.find(marker=LOGOUT_MARKER).click()

    await ui_user.should_see(marker=SIGN_IN_MARKER)
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_signing_back_in_reaches_the_same_account(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)
    ui_user.find(marker=LOGOUT_MARKER).click()
    await ui_user.should_see(marker=SIGN_IN_MARKER)

    await sign_in(ui_user, USERNAME, PASSWORD)

    await ui_user.should_see(_SIGNED_IN_TEXT)
    assert _count_users(db_schema) == 1


# ---------------------------------------------------------------------------
# Account activity logging
# ---------------------------------------------------------------------------


async def test_registration_writes_one_info_line_with_the_user_id(
    ui_user: User,
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    app_logs: pytest.LogCaptureFixture,
) -> None:
    await _create_account(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)

    user_id = _stored_user_id(db_schema, USERNAME)
    assert [line.levelno for line in _auth_lines(app_logs, user_id)] == [logging.INFO]


async def test_sign_in_writes_one_info_line_with_the_user_id(
    ui_user: User,
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    app_logs: pytest.LogCaptureFixture,
) -> None:
    user_id = _register_directly(db_schema)

    await sign_in(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)

    assert [line.levelno for line in _auth_lines(app_logs, user_id)] == [logging.INFO]


async def test_log_out_writes_one_info_line_with_the_user_id(
    ui_user: User,
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    app_logs: pytest.LogCaptureFixture,
) -> None:
    user_id = _register_directly(db_schema)
    await sign_in(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)
    start = len(app_logs.records)

    await _log_out(ui_user)

    lines = _auth_lines(app_logs, user_id, start)
    assert [line.levelno for line in lines] == [logging.INFO]


async def test_registration_sign_in_and_log_out_lines_are_each_their_own_kind(
    ui_user: User,
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    app_logs: pytest.LogCaptureFixture,
) -> None:
    await _create_account(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)
    await _log_out(ui_user)
    await sign_in(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)

    user_id = _stored_user_id(db_schema, USERNAME)
    lines = _auth_lines(app_logs, user_id)
    # With the id removed, wording is all that is left, so this does not pin any of it.
    kinds = [line.getMessage().replace(str(user_id), "") for line in lines]
    assert len(kinds) == 3
    assert len(set(kinds)) == 3


@pytest.mark.parametrize(
    ("submitted_username", "submitted_password"),
    [
        pytest.param(_SCANNED_USERNAME, _WRONG_PASSWORD, id="wrong-password"),
        pytest.param(_SCANNED_UNKNOWN_USERNAME, PASSWORD, id="unknown-username"),
        pytest.param(_SCANNED_USERNAME, _OVER_LENGTH_PASSWORD, id="over-length-password"),
    ],
)
async def test_failed_sign_in_writes_one_line_without_the_submitted_username(
    ui_user: User,
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    app_logs: pytest.LogCaptureFixture,
    submitted_username: str,
    submitted_password: str,
) -> None:
    register(_SCANNED_USERNAME, PASSWORD, conn=db_schema, rounds=TEST_BCRYPT_ROUNDS)
    await fill_sign_in_form(ui_user, submitted_username, submitted_password)
    start = len(app_logs.records)

    ui_user.find(marker=SIGN_IN_MARKER).click()
    await ui_user.should_see(_SIGN_IN_FAILED_TEXT)

    assert len(_all_auth_lines(app_logs, start)) == 1
    assert submitted_username.casefold() not in app_logs.text.casefold()


async def test_empty_sign_in_form_writes_no_line(
    ui_user: User, app_logs: pytest.LogCaptureFixture
) -> None:
    await fill_sign_in_form(ui_user, "", "")
    start = len(app_logs.records)

    ui_user.find(marker=SIGN_IN_MARKER).click()
    await ui_user.should_see(_MISSING_INPUT_TEXT)

    assert _all_auth_lines(app_logs, start) == []


@pytest.mark.parametrize(
    ("username", "password", "confirmation", "expected_error"),
    [
        pytest.param(_SCANNED_USERNAME, _SCANNED_PASSWORD, None, _USERNAME_TAKEN_TEXT, id="taken-username"),
        pytest.param(_SCANNED_INVALID_USERNAME, _SCANNED_PASSWORD, None, _USERNAME_RULES_ERROR_TEXT, id="invalid-username"),
        pytest.param(_SCANNED_UNKNOWN_USERNAME, _SCANNED_OVER_LENGTH_PASSWORD, None, _PASSWORD_RULES_ERROR_TEXT, id="invalid-password"),
        pytest.param(_SCANNED_UNKNOWN_USERNAME, _SCANNED_PASSWORD, _SCANNED_WRONG_PASSWORD, _PASSWORD_MISMATCH_TEXT, id="mismatched-confirmation"),
        pytest.param("", "", None, _MISSING_INPUT_TEXT, id="empty-form"),
    ],
)
async def test_rejected_registration_writes_one_line_with_the_reason_and_no_submitted_values(
    ui_user: User,
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    app_logs: pytest.LogCaptureFixture,
    username: str,
    password: str,
    confirmation: str | None,
    expected_error: str,
) -> None:
    register(_SCANNED_USERNAME, PASSWORD, conn=db_schema, rounds=TEST_BCRYPT_ROUNDS)
    await _fill_account_form(ui_user, username, password, confirmation)
    start = len(app_logs.records)

    ui_user.find(marker=_CREATE_ACCOUNT_MARKER).click()
    await ui_user.should_see(expected_error)

    lines = _all_auth_lines(app_logs, start)
    assert [line.levelno for line in lines] == [logging.INFO]
    assert expected_error in lines[0].getMessage()
    for submitted in filter(None, (username, password, confirmation)):
        assert submitted.casefold() not in app_logs.text.casefold()


async def test_logs_never_contain_submitted_passwords(
    ui_user: User, app_logs: pytest.LogCaptureFixture
) -> None:
    await _create_account(ui_user, USERNAME, _SCANNED_PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)
    await _log_out(ui_user)
    await sign_in(ui_user, USERNAME, _SCANNED_PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)
    await _log_out(ui_user)
    await sign_in(ui_user, USERNAME, _SCANNED_WRONG_PASSWORD)
    await ui_user.should_see(_SIGN_IN_FAILED_TEXT)
    await sign_in(ui_user, USERNAME, _SCANNED_OVER_LENGTH_PASSWORD)
    await ui_user.should_see(_SIGN_IN_FAILED_TEXT)

    assert app_logs.records
    for password in (_SCANNED_PASSWORD, _SCANNED_WRONG_PASSWORD, _SCANNED_OVER_LENGTH_PASSWORD):
        assert password not in app_logs.text


# ---------------------------------------------------------------------------
# Unhandled page errors
# ---------------------------------------------------------------------------


async def test_unhandled_error_in_an_event_handler_is_logged_with_its_traceback(
    ui_user: User, app_logs: pytest.LogCaptureFixture
) -> None:
    await _create_account(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)

    @ui.page(_FAILING_PAGE_PATH)
    def failing_page() -> None:
        ui.button("Fail", on_click=_raise_sample_error).mark(_FAIL_MARKER)

    await ui_user.open(_FAILING_PAGE_PATH)
    ui_user.find(marker=_FAIL_MARKER).click()

    assert _TRACEBACK_HEADER in app_logs.text
    assert _PAGE_ERROR_TEXT in app_logs.text


async def test_unhandled_error_while_building_a_page_is_logged_with_its_traceback(
    ui_user: User, app_logs: pytest.LogCaptureFixture
) -> None:
    await _create_account(ui_user, USERNAME, PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)

    @ui.page(_FAILING_PAGE_PATH)
    def failing_page() -> None:
        _raise_sample_error()

    # The simulated transport re-raises app errors; a real server answers with a 500 page.
    with pytest.raises(RuntimeError, match=_PAGE_ERROR_TEXT):
        await ui_user.http_client.get(_FAILING_PAGE_PATH)

    assert _TRACEBACK_HEADER in app_logs.text
    assert _PAGE_ERROR_TEXT in app_logs.text
