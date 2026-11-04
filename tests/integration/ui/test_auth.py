"""
Integration tests for the web app authentication.

NiceGUI's User fixture simulates the browser in Python (no real browser).
Register and sign-in run through the real users.service against the per-test schema.
"""

import nicegui
import psycopg
import psycopg.rows
from nicegui.testing import User

from civica.users.service import register
from tests.support.users import TEST_BCRYPT_ROUNDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USERNAME = "test-user"
_PASSWORD = "sample-password"
_WRONG_PASSWORD = "wrong-password"
_OTHER_PASSWORD = "other-password"
_UNKNOWN_USERNAME = "nobody-registered"

# One under the 8-character minimum, and one byte past bcrypt's 72-byte limit.
_UNDER_LENGTH_PASSWORD = "a" * 7
_OVER_LENGTH_PASSWORD = "a" * 73

_SIGNED_IN_TEXT = f"Signed in as {_USERNAME}"
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

_USERNAME_MARKER = "username"
_PASSWORD_MARKER = "password"
_CONFIRM_PASSWORD_MARKER = "confirm-password"
_SIGN_IN_MARKER = "sign-in"
_REGISTER_LINK_MARKER = "register-link"
_SIGN_IN_LINK_MARKER = "sign-in-link"
_CREATE_ACCOUNT_MARKER = "create-account"
_PASSWORD_RULES_MARKER = "password-rules"
_USERNAME_RULES_MARKER = "username-rules"
_LOGOUT_MARKER = "logout"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_users(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> int:
    result = conn.execute("SELECT COUNT(*) FROM users").fetchone()
    assert result is not None
    return int(result[0])


def _register_directly(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> None:
    register(_USERNAME, _PASSWORD, conn=conn, rounds=TEST_BCRYPT_ROUNDS)


async def _redirect_location(user: User, path: str) -> str | None:
    """Where a plain GET of path redirects this browser, or None when it is served."""
    response = await user.http_client.get(path, follow_redirects=False)
    return response.headers.get("location") if response.is_redirect else None


async def _fill_sign_in_form(user: User, username: str, password: str) -> None:
    await user.open("/login")
    user.find(marker=_USERNAME_MARKER).type(username)
    user.find(marker=_PASSWORD_MARKER).type(password)


async def _sign_in(user: User, username: str, password: str) -> None:
    await _fill_sign_in_form(user, username, password)
    user.find(marker=_SIGN_IN_MARKER).click()


async def _create_account(
    user: User, username: str, password: str, confirmation: str | None = None
) -> None:
    await user.open("/register")
    user.find(marker=_USERNAME_MARKER).type(username)
    user.find(marker=_PASSWORD_MARKER).type(password)
    user.find(marker=_CONFIRM_PASSWORD_MARKER).type(
        password if confirmation is None else confirmation
    )
    user.find(marker=_CREATE_ACCOUNT_MARKER).click()


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


async def test_unauthenticated_visit_redirects_to_login(ui_user: User) -> None:
    assert await _redirect_location(ui_user, "/") == "/login"

    await ui_user.open("/")
    await ui_user.should_see(marker=_SIGN_IN_MARKER)


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

    await _sign_in(ui_user, _USERNAME, _PASSWORD)

    await ui_user.should_see(_SIGNED_IN_TEXT)


async def test_enter_key_signs_in(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    _register_directly(db_schema)

    await _fill_sign_in_form(ui_user, _USERNAME, _PASSWORD)
    ui_user.find(marker=_PASSWORD_MARKER).trigger("keydown.enter")

    await ui_user.should_see(_SIGNED_IN_TEXT)


async def test_sign_in_with_wrong_password_shows_error_and_stays_on_login(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    _register_directly(db_schema)

    await _sign_in(ui_user, _USERNAME, _WRONG_PASSWORD)

    await ui_user.should_see(_SIGN_IN_FAILED_TEXT)
    await ui_user.should_not_see(_SIGNED_IN_TEXT)
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_sign_in_with_unknown_username_shows_error_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _sign_in(ui_user, _UNKNOWN_USERNAME, _PASSWORD)

    await ui_user.should_see(_SIGN_IN_FAILED_TEXT)
    assert _count_users(db_schema) == 0
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_sign_in_with_over_length_password_shows_error(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    _register_directly(db_schema)

    await _sign_in(ui_user, _USERNAME, _OVER_LENGTH_PASSWORD)

    await ui_user.should_see(_SIGN_IN_FAILED_TEXT)
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_sign_in_with_empty_input_shows_error(ui_user: User) -> None:
    await _sign_in(ui_user, "", "")

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
    await _create_account(ui_user, _INVALID_USERNAME, _PASSWORD)

    await ui_user.should_see(_USERNAME_RULES_ERROR_TEXT)
    assert _count_users(db_schema) == 0


async def test_common_password_shows_error_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, _USERNAME, _COMMON_PASSWORD)

    await ui_user.should_see(_COMMON_PASSWORD_TEXT)
    assert _count_users(db_schema) == 0


async def test_sign_in_link_returns_to_sign_in_page(ui_user: User) -> None:
    await ui_user.open("/register")

    ui_user.find(marker=_SIGN_IN_LINK_MARKER).click()

    await ui_user.should_see(marker=_SIGN_IN_MARKER)


async def test_create_account_registers_and_signs_in(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, _USERNAME, _PASSWORD)

    await ui_user.should_see(_SIGNED_IN_TEXT)
    assert _count_users(db_schema) == 1


async def test_create_account_with_taken_username_shows_error(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    _register_directly(db_schema)

    await _create_account(ui_user, _USERNAME, _OTHER_PASSWORD)

    await ui_user.should_see(_USERNAME_TAKEN_TEXT)
    assert _count_users(db_schema) == 1
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_mismatched_confirmation_shows_error_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, _USERNAME, _PASSWORD, confirmation=_OTHER_PASSWORD)

    await ui_user.should_see(_PASSWORD_MISMATCH_TEXT)
    assert _count_users(db_schema) == 0


async def test_under_length_password_shows_rules_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, _USERNAME, _UNDER_LENGTH_PASSWORD)

    await ui_user.should_see(_PASSWORD_RULES_ERROR_TEXT)
    assert _count_users(db_schema) == 0


async def test_over_length_password_shows_rules_and_creates_no_user(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, _USERNAME, _OVER_LENGTH_PASSWORD)

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
    await _create_account(ui_user, _USERNAME, _PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)

    await ui_user.open("/")

    await ui_user.should_see(_SIGNED_IN_TEXT)
    await ui_user.should_not_see(marker=_SIGN_IN_MARKER)


async def test_logout_shows_login_and_blocks_signed_in_page(ui_user: User) -> None:
    await _create_account(ui_user, _USERNAME, _PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)

    ui_user.find(marker=_LOGOUT_MARKER).click()

    await ui_user.should_see(marker=_SIGN_IN_MARKER)
    assert await _redirect_location(ui_user, "/") == "/login"


async def test_signing_back_in_reaches_the_same_account(
    ui_user: User, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    await _create_account(ui_user, _USERNAME, _PASSWORD)
    await ui_user.should_see(_SIGNED_IN_TEXT)
    ui_user.find(marker=_LOGOUT_MARKER).click()
    await ui_user.should_see(marker=_SIGN_IN_MARKER)

    await _sign_in(ui_user, _USERNAME, _PASSWORD)

    await ui_user.should_see(_SIGNED_IN_TEXT)
    assert _count_users(db_schema) == 1
