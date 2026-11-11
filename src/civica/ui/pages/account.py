"""User account pages such as sign in and registration."""

import logging
from enum import StrEnum

from nicegui import ui

from civica.domain.user import UserId
from civica.ui.auth import HOME_PATH, LOGIN_PATH, REGISTER_PATH, sign_in
from civica.ui.deps import AppDeps
from civica.users.rules import SECRET_RULES, USERNAME_RULES, InvalidSecret, InvalidUsername
from civica.users.service import UsernameTaken, normalize_username, verify
from civica.users.service import register as register_user

logger = logging.getLogger(__name__)

USERNAME_LABEL = "username"
PASSWORD_LABEL = "password"
_MISSING_INPUT_MESSAGE = "Enter a username and a password."
_SIGN_IN_FAILED_MESSAGE = "Wrong username or password."
_USERNAME_TAKEN_MESSAGE = "That username is taken."
_PASSWORD_MISMATCH_MESSAGE = "Passwords do not match."


class _Reason(StrEnum):
    BAD_CREDENTIALS = "bad_credentials"
    MISSING_INPUT = "missing_input"
    PASSWORD_MISMATCH = "password_mismatch"
    USERNAME_TAKEN = "username_taken"
    INVALID_USERNAME = "invalid_username"
    INVALID_PASSWORD = "invalid_password"


class _FormRejected(Exception):
    """The form submission failed with a message safe to show the user and a reason code for the logs."""

    def __init__(self, message: str, reason: _Reason) -> None:
        super().__init__(message)
        self.reason = reason


def _require_input(username: str, password: str) -> None:
    if not username.strip() or not password:
        raise _FormRejected(_MISSING_INPUT_MESSAGE, _Reason.MISSING_INPUT)


def _check_credentials(deps: AppDeps, username: str, password: str) -> UserId:
    _require_input(username, password)
    user_id = verify(username, password, deps.conn)
    if user_id is None:
        raise _FormRejected(_SIGN_IN_FAILED_MESSAGE, _Reason.BAD_CREDENTIALS)
    return user_id


def _create_account(deps: AppDeps, username: str, password: str, confirmation: str) -> UserId:
    _require_input(username, password)
    if password != confirmation:
        raise _FormRejected(_PASSWORD_MISMATCH_MESSAGE, _Reason.PASSWORD_MISMATCH)
    try:
        return register_user(username, password, deps.conn, rounds=deps.bcrypt_rounds)
    except UsernameTaken as err:
        raise _FormRejected(_USERNAME_TAKEN_MESSAGE, _Reason.USERNAME_TAKEN) from err
    except InvalidUsername as err:
        raise _FormRejected(str(err), _Reason.INVALID_USERNAME) from err
    except InvalidSecret as err:
        raise _FormRejected(str(err), _Reason.INVALID_PASSWORD) from err


def _complete_sign_in(user_id: UserId, username: str) -> None:
    sign_in(user_id, normalize_username(username))
    ui.navigate.to(HOME_PATH)


def register(deps: AppDeps) -> None:
    """Define the login and register routes and content."""

    @ui.page(LOGIN_PATH)
    def login_page() -> None:
        username = ui.input(USERNAME_LABEL).mark("username")
        password = ui.input(PASSWORD_LABEL, password=True).mark("password")
        error = ui.label().classes("text-negative")

        def submit() -> None:
            try:
                user_id = _check_credentials(deps, username.value, password.value)
            except _FormRejected as err:
                if err.reason is _Reason.BAD_CREDENTIALS:
                    logger.info("sign_in_rejected reason=%s", err.reason)
                error.set_text(str(err))
                return
            logger.info("sign_in user_id=%s", user_id)
            _complete_sign_in(user_id, username.value)

        username.on("keydown.enter", submit)
        password.on("keydown.enter", submit)
        ui.button("Sign in", on_click=submit).mark("sign-in")
        with ui.row().classes("gap-1"):
            ui.label("If you don't have an account,")
            ui.link("register here.", REGISTER_PATH).mark("register-link")

    @ui.page(REGISTER_PATH)
    def register_page() -> None:
        username = ui.input(USERNAME_LABEL).mark("username")
        ui.label(f"Use {USERNAME_RULES}.").classes("text-caption").mark("username-rules")
        password = ui.input(PASSWORD_LABEL, password=True).mark("password")
        confirmation = ui.input("confirm password", password=True).mark("confirm-password")
        ui.label(f"Use {SECRET_RULES}. Avoid common passwords and your username.").classes("text-caption").mark("password-rules")
        error = ui.label().classes("text-negative")

        def submit() -> None:
            try:
                user_id = _create_account(
                    deps, username.value, password.value, confirmation.value
                )
            except _FormRejected as err:
                logger.info("register_rejected reason=%s", err.reason)
                error.set_text(str(err))
                return
            logger.info("register user_id=%s", user_id)
            _complete_sign_in(user_id, username.value)

        ui.button("Create account", on_click=submit).mark("create-account")
        with ui.row().classes("gap-1"):
            ui.label("Already have an account?")
            ui.link("Sign in here.", LOGIN_PATH).mark("sign-in-link")
