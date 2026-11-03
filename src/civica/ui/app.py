"""NiceGUI web app entrypoint. Run with: uv run python -m civica.ui.app"""

from dotenv import load_dotenv
from nicegui import ui

from civica.domain.user import UserId
from civica.ui.auth import (
    LOGIN_PATH,
    REGISTER_PATH,
    current_user,
    install_auth_middleware,
    sign_in,
    sign_out,
)
from civica.ui.deps import AppDeps, default_app_deps
from civica.ui.settings import SESSION_MAX_AGE, session_secret
from civica.users.rules import SECRET_RULES, USERNAME_RULES, InvalidSecret, InvalidUsername
from civica.users.service import UsernameTaken, normalize_username, register, verify

_HOME_PATH = "/"
_MISSING_INPUT_MESSAGE = "Enter a username and a password."
_SIGN_IN_FAILED_MESSAGE = "Wrong username or password."
_USERNAME_TAKEN_MESSAGE = "That username is taken."
_PASSWORD_MISMATCH_MESSAGE = "Passwords do not match."


class _LoginRejected(Exception):
    """The login attempt failed with a message safe to show the user."""


def _require_input(username: str, password: str) -> None:
    if not username.strip() or not password:
        raise _LoginRejected(_MISSING_INPUT_MESSAGE)


def _sign_in(deps: AppDeps, username: str, password: str) -> UserId:
    _require_input(username, password)
    try:
        user_id = verify(username, password, deps.conn)
    except ValueError as err:
        raise _LoginRejected(_SIGN_IN_FAILED_MESSAGE) from err
    if user_id is None:
        raise _LoginRejected(_SIGN_IN_FAILED_MESSAGE)
    return user_id


def _create_account(deps: AppDeps, username: str, password: str, confirmation: str) -> UserId:
    _require_input(username, password)
    if password != confirmation:
        raise _LoginRejected(_PASSWORD_MISMATCH_MESSAGE)
    try:
        return register(username, password, deps.conn, rounds=deps.bcrypt_rounds)
    except UsernameTaken as err:
        raise _LoginRejected(_USERNAME_TAKEN_MESSAGE) from err
    except (InvalidUsername, InvalidSecret) as err:
        raise _LoginRejected(str(err)) from err


def _complete_sign_in(user_id: UserId, username: str) -> None:
    sign_in(user_id, normalize_username(username))
    ui.navigate.to(_HOME_PATH)


def register_pages(deps: AppDeps) -> None:
    """Install auth and define the app's pages. Tests call this with their own deps."""
    install_auth_middleware()

    @ui.page(LOGIN_PATH)
    def login_page() -> None:
        username = ui.input("username").mark("username")
        password = ui.input("password", password=True).mark("password")
        error = ui.label().classes("text-negative")

        def submit() -> None:
            try:
                user_id = _sign_in(deps, username.value, password.value)
            except _LoginRejected as err:
                error.set_text(str(err))
                return
            _complete_sign_in(user_id, username.value)

        username.on("keydown.enter", submit)
        password.on("keydown.enter", submit)
        ui.button("Sign in", on_click=submit).mark("sign-in")
        with ui.row().classes("gap-1"):
            ui.label("If you don't have an account,")
            ui.link("register here.", REGISTER_PATH).mark("register-link")

    @ui.page(REGISTER_PATH)
    def register_page() -> None:
        username = ui.input("username").mark("username")
        ui.label(f"Use {USERNAME_RULES}.").classes("text-caption").mark("username-rules")
        password = ui.input("password", password=True).mark("password")
        confirmation = ui.input("confirm password", password=True).mark("confirm-password")
        ui.label(f"Use {SECRET_RULES}. Avoid common passwords and your username.").classes("text-caption").mark("password-rules")
        error = ui.label().classes("text-negative")

        def submit() -> None:
            try:
                user_id = _create_account(
                    deps, username.value, password.value, confirmation.value
                )
            except _LoginRejected as err:
                error.set_text(str(err))
                return
            _complete_sign_in(user_id, username.value)

        ui.button("Create account", on_click=submit).mark("create-account")
        with ui.row().classes("gap-1"):
            ui.label("Already have an account?")
            ui.link("Sign in here.", LOGIN_PATH).mark("sign-in-link")

    @ui.page(_HOME_PATH)
    def home_page() -> None:
        user = current_user()
        if user is None:
            ui.navigate.to(LOGIN_PATH)
            return

        def log_out() -> None:
            sign_out()
            ui.navigate.to(LOGIN_PATH)

        ui.label(f"Signed in as {user.username}")
        ui.button("Log out", on_click=log_out).mark("logout")
        # Placeholder until Step 12 adds the coach.
        ui.label("Your coach will appear here.")


if __name__ in {"__main__", "__mp_main__"}:  # __mp_main__ is needed for NiceGUI's auto-reload
    load_dotenv()
    register_pages(default_app_deps())
    ui.run(
        storage_secret=session_secret(),
        session_middleware_kwargs={"max_age": SESSION_MAX_AGE},
        title="Civica",
        favicon="📘",
    )
