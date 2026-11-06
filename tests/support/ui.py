"""Credentials, markers, and browser actions shared by the web app tests."""

from nicegui.testing import User

USERNAME = "test-user"
PASSWORD = "sample-password"

USERNAME_MARKER = "username"
PASSWORD_MARKER = "password"
SIGN_IN_MARKER = "sign-in"

PROMPT_INPUT_MARKER = "prompt-input"
SUBMIT_MARKER = "prompt-submit"
CHAT_MARKER = "coach-chat"
QUIZ_MARKER = "coach-quiz"

INTRO_CARD_MARKER = "intro-card"
INTRO_OPEN_MARKER = "intro-open"
INTRO_DIALOG_MARKER = "intro-dialog"
INTRO_CLOSE_MARKER = "intro-close"


async def fill_sign_in_form(user: User, username: str, password: str) -> None:
    await user.open("/login")
    user.find(marker=USERNAME_MARKER).type(username)
    user.find(marker=PASSWORD_MARKER).type(password)


async def sign_in(user: User, username: str, password: str) -> None:
    await fill_sign_in_form(user, username, password)
    user.find(marker=SIGN_IN_MARKER).click()
