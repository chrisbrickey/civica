"""Credentials, markers, and browser actions shared by the web app tests."""

from nicegui.storage import request_contextvar
from nicegui.testing import User

from civica.domain.user import UserId
from civica.ui import auth

USERNAME = "test-user"
PASSWORD = "sample-password"

USERNAME_MARKER = "username"
PASSWORD_MARKER = "password"
SIGN_IN_MARKER = "sign-in"
LOGOUT_MARKER = "logout"

PROMPT_INPUT_MARKER = "prompt-input"
SUBMIT_MARKER = "prompt-submit"
CHAT_MARKER = "coach-chat"
LESSON_MARKER = "coach-lesson"
QUIZ_MARKER = "coach-quiz"
QUIZ_HOOK_MARKER = "coach-quiz-hook"
SPINNER_VERB_MARKER = "spinner-verb"

TEACH_ME_MARKER = "teach-me"
QUIZ_ME_MARKER = "quiz-me"
# One Check button per question, and the element(s) showing a question's check result.
QUIZ_CHECK_MARKER = "quiz-check"
QUIZ_RESULT_MARKER = "quiz-result"

INTRO_CARD_MARKER = "intro-card"
INTRO_OPEN_MARKER = "intro-open"
INTRO_DIALOG_MARKER = "intro-dialog"
INTRO_CLOSE_MARKER = "intro-close"


async def fill_sign_in_form(user: User, username: str, password: str) -> None:
    await user.open(auth.LOGIN_PATH)
    user.find(marker=USERNAME_MARKER).type(username)
    user.find(marker=PASSWORD_MARKER).type(password)


async def sign_in(user: User, username: str, password: str) -> None:
    await fill_sign_in_form(user, username, password)
    user.find(marker=SIGN_IN_MARKER).click()


async def open_signed_in(user: User, user_id: UserId, username: str = USERNAME) -> None:
    """Sign the browser in through session storage, skipping the form, then open the home page."""
    client = await user.open(auth.LOGIN_PATH)
    token = request_contextvar.set(client.request)
    try:
        auth.sign_in(user_id, username)
    finally:
        request_contextvar.reset(token)
    await user.open(auth.HOME_PATH)
