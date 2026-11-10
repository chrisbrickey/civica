"""Home page. This is what a signed-in user sees."""

import logging

from nicegui import ui

from civica.ui.auth import HOME_PATH, LOGIN_PATH, current_user, sign_out
from civica.ui.deps import AppDeps
from civica.ui.intro import build_intro_dialog_button
from civica.ui.pages import coach

logger = logging.getLogger(__name__)


def register(deps: AppDeps) -> None:
    """Define the home page route and content."""

    @ui.page(HOME_PATH)
    async def home_page() -> None:
        user = current_user()
        if user is None:
            ui.navigate.to(LOGIN_PATH)
            return

        def log_out() -> None:
            logger.info("User %s signed out", user.user_id)
            sign_out()
            ui.navigate.to(LOGIN_PATH)

        with ui.header().classes("items-center justify-between"):
            ui.label(f"Signed in as {user.username}")
            with ui.row().classes("items-center"):
                build_intro_dialog_button()
                ui.button("Log out", on_click=log_out).mark("logout")
        await coach.render(deps, user.user_id)
