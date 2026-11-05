"""Home page. This is what a signed-in user sees."""

from nicegui import ui

from civica.ui.auth import HOME_PATH, LOGIN_PATH, current_user, sign_out


def register() -> None:
    """Define the home page route and content."""

    @ui.page(HOME_PATH)
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
        ui.label("Your coach will appear here.") # placeholder
