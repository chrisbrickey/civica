"""NiceGUI web app entrypoint. Run with: uv run python -m civica.ui.app"""

from dotenv import load_dotenv
from nicegui import ui

from civica.ui.deps import default_app_deps
from civica.ui.pages import register_pages
from civica.ui.settings import SESSION_MAX_AGE, session_secret

if __name__ in {"__main__", "__mp_main__"}:  # __mp_main__ is needed for NiceGUI's auto-reload
    load_dotenv()
    register_pages(default_app_deps())
    ui.run(
        storage_secret=session_secret(),
        session_middleware_kwargs={"max_age": SESSION_MAX_AGE},
        title="Civica",
        favicon="📘",
    )
