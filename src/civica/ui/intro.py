"""The welcome text for new learners, shown as a card and reopened from the header in a dialog."""

from typing import Final

from nicegui import ui

from civica.domain.themes import THEMES_BY_SLUG

INTRO_TITLE: Final = "Bienvenue sur Civica · Welcome to Civica"
_ABOUT: Final = (
    "Civica is an agentic coach for the French naturalization civics exam. "
    "Its lessons are grounded in the official study material published by le Ministère de l’Intérieur."
)
_THEMES_HEADING: Final = "The five official themes of the exam (as of 2026)"
_ASK_HEADING: Final = "How to use Civica"
_ASK: Final = "You can ask about specific subjects or you can ask for general coaching. The coach tracks your progress and mastery to ensure that you are learning all five themes."
_SCOPE: Final = "Civica only covers the civics exam. It is not intended for general chat."
_EXAMPLES_HEADING: Final = "Try one of these"
_EXAMPLES: Final = (
    "Start coaching me for this exam",
    "Teach me about the Republic's values",
    "Quiz me on Droits et devoirs",
)

_CLOSE_TEXT: Final = "Fermer / Close"
_OPEN_TOOLTIP: Final = "À propos de Civica / About Civica"


def _bullets(items: list[str] | tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def draw_intro() -> None:
    """Draw the intro text; the card and the dialog share it."""
    themes = [theme.display_name_fr for theme in THEMES_BY_SLUG.values()]
    ui.label(INTRO_TITLE).classes("text-h6")
    ui.markdown(_ABOUT)
    ui.markdown(f"**{_THEMES_HEADING}**\n\n{_bullets(themes)}")
    ui.markdown(f"**{_ASK_HEADING}**\n\n{_ASK}\n\n*{_SCOPE}*")
    ui.markdown(f"**{_EXAMPLES_HEADING}**\n\n{_bullets(_EXAMPLES)}")


def draw_intro_card() -> None:
    """Draw the intro as a card in the turn area."""
    with ui.card().classes("w-full").mark("intro-card"):
        draw_intro()


def build_intro_dialog_button() -> None:
    """Add a closed intro dialog and the button that opens it."""
    with ui.dialog().mark("intro-dialog") as dialog, ui.card():
        draw_intro()
        ui.button(_CLOSE_TEXT, on_click=dialog.close).props("flat").mark("intro-close")
    button = ui.button(icon="help_outline", on_click=dialog.open).props("flat round color=white").mark("intro-open")
    with button:
        ui.tooltip(_OPEN_TOOLTIP)
