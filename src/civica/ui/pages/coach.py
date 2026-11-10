"""The coach: the learner's message and the finished lesson with its sources and questions."""

import logging
import random
from collections.abc import Callable
from string import ascii_uppercase
from typing import Final

from nicegui import run, ui

from civica.domain.user import UserId
from civica.graph.graph import GraphInput, InputKind
from civica.ui.deps import AppDeps
from civica.ui.intro import draw_intro_card
from civica.ui.turn_view import QuizItemView, SourceView, TurnView, build_turn_view, group_by_page
from civica.ui.turns import ActiveTurn, TurnAlreadyRunning, TurnStatus, coach_config

logger = logging.getLogger(__name__)

# Decoration while a turn runs, not real progress: one generic word that names no tool, step, or outside source.
SPINNER_VERBS: Final = (
    "Thinking…",
    "Pondering…",
    "Reflecting…",
    "Composing…",
    "Drafting…",
    "Preparing…",
)

PROMPT_HINT: Final = "Ask for a lesson or a quiz"
_MAX_PROMPT_LENGTH: Final = 1000
_FAILED_TEXT: Final = "Something went wrong. Please try again."
_ALREADY_RUNNING_TEXT: Final = "A turn is already running"
_FALLBACK_NOTE: Final = "This is a fallback lesson."
_SOURCES_TITLE: Final = "Sources for this lesson"
_UNTITLED_SECTION: Final = "Passage"
_QUIZ_TITLE: Final = "Check your comprehension"
_COMPACT_PROPS: Final = "dense dense-toggle switch-toggle-side"
_POLL_SECONDS: Final = 0.1
_VERB_SECONDS: Final = 3.0

# Lesson lines stay readable and headings near body size; the sources sit as a quiet footer in the lesson bubble.
_LESSON_CSS: Final = """
:root {
  --civica-bubble: #81c784; --civica-panel-width: min(56rem, calc(100vw - 48px));
  --civica-bubble-padding: 8px; --civica-hook-width: 40px;
}
.q-message-text--received { color: var(--civica-bubble); }
.q-message-sent { align-self: flex-end; }
/* The lesson bubble and the welcome card share one outer width; the bubble's padding sits inside it. */
.civica-panel { width: var(--civica-panel-width); }
.civica-lesson-body { width: calc(var(--civica-panel-width) - 2 * var(--civica-bubble-padding)); }
/* After a lesson the quiz hangs off it by a hooked line; its right edge meets the bubble's. */
.civica-quiz-thread { margin-top: -8px; gap: 0; flex-wrap: nowrap; }
.civica-quiz-hook {
  flex-shrink: 0; width: 24px; height: 36px; margin-left: 16px;
  border-left: 3px solid var(--civica-bubble); border-bottom: 3px solid var(--civica-bubble); border-radius: 0 0 0 10px;
}
.civica-quiz { width: var(--civica-panel-width); border: 3px solid var(--civica-bubble); }
.civica-quiz-thread .civica-quiz { width: calc(var(--civica-panel-width) - var(--civica-hook-width)); }
.civica-quiz-thread .civica-quiz { margin-top: 18px; }
.nicegui-markdown.civica-lesson h1 { font-size: 1.25rem; line-height: 1.3; font-weight: 500; margin: 0 0 0.4rem; }
.nicegui-markdown.civica-lesson :is(h2, h3, h4, h5, h6) {
  font-size: 1rem; line-height: 1.4; font-weight: 700; margin: 0.75rem 0 0.25rem;
}
.nicegui-markdown.civica-lesson > :first-child { margin-top: 0; }
.civica-sources { margin-top: 10px; border-top: 1px solid rgba(0, 0, 0, 0.18); font-size: 13.5px; }
.civica-sources .q-item { min-height: 0; padding: 3px 0; }
.civica-sources .q-item__section--side { min-width: 0; padding-right: 6px; color: inherit; }
.civica-sources .nicegui-expansion-content { padding: 0 0 0 18px; gap: 0; align-items: stretch; }
.civica-source-page { font-weight: 500; opacity: 0.75; margin-top: 4px; }
.civica-source-passage {
  white-space: pre-line; background: rgba(255, 255, 255, 0.45); border-radius: 4px; padding: 8px 10px; margin: 2px 0 6px;
}
"""


# ---------------------------------------------------------------------------
# Drawing a turn
# ---------------------------------------------------------------------------


def _draw_sources(sources: tuple[SourceView, ...]) -> None:
    """One closed footer; sources grouped under their page, each opening to its passage."""
    pages = group_by_page(sources)
    with ui.expansion(f"{_SOURCES_TITLE} ({len(pages)})").props(_COMPACT_PROPS).classes("w-full civica-sources"):
        for page in pages:
            ui.label(page.page_title).classes("civica-source-page")
            for source in page.sources:
                with ui.expansion(source.section_heading or _UNTITLED_SECTION).props(_COMPACT_PROPS).classes("w-full"):
                    ui.label(source.passage).classes("civica-source-passage")


def _draw_lesson(view: TurnView, lesson_text: str) -> None:
    # One column, so the lesson and its sources share a single bubble.
    with ui.chat_message().mark("coach-lesson"), ui.column().classes("w-full gap-0 civica-lesson-body"):
        ui.markdown(lesson_text).classes("civica-lesson")
        if view.sources:
            _draw_sources(view.sources)


def _draw_quiz_card(items: tuple[QuizItemView, ...]) -> None:
    """All questions in one read-only card, numbered and lettered."""
    with ui.card().classes("civica-quiz").mark("coach-quiz"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("quiz")
            ui.label(_QUIZ_TITLE).classes("text-weight-medium")
        for number, item in enumerate(items, start=1):
            ui.separator()
            ui.label(f"{number}. {item.text}").classes("text-weight-medium")
            for letter, option in zip(ascii_uppercase, item.options):
                ui.label(f"{letter}. {option}")


def _draw_quiz(items: tuple[QuizItemView, ...], after_lesson: bool) -> None:
    """After a lesson the card hangs off it by a hooked line; on its own it has none."""
    if not after_lesson:
        _draw_quiz_card(items)
        return
    with ui.row().classes("civica-quiz-thread"):
        ui.element().classes("civica-quiz-hook").mark("coach-quiz-hook")
        _draw_quiz_card(items)


def _draw_view(view: TurnView) -> None:
    """A saved turn, in generation order: message, lesson, questions, reply, fallback note."""
    ui.chat_message(view.learner_message, sent=True)
    if view.notice is not None:
        ui.label(view.notice).classes("text-negative")
        return
    if view.lesson_text is not None:
        _draw_lesson(view, view.lesson_text)
    if view.quiz_items:
        _draw_quiz(view.quiz_items, after_lesson=view.lesson_text is not None)
    if view.reply:
        with ui.chat_message():
            ui.markdown(view.reply)
    if view.is_fallback:
        ui.label(_FALLBACK_NOTE).classes("text-caption")


def _draw_running(turn: ActiveTurn) -> None:
    """The learner's message and a spinner whose verb changes every few seconds."""
    ui.chat_message(turn.learner_message, sent=True)
    with ui.row().classes("items-center gap-2"):
        ui.spinner()
        verb = ui.label(random.choice(SPINNER_VERBS)).mark("spinner-verb")

    def next_verb() -> None:
        verb.set_text(random.choice([text for text in SPINNER_VERBS if text != verb.text]))

    ui.timer(_VERB_SECONDS, next_verb)


def _draw_failed(turn: ActiveTurn) -> None:
    ui.chat_message(turn.learner_message, sent=True)
    ui.label(_FAILED_TEXT).classes("text-negative")


def _draw_read_failed() -> None:
    ui.label(_FAILED_TEXT).classes("text-negative")


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


class _CoachPage:
    """One page visit: draws the saved turn and follows any live turn of the user, from this tab or another."""

    def __init__(self, deps: AppDeps, user_id: UserId) -> None:
        self._deps = deps
        self._user_id = user_id
        self._followed: ActiveTurn | None = None
        self._drawn_running = False

        ui.add_css(_LESSON_CSS)
        self._turn_area = ui.column().classes("w-full").mark("coach-chat")
        # The prompt is page content right after the turn area, not a footer.
        with ui.row().classes("w-full items-center no-wrap"):
            self._prompt = ui.input(placeholder=PROMPT_HINT).props(f"maxlength={_MAX_PROMPT_LENGTH}").classes("grow")
            self._prompt.mark("prompt-input")
            self._send = ui.button(icon="send").mark("prompt-submit")
        self._send.on_click(self._submit)
        self._prompt.on("keydown.enter", self._submit)

    async def open(self) -> None:
        """Draw what the thread holds now, then keep following the user's turns."""
        live = self._deps.turn_runner.active(self._user_id)
        if live is not None:
            self._follow(live)
            found = "running turn"
        else:
            found = await self._show_saved_turn()
        logger.info("Coach page opened for user %s: %s", self._user_id, found)
        ui.timer(_POLL_SECONDS, self._tick)

    def _show(self, draw: Callable[[], None]) -> None:
        self._turn_area.clear()
        with self._turn_area:
            draw()

    def _set_locked(self, locked: bool) -> None:
        self._prompt.set_enabled(not locked)
        self._send.set_enabled(not locked)

    def _follow(self, turn: ActiveTurn) -> None:
        self._followed, self._drawn_running = turn, False
        self._set_locked(True)

    def _submit(self) -> None:
        text = self._prompt.value.strip()
        if not text or self._followed is not None:
            return
        graph_input: GraphInput = {"kind": InputKind.MESSAGE, "learner_message": text}
        try:
            turn = self._deps.turn_runner.start(self._user_id, graph_input)
        except TurnAlreadyRunning:
            ui.notify(_ALREADY_RUNNING_TEXT)
            return
        self._prompt.set_value("")
        self._follow(turn)

    async def _tick(self) -> None:
        live = self._deps.turn_runner.active(self._user_id)
        if live is not None and live is not self._followed:
            self._follow(live)
        turn = self._followed
        if turn is None:
            return
        if live is not None:
            if not self._drawn_running:
                self._drawn_running = True
                self._show(lambda: _draw_running(turn))
            return
        # The turn has ended: let go of it and show the outcome.
        self._followed = None
        if turn.snapshot().status is TurnStatus.FAILED:
            self._show(lambda: _draw_failed(turn))
        else:
            await self._show_saved_turn()
        self._set_locked(False)

    async def _show_saved_turn(self) -> str:
        """Draw the turn saved on the thread and say what was found."""
        try:
            snapshot = await run.io_bound(self._deps.graph.get_state, coach_config(self._user_id))
            view = build_turn_view(snapshot) if snapshot is not None else None
        except Exception:
            logger.exception("Could not read the coach turn for user %s", self._user_id)
            self._show(_draw_read_failed)
            return "read failed"
        if view is None:
            self._show(draw_intro_card)
            return "no turn yet"
        self._show(lambda: _draw_view(view))
        return "unfinished turn" if view.notice is not None else "saved turn"


async def render(deps: AppDeps, user_id: UserId) -> None:
    """Build the coach for one page visit."""
    await _CoachPage(deps, user_id).open()
