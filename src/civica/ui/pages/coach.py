"""The coach: the learner's message and the finished lesson with its sources and questions."""

import logging
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

_HINT: Final = "Ask for a lesson or a quiz"
_MAX_PROMPT_LENGTH: Final = 1000
_FAILED_TEXT: Final = "Something went wrong. Please try again."
_ALREADY_RUNNING_TEXT: Final = "A turn is already running"
_POLL_SECONDS: Final = 0.1
_SOURCES_TITLE: Final = "Sources for this lesson"
_UNTITLED_SECTION: Final = "Passage"
_COMPACT_PROPS: Final = "dense dense-toggle switch-toggle-side"
_QUIZ_TITLE: Final = "Check your comprehension"

# Lesson lines stay readable and headings near body size; the sources sit as a quiet footer in the lesson bubble.
_LESSON_CSS: Final = """
:root { --civica-bubble: #81c784; --civica-panel-width: min(56rem, calc(100vw - 48px)); }
.q-message-text--received { color: var(--civica-bubble); }
.q-message-sent { align-self: flex-end; }
.civica-lesson-body { width: var(--civica-panel-width); }
/* The quiz hangs off the lesson by a hooked line; its right edge meets the bubble's (content plus 8px padding a side). */
.civica-quiz-thread { margin-top: -8px; gap: 0; flex-wrap: nowrap; }
.civica-quiz-connector {
  flex-shrink: 0; width: 24px; height: 36px; margin-left: 16px;
  border-left: 3px solid var(--civica-bubble); border-bottom: 3px solid var(--civica-bubble); border-radius: 0 0 0 10px;
}
.civica-quiz { width: calc(var(--civica-panel-width) - 24px); margin-top: 18px; border: 3px solid var(--civica-bubble); }
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


def _draw_sources(sources: tuple[SourceView, ...]) -> None:
    """One closed footer; sources grouped under their page, each opening to its original passage."""
    title = f"{_SOURCES_TITLE} ({len(sources)})"
    with ui.expansion(title).props(_COMPACT_PROPS).classes("w-full civica-sources"):
        for page in group_by_page(sources):
            ui.label(page.page_title).classes("civica-source-page")
            for source in page.sources:
                with ui.expansion(source.section_heading or _UNTITLED_SECTION).props(_COMPACT_PROPS).classes("w-full"):
                    ui.label(source.passage).classes("civica-source-passage")


def _draw_quiz(items: tuple[QuizItemView, ...]) -> None:
    """One labelled panel for all questions; options are plain text, numbered and lettered."""
    with ui.row().classes("civica-quiz-thread"):
        ui.element().classes("civica-quiz-connector")
        with ui.card().classes("civica-quiz").mark("coach-quiz"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("quiz")
                ui.label(_QUIZ_TITLE).classes("text-weight-medium")
            for number, item in enumerate(items, start=1):
                ui.separator()
                ui.label(f"{number}. {item.text}").classes("text-weight-medium")
                for letter, option in zip(ascii_uppercase, item.options):
                    ui.label(f"{letter}. {option}")


def _draw_view(view: TurnView) -> None:
    """A finished turn, in generation order: message, lesson, sources, questions, reply."""
    ui.chat_message(view.learner_message, sent=True)
    if view.notice is not None:
        ui.label(view.notice).classes("text-negative")
        return
    if view.lesson_text is not None:
        # One column, so the lesson and its sources share a single bubble.
        with ui.chat_message(), ui.column().classes("w-full gap-0 civica-lesson-body"):
            ui.markdown(view.lesson_text).classes("civica-lesson")
            if view.sources:
                _draw_sources(view.sources)
    if view.quiz_items:
        _draw_quiz(view.quiz_items)
    if view.reply:
        with ui.chat_message():
            ui.markdown(view.reply)
    if view.is_fallback:
        ui.label("This is a fallback lesson.").classes("text-caption")


def _draw_live(turn: ActiveTurn) -> None:
    ui.chat_message(turn.learner_message, sent=True)
    ui.spinner()


def _draw_failed(turn: ActiveTurn) -> None:
    ui.chat_message(turn.learner_message, sent=True)
    ui.label(_FAILED_TEXT).classes("text-negative")


def _draw_read_failed() -> None:
    ui.label(_FAILED_TEXT).classes("text-negative")



# TODO: break up this monster method. Extract the major pieces into private methods.
async def render(deps: AppDeps, user_id: UserId) -> None:
    """Build the coach for one page visit and follow any turn of this user."""
    config = coach_config(user_id)
    ui.add_css(_LESSON_CSS)
    turn_area = ui.column().classes("w-full").mark("coach-chat")

    def show(draw: Callable[[], None]) -> None:
        turn_area.clear()
        with turn_area:
            draw()

    async def read_view() -> TurnView | None:
        snapshot = await run.io_bound(deps.graph.get_state, config)
        assert snapshot is not None
        return build_turn_view(snapshot)

    # The prompt is page content right after the turn area, not a footer.
    with ui.row().classes("w-full items-center no-wrap"):
        prompt = ui.input(placeholder=_HINT).props(f"maxlength={_MAX_PROMPT_LENGTH}").classes("grow")
        prompt.mark("prompt-input")
        send = ui.button(icon="send").mark("prompt-submit")

    def set_locked(locked: bool) -> None:
        prompt.set_enabled(not locked)
        send.set_enabled(not locked)

    followed: ActiveTurn | None = None
    drawn: TurnStatus | None = None

    def follow(turn: ActiveTurn) -> None:
        nonlocal followed, drawn
        followed, drawn = turn, None
        set_locked(True)

    def submit() -> None:
        text = prompt.value.strip()
        if not text:
            return
        graph_input: GraphInput = {"kind": InputKind.MESSAGE, "learner_message": text}
        try:
            turn = deps.turn_runner.start(user_id, graph_input)
        except TurnAlreadyRunning:
            ui.notify(_ALREADY_RUNNING_TEXT)
            return
        prompt.set_value("")
        follow(turn)

    async def tick() -> None:
        nonlocal followed, drawn
        live = deps.turn_runner.active(user_id)
        if live is not None and live is not followed:
            follow(live)
        turn = followed
        if turn is None:
            return
        snapshot = turn.snapshot()
        if live is not None:
            if snapshot.status != drawn:
                drawn = snapshot.status
                show(lambda: _draw_live(turn))
            return
        # The turn has ended: this page lets go of it and shows the outcome.
        followed = None
        if snapshot.status is TurnStatus.FAILED:
            show(lambda: _draw_failed(turn))
        else:
            await show_saved_turn()
        set_locked(False)

    async def show_saved_turn() -> str:
        """Draw the turn saved on the thread and say what was found."""
        try:
            view = await read_view()
        except Exception:
            logger.exception("Could not read the coach turn for user %s", user_id)
            show(_draw_read_failed)
            return "read failed"
        if view is None:
            show(draw_intro_card)
            return "no turn yet"
        show(lambda: _draw_view(view))
        return "unfinished turn" if view.notice is not None else "saved turn"

    send.on_click(submit)
    prompt.on("keydown.enter", submit)

    live = deps.turn_runner.active(user_id)
    if live is not None:
        follow(live)
        found = "running turn"
    else:
        found = await show_saved_turn()
    logger.info("Coach page opened for user %s: %s", user_id, found)
    ui.timer(_POLL_SECONDS, tick)
