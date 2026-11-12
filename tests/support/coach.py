"""Planner scripts, page lookups, and waits shared by the coach page tests."""

import asyncio
import re
from collections.abc import Callable, Iterable
from typing import Final

import pytest
from langchain_core.messages import AIMessage
from nicegui import ElementFilter, ui
from nicegui.elements.mixins.text_element import TextElement
from nicegui.testing import User, UserInteraction

from civica.domain.user import UserId
from civica.ui.turns import TurnRunner
from tests.support.assessment import (
    SAMPLE_OPTIONS,
    FakeChatModel,
    default_generation_reply,
    generated_item_text,
    make_question,
)
from tests.support.logs import LogLine, lines_from
from tests.support.planner import DEFAULT_THEME, ControlledPlannerModel, final_message, tool_call
from tests.support.ui import (
    PROMPT_INPUT_MARKER,
    QUIZ_CHECK_MARKER,
    QUIZ_MARKER,
    QUIZ_RESULT_MARKER,
    SUBMIT_MARKER,
)

PROMPT: Final = "sample-prompt-001"
SECOND_PROMPT: Final = "sample-prompt-002"
LESSON_TEXT: Final = "sample-lesson-text-001"
SECOND_LESSON_TEXT: Final = "sample-lesson-text-002"
FALLBACK_TEXT: Final = "sample-fallback-lesson-text"
REPLY: Final = "sample-final-reply-001"
SECOND_REPLY: Final = "sample-final-reply-002"
FOCUS: Final = "sample-focus"

READ_STATE: Final = "read_learner_state"
TEACH: Final = "teach"
GENERATE_QUIZ: Final = "generate_quiz"
SUGGEST_THEME: Final = "suggest_review_theme"

TEACH_ME_TEXT: Final = "Teach me"
QUIZ_ME_TEXT: Final = "Quiz me"
PRACTICE_LABEL: Final = "not scored or saved yet"

COACH_LOGGER: Final = "civica.ui.pages.coach"

FAILED_TEXT: Final = "Something went wrong. Please try again."
UNFINISHED_TEXT: Final = "Your last request didn't finish. Please try again."

# The first generate_quiz call follows the one teach call on the content model.
QUIZ_AFTER_LESSON_CALL: Final = 1
# A quiz-only turn's generate_quiz call is the content model's first.
QUIZ_ONLY_CALL: Final = 0

# How the planner calls of `lesson_script` line up: hold the last one to pause mid-turn.
FINAL_CALL: Final = 2
CALLS_PER_LESSON: Final = 3

# should_see polls every 0.1 seconds, so a turn has up to 5 seconds to reach the page.
TURN_RETRIES: Final = 50
# Upper bound on a turn; ActiveTurn.wait returns as soon as it ends.
TURN_TIMEOUT_SECONDS: Final = 5.0
# Page state such as an enabled input has no event to await, so it is polled briefly.
_POLL_SECONDS: Final = 0.01
_POLL_LIMIT: Final = 500


def lesson_script(reply: str = REPLY, call_suffix: str = "001") -> tuple[AIMessage, ...]:
    return (
        tool_call(READ_STATE, f"call-read-{call_suffix}"),
        tool_call(TEACH, f"call-teach-{call_suffix}", theme=DEFAULT_THEME.slug, focus=FOCUS),
        final_message(reply),
    )


def script_lesson(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    planner_model.script = lesson_script()
    content_model.replies = (LESSON_TEXT,)


def script_lesson_with_quiz(
    planner_model: ControlledPlannerModel, content_model: FakeChatModel, question_count: int = 1
) -> None:
    planner_model.script = (
        tool_call(TEACH, "call-teach", theme=DEFAULT_THEME.slug, focus=FOCUS),
        tool_call(GENERATE_QUIZ, "call-quiz", theme=DEFAULT_THEME.slug, n=question_count),
        final_message(REPLY),
    )
    content_model.replies = (LESSON_TEXT,)


def script_quiz_only(planner_model: ControlledPlannerModel, question_count: int = 1) -> None:
    planner_model.script = (
        tool_call(GENERATE_QUIZ, "call-quiz", theme=DEFAULT_THEME.slug, n=question_count),
        final_message(REPLY),
    )


def script_quiz_then_lesson(
    planner_model: ControlledPlannerModel, content_model: FakeChatModel, question_count: int = 1
) -> None:
    """A quiz-only turn, then a lesson turn."""
    script_quiz_only(planner_model, question_count)
    planner_model.script = (*planner_model.script, *lesson_script(SECOND_REPLY, "002"))
    content_model.replies = (default_generation_reply(QUIZ_ONLY_CALL), LESSON_TEXT)


def script_reply(planner_model: ControlledPlannerModel) -> None:
    planner_model.script = (final_message(REPLY),)


def script_two_lessons(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    planner_model.script = (*lesson_script(), *lesson_script(SECOND_REPLY, "002"))
    content_model.replies = (LESSON_TEXT, SECOND_LESSON_TEXT)


def script_fallback(planner_model: ControlledPlannerModel, content_model: FakeChatModel) -> None:
    """A planner that never stops calling tools, so the step limit serves the fallback lesson."""
    planner_model.repeat = tool_call(SUGGEST_THEME, "call-forever")
    content_model.replies = (FALLBACK_TEXT,)


def script_failure(planner_model: ControlledPlannerModel) -> None:
    """A lesson whose second planner call raises."""
    planner_model.script = lesson_script()
    planner_model.fail_at_call = 1


def question_texts(call_index: int, question_count: int) -> tuple[str, ...]:
    return tuple(generated_item_text(call_index, index) for index in range(question_count))


def question_id_of(text: str) -> str:
    """The question_id of the question with this text."""
    return make_question(text=text).question_id


def type_prompt(user: User, prompt: str) -> None:
    user.find(marker=PROMPT_INPUT_MARKER).type(prompt)


def click_submit(user: User) -> None:
    user.find(marker=SUBMIT_MARKER).click()


def press_enter(user: User) -> None:
    user.find(marker=PROMPT_INPUT_MARKER).trigger("keydown.enter")


def submit(user: User, prompt: str = PROMPT) -> None:
    type_prompt(user, prompt)
    click_submit(user)


def the(user: User, marker: str) -> ui.element:
    """The one element carrying `marker`."""
    marked = list(user.find(marker=marker).elements)
    assert len(marked) == 1, f"expected one element marked {marker!r}, found {len(marked)}"
    return marked[0]


def prompt_input(user: User) -> ui.input:
    element = the(user, PROMPT_INPUT_MARKER)
    assert isinstance(element, ui.input)
    return element


def submit_button(user: User) -> ui.button:
    element = the(user, SUBMIT_MARKER)
    assert isinstance(element, ui.button)
    return element


async def wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(_POLL_LIMIT):
        if condition():
            return
        await asyncio.sleep(_POLL_SECONDS)
    raise AssertionError("condition not met in time")


async def prompt_is_locked(user: User) -> None:
    await wait_until(lambda: not prompt_input(user).enabled and not submit_button(user).enabled)


async def prompt_is_open(user: User) -> None:
    await wait_until(lambda: prompt_input(user).enabled and submit_button(user).enabled)


def wait_for_turn_end(turn_runner: TurnRunner, user_id: UserId) -> None:
    """Block until the user's turn has ended. A turn that already ended is not live, so there is nothing to wait for."""
    turn = turn_runner.active(user_id)
    if turn is not None:
        assert turn.wait(TURN_TIMEOUT_SECONDS), "the turn never ended"


async def held_turn_is_shown(user: User, planner_model: ControlledPlannerModel) -> None:
    """Wait until the planner is paused at its held call and the page shows the turn running."""
    planner_model.wait_until_held()
    await prompt_is_locked(user)


async def finish_turn(user: User, turn_runner: TurnRunner, user_id: UserId, prompt: str = PROMPT) -> None:
    """Send a prompt, wait for its turn to end, and wait for the page to reopen the prompt."""
    submit(user, prompt)
    wait_for_turn_end(turn_runner, user_id)
    await prompt_is_open(user)


async def see(user: User, *contents: str) -> None:
    """Wait until each text is on the page or in a notification, allowing for a turn in flight."""
    for content in contents:
        await user.should_see(content, retries=TURN_RETRIES)


def coach_lines(caplog: pytest.LogCaptureFixture) -> list[LogLine]:
    return lines_from(caplog.records, COACH_LOGGER)


# ---------------------------------------------------------------------------
# Quiz card
# ---------------------------------------------------------------------------


def marked(user: User, marker: str) -> list[ui.element]:
    """Every element carrying `marker`, in page order; empty when there is none."""
    assert user.client is not None
    with user.client:
        return list(ElementFilter(marker=marker))


def _texts(elements: Iterable[ui.element]) -> list[str]:
    """The text of each element and of each radio option, so assertions do not depend on layout."""
    texts: list[str] = []
    for element in elements:
        if isinstance(element, TextElement):
            texts.append(element.text)
        elif isinstance(element, ui.radio):
            options = element.options
            texts.extend(str(label) for label in (options.values() if isinstance(options, dict) else options))
    return texts


def labelled(marker: str, text: str) -> re.Pattern[str]:
    """A marker such as "1" or "A" directly before the text, as "1. text", "A) text", or on its own line."""
    return re.compile(rf"(?<!\w){re.escape(marker)}[.)]?\s+{re.escape(text)}")


def quiz_text(user: User) -> str:
    """Every piece of text inside the quiz card, in page order."""
    return "\n".join(_texts(the(user, QUIZ_MARKER).descendants()))


def result_text(user: User) -> str:
    """The text of every shown check result, in page order."""
    return "\n".join(text for found in marked(user, QUIZ_RESULT_MARKER) for text in _texts(found.descendants(include_self=True)))


def quiz_radios(user: User) -> list[ui.radio]:
    """One answer radio per question, in question order."""
    return [element for element in the(user, QUIZ_MARKER).descendants() if isinstance(element, ui.radio)]


def check_buttons(user: User) -> list[ui.button]:
    """One Check button per question, in question order."""
    buttons = [element for element in marked(user, QUIZ_CHECK_MARKER) if isinstance(element, ui.button)]
    assert buttons, "the quiz card has no Check buttons"
    return buttons


def choose_option(user: User, question: int, option: int) -> None:
    """Pick the option holding SAMPLE_OPTIONS[option] on the question at index `question`, as a click would."""
    radios = quiz_radios(user)
    assert question < len(radios), f"expected an answer radio for question {question + 1}, found {len(radios)} radios"
    radio = radios[question]
    labels = _texts([radio])
    matching = [label for label in labels if SAMPLE_OPTIONS[option] in label]
    assert len(matching) == 1, f"expected one radio option holding {SAMPLE_OPTIONS[option]!r}, found {labels}"
    UserInteraction(user, {radio}, matching[0]).click()


def click_check(user: User, question: int) -> None:
    """Press the Check button of the question at index `question`; a disabled button ignores it."""
    UserInteraction(user, {check_buttons(user)[question]}, None).click()


def click_marked(user: User, marker: str) -> None:
    UserInteraction(user, {the(user, marker)}, None).click()
