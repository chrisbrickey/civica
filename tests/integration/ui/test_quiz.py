"""Integration tests for asking the coach for a quiz and answering its questions."""

from string import ascii_uppercase

import pytest
from nicegui import ui
from nicegui.testing import User

from civica.domain.question import Question
from civica.domain.user import UserId
from civica.graph.graph import InputKind
from civica.graph.nodes import PlannerContext
from civica.ui.pages.coach import CORRECT_TEXTS, MISSED_TEXT
from civica.ui.turns import CoachGraph, TurnRunner, coach_config
from tests.support.answers import EXPIRED_REPLY, VIA_ANSWER, answer_message
from tests.support.assessment import (
    OPTION_COUNT,
    SAMPLE_CORRECT_INDEX,
    SAMPLE_OPTIONS,
    FakeChatModel,
    LoggedAnswer,
    default_generation_reply,
    generated_item_explanation,
)
from tests.support.coach import (
    COACH_LOGGER,
    FINAL_CALL,
    LESSON_TEXT,
    PRACTICE_LABEL,
    PROMPT,
    QUIZ_AFTER_LESSON_CALL,
    QUIZ_ME_TEXT,
    QUIZ_ONLY_CALL,
    SECOND_PROMPT,
    SECOND_REPLY,
    TEACH_ME_TEXT,
    answer_question,
    check_buttons,
    choose_option,
    click_check,
    click_marked,
    coach_lines,
    finish_quiz_me,
    held_turn_is_shown,
    is_locked,
    labelled,
    prompt_is_open,
    question_texts,
    quiz_radios,
    quiz_text,
    result_text,
    script_lesson,
    script_lesson_with_quiz,
    script_quiz_only,
    script_quiz_then_lesson,
    scoring_graph,
    see,
    submit,
    the,
    wait_for_turn_end,
    wait_until,
)
from tests.support.logs import UNKNOWN_TURN_ID, events, lines_from, only
from tests.support.planner import DEFAULT_THEME, ControlledPlannerModel, PlannerFakes, final_message
from tests.support.ui import INTRO_OPEN_MARKER, LOGOUT_MARKER, QUIZ_MARKER, QUIZ_ME_MARKER, TEACH_ME_MARKER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THATS_CORRECT = "That's correct!"
_QUICK_ACTIONS = (TEACH_ME_MARKER, QUIZ_ME_MARKER)

_QUIZ_QUESTION = question_texts(QUIZ_ONLY_CALL, 1)[0]
_QUIZ_EXPLANATION = generated_item_explanation(QUIZ_ONLY_CALL, 0)
_TWO_QUESTIONS = 2

_WRONG_INDEX = (SAMPLE_CORRECT_INDEX + 1) % OPTION_COUNT
_CORRECT_LETTER = ascii_uppercase[SAMPLE_CORRECT_INDEX]
_CORRECT_OPTION = SAMPLE_OPTIONS[SAMPLE_CORRECT_INDEX]

# Log line events and their fixed field values.
_TURN_REQUEST = "turn_request"
_PRACTICE_CHECK = "practice_check"
_TURNS_LOGGER = "civica.ui.turns"
_TURN_START = "turn_start"
_TURN_END = "turn_end"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _says_correct(text: str) -> bool:
    return any(verdict in text for verdict in CORRECT_TEXTS)


def _quick_action_enabled(user: User, marker: str) -> bool:
    button = the(user, marker)
    assert isinstance(button, ui.button)
    return bool(button.enabled)


def _saved_question(graph: CoachGraph, user_id: UserId, index: int = 0) -> Question:
    """A question of the quiz checkpointed on the user's coach thread."""
    question = graph.get_state(coach_config(user_id)).values["quiz_questions"][index]
    assert isinstance(question, Question)
    return question


def _passages(fakes: PlannerFakes) -> list[str]:
    """Texts of every fake corpus passage on the quiz theme."""
    return [chunk.chunk.text for chunk in fakes.retriever.chunks(DEFAULT_THEME)]


def _assert_missed_then_stored_explanation(result: str, explanation: str) -> None:
    """The MISSED verdict with the lettered correct option, then the question's stored explanation."""
    assert result.startswith(MISSED_TEXT), f"no missed verdict first in {result!r}"
    correct = labelled(_CORRECT_LETTER, _CORRECT_OPTION).search(result)
    assert correct, f"no lettered correct option in {result!r}"
    assert explanation in result[correct.end():], f"stored explanation should follow the correct option in {result!r}"


def _answer_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, str]]:
    return [fields for event, fields in coach_lines(caplog) if event == _TURN_REQUEST and fields["via"] == VIA_ANSWER]


# ---------------------------------------------------------------------------
# Quick actions
# ---------------------------------------------------------------------------


async def test_quick_actions_sit_in_the_header_between_the_intro_button_and_log_out(signed_in_user: User) -> None:
    header = next(iter(signed_in_user.find(kind=ui.header).elements))
    header_order = list(header.descendants())
    markers = (INTRO_OPEN_MARKER, *_QUICK_ACTIONS, LOGOUT_MARKER)

    for marker in markers:
        assert the(signed_in_user, marker) in header_order, f"{marker!r} should sit in the header"
    positions = [header_order.index(the(signed_in_user, marker)) for marker in markers]
    assert positions == sorted(positions), f"header order should be {markers}"


@pytest.mark.parametrize(
    ("marker", "text"),
    [(TEACH_ME_MARKER, TEACH_ME_TEXT), (QUIZ_ME_MARKER, QUIZ_ME_TEXT)],
    ids=["teach-me", "quiz-me"],
)
async def test_quick_action_has_a_tooltip_naming_it(signed_in_user: User, marker: str, text: str) -> None:
    # On narrow screens only the icon shows, so the tooltip carries the name.
    tooltips = [child for child in the(signed_in_user, marker).descendants() if isinstance(child, ui.tooltip)]

    assert [tooltip.text for tooltip in tooltips] == [text]


@pytest.mark.parametrize(
    ("marker", "text"),
    [(TEACH_ME_MARKER, TEACH_ME_TEXT), (QUIZ_ME_MARKER, QUIZ_ME_TEXT)],
    ids=["teach-me", "quiz-me"],
)
async def test_quick_action_sends_its_text_as_the_learner_message(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    marker: str,
    text: str,
) -> None:
    script_lesson(planner_model, content_model)

    click_marked(signed_in_user, marker)
    wait_for_turn_end(turn_runner, learner_id)

    await see(signed_in_user, text, LESSON_TEXT)
    assert set(planner_model.human_texts()) == {text}


async def test_quiz_me_renders_the_questions_with_no_option_selected(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    questions = question_texts(QUIZ_ONLY_CALL, _TWO_QUESTIONS)
    script_quiz_only(planner_model, _TWO_QUESTIONS)

    await finish_quiz_me(signed_in_user, turn_runner, learner_id, _TWO_QUESTIONS)

    text = quiz_text(signed_in_user)
    for question in questions:
        assert question in text
    for option in SAMPLE_OPTIONS:
        assert option in text
    radios = quiz_radios(signed_in_user)
    assert len(radios) == len(questions)
    assert all(radio.value is None for radio in radios)


async def test_quick_actions_are_disabled_while_a_turn_runs(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_lesson(planner_model, content_model)
    planner_model.hold_at_call = FINAL_CALL
    submit(signed_in_user)
    await held_turn_is_shown(signed_in_user, planner_model)

    await wait_until(lambda: not any(_quick_action_enabled(signed_in_user, marker) for marker in _QUICK_ACTIONS))
    for marker in _QUICK_ACTIONS:
        click_marked(signed_in_user, marker)
    planner_model.release()
    wait_for_turn_end(turn_runner, learner_id)

    await prompt_is_open(signed_in_user)
    await wait_until(lambda: all(_quick_action_enabled(signed_in_user, marker) for marker in _QUICK_ACTIONS))
    assert set(planner_model.human_texts()) == {PROMPT}


# ---------------------------------------------------------------------------
# Before answering
# ---------------------------------------------------------------------------


def test_the_correct_verdicts_include_thats_correct() -> None:
    assert _THATS_CORRECT in CORRECT_TEXTS


async def test_submit_is_disabled_until_an_option_is_selected(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await finish_quiz_me(signed_in_user, turn_runner, learner_id)

    assert not check_buttons(signed_in_user)[0].enabled

    choose_option(signed_in_user, 0, _WRONG_INDEX)

    await wait_until(lambda: check_buttons(signed_in_user)[0].enabled)


async def test_before_an_answer_the_card_shows_no_explanation_or_verdict(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await finish_quiz_me(signed_in_user, turn_runner, learner_id)
    choose_option(signed_in_user, 0, SAMPLE_CORRECT_INDEX)

    text = quiz_text(signed_in_user)
    assert _QUIZ_EXPLANATION not in text
    assert "correct" not in text.lower()
    assert result_text(signed_in_user) == ""


async def test_reload_before_answering_shows_the_same_questions_unanswered(
    signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
) -> None:
    script_quiz_only(planner_model)
    await finish_quiz_me(signed_in_user, turn_runner, learner_id)
    choose_option(signed_in_user, 0, _WRONG_INDEX)

    await signed_in_user.open("/")

    await see(signed_in_user, _QUIZ_QUESTION)
    (radio,) = quiz_radios(signed_in_user)
    assert radio.value is None
    assert radio.enabled
    assert result_text(signed_in_user) == ""


async def test_a_lesson_after_a_quiz_replaces_the_questions(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
) -> None:
    script_quiz_then_lesson(planner_model, content_model)
    await finish_quiz_me(signed_in_user, turn_runner, learner_id)

    submit(signed_in_user, SECOND_PROMPT)
    wait_for_turn_end(turn_runner, learner_id)

    await see(signed_in_user, SECOND_PROMPT, LESSON_TEXT)
    await signed_in_user.should_not_see(content=_QUIZ_QUESTION)
    await signed_in_user.should_not_see(marker=QUIZ_MARKER)


@pytest.mark.parametrize(
    ("marker", "via"), [(TEACH_ME_MARKER, "teach_me"), (QUIZ_ME_MARKER, "quiz_me")], ids=["teach-me", "quiz-me"]
)
async def test_quick_action_logs_one_request_line_naming_it(
    signed_in_user: User,
    planner_model: ControlledPlannerModel,
    content_model: FakeChatModel,
    turn_runner: TurnRunner,
    learner_id: UserId,
    app_logs: pytest.LogCaptureFixture,
    marker: str,
    via: str,
) -> None:
    script_lesson(planner_model, content_model)

    click_marked(signed_in_user, marker)
    wait_for_turn_end(turn_runner, learner_id)
    await see(signed_in_user, LESSON_TEXT)

    assert only(coach_lines(app_logs), _TURN_REQUEST) == {"user_id": str(learner_id), "via": via}


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


class TestAnswering:
    """Submitting a choice runs an answer turn, which scores, logs, and locks the question; no model is called."""

    @pytest.fixture()
    def coach_graph(self, planner_fakes: PlannerFakes, planner_model: ControlledPlannerModel) -> CoachGraph:
        return scoring_graph(planner_fakes, planner_model)

    async def test_the_card_no_longer_says_answers_are_practice(
        self, signed_in_user: User, planner_model: ControlledPlannerModel, turn_runner: TurnRunner, learner_id: UserId
    ) -> None:
        script_quiz_only(planner_model)
        await finish_quiz_me(signed_in_user, turn_runner, learner_id)

        assert PRACTICE_LABEL not in quiz_text(signed_in_user)

    async def test_submitting_a_choice_scores_it_through_the_graph_without_the_planner(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        planner_fakes: PlannerFakes,
        coach_graph: CoachGraph,
        turn_runner: TurnRunner,
        learner_id: UserId,
    ) -> None:
        script_quiz_only(planner_model)
        await finish_quiz_me(signed_in_user, turn_runner, learner_id)
        question = _saved_question(coach_graph, learner_id)
        planner_calls = planner_model.invocation_count

        await answer_question(signed_in_user, turn_runner, learner_id, 0, _WRONG_INDEX)

        assert planner_fakes.answer_logger.calls == [
            LoggedAnswer(learner_id, DEFAULT_THEME, question.question_id, _WRONG_INDEX, SAMPLE_CORRECT_INDEX, False)
        ]
        assert planner_model.invocation_count == planner_calls
        await see(signed_in_user, answer_message(1))

    async def test_a_right_answer_says_it_is_correct(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        planner_fakes: PlannerFakes,
        turn_runner: TurnRunner,
        learner_id: UserId,
    ) -> None:
        script_quiz_only(planner_model)
        await finish_quiz_me(signed_in_user, turn_runner, learner_id)

        await answer_question(signed_in_user, turn_runner, learner_id, 0, SAMPLE_CORRECT_INDEX)

        await wait_until(lambda: _says_correct(result_text(signed_in_user)))
        assert [answer.is_correct for answer in planner_fakes.answer_logger.calls] == [True]

    async def test_a_wrong_answer_shows_the_correct_option_then_the_stored_explanation_without_a_model_call(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        content_model: FakeChatModel,
        planner_fakes: PlannerFakes,
        turn_runner: TurnRunner,
        learner_id: UserId,
    ) -> None:
        script_quiz_only(planner_model)
        await finish_quiz_me(signed_in_user, turn_runner, learner_id)
        content_calls = content_model.invocation_count

        await answer_question(signed_in_user, turn_runner, learner_id, 0, _WRONG_INDEX)

        await wait_until(lambda: result_text(signed_in_user).startswith(MISSED_TEXT))
        result = result_text(signed_in_user)
        _assert_missed_then_stored_explanation(result, _QUIZ_EXPLANATION)
        assert not _says_correct(result)
        for passage in _passages(planner_fakes):
            assert passage not in result
        assert content_model.invocation_count == content_calls

    async def test_an_answered_question_locks_and_unanswered_questions_stay_usable(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        planner_fakes: PlannerFakes,
        turn_runner: TurnRunner,
        learner_id: UserId,
    ) -> None:
        script_quiz_only(planner_model, _TWO_QUESTIONS)
        await finish_quiz_me(signed_in_user, turn_runner, learner_id, _TWO_QUESTIONS)

        await answer_question(signed_in_user, turn_runner, learner_id, 0, SAMPLE_CORRECT_INDEX)

        first, second = quiz_radios(signed_in_user)
        assert not first.enabled
        assert not check_buttons(signed_in_user)[0].enabled
        assert second.enabled
        assert second.value is None
        await answer_question(signed_in_user, turn_runner, learner_id, 1, _WRONG_INDEX)
        assert is_locked(signed_in_user, 0)
        await wait_until(lambda: MISSED_TEXT in result_text(signed_in_user))
        assert _says_correct(result_text(signed_in_user))
        assert [answer.is_correct for answer in planner_fakes.answer_logger.calls] == [True, False]

    async def test_an_answered_question_stays_locked_with_its_result_after_a_reload(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        turn_runner: TurnRunner,
        learner_id: UserId,
    ) -> None:
        script_quiz_only(planner_model)
        await finish_quiz_me(signed_in_user, turn_runner, learner_id)
        await answer_question(signed_in_user, turn_runner, learner_id, 0, _WRONG_INDEX)

        await signed_in_user.open("/")

        await see(signed_in_user, _QUIZ_QUESTION)
        await wait_until(lambda: is_locked(signed_in_user, 0))
        (radio,) = quiz_radios(signed_in_user)
        assert radio.value == _WRONG_INDEX
        _assert_missed_then_stored_explanation(result_text(signed_in_user), _QUIZ_EXPLANATION)

    async def test_answering_a_question_that_has_expired_shows_the_expired_reply(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        planner_fakes: PlannerFakes,
        coach_graph: CoachGraph,
        turn_runner: TurnRunner,
        learner_id: UserId,
    ) -> None:
        script_quiz_only(planner_model)
        planner_model.script = (*planner_model.script, final_message(SECOND_REPLY))
        await finish_quiz_me(signed_in_user, turn_runner, learner_id)
        # Another tab moves on; this page still shows the old question.
        coach_graph.invoke(
            {"kind": InputKind.MESSAGE, "learner_message": SECOND_PROMPT},
            coach_config(learner_id),
            context=PlannerContext(user_id=learner_id),
        )

        choose_option(signed_in_user, 0, _WRONG_INDEX)
        await wait_until(lambda: check_buttons(signed_in_user)[0].enabled)
        click_check(signed_in_user, 0)
        wait_for_turn_end(turn_runner, learner_id)

        await see(signed_in_user, EXPIRED_REPLY)
        assert planner_fakes.answer_logger.calls == []

    async def test_a_lesson_comprehension_question_can_be_answered(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        content_model: FakeChatModel,
        turn_runner: TurnRunner,
        learner_id: UserId,
    ) -> None:
        script_lesson_with_quiz(planner_model, content_model)
        content_model.replies = (LESSON_TEXT, default_generation_reply(QUIZ_AFTER_LESSON_CALL))
        submit(signed_in_user)
        wait_for_turn_end(turn_runner, learner_id)
        await see(signed_in_user, LESSON_TEXT, *question_texts(QUIZ_AFTER_LESSON_CALL, 1))

        await answer_question(signed_in_user, turn_runner, learner_id, 0, _WRONG_INDEX)

        await wait_until(lambda: result_text(signed_in_user).startswith(MISSED_TEXT))
        _assert_missed_then_stored_explanation(
            result_text(signed_in_user), generated_item_explanation(QUIZ_AFTER_LESSON_CALL, 0)
        )

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    async def test_an_answer_turn_logs_its_start_and_end_with_its_own_turn_id(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        turn_runner: TurnRunner,
        learner_id: UserId,
        app_logs: pytest.LogCaptureFixture,
    ) -> None:
        script_quiz_only(planner_model)
        await finish_quiz_me(signed_in_user, turn_runner, learner_id)

        await answer_question(signed_in_user, turn_runner, learner_id, 0, _WRONG_INDEX)

        lines = [line for line in lines_from(app_logs.records, _TURNS_LOGGER) if line[0] in (_TURN_START, _TURN_END)]
        assert events(lines) == [_TURN_START, _TURN_END, _TURN_START, _TURN_END]
        quiz_id, _, answer_id, answer_end_id = (fields["turn_id"] for _event, fields in lines)
        assert answer_id == answer_end_id
        assert answer_id not in (UNKNOWN_TURN_ID, quiz_id)

    @pytest.mark.parametrize("choice", [SAMPLE_CORRECT_INDEX, _WRONG_INDEX], ids=["right", "wrong"])
    async def test_an_answer_logs_one_request_line_with_its_question_but_no_option_text(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        coach_graph: CoachGraph,
        turn_runner: TurnRunner,
        learner_id: UserId,
        app_logs: pytest.LogCaptureFixture,
        choice: int,
    ) -> None:
        script_quiz_only(planner_model)
        await finish_quiz_me(signed_in_user, turn_runner, learner_id)
        question = _saved_question(coach_graph, learner_id)

        await answer_question(signed_in_user, turn_runner, learner_id, 0, choice)

        assert _answer_lines(app_logs) == [
            {"user_id": str(learner_id), "via": VIA_ANSWER, "question_id": question.question_id}
        ]
        assert _PRACTICE_CHECK not in events(coach_lines(app_logs))
        coach_messages = [record.getMessage() for record in app_logs.records if record.name == COACH_LOGGER]
        assert not [message for message in coach_messages if SAMPLE_OPTIONS[choice] in message]

    async def test_a_quiz_turn_and_answer_turns_log_no_question_option_or_stored_explanation_text(
        self,
        signed_in_user: User,
        planner_model: ControlledPlannerModel,
        planner_fakes: PlannerFakes,
        turn_runner: TurnRunner,
        learner_id: UserId,
        app_logs: pytest.LogCaptureFixture,
    ) -> None:
        # Extends the lesson turn's no-learner-text test in test_lessons.py to quizzes and answers.
        script_quiz_only(planner_model, _TWO_QUESTIONS)
        await finish_quiz_me(signed_in_user, turn_runner, learner_id, _TWO_QUESTIONS)
        await answer_question(signed_in_user, turn_runner, learner_id, 0, SAMPLE_CORRECT_INDEX)
        await answer_question(signed_in_user, turn_runner, learner_id, 1, _WRONG_INDEX)
        await signed_in_user.open("/")
        # The wrong answer's stored explanation is on screen, so the scan below covers it.
        await see(signed_in_user, _QUIZ_QUESTION, generated_item_explanation(QUIZ_ONLY_CALL, 1))

        assert len(_answer_lines(app_logs)) == _TWO_QUESTIONS, "no answer was logged, so the scan proves nothing"
        explanations = [generated_item_explanation(QUIZ_ONLY_CALL, index) for index in range(_TWO_QUESTIONS)]
        learner_texts = (
            *question_texts(QUIZ_ONLY_CALL, _TWO_QUESTIONS),
            *SAMPLE_OPTIONS,
            *explanations,
            *_passages(planner_fakes),
        )
        for learner_text in learner_texts:
            assert learner_text not in app_logs.text
