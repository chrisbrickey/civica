"""Integration tests for civica.scripts.measure_tokens.

Uses the real graph and checkpointer with fake models."""

import uuid
from collections.abc import Generator, Sequence
from typing import Final

import psycopg
import psycopg.rows
import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from civica.domain.user import UserId
from civica.graph.graph import build_graph
from civica.llm.usage import TokenCounts
from civica.memory.checkpointer import get_saver
from civica.scripts.measure_tokens import MEASUREMENT_USERNAME, Scenario, ScenarioResult, measurement_user_id, run_scenarios
from civica.ui.turns import coach_config
from civica.users.service import find_user_id
from tests.support.assessment import FakeChatModel
from tests.support.llm import EXPLANATION_TAG, PLANNER_TAG, QUIZ_TAG, mapped_usage, raw_usage
from tests.support.planner import (
    DEFAULT_THEME,
    READ_STATE,
    TEACH,
    FakePlannerModel,
    final_message,
    make_planner_fakes,
    planner_deps,
    tool_call,
)
from tests.support.usage import turn_usage
from tests.support.users import TEST_BCRYPT_ROUNDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LESSON_TEXT: Final = "sample-lesson-text"
_FOCUS: Final = "sample-focus"
_REPLY_TEXT: Final = "sample-final-reply"
_END_TURN: Final = "end_turn"

# Distinct numbers per model, with cached input, so a dropped or mixed-up count is caught.
_PLANNER_RAW: Final = raw_usage(input_tokens=100, output_tokens=20, cache_read=40, cache_creation=3)
_CONTENT_RAW: Final = raw_usage(input_tokens=3000, output_tokens=400, cache_read=500, cache_creation=11)
_PLANNER_REASONING: Final = 5
_CONTENT_REASONING: Final = 70

_LESSON: Final = Scenario(
    name="sample-lesson", learner_message="sample-lesson-prompt", expected_callers=frozenset({PLANNER_TAG, EXPLANATION_TAG})
)
_REPLY_ONLY: Final = Scenario(
    name="sample-reply-only", learner_message="sample-reply-prompt", expected_callers=frozenset({PLANNER_TAG})
)
_PLANNER_CALLS_PER_LESSON: Final = 3

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def user_id() -> UserId:
    """A fresh id per test, so parallel workers never share a scenario thread."""
    return UserId(uuid.uuid4())


@pytest.fixture()
def cleanup_threads() -> Generator[list[ScenarioResult], None, None]:
    """Collects results whose checkpoint threads are deleted after the test."""
    results: list[ScenarioResult] = []
    try:
        yield results
    finally:
        for result in results:
            get_saver().delete_thread(result.thread_id)


def _lesson_script() -> tuple[AIMessage, ...]:
    return (
        tool_call(READ_STATE, "call-state"),
        tool_call(TEACH, "call-teach", theme=DEFAULT_THEME.slug, focus=_FOCUS),
        final_message(_REPLY_TEXT),
    )


def _planner_model(*script: AIMessage) -> FakePlannerModel:
    return FakePlannerModel(
        script=script,
        tags=[PLANNER_TAG],
        raw_usage=_PLANNER_RAW,
        usage=mapped_usage(_PLANNER_RAW, _PLANNER_REASONING),
        stop_reason=_END_TURN,
    )


def _content_model(drop_cached_input: bool = False) -> FakeChatModel:
    """Writes the lesson. `drop_cached_input` makes the mapped usage disagree with the raw usage."""
    mapped = (
        mapped_usage(raw_usage(_CONTENT_RAW["input_tokens"], _CONTENT_RAW["output_tokens"]), _CONTENT_REASONING)
        if drop_cached_input
        else mapped_usage(_CONTENT_RAW, _CONTENT_REASONING)
    )
    return FakeChatModel(
        replies=(_LESSON_TEXT,),
        tags=[EXPLANATION_TAG],
        raw_usage=_CONTENT_RAW,
        usage=mapped,
        stop_reason=_END_TURN,
    )


def _run(
    user_id: UserId,
    cleanup: list[ScenarioResult],
    scenarios: Sequence[Scenario],
    script: tuple[AIMessage, ...],
    content_model: FakeChatModel | None = None,
) -> list[ScenarioResult]:
    """Run `scenarios` through the compiled graph on the real checkpointer, with every model faked."""
    fakes = make_planner_fakes(InMemoryStore(), content_model=content_model or _content_model())
    graph = build_graph(planner_deps(fakes, _planner_model(*script)))
    results = list(run_scenarios(graph, scenarios, user_id))
    cleanup.extend(results)
    return results


def _callers(result: ScenarioResult) -> list[str]:
    return [record.caller for record in result.records]


def _planner_counts(calls: int) -> TokenCounts:
    mapped = mapped_usage(_PLANNER_RAW, _PLANNER_REASONING)
    return TokenCounts(
        input=mapped["input_tokens"] * calls, output=mapped["output_tokens"] * calls, reasoning=_PLANNER_REASONING * calls
    )


# ---------------------------------------------------------------------------
# run_scenarios
# ---------------------------------------------------------------------------


def test_every_model_call_of_each_scenario_is_recorded(
    user_id: UserId, cleanup_threads: list[ScenarioResult]
) -> None:
    lesson, reply = _run(user_id, cleanup_threads, [_LESSON, _REPLY_ONLY], (*_lesson_script(), final_message(_REPLY_TEXT)))

    assert sorted(_callers(lesson)) == sorted([PLANNER_TAG] * _PLANNER_CALLS_PER_LESSON + [EXPLANATION_TAG])
    assert _callers(reply) == [PLANNER_TAG]
    assert {record.scenario for record in lesson.records} == {_LESSON.name}
    assert {record.scenario for record in reply.records} == {_REPLY_ONLY.name}


def test_records_carry_raw_token_counts_including_cached_input(
    user_id: UserId, cleanup_threads: list[ScenarioResult]
) -> None:
    (reply,) = _run(user_id, cleanup_threads, [_REPLY_ONLY], (final_message(_REPLY_TEXT),))

    (record,) = reply.records
    assert record.input_tokens == mapped_usage(_PLANNER_RAW)["input_tokens"]
    assert record.output_tokens == _PLANNER_RAW["output_tokens"]
    assert record.reasoning_tokens == _PLANNER_REASONING
    assert record.stop_reason == _END_TURN


def test_lesson_and_reply_only_turns_pass_every_accounting_check(
    user_id: UserId, cleanup_threads: list[ScenarioResult]
) -> None:
    results = _run(user_id, cleanup_threads, [_LESSON, _REPLY_ONLY], (*_lesson_script(), final_message(_REPLY_TEXT)))

    assert [result.problems for result in results] == [(), ()]
    assert [result.notes for result in results] == [(), ()]


def test_saved_usage_is_read_back_through_the_checkpointer(
    user_id: UserId, cleanup_threads: list[ScenarioResult]
) -> None:
    (reply,) = _run(user_id, cleanup_threads, [_REPLY_ONLY], (final_message(_REPLY_TEXT),))

    assert reply.saved_usage == turn_usage({PLANNER_TAG: _planner_counts(1)})
    saved = get_saver().get_tuple({"configurable": {"thread_id": reply.thread_id}})
    assert saved is not None, "the scenario's turn was not checkpointed on its thread"


def test_each_scenario_runs_on_its_own_thread_that_is_not_the_coach_thread(
    user_id: UserId, cleanup_threads: list[ScenarioResult]
) -> None:
    results = _run(user_id, cleanup_threads, [_LESSON, _REPLY_ONLY], (*_lesson_script(), final_message(_REPLY_TEXT)))

    thread_ids = [result.thread_id for result in results]
    assert len(set(thread_ids)) == len(thread_ids)
    assert coach_config(user_id)["configurable"]["thread_id"] not in thread_ids
    assert get_saver().get_tuple(coach_config(user_id)) is None


def test_mapped_usage_that_disagrees_with_the_raw_usage_is_reported(
    user_id: UserId, cleanup_threads: list[ScenarioResult]
) -> None:
    (lesson,) = _run(user_id, cleanup_threads, [_LESSON], _lesson_script(), _content_model(drop_cached_input=True))

    assert lesson.problems != ()


def test_unexpected_callers_are_a_note_not_a_problem(
    user_id: UserId, cleanup_threads: list[ScenarioResult]
) -> None:
    expects_a_quiz = Scenario(
        name="sample-expects-quiz", learner_message="sample-quiz-prompt", expected_callers=frozenset({PLANNER_TAG, QUIZ_TAG})
    )

    (result,) = _run(user_id, cleanup_threads, [expects_a_quiz], (final_message(_REPLY_TEXT),))

    assert result.problems == ()
    assert result.notes != ()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_measurement_account_is_created_when_missing(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> None:
    created = measurement_user_id(conn=db_schema, rounds=TEST_BCRYPT_ROUNDS)

    assert find_user_id(MEASUREMENT_USERNAME, conn=db_schema) == created


def test_measurement_account_is_reused_on_later_runs(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> None:
    first = measurement_user_id(conn=db_schema, rounds=TEST_BCRYPT_ROUNDS)

    assert measurement_user_id(conn=db_schema, rounds=TEST_BCRYPT_ROUNDS) == first
