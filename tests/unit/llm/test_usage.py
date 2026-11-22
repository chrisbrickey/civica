"""Unit tests for civica.llm.usage.

Uses fake chat models whose replies carry usage. No network calls."""

from concurrent.futures import ThreadPoolExecutor
from typing import Final

from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import UsageMetadata

from civica.llm.client import ModelCaller
from civica.llm.usage import UNTAGGED, TokenCounts, TurnUsage, UsageTally
from tests.support.assessment import FakeChatModel
from tests.support.llm import CONTENT_USAGE, CRITIC_TAG, EXPLANATION_TAG, PLANNER_TAG, PLANNER_USAGE, QUIZ_TAG, token_usage
from tests.support.usage import counts_of, sum_counts, turn_usage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MESSAGES: Final = [HumanMessage(content="sample-prompt")]
_REPLY_TEXT: Final = "sample-reply"

_PLANNER_COUNTS: Final = counts_of(PLANNER_USAGE)
_CONTENT_COUNTS: Final = counts_of(CONTENT_USAGE)
_EMPTY: Final = TurnUsage(by_caller={}, total=TokenCounts())

# A tag LangChain adds on its own, which names no caller.
_STEP_TAG: Final = "seq:step:1"
_NOT_A_CALLER_TAG: Final = "sample-tag"

_THREADS: Final = 8
_CALLS_PER_THREAD: Final = 25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(usage: UsageMetadata | None, *tags: str) -> FakeChatModel:
    return FakeChatModel(canned_text=_REPLY_TEXT, usage=usage, tags=list(tags) or None)


def _call(model: FakeChatModel, tally: UsageTally, *run_tags: str) -> None:
    """One model call that reports to the tally the way a graph run does: through the config."""
    model.invoke(_MESSAGES, config={"callbacks": [tally], "tags": list(run_tags)})


# ---------------------------------------------------------------------------
# Callers
# ---------------------------------------------------------------------------


def test_caller_names_are_the_tags_the_profiles_carry() -> None:
    assert {caller.value for caller in ModelCaller} == {PLANNER_TAG, EXPLANATION_TAG, QUIZ_TAG, CRITIC_TAG}


# ---------------------------------------------------------------------------
# Token counts
# ---------------------------------------------------------------------------


def test_token_counts_default_to_zero() -> None:
    assert TokenCounts() == TokenCounts(input=0, output=0, reasoning=0)


def test_total_is_input_plus_output_because_output_already_includes_reasoning() -> None:
    counts = _CONTENT_COUNTS

    assert counts.total == counts.input + counts.output


# ---------------------------------------------------------------------------
# Tallying a run
# ---------------------------------------------------------------------------


def test_fresh_tally_starts_at_zero() -> None:
    assert UsageTally().totals() == _EMPTY


def test_calls_through_two_callers_are_totaled_separately_and_the_total_is_their_sum() -> None:
    tally = UsageTally()

    _call(_model(PLANNER_USAGE, ModelCaller.PLANNER.value), tally)
    _call(_model(CONTENT_USAGE, ModelCaller.EXPLANATION.value), tally)

    totals = tally.totals()
    assert totals.by_caller == {ModelCaller.PLANNER.value: _PLANNER_COUNTS, ModelCaller.EXPLANATION.value: _CONTENT_COUNTS}
    assert totals.total == sum_counts(_PLANNER_COUNTS, _CONTENT_COUNTS)


def test_repeated_calls_by_one_caller_sum_input_output_and_reasoning() -> None:
    tally = UsageTally()
    model = _model(CONTENT_USAGE, ModelCaller.QUIZ.value)
    calls = 3

    for _ in range(calls):
        _call(model, tally)

    assert tally.totals() == turn_usage({ModelCaller.QUIZ.value: counts_of(CONTENT_USAGE, calls)})


def test_caller_that_made_no_call_is_absent() -> None:
    tally = UsageTally()

    _call(_model(PLANNER_USAGE, ModelCaller.PLANNER.value), tally)

    assert set(tally.totals().by_caller) == {ModelCaller.PLANNER.value}


def test_reply_without_usage_metadata_adds_nothing_and_does_not_error() -> None:
    tally = UsageTally()
    _call(_model(PLANNER_USAGE, ModelCaller.PLANNER.value), tally)

    _call(_model(None, ModelCaller.EXPLANATION.value), tally)

    assert tally.totals() == turn_usage({ModelCaller.PLANNER.value: _PLANNER_COUNTS})


def test_usage_without_a_reasoning_detail_counts_zero_reasoning() -> None:
    tally = UsageTally()
    usage = token_usage(input_tokens=_PLANNER_COUNTS.input, output_tokens=_PLANNER_COUNTS.output)

    _call(_model(usage, ModelCaller.PLANNER.value), tally)

    expected = TokenCounts(input=_PLANNER_COUNTS.input, output=_PLANNER_COUNTS.output, reasoning=0)
    assert tally.totals().by_caller == {ModelCaller.PLANNER.value: expected}


def test_tags_that_name_no_caller_do_not_change_the_caller() -> None:
    tally = UsageTally()

    _call(_model(PLANNER_USAGE, ModelCaller.PLANNER.value), tally, _STEP_TAG)

    assert tally.totals().by_caller == {ModelCaller.PLANNER.value: _PLANNER_COUNTS}


def test_call_by_a_model_with_no_caller_tag_is_counted_as_untagged() -> None:
    tally = UsageTally()

    _call(_model(PLANNER_USAGE), tally)
    _call(_model(CONTENT_USAGE, _NOT_A_CALLER_TAG), tally, _STEP_TAG)

    assert tally.totals() == turn_usage({UNTAGGED: sum_counts(_PLANNER_COUNTS, _CONTENT_COUNTS)})


def test_one_tallys_usage_never_leaks_into_another() -> None:
    model = _model(PLANNER_USAGE, ModelCaller.PLANNER.value)
    first, second = UsageTally(), UsageTally()

    _call(model, first)

    assert second.totals() == _EMPTY
    assert UsageTally().totals() == _EMPTY


def test_totals_are_a_snapshot_that_later_calls_do_not_change() -> None:
    tally = UsageTally()
    model = _model(PLANNER_USAGE, ModelCaller.PLANNER.value)
    _call(model, tally)
    before = tally.totals()

    _call(model, tally)

    assert before == turn_usage({ModelCaller.PLANNER.value: _PLANNER_COUNTS})


def test_calls_from_many_threads_are_all_counted() -> None:
    tally = UsageTally()
    model = _model(PLANNER_USAGE, ModelCaller.PLANNER.value)

    def call_repeatedly(_thread: int) -> None:
        for _ in range(_CALLS_PER_THREAD):
            _call(model, tally)

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        list(pool.map(call_repeatedly, range(_THREADS)))

    assert tally.totals().total == counts_of(PLANNER_USAGE, _THREADS * _CALLS_PER_THREAD)
