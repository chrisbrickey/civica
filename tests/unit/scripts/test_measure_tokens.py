"""Unit tests for civica.scripts.measure_tokens.

Uses fake chat models. No network or database calls."""

import csv
import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import UsageMetadata

from civica.llm.client import EXPLANATION_PROFILE, PLANNER_PROFILE, QUIZ_PROFILE
from civica.llm.usage import UNTAGGED, TokenCounts, TurnUsage
from civica.scripts.measure_tokens import CallRecord, CallRecorder, check_turn, default_scenarios, summarize, write_csv
from tests.support.assessment import FakeChatModel
from tests.support.llm import EXPLANATION_TAG, PLANNER_TAG, QUIZ_TAG, mapped_usage, raw_usage
from tests.support.usage import token_line, turn_usage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCENARIO: Final = "sample-scenario-001"
_MESSAGES: Final = [HumanMessage(content="sample-prompt")]
_REPLY_TEXT: Final = "sample-reply"
_NOT_A_CALLER_TAG: Final = "sample-tag"

_END_TURN: Final = "end_turn"
_MAX_TOKENS: Final = "max_tokens"

_RAW: Final = raw_usage(input_tokens=100, output_tokens=20, cache_read=30, cache_creation=7)
_RAW_INPUT_TOTAL: Final = 137
_REASONING: Final = 5

_CLOCK_START: Final = 100.0
_CLOCK_STEP: Final = 2.5

_CSV_HEADER: Final = [
    "scenario",
    "caller",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "stop_reason",
    "latency_seconds",
]

_PLANNER_COUNTS: Final = TokenCounts(input=100, output=20, reasoning=5)
_CONTENT_COUNTS: Final = TokenCounts(input=3000, output=400, reasoning=70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clock() -> Callable[[], float]:
    """A clock that moves forward by _CLOCK_STEP on every read."""
    ticks = itertools.count(_CLOCK_START, _CLOCK_STEP)
    return lambda: next(ticks)


def _recorder() -> CallRecorder:
    return CallRecorder(_SCENARIO, clock=_clock())


def _model(
    *tags: str,
    raw: dict[str, int] | None = _RAW,
    usage: UsageMetadata | None = None,
    stop_reason: str | None = _END_TURN,
) -> FakeChatModel:
    return FakeChatModel(
        canned_text=_REPLY_TEXT,
        tags=list(tags) or None,
        raw_usage=raw,
        usage=usage if usage is not None else (mapped_usage(raw, _REASONING) if raw is not None else None),
        stop_reason=stop_reason,
    )


def _call(model: FakeChatModel, recorder: CallRecorder, *run_tags: str) -> None:
    """One model call that reports to the recorder the way a graph run does: through the config."""
    model.invoke(_MESSAGES, config={"callbacks": [recorder], "tags": list(run_tags)})


def _record(
    caller: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
    stop_reason: str | None = _END_TURN,
    latency_seconds: float = 1.0,
    scenario: str = _SCENARIO,
) -> CallRecord:
    return CallRecord(
        scenario=scenario,
        caller=caller,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        stop_reason=stop_reason,
        latency_seconds=latency_seconds,
    )


def _records_of(by_caller: dict[str, TokenCounts]) -> list[CallRecord]:
    """One record per caller carrying that caller's counts."""
    return [
        _record(name, counts.input, counts.output, counts.reasoning) for name, counts in by_caller.items()
    ]


def _panel_values(usage: TurnUsage) -> tuple[str, ...]:
    """The Token Counts values the developer panel shows for `usage`."""
    callers = (token_line(name, usage.by_caller[name]) for name in sorted(usage.by_caller))
    return (token_line("total", usage.total), *callers)


_BY_CALLER: Final = {PLANNER_TAG: _PLANNER_COUNTS, EXPLANATION_TAG: _CONTENT_COUNTS}


# ---------------------------------------------------------------------------
# CallRecorder
# ---------------------------------------------------------------------------


def test_fresh_recorder_has_no_records() -> None:
    assert _recorder().records() == ()


def test_each_model_call_is_one_record_named_for_the_scenario() -> None:
    recorder = _recorder()
    model = _model(PLANNER_TAG)

    _call(model, recorder)
    _call(model, recorder)

    records = recorder.records()
    assert len(records) == 2
    assert {record.scenario for record in records} == {_SCENARIO}


@pytest.mark.parametrize("tag", [PLANNER_TAG, EXPLANATION_TAG, QUIZ_TAG])
def test_caller_is_read_from_the_model_tag(tag: str) -> None:
    recorder = _recorder()

    _call(_model(tag), recorder)

    assert [record.caller for record in recorder.records()] == [tag]


def test_caller_is_read_from_a_run_tag() -> None:
    recorder = _recorder()

    _call(_model(), recorder, QUIZ_TAG)

    assert [record.caller for record in recorder.records()] == [QUIZ_TAG]


@pytest.mark.parametrize("tags", [(), (_NOT_A_CALLER_TAG,)])
def test_call_without_a_caller_tag_is_recorded_as_untagged(tags: tuple[str, ...]) -> None:
    recorder = _recorder()

    _call(_model(*tags), recorder)

    assert [record.caller for record in recorder.records()] == [UNTAGGED]


def test_input_counts_cached_tokens_from_the_raw_provider_usage() -> None:
    recorder = _recorder()

    _call(_model(PLANNER_TAG), recorder)

    (record,) = recorder.records()
    assert record.input_tokens == _RAW_INPUT_TOTAL
    assert record.output_tokens == _RAW["output_tokens"]


def test_tokens_come_from_the_raw_usage_not_the_mapped_usage() -> None:
    """A mapped usage that disagrees with the raw one must not leak into the record."""
    recorder = _recorder()
    disagreeing = mapped_usage(raw_usage(input_tokens=1, output_tokens=2), _REASONING)

    _call(_model(PLANNER_TAG, usage=disagreeing), recorder)

    (record,) = recorder.records()
    assert (record.input_tokens, record.output_tokens) == (_RAW_INPUT_TOTAL, _RAW["output_tokens"])


def test_reasoning_tokens_come_from_the_mapped_usage() -> None:
    recorder = _recorder()

    _call(_model(PLANNER_TAG), recorder)

    (record,) = recorder.records()
    assert record.reasoning_tokens == _REASONING


def test_reasoning_is_zero_when_the_reply_reports_none() -> None:
    recorder = _recorder()
    without_reasoning = mapped_usage(_RAW)

    _call(_model(PLANNER_TAG, usage=without_reasoning), recorder)

    (record,) = recorder.records()
    assert record.reasoning_tokens == 0


def test_reply_without_usage_is_recorded_with_zero_tokens_and_its_stop_reason() -> None:
    recorder = _recorder()

    _call(_model(PLANNER_TAG, raw=None, stop_reason=_END_TURN), recorder)

    (record,) = recorder.records()
    assert (record.input_tokens, record.output_tokens, record.reasoning_tokens) == (0, 0, 0)
    assert record.stop_reason == _END_TURN


@pytest.mark.parametrize("stop_reason", [_END_TURN, _MAX_TOKENS])
def test_stop_reason_is_recorded(stop_reason: str) -> None:
    recorder = _recorder()

    _call(_model(PLANNER_TAG, stop_reason=stop_reason), recorder)

    (record,) = recorder.records()
    assert record.stop_reason == stop_reason


def test_latency_is_the_time_from_call_start_to_call_end() -> None:
    recorder = _recorder()
    model = _model(PLANNER_TAG)

    _call(model, recorder)
    _call(model, recorder)

    assert [record.latency_seconds for record in recorder.records()] == [_CLOCK_STEP, _CLOCK_STEP]


# ---------------------------------------------------------------------------
# check_turn
# ---------------------------------------------------------------------------


def test_turn_whose_accounting_holds_has_no_problems() -> None:
    saved = turn_usage(_BY_CALLER)

    assert check_turn(_records_of(_BY_CALLER), saved, _panel_values(saved)) == []


def test_several_calls_of_one_caller_are_summed_before_comparing() -> None:
    saved = turn_usage({PLANNER_TAG: TokenCounts(input=200, output=40)})
    records = [_record(PLANNER_TAG, 100, 20), _record(PLANNER_TAG, 100, 20)]

    assert check_turn(records, saved, _panel_values(saved)) == []


@pytest.mark.parametrize(
    "recorded",
    [
        TokenCounts(input=_PLANNER_COUNTS.input + 1, output=_PLANNER_COUNTS.output),
        TokenCounts(input=_PLANNER_COUNTS.input, output=_PLANNER_COUNTS.output + 1),
    ],
    ids=["input", "output"],
)
def test_caller_sum_that_differs_from_the_saved_usage_is_a_problem(recorded: TokenCounts) -> None:
    saved = turn_usage(_BY_CALLER)
    records = _records_of({**_BY_CALLER, PLANNER_TAG: recorded})

    assert check_turn(records, saved, _panel_values(saved)) != []


def test_caller_recorded_but_missing_from_the_saved_usage_is_a_problem() -> None:
    saved = turn_usage({PLANNER_TAG: _PLANNER_COUNTS})
    records = _records_of({PLANNER_TAG: _PLANNER_COUNTS, QUIZ_TAG: TokenCounts()})

    assert check_turn(records, saved, _panel_values(saved)) != []


def test_caller_saved_but_never_recorded_is_a_problem() -> None:
    saved = turn_usage(_BY_CALLER)
    records = _records_of({PLANNER_TAG: _PLANNER_COUNTS})

    assert check_turn(records, saved, _panel_values(saved)) != []


def test_saved_total_that_is_not_the_sum_of_its_callers_is_a_problem() -> None:
    saved = TurnUsage(by_caller=_BY_CALLER, total=_PLANNER_COUNTS)

    assert check_turn(_records_of(_BY_CALLER), saved, _panel_values(saved)) != []


def test_panel_values_that_differ_from_the_saved_usage_are_a_problem() -> None:
    saved = turn_usage(_BY_CALLER)
    other = turn_usage({PLANNER_TAG: _PLANNER_COUNTS})

    assert check_turn(_records_of(_BY_CALLER), saved, _panel_values(other)) != []


def test_no_saved_usage_while_calls_were_recorded_is_a_problem() -> None:
    assert check_turn(_records_of(_BY_CALLER), None, ("none",)) != []


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_max_output_is_per_call_not_per_turn() -> None:
    records = [_record(PLANNER_TAG, output_tokens=600), _record(PLANNER_TAG, output_tokens=600)]

    (row,) = summarize(records)

    assert row.caller == PLANNER_TAG
    assert row.calls == 2
    assert row.max_output == 600


def test_each_caller_is_summarized_against_its_profile_cap() -> None:
    records = [_record(PLANNER_TAG, output_tokens=512), _record(EXPLANATION_TAG, output_tokens=1024)]

    rows = {row.caller: row for row in summarize(records)}

    assert rows[PLANNER_TAG].cap == PLANNER_PROFILE.max_tokens
    assert rows[PLANNER_TAG].cap_share == pytest.approx(512 / PLANNER_PROFILE.max_tokens)
    assert rows[EXPLANATION_TAG].cap == EXPLANATION_PROFILE.max_tokens
    assert rows[EXPLANATION_TAG].cap_share == pytest.approx(1024 / EXPLANATION_PROFILE.max_tokens)


def test_critic_and_question_calls_share_the_quiz_cap() -> None:
    """The critic is built from the quiz profile, so its calls carry the quiz tag."""
    records = [_record(QUIZ_TAG, output_tokens=3000), _record(QUIZ_TAG, output_tokens=200)]

    (row,) = summarize(records)

    assert row.caller == QUIZ_TAG
    assert row.calls == 2
    assert row.cap == QUIZ_PROFILE.max_tokens
    assert row.max_output == 3000


def test_reasoning_is_totalled_per_caller() -> None:
    records = [_record(PLANNER_TAG, reasoning_tokens=5), _record(PLANNER_TAG, reasoning_tokens=7)]

    (row,) = summarize(records)

    assert row.reasoning_tokens == 12


def test_truncations_count_calls_that_stopped_at_the_cap() -> None:
    records = [
        _record(PLANNER_TAG, stop_reason=_MAX_TOKENS),
        _record(PLANNER_TAG, stop_reason=_END_TURN),
        _record(PLANNER_TAG, stop_reason=_MAX_TOKENS),
    ]

    (row,) = summarize(records)

    assert row.truncations == 2


def test_max_latency_is_the_slowest_call() -> None:
    records = [_record(PLANNER_TAG, latency_seconds=1.5), _record(PLANNER_TAG, latency_seconds=4.0)]

    (row,) = summarize(records)

    assert row.max_latency_seconds == 4.0


def test_no_records_summarize_to_no_rows() -> None:
    assert list(summarize([])) == []


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


def test_csv_has_a_header_and_one_row_per_call(tmp_path: Path) -> None:
    records = [
        _record(PLANNER_TAG, 100, 20, 5, _END_TURN, 1.5),
        _record(QUIZ_TAG, 3000, 400, 0, _MAX_TOKENS, 4.25, scenario="sample-scenario-002"),
    ]
    path = tmp_path / "nested" / "calls.csv"

    write_csv(records, path)

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == _CSV_HEADER
    assert [
        CallRecord(
            scenario=row["scenario"],
            caller=row["caller"],
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            reasoning_tokens=int(row["reasoning_tokens"]),
            stop_reason=row["stop_reason"],
            latency_seconds=float(row["latency_seconds"]),
        )
        for row in rows
    ] == records


def test_csv_of_no_calls_has_only_the_header(tmp_path: Path) -> None:
    path = tmp_path / "calls.csv"

    write_csv([], path)

    with path.open(newline="") as handle:
        assert list(csv.reader(handle)) == [_CSV_HEADER]


# ---------------------------------------------------------------------------
# default_scenarios
# ---------------------------------------------------------------------------

_WITH_CHECK: Final = frozenset({PLANNER_TAG, EXPLANATION_TAG, QUIZ_TAG})
_WITHOUT_CHECK: Final = frozenset({PLANNER_TAG, EXPLANATION_TAG})


def _expected_by_prefix(prefix: str) -> set[frozenset[str]]:
    return {s.expected_callers for s in default_scenarios() if s.name.startswith(prefix)}


@pytest.mark.parametrize("prefix", ["lesson-", "check-"])
def test_full_lessons_expect_the_comprehension_check(prefix: str) -> None:
    # The planner prompt runs a one-question check after a full lesson.
    full = {
        s.expected_callers
        for s in default_scenarios()
        if s.name.startswith(prefix) and not s.name.startswith("lesson-brief-")
    }
    assert full == {_WITH_CHECK}


@pytest.mark.parametrize("prefix", ["lesson-brief-", "hard-"])
def test_brief_lessons_and_direct_questions_expect_no_check(prefix: str) -> None:
    assert _expected_by_prefix(prefix) == {_WITHOUT_CHECK}
