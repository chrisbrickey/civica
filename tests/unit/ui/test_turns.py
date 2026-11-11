"""Unit tests for civica.ui.turns, on a fake graph and plain threads. No NiceGUI, no database."""

import logging
import re
import threading
from collections.abc import Generator, Iterator, Sequence
from typing import Final

import pytest

from civica.graph.graph import MEMORY_WRITER, PLANNER, InputKind
from civica.graph.nodes import PLANNER_MODEL, PLANNER_TOOLS, START_TURN, PlannerContext
from civica.graph.planner import HaltReason, ToolCall
from civica.ui.turns import ActiveTurn, TurnAlreadyRunning, TurnRunner, TurnStatus
from tests.support.logs import UNKNOWN_TURN_ID, LogLine, events, lines_from, only, parse_line
from tests.support.users import OTHER_USER_ID, USER_ID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRAPH_INPUT: Final = {"kind": InputKind.MESSAGE, "learner_message": "sample-learner-message-001"}
_SUBGRAPH: Final = ("planner:sample-task-001",)
_TURN_ID: Final = "sample-turn-id-001"
_FAILURE_TEXT: Final = "sample-graph-failure"
_TURNS_LOGGER: Final = "civica.ui.turns"

# Log line events and the fields each carries.
_TURN_START: Final = "turn_start"
_TURN_FAILED: Final = "turn_failed"
_TURN_END: Final = "turn_end"
_NONE: Final = "none"
_TURN_KEYS: Final = {"user_id", "turn_id"}
_END_KEYS: Final = _TURN_KEYS | {"status", "halt_reason", "duration"}
_DURATION_FORMAT: Final = re.compile(r"^\d+\.\ds$")

_READ_CALL: Final = ToolCall(name="read_learner_state", args={})
_TEACH_CALL: Final = ToolCall(name="teach", args={"focus": "sample-focus"})

# Upper bound only: passing tests return as soon as the turn ends.
_WAIT_TIMEOUT_SECONDS: Final = 5.0
# A wait that is expected to time out, kept short so the test stays fast.
_SHORT_TIMEOUT_SECONDS: Final = 0.01

_Namespace = tuple[str, ...]
_StreamEvent = tuple[_Namespace, str, dict[str, dict[str, object] | None]]


class _Pause:
    """Placed in a script to stop the stream there until the test resumes it."""


_PAUSE: Final = _Pause()


def _update(node: str, values: dict[str, object] | None, namespace: _Namespace = _SUBGRAPH) -> _StreamEvent:
    """One `updates` item as graph.stream emits it with subgraphs=True and a stream_mode list."""
    return namespace, "updates", {node: values}


def _model_update(*calls: ToolCall) -> _StreamEvent:
    """A planner_model update, whose tool_calls list holds every call of the turn so far."""
    return _update(PLANNER_MODEL, {"model_calls": 1, "tool_calls": list(calls)})


_START: Final = _update(START_TURN, {"turn_id": _TURN_ID, "tool_calls": []})
_TOOLS_RAN: Final = _update(PLANNER_TOOLS, {"messages": []})
_FINAL_MESSAGE: Final = _update(
    PLANNER_MODEL, {"model_calls": 3, "reply": "sample-final-reply", "halt_reason": HaltReason.FINAL_MESSAGE}
)
_PLANNER_DONE: Final = _update(
    PLANNER, {"tool_calls": [_READ_CALL, _TEACH_CALL], "halt_reason": HaltReason.FINAL_MESSAGE}, namespace=()
)
_WRITER_DONE: Final = _update(MEMORY_WRITER, None, namespace=())

_FULL_TURN: Final = (
    _START,
    _model_update(_READ_CALL),
    _TOOLS_RAN,
    _model_update(_READ_CALL, _TEACH_CALL),
    _TOOLS_RAN,
    _FINAL_MESSAGE,
    _PLANNER_DONE,
    _WRITER_DONE,
)


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Replays a script of stream events, then raises `failure` if one is set."""

    def __init__(self, script: Sequence[_StreamEvent | _Pause], failure: Exception | None) -> None:
        self._script = script
        self._failure = failure
        self.paused = threading.Event()
        self.resume = threading.Event()
        self.contexts: list[PlannerContext] = []

    def stream(
        self,
        input: dict[str, object],
        config: dict[str, object],
        *,
        context: PlannerContext,
        stream_mode: list[str],
        subgraphs: bool,
    ) -> Iterator[_StreamEvent]:
        self.contexts.append(context)
        for item in self._script:
            if isinstance(item, _Pause):
                self.paused.set()
                assert self.resume.wait(_WAIT_TIMEOUT_SECONDS), "the paused stream was never resumed"
            else:
                yield item
        if self._failure is not None:
            raise self._failure


class _GraphFactory:
    """Builds fake graphs and resumes them all at teardown, so no worker thread outlives its test."""

    def __init__(self) -> None:
        self._graphs: list[_FakeGraph] = []

    def __call__(self, *script: _StreamEvent | _Pause, failure: Exception | None = None) -> _FakeGraph:
        graph = _FakeGraph(script, failure)
        self._graphs.append(graph)
        return graph

    def resume_all(self) -> None:
        for graph in self._graphs:
            graph.resume.set()


@pytest.fixture()
def make_graph() -> Generator[_GraphFactory, None, None]:
    factory = _GraphFactory()
    yield factory
    factory.resume_all()


def _run_to_end(runner: TurnRunner) -> ActiveTurn:
    turn = runner.start(USER_ID, _GRAPH_INPUT)
    assert turn.wait(_WAIT_TIMEOUT_SECONDS), "the turn never ended"
    return turn


def _start_and_pause(runner: TurnRunner, graph: _FakeGraph) -> ActiveTurn:
    turn = runner.start(USER_ID, _GRAPH_INPUT)
    assert graph.paused.wait(_WAIT_TIMEOUT_SECONDS), "the stream never reached its pause"
    return turn


def _failing_graph(make_graph: _GraphFactory) -> _FakeGraph:
    return make_graph(_START, _model_update(_READ_CALL), failure=ValueError(_FAILURE_TEXT))


@pytest.fixture()
def turn_logs(caplog: pytest.LogCaptureFixture) -> Generator[pytest.LogCaptureFixture, None, None]:
    """Everything the turns logger writes at INFO and above during the test."""
    with caplog.at_level(logging.INFO, logger=_TURNS_LOGGER):
        yield caplog


def _turn_lines(caplog: pytest.LogCaptureFixture) -> list[LogLine]:
    return lines_from(caplog.records, _TURNS_LOGGER)


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def test_turn_that_streams_to_its_end_finishes_with_its_turn_id_and_halt_reason(
    make_graph: _GraphFactory,
) -> None:
    turn = _run_to_end(TurnRunner(make_graph(*_FULL_TURN)))

    snapshot = turn.snapshot()
    assert snapshot.status is TurnStatus.FINISHED
    assert snapshot.error_type is None
    assert snapshot.turn_id == _TURN_ID
    assert snapshot.halt_reason == HaltReason.FINAL_MESSAGE


def test_turn_keeps_the_learner_message(make_graph: _GraphFactory) -> None:
    turn = _run_to_end(TurnRunner(make_graph(*_FULL_TURN)))

    assert turn.learner_message == _GRAPH_INPUT["learner_message"]


def test_turn_runs_with_the_user_in_its_context(make_graph: _GraphFactory) -> None:
    graph = make_graph(*_FULL_TURN)

    _run_to_end(TurnRunner(graph))

    assert graph.contexts == [PlannerContext(user_id=USER_ID)]


# ---------------------------------------------------------------------------
# One turn per user
# ---------------------------------------------------------------------------


def test_active_finds_the_live_turn_and_nothing_once_it_ended(make_graph: _GraphFactory) -> None:
    graph = make_graph(_START, _PAUSE)
    runner = TurnRunner(graph)
    assert runner.active(USER_ID) is None

    turn = _start_and_pause(runner, graph)
    assert runner.active(USER_ID) is turn

    graph.resume.set()
    assert turn.wait(_WAIT_TIMEOUT_SECONDS)
    assert runner.active(USER_ID) is None


def test_second_start_for_the_same_user_raises(make_graph: _GraphFactory) -> None:
    graph = make_graph(_START, _PAUSE)
    runner = TurnRunner(graph)
    turn = _start_and_pause(runner, graph)

    with pytest.raises(TurnAlreadyRunning):
        runner.start(USER_ID, _GRAPH_INPUT)

    assert runner.active(USER_ID) is turn
    assert graph.contexts == [PlannerContext(user_id=USER_ID)]


def test_a_different_user_is_unaffected_by_a_running_turn(make_graph: _GraphFactory) -> None:
    graph = make_graph(_START, _PAUSE)
    runner = TurnRunner(graph)
    turn = _start_and_pause(runner, graph)

    other_turn = runner.start(OTHER_USER_ID, _GRAPH_INPUT)

    assert other_turn is not turn
    assert runner.active(OTHER_USER_ID) is other_turn
    assert runner.active(USER_ID) is turn


def test_the_same_user_can_start_again_after_a_finished_turn(make_graph: _GraphFactory) -> None:
    runner = TurnRunner(make_graph(*_FULL_TURN))
    _run_to_end(runner)

    next_turn = _run_to_end(runner)

    assert next_turn.snapshot().status is TurnStatus.FINISHED


# ---------------------------------------------------------------------------
# Failed turns
# ---------------------------------------------------------------------------


def test_graph_that_raises_marks_the_turn_failed_with_the_error_type(make_graph: _GraphFactory) -> None:
    turn = _run_to_end(TurnRunner(_failing_graph(make_graph)))

    snapshot = turn.snapshot()
    assert snapshot.status is TurnStatus.FAILED
    assert snapshot.error_type == ValueError.__name__


def test_graph_that_raises_logs_the_exception_once_as_a_turn_failed_line(
    make_graph: _GraphFactory, turn_logs: pytest.LogCaptureFixture
) -> None:
    _run_to_end(TurnRunner(_failing_graph(make_graph)))

    errors = [
        record
        for record in turn_logs.records
        if record.name == _TURNS_LOGGER and record.levelno == logging.ERROR
    ]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    assert _FAILURE_TEXT in turn_logs.text
    event, fields = parse_line(errors[0].getMessage())
    assert event == _TURN_FAILED
    assert fields == {"user_id": str(USER_ID), "turn_id": _TURN_ID, "error_type": ValueError.__name__}


def test_graph_that_raises_frees_the_user_to_start_again(make_graph: _GraphFactory) -> None:
    runner = TurnRunner(_failing_graph(make_graph))
    failed_turn = _run_to_end(runner)

    retry = _run_to_end(runner)

    assert retry is not failed_turn


# ---------------------------------------------------------------------------
# Never blocking the caller, and waiting on a turn
# ---------------------------------------------------------------------------


def test_start_returns_while_the_turn_is_still_running(make_graph: _GraphFactory) -> None:
    # A runner that ran the graph on the caller would only return after the pause gave up.
    turn = TurnRunner(make_graph(_START, _PAUSE)).start(USER_ID, _GRAPH_INPUT)

    assert turn.snapshot().status is TurnStatus.RUNNING


@pytest.mark.parametrize(
    ("script", "failure", "status"),
    [
        pytest.param(_FULL_TURN, None, TurnStatus.FINISHED, id="finished"),
        pytest.param((_START,), ValueError(_FAILURE_TEXT), TurnStatus.FAILED, id="failed"),
    ],
)
def test_wait_returns_true_once_the_turn_ends(
    make_graph: _GraphFactory,
    script: tuple[_StreamEvent, ...],
    failure: Exception | None,
    status: TurnStatus,
) -> None:
    turn = TurnRunner(make_graph(*script, failure=failure)).start(USER_ID, _GRAPH_INPUT)

    assert turn.wait(_WAIT_TIMEOUT_SECONDS)
    assert turn.snapshot().status is status


def test_wait_returns_false_at_its_timeout_while_the_turn_runs(make_graph: _GraphFactory) -> None:
    graph = make_graph(_START, _PAUSE)
    turn = _start_and_pause(TurnRunner(graph), graph)

    assert not turn.wait(_SHORT_TIMEOUT_SECONDS)
    assert turn.snapshot().status is TurnStatus.RUNNING


# ---------------------------------------------------------------------------
# Turn log lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("script", "failure", "expected_events"),
    [
        pytest.param(_FULL_TURN, None, [_TURN_START, _TURN_END], id="finished"),
        pytest.param(
            (_START, _model_update(_READ_CALL)),
            ValueError(_FAILURE_TEXT),
            [_TURN_START, _TURN_FAILED, _TURN_END],
            id="failed",
        ),
    ],
)
def test_every_line_of_a_turn_carries_its_user_and_turn_id(
    make_graph: _GraphFactory,
    turn_logs: pytest.LogCaptureFixture,
    script: tuple[_StreamEvent, ...],
    failure: Exception | None,
    expected_events: list[str],
) -> None:
    _run_to_end(TurnRunner(make_graph(*script, failure=failure)))

    lines = _turn_lines(turn_logs)
    assert events(lines) == expected_events
    assert only(lines, _TURN_START) == {"user_id": str(USER_ID), "turn_id": _TURN_ID}
    for _event, fields in lines:
        assert fields["user_id"] == str(USER_ID)
        assert fields["turn_id"] == _TURN_ID


def test_turn_that_fails_before_start_turn_logs_its_failure_and_end_with_an_unknown_turn_id(
    make_graph: _GraphFactory, turn_logs: pytest.LogCaptureFixture
) -> None:
    _run_to_end(TurnRunner(make_graph(failure=ValueError(_FAILURE_TEXT))))

    lines = _turn_lines(turn_logs)
    assert events(lines) == [_TURN_FAILED, _TURN_END]
    assert only(lines, _TURN_FAILED)["turn_id"] == UNKNOWN_TURN_ID
    assert only(lines, _TURN_END)["turn_id"] == UNKNOWN_TURN_ID


def test_turn_start_is_logged_when_start_turn_arrives_not_when_the_turn_begins(
    make_graph: _GraphFactory, turn_logs: pytest.LogCaptureFixture
) -> None:
    graph = make_graph(_PAUSE, *_FULL_TURN)
    turn = _start_and_pause(TurnRunner(graph), graph)

    assert _TURN_START not in events(_turn_lines(turn_logs))

    graph.resume.set()
    assert turn.wait(_WAIT_TIMEOUT_SECONDS), "the turn never ended"
    assert only(_turn_lines(turn_logs), _TURN_START)["turn_id"] == _TURN_ID


@pytest.mark.parametrize(
    ("script", "failure", "expected"),
    [
        pytest.param(
            _FULL_TURN,
            None,
            {"status": TurnStatus.FINISHED.value, "halt_reason": HaltReason.FINAL_MESSAGE.value},
            id="finished",
        ),
        pytest.param(
            (_START, _model_update(_READ_CALL)),
            ValueError(_FAILURE_TEXT),
            {"status": TurnStatus.FAILED.value, "halt_reason": _NONE, "error_type": ValueError.__name__},
            id="failed",
        ),
    ],
)
def test_end_line_carries_status_halt_reason_duration_and_error_type_only_when_failed(
    make_graph: _GraphFactory,
    turn_logs: pytest.LogCaptureFixture,
    script: tuple[_StreamEvent, ...],
    failure: Exception | None,
    expected: dict[str, str],
) -> None:
    _run_to_end(TurnRunner(make_graph(*script, failure=failure)))

    end = only(_turn_lines(turn_logs), _TURN_END)
    assert set(end) == _END_KEYS | set(expected)
    assert _DURATION_FORMAT.match(end["duration"]), f"duration is not in seconds: {end['duration']!r}"
    assert {key: end[key] for key in expected} == expected


def test_end_line_is_logged_by_the_time_a_waiter_sees_the_turn_end(
    make_graph: _GraphFactory, turn_logs: pytest.LogCaptureFixture
) -> None:
    # Pages and tests read the log right after wait returns, so the end line must already be there.
    _run_to_end(TurnRunner(make_graph(*_FULL_TURN)))

    assert _TURN_END in events(_turn_lines(turn_logs))
