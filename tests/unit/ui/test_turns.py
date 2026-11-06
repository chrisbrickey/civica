"""Unit tests for civica.ui.turns, on a fake graph and plain threads. No NiceGUI, no database."""

import logging
import threading
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from typing import Final

import pytest

from civica.graph.graph import MEMORY_WRITER, PLANNER, InputKind
from civica.graph.nodes import PLANNER_MODEL, PLANNER_TOOLS, START_TURN, PlannerContext
from civica.graph.planner import ToolCall
from civica.ui.turns import ActiveTurn, TurnAlreadyRunning, TurnRunner, TurnStatus
from tests.support.users import OTHER_USER_ID, USER_ID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRAPH_INPUT: Final = {"kind": InputKind.MESSAGE, "learner_message": "sample-learner-message-001"}
_SUBGRAPH: Final = ("planner:sample-task-001",)
_TURN_ID: Final = "sample-turn-id-001"
_FAILURE_TEXT: Final = "sample-graph-failure"
_TURNS_LOGGER: Final = "civica.ui.turns"

_READ_TOOL: Final = "read_learner_state"
_TEACH_TOOL: Final = "teach"

_READ_CALL: Final = ToolCall(name=_READ_TOOL, args={})
_TEACH_CALL: Final = ToolCall(name=_TEACH_TOOL, args={"focus": "sample-focus"})

# Upper bounds only: passing tests return as soon as the condition holds.
_WAIT_TIMEOUT_SECONDS: Final = 5.0
_POLL_SECONDS: Final = 0.005

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
_FINAL_MESSAGE: Final = _update(PLANNER_MODEL, {"model_calls": 3, "reply": "sample-final-reply"})
_PLANNER_DONE: Final = _update(PLANNER, {"tool_calls": [_READ_CALL, _TEACH_CALL]}, namespace=())
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


_MakeGraph = Callable[..., _FakeGraph]


@pytest.fixture()
def make_graph() -> Generator[_MakeGraph, None, None]:
    """Builds fake graphs and resumes them all at teardown, so no worker thread outlives its test."""
    graphs: list[_FakeGraph] = []

    def _make(*script: _StreamEvent | _Pause, failure: Exception | None = None) -> _FakeGraph:
        graph = _FakeGraph(script, failure)
        graphs.append(graph)
        return graph

    yield _make
    for graph in graphs:
        graph.resume.set()


def _wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_SECONDS
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(_POLL_SECONDS)


def _wait_for_end(runner: TurnRunner, turn: ActiveTurn) -> None:
    """Wait until the runner has freed the user, which only happens once the turn has ended."""
    _wait_until(lambda: runner.active(USER_ID) is None)
    assert turn.snapshot().status is not TurnStatus.RUNNING


def _start_and_pause(runner: TurnRunner, graph: _FakeGraph) -> ActiveTurn:
    turn = runner.start(USER_ID, _GRAPH_INPUT)
    assert graph.paused.wait(_WAIT_TIMEOUT_SECONDS), "the stream never reached its pause"
    return turn


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def test_turn_that_streams_to_its_end_finishes(make_graph: _MakeGraph) -> None:
    runner = TurnRunner(make_graph(*_FULL_TURN))

    turn = runner.start(USER_ID, _GRAPH_INPUT)
    _wait_for_end(runner, turn)

    snapshot = turn.snapshot()
    assert snapshot.status is TurnStatus.FINISHED
    assert snapshot.error_type is None




def test_turn_runs_with_the_user_in_its_context(make_graph: _MakeGraph) -> None:
    graph = make_graph(*_FULL_TURN)
    runner = TurnRunner(graph)

    _wait_for_end(runner, runner.start(USER_ID, _GRAPH_INPUT))

    assert graph.contexts == [PlannerContext(user_id=USER_ID)]


# ---------------------------------------------------------------------------
# One turn per user
# ---------------------------------------------------------------------------


def test_active_finds_the_live_turn_and_nothing_once_it_ended(make_graph: _MakeGraph) -> None:
    graph = make_graph(_START, _PAUSE)
    runner = TurnRunner(graph)
    assert runner.active(USER_ID) is None

    turn = _start_and_pause(runner, graph)
    assert runner.active(USER_ID) is turn

    graph.resume.set()
    _wait_for_end(runner, turn)
    assert runner.active(USER_ID) is None


def test_second_start_for_the_same_user_raises(make_graph: _MakeGraph) -> None:
    graph = make_graph(_START, _PAUSE)
    runner = TurnRunner(graph)
    turn = _start_and_pause(runner, graph)

    with pytest.raises(TurnAlreadyRunning):
        runner.start(USER_ID, _GRAPH_INPUT)

    assert runner.active(USER_ID) is turn
    assert graph.contexts == [PlannerContext(user_id=USER_ID)]


def test_a_different_user_is_unaffected_by_a_running_turn(make_graph: _MakeGraph) -> None:
    graph = make_graph(_START, _PAUSE)
    runner = TurnRunner(graph)
    turn = _start_and_pause(runner, graph)

    other_turn = runner.start(OTHER_USER_ID, _GRAPH_INPUT)

    assert other_turn is not turn
    assert runner.active(OTHER_USER_ID) is other_turn
    assert runner.active(USER_ID) is turn


def test_the_same_user_can_start_again_after_a_finished_turn(make_graph: _MakeGraph) -> None:
    runner = TurnRunner(make_graph(*_FULL_TURN))
    _wait_for_end(runner, runner.start(USER_ID, _GRAPH_INPUT))

    next_turn = runner.start(USER_ID, _GRAPH_INPUT)
    _wait_for_end(runner, next_turn)

    assert next_turn.snapshot().status is TurnStatus.FINISHED


# ---------------------------------------------------------------------------
# Failed turns
# ---------------------------------------------------------------------------


def _failing_graph(make_graph: _MakeGraph) -> _FakeGraph:
    failure = ValueError(_FAILURE_TEXT)
    return make_graph(_START, _model_update(_READ_CALL), failure=failure)


def test_graph_that_raises_marks_the_turn_failed_with_the_error_type(make_graph: _MakeGraph) -> None:
    runner = TurnRunner(_failing_graph(make_graph))

    turn = runner.start(USER_ID, _GRAPH_INPUT)
    _wait_for_end(runner, turn)

    snapshot = turn.snapshot()
    assert snapshot.status is TurnStatus.FAILED
    assert snapshot.error_type == ValueError.__name__


def test_graph_that_raises_logs_the_exception(
    make_graph: _MakeGraph, caplog: pytest.LogCaptureFixture
) -> None:
    runner = TurnRunner(_failing_graph(make_graph))

    with caplog.at_level(logging.ERROR):
        _wait_for_end(runner, runner.start(USER_ID, _GRAPH_INPUT))

    errors = [
        record
        for record in caplog.records
        if record.name == _TURNS_LOGGER and record.levelno == logging.ERROR
    ]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    assert _FAILURE_TEXT in caplog.text


def test_graph_that_raises_frees_the_user_to_start_again(make_graph: _MakeGraph) -> None:
    runner = TurnRunner(_failing_graph(make_graph))
    failed_turn = runner.start(USER_ID, _GRAPH_INPUT)
    _wait_for_end(runner, failed_turn)

    retry = runner.start(USER_ID, _GRAPH_INPUT)
    _wait_for_end(runner, retry)

    assert retry is not failed_turn


# ---------------------------------------------------------------------------
# Never blocking the caller
# ---------------------------------------------------------------------------


def test_start_returns_while_the_turn_is_still_running(make_graph: _MakeGraph) -> None:
    # A runner that ran the graph on the caller would return only after the pause gave up.
    runner = TurnRunner(make_graph(_START, _PAUSE))

    turn = runner.start(USER_ID, _GRAPH_INPUT)

    assert turn.snapshot().status is TurnStatus.RUNNING
