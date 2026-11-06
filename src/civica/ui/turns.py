"""Server-owned coach turns: each runs in its own worker thread, so pages only display it.

No NiceGUI import. One turn per user at a time, tracked per process.
"""

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamMode

from civica.domain.user import UserId
from civica.graph.graph import GraphInput, GraphState
from civica.graph.nodes import START_TURN, PlannerContext

logger = logging.getLogger(__name__)

CoachGraph = CompiledStateGraph[GraphState, PlannerContext, GraphInput, GraphState]

_STREAM_MODES: Final[list[StreamMode]] = ["updates"]


def coach_config(user_id: UserId) -> RunnableConfig:
    """The config of the user's one coach conversation, which PostgresSaver restores."""
    return {"configurable": {"thread_id": f"{user_id}:coach"}}


class TurnStatus(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class TurnAlreadyRunning(Exception):
    """The user already has a live turn."""


@dataclass(frozen=True)
class TurnSnapshot:
    """A frozen view of a turn's progress."""

    status: TurnStatus
    error_type: str | None


class ActiveTurn:
    """Thread-safe progress of one turn. The worker writes it and pages read it."""

    def __init__(self, learner_message: str) -> None:
        self.learner_message = learner_message
        self.turn_id: str | None = None
        self.halt_reason: str | None = None
        self._lock = threading.Lock()
        self._status = TurnStatus.RUNNING
        self._error_type: str | None = None

    def snapshot(self) -> TurnSnapshot:
        with self._lock:
            return TurnSnapshot(self._status, self._error_type)

    def end(self, status: TurnStatus, error_type: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._error_type = error_type


def _apply(turn: ActiveTurn, node: str, values: Mapping[str, object] | None) -> None:
    """Fold one node update into the turn's progress."""
    if values is None:
        return
    if node == START_TURN:
        turn.turn_id = str(values["turn_id"])
    if values.get("halt_reason") is not None:
        turn.halt_reason = str(values["halt_reason"])


class TurnRunner:
    """Starts turns on worker threads and tracks the live one of each user."""

    def __init__(self, graph: CoachGraph) -> None:
        self._graph = graph
        self._lock = threading.Lock()
        self._live: dict[UserId, ActiveTurn] = {}

    def start(self, user_id: UserId, graph_input: GraphInput) -> ActiveTurn:
        """Run a turn in the background. Raises TurnAlreadyRunning if the user has a live one."""
        with self._lock:
            if user_id in self._live:
                raise TurnAlreadyRunning(f"User {user_id} already has a turn running")
            turn = ActiveTurn(graph_input["learner_message"])
            self._live[user_id] = turn
        threading.Thread(target=self._run, args=(user_id, graph_input, turn), daemon=True).start()
        return turn

    def active(self, user_id: UserId) -> ActiveTurn | None:
        """The user's live turn, or None once it has ended."""
        with self._lock:
            return self._live.get(user_id)

    def _run(self, user_id: UserId, graph_input: GraphInput, turn: ActiveTurn) -> None:
        started = time.monotonic()
        logger.info("Turn started for user %s", user_id)
        status, error_type = TurnStatus.FINISHED, None
        try:
            for _namespace, _mode, update in self._graph.stream(
                graph_input, coach_config(user_id), context=PlannerContext(user_id=user_id),
                stream_mode=_STREAM_MODES, subgraphs=True,
            ):
                assert isinstance(update, dict)
                for node, values in update.items():
                    _apply(turn, node, values)
        except Exception as error:
            logger.exception("Turn failed for user %s", user_id)
            status, error_type = TurnStatus.FAILED, type(error).__name__
        finally:
            # Final status first, so a user is freed only after the turn has ended.
            turn.end(status, error_type)
            with self._lock:
                del self._live[user_id]
        logger.info(
            "Turn %s for user %s: turn_id=%s halt_reason=%s duration=%.1fs",
            status, user_id, turn.turn_id, turn.halt_reason, time.monotonic() - started,
        )
