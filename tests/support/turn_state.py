"""Hand-built saved turns for tests that shape a thread's state without running the graph."""

from collections.abc import Mapping, Sequence
from typing import Final

from langgraph.types import StateSnapshot

from civica.graph.graph import InputKind
from civica.graph.planner import HaltReason

LEARNER_MESSAGE: Final = "sample-learner-message-001"
REPLY: Final = "sample-final-reply"


def finished_state(**overrides: object) -> dict[str, object]:
    """The graph's values at the end of a turn that only sent a reply."""
    state: dict[str, object] = {
        "kind": InputKind.MESSAGE,
        "learner_message": LEARNER_MESSAGE,
        "reply": REPLY,
        "lesson": None,
        "quiz_questions": [],
        "tool_calls": [],
        "halt_reason": HaltReason.FINAL_MESSAGE,
    }
    return state | overrides


def snapshot_of(values: Mapping[str, object], next_nodes: Sequence[str] = ()) -> StateSnapshot:
    """A thread snapshot holding `values`; a non-empty `next_nodes` marks the turn unfinished."""
    return StateSnapshot(
        values=dict(values),
        next=tuple(next_nodes),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )
