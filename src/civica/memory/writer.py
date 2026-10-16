"""Allowlist-guarded writer over the LangGraph long-term memory store.

Store key conventions:
- LearnerProfile / SessionSummary: one per user, constant key like "current"
- TopicMastery: keyed by theme.slug
- MistakeEpisode: unique key per episode (e.g. question_id + timestamp)
"""

from collections.abc import Mapping
from typing import TypeVar

from langgraph.store.base import BaseStore

from civica.domain.user import UserId
from civica.memory.records import (
    LearnerProfile,
    MemoryKind,
    MemoryRecord,
    MistakeEpisode,
    SessionSummary,
    TopicMastery,
)
from civica.memory.store import get_store

MEMORY_WRITE_ALLOWLIST: frozenset[MemoryKind] = frozenset(MemoryKind)

# Hand-enumerated on purpose so new memory kinds never widen the planner's write scope.
PLANNER_WRITE_ALLOWLIST: frozenset[MemoryKind] = frozenset(
    {MemoryKind.MISTAKE_EPISODE, MemoryKind.SESSION_SUMMARY}
)

_RECORD_MODELS: dict[MemoryKind, type[MemoryRecord]] = {
    model.kind: model for model in (LearnerProfile, TopicMastery, MistakeEpisode, SessionSummary)
}

R = TypeVar("R", bound=MemoryRecord)

_PAGE_SIZE = 100


class MemoryNotAllowed(Exception):
    """Raised when a write targets a memory kind outside the allowlist."""


def put_raw(
    user_id: UserId,
    kind: str,
    key: str,
    value: Mapping[str, object],
    store: BaseStore | None = None,
) -> None:
    """Write a raw value to the store, enforcing the allowlist on `kind`.

    This is the graph/LLM boundary where `kind` arrives as an untrusted string.

    The allowlist is checked before the store is touched, so a rejected write never reaches it.
    """
    if kind not in MEMORY_WRITE_ALLOWLIST:
        raise MemoryNotAllowed(
            f"Memory kind {kind!r} is not in the write allowlist: "
            f"{sorted(MEMORY_WRITE_ALLOWLIST)}"
        )
    (store or get_store()).put((kind, str(user_id)), key, dict(value))


def put(
    user_id: UserId,
    key: str,
    record: MemoryRecord,
    store: BaseStore | None = None,
) -> None:
    """Typed convenience wrapper for in-process callers."""
    put_raw(user_id, record.kind.value, key, record.model_dump(mode="json"), store)


def put_planner(
    user_id: UserId,
    kind: str,
    key: str,
    value: Mapping[str, object],
    store: BaseStore | None = None,
) -> None:
    """Write on behalf of the planner: planner allowlist first, then validate `value` for `kind`."""
    if kind not in PLANNER_WRITE_ALLOWLIST:
        raise MemoryNotAllowed(
            f"Memory kind {kind!r} is not in the planner write allowlist: "
            f"{sorted(PLANNER_WRITE_ALLOWLIST)}"
        )
    record = _RECORD_MODELS[MemoryKind(kind)].model_validate(value)
    put_raw(user_id, kind, key, record.model_dump(mode="json"), store)


def get(
    user_id: UserId,
    key: str,
    kind: type[R],
    store: BaseStore | None = None,
) -> R | None:
    """Read a typed record from the store, or None if it does not exist."""
    namespace = (kind.kind.value, str(user_id))
    item = (store or get_store()).get(namespace, key)
    if item is None:
        return None
    return kind.model_validate(item.value)


def list_records(
    user_id: UserId,
    kind: type[R],
    store: BaseStore | None = None,
) -> list[R]:
    """Read every record of a kind for a user, paging past the store's default limit."""
    namespace = (kind.kind.value, str(user_id))
    resolved = store or get_store()
    records: list[R] = []
    while True:
        page = resolved.search(namespace, limit=_PAGE_SIZE, offset=len(records))
        records.extend(kind.model_validate(item.value) for item in page)
        if len(page) < _PAGE_SIZE:
            return records
