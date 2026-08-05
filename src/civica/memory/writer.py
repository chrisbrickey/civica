"""Allowlist-guarded writer over the LangGraph long-term memory store.

Store key conventions:
- LearnerProfile / SessionSummary: one per user, constant key like "current"
- TopicMastery: keyed by theme.slug
- MistakeEpisode: unique key per episode (e.g. question_id + timestamp)
"""

from collections.abc import Mapping
from typing import TypeVar

from civica.domain.user import UserId
from civica.memory.records import MemoryKind, MemoryRecord
from civica.memory.store import get_store

MEMORY_WRITE_ALLOWLIST: frozenset[MemoryKind] = frozenset(MemoryKind)

R = TypeVar("R", bound=MemoryRecord)


class MemoryNotAllowed(Exception):
    """Raised when a write targets a memory kind outside the allowlist."""


def put_raw(user_id: UserId, kind: str, key: str, value: Mapping[str, object]) -> None:
    """Write a raw value to the store, enforcing the allowlist on `kind`.

    This is the graph/LLM boundary where `kind` arrives as an untrusted string.
    """
    if kind not in MEMORY_WRITE_ALLOWLIST:
        raise MemoryNotAllowed(
            f"Memory kind {kind!r} is not in the write allowlist: "
            f"{sorted(MEMORY_WRITE_ALLOWLIST)}"
        )
    get_store().put((kind, str(user_id)), key, dict(value))


def put(user_id: UserId, key: str, record: MemoryRecord) -> None:
    """Typed convenience wrapper for in-process callers."""
    put_raw(user_id, record.kind.value, key, record.model_dump(mode="json"))


def get(user_id: UserId, key: str, kind: type[R]) -> R | None:
    """Read a typed record from the store, or None if it does not exist."""
    namespace = (kind.kind.value, str(user_id))
    item = get_store().get(namespace, key)
    if item is None:
        return None
    return kind.model_validate(item.value)
