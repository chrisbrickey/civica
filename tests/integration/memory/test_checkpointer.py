"""
Integration tests for civica.memory.checkpointer.

Exercises the real PostgresSaver against the test database. LangGraph's
setup() creates its own tables in the public schema, outside schema.sql, so
the db_schema fixture does not isolate them. These tests therefore use unique
per-test thread ids and delete every thread they create on teardown.
"""

from collections.abc import Generator

import pytest
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.runnables import RunnableConfig

from civica.memory.checkpointer import get_saver

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THREAD_ID = "test-thread-checkpointer"
_UNRELATED_THREAD_ID = "test-thread-unrelated"
_EMPTY_NAMESPACE = ""


def _config_for(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": _EMPTY_NAMESPACE}}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def saver() -> Generator[PostgresSaver, None, None]:
    """The singleton saver, cleaning up any threads created by the test."""
    live_saver = get_saver()
    try:
        yield live_saver
    finally:
        live_saver.delete_thread(_THREAD_ID)
        live_saver.delete_thread(_UNRELATED_THREAD_ID)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_saved_checkpoint_can_be_restored(saver: PostgresSaver) -> None:
    """A checkpoint written for a thread is readable back for that thread."""
    checkpoint = empty_checkpoint()

    saver.put(
        _config_for(_THREAD_ID),
        checkpoint,
        CheckpointMetadata(),
        {},
    )

    restored = saver.get_tuple(_config_for(_THREAD_ID))

    assert restored is not None
    assert restored.checkpoint["id"] == checkpoint["id"]


def test_unrelated_thread_returns_none(saver: PostgresSaver) -> None:
    """A thread that was never written returns no checkpoint."""
    saver.put(
        _config_for(_THREAD_ID),
        empty_checkpoint(),
        CheckpointMetadata(),
        {},
    )

    restored = saver.get_tuple(_config_for(_UNRELATED_THREAD_ID))

    assert restored is None
