"""Integration tests for civica.graph.trace.

Exercises the real schema via the db_schema fixture. No network calls.
planner_trace.user_id references users so a real user is registered first.
"""

import uuid
from dataclasses import replace

import psycopg
import psycopg.errors
import psycopg.rows
import pytest

from civica.domain.themes import DROITS_ET_DEVOIRS
from civica.domain.user import UserId
from civica.graph.trace import (
    HaltReason,
    RetrievalGap,
    TraceStep,
    record_gap,
    record_step,
)
from civica.users.service import register

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USERNAME = "test-user"
_SECRET = "sample-secret"
# bcrypt cost factor: lowered for tests, matching the users-service tests.
_TEST_BCRYPT_ROUNDS = 4

_THREAD_ID = "sample-thread-001"
_OTHER_THREAD_ID = "sample-thread-002"
_TURN_ID = uuid.UUID("00000000-0000-0000-0000-0000000000a1")
_PROMPT_VERSION = "sample-prompt-v1"
_TOOL_NAME = "sample-tool"
_OTHER_TOOL_NAME = "sample-other-tool"
_TOOL_ARGS: dict[str, object] = {"theme": "sample-theme", "n": 2, "nested": {"key": "sample-value"}}
_UNREGISTERED_USER_ID = UserId(uuid.UUID("00000000-0000-0000-0000-0000000000ff"))

_THEME = DROITS_ET_DEVOIRS
_QUERY = "sample-query"

_STEP_ROWS_SQL = """
SELECT user_id, thread_id, turn_id, step_index, tool_name, tool_args, halt_reason,
       prompt_version, created_at
FROM planner_trace
WHERE user_id = %s AND thread_id = %s
ORDER BY step_index
"""

_GAP_ROWS_SQL = "SELECT theme, query, occurred_at FROM retrieval_gaps ORDER BY id"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def user_id(db_schema: psycopg.Connection[psycopg.rows.TupleRow]) -> UserId:
    return register(_USERNAME, _SECRET, conn=db_schema, rounds=_TEST_BCRYPT_ROUNDS)


def _tool_step(user_id: UserId, step_index: int, tool_name: str = _TOOL_NAME) -> TraceStep:
    return TraceStep(
        user_id=user_id,
        thread_id=_THREAD_ID,
        turn_id=_TURN_ID,
        step_index=step_index,
        tool_name=tool_name,
        tool_args=_TOOL_ARGS,
        halt_reason=None,
        prompt_version=_PROMPT_VERSION,
    )


def _terminal_step(user_id: UserId, step_index: int) -> TraceStep:
    return replace(
        _tool_step(user_id, step_index),
        tool_name=None,
        tool_args=None,
        halt_reason=HaltReason.MAX_STEPS,
    )


def _step_rows(
    conn: psycopg.Connection[psycopg.rows.TupleRow], user_id: UserId, thread_id: str = _THREAD_ID
) -> list[dict[str, object]]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(_STEP_ROWS_SQL, (user_id, thread_id))
        return cursor.fetchall()


def _gap_rows(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> list[dict[str, object]]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(_GAP_ROWS_SQL)
        return cursor.fetchall()


# ---------------------------------------------------------------------------
# record_step
# ---------------------------------------------------------------------------


class TestRecordStep:
    """A step row round-trips every field, with tool_args as JSONB."""

    def test_tool_step_round_trips(
        self, user_id: UserId, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
    ) -> None:
        record_step(_tool_step(user_id, 0), conn=db_schema)

        (row,) = _step_rows(db_schema, user_id)
        assert row["user_id"] == user_id
        assert row["thread_id"] == _THREAD_ID
        assert row["turn_id"] == _TURN_ID
        assert row["step_index"] == 0
        assert row["tool_name"] == _TOOL_NAME
        assert row["tool_args"] == _TOOL_ARGS
        assert row["halt_reason"] is None
        assert row["prompt_version"] == _PROMPT_VERSION
        assert row["created_at"] is not None

    def test_terminal_step_stores_the_halt_reason(
        self, user_id: UserId, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
    ) -> None:
        record_step(_terminal_step(user_id, 0), conn=db_schema)

        (row,) = _step_rows(db_schema, user_id)
        assert row["tool_name"] is None
        assert row["tool_args"] is None
        assert row["halt_reason"] == HaltReason.MAX_STEPS.value

    def test_rows_read_back_by_user_and_thread_in_step_order(
        self, user_id: UserId, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
    ) -> None:
        steps = [
            _tool_step(user_id, 0),
            _tool_step(user_id, 1, _OTHER_TOOL_NAME),
            _terminal_step(user_id, 2),
        ]
        for step in reversed(steps):
            record_step(step, conn=db_schema)
        record_step(replace(_tool_step(user_id, 0), thread_id=_OTHER_THREAD_ID), conn=db_schema)

        rows = _step_rows(db_schema, user_id)
        assert [(r["step_index"], r["tool_name"]) for r in rows] == [
            (s.step_index, s.tool_name) for s in steps
        ]

    def test_unregistered_user_is_rejected(
        self, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
    ) -> None:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            record_step(_tool_step(_UNREGISTERED_USER_ID, 0), conn=db_schema)

    @pytest.mark.parametrize(
        ("tool_name", "halt_reason"),
        [(_TOOL_NAME, HaltReason.FINAL_MESSAGE), (None, None)],
        ids=["both", "neither"],
    )
    def test_row_needs_exactly_one_of_tool_name_or_halt_reason(
        self,
        user_id: UserId,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
        tool_name: str | None,
        halt_reason: HaltReason | None,
    ) -> None:
        step = replace(_tool_step(user_id, 0), tool_name=tool_name, halt_reason=halt_reason)

        with pytest.raises(psycopg.errors.CheckViolation):
            record_step(step, conn=db_schema)


# ---------------------------------------------------------------------------
# record_gap
# ---------------------------------------------------------------------------


class TestRecordGap:
    """Each gap is one timestamped row keyed by theme slug; the log is append-only."""

    def test_gap_round_trips(self, db_schema: psycopg.Connection[psycopg.rows.TupleRow]) -> None:
        record_gap(RetrievalGap(theme=_THEME, query=_QUERY), conn=db_schema)

        (row,) = _gap_rows(db_schema)
        assert row["theme"] == _THEME.slug
        assert row["query"] == _QUERY
        assert row["occurred_at"] is not None

    def test_repeated_gap_appends(self, db_schema: psycopg.Connection[psycopg.rows.TupleRow]) -> None:
        gap = RetrievalGap(theme=_THEME, query=_QUERY)

        record_gap(gap, conn=db_schema)
        record_gap(gap, conn=db_schema)

        assert len(_gap_rows(db_schema)) == 2

