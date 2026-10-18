"""Operator telemetry in plain SQL, outside the LangGraph store and the memory allowlist."""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import psycopg
import psycopg.rows
from psycopg.types.json import Jsonb

from civica.db.session import run_on_connection
from civica.domain.themes import Theme
from civica.domain.user import UserId

_INSERT_STEP_SQL = """
INSERT INTO planner_trace (
    user_id, thread_id, turn_id, step_index, tool_name, tool_args, halt_reason, prompt_version
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_GAP_SQL = "INSERT INTO retrieval_gaps (theme, query) VALUES (%s, %s)"


class HaltReason(StrEnum):
    FINAL_MESSAGE = "final_message"
    MAX_STEPS = "max_steps"
    FORBIDDEN_WRITE = "forbidden_write"


@dataclass(frozen=True)
class TraceStep:
    """One planner row: a tool call or the single terminal row of a turn."""

    user_id: UserId
    thread_id: str
    turn_id: uuid.UUID
    step_index: int
    tool_name: str | None
    tool_args: Mapping[str, object] | None
    halt_reason: HaltReason | None
    prompt_version: str


@dataclass(frozen=True)
class RetrievalGap:
    """A query that retrieved nothing for a theme."""

    theme: Theme
    query: str


class TraceWriter(Protocol):
    """Appends planner steps and retrieval gaps to the operator log."""

    def record_step(self, step: TraceStep) -> None: ...

    def record_gap(self, gap: RetrievalGap) -> None: ...


def record_step(step: TraceStep, conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None) -> None:
    """Append one planner_trace row; the database enforces the tool/halt_reason pairing."""
    params = (
        step.user_id,
        step.thread_id,
        step.turn_id,
        step.step_index,
        step.tool_name,
        Jsonb(dict(step.tool_args)) if step.tool_args is not None else None,
        step.halt_reason.value if step.halt_reason is not None else None,
        step.prompt_version,
    )
    run_on_connection(lambda connection: connection.execute(_INSERT_STEP_SQL, params), conn)


def record_gap(gap: RetrievalGap, conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None) -> None:
    """Append one retrieval_gaps row."""
    run_on_connection(
        lambda connection: connection.execute(_INSERT_GAP_SQL, (gap.theme.slug, gap.query)), conn
    )


class PostgresTraceWriter:
    """TraceWriter backed by the planner_trace and retrieval_gaps tables."""

    def record_step(self, step: TraceStep) -> None:
        record_step(step)

    def record_gap(self, gap: RetrievalGap) -> None:
        record_gap(gap)
