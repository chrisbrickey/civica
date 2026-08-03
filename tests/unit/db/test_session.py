"""Unit tests for civica.db.session: the connection-acquisition dispatch.

The shared pool is mocked, so no real calls to database.
"""

from typing import cast
from unittest.mock import MagicMock

import psycopg
import psycopg.rows
import pytest

from civica.db import session

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIRECT_RESULT = "direct-result"
_POOLED_RESULT = "pooled-result"


# ---------------------------------------------------------------------------
# Branch 1: a connection is supplied
# ---------------------------------------------------------------------------


def test_runs_operation_on_supplied_connection_without_touching_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied_conn = cast(psycopg.Connection[psycopg.rows.TupleRow], object())
    pool = MagicMock()
    monkeypatch.setattr(session, "get_pool", lambda: pool)

    received: list[object] = []

    def operation(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> str:
        received.append(conn)
        return _DIRECT_RESULT

    result = session.run_on_connection(operation, supplied_conn)

    assert result == _DIRECT_RESULT
    assert received == [supplied_conn]
    pool.connection.assert_not_called()


# ---------------------------------------------------------------------------
# Branch 2: no connection supplied, one is borrowed from the pool
# ---------------------------------------------------------------------------


def test_checks_out_pool_connection_when_none_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pooled_conn = object()
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = pooled_conn
    monkeypatch.setattr(session, "get_pool", lambda: pool)

    received: list[object] = []

    def operation(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> str:
        received.append(conn)
        return _POOLED_RESULT

    result = session.run_on_connection(operation, None)

    assert result == _POOLED_RESULT
    assert received == [pooled_conn]
    pool.connection.assert_called_once_with()
    pool.connection.return_value.__exit__.assert_called_once()
