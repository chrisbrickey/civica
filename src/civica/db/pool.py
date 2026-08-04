import os

import psycopg
import psycopg.rows
import psycopg_pool

_pool: psycopg_pool.ConnectionPool | None = None


def get_pool() -> psycopg_pool.ConnectionPool:
    """Return the singleton connection pool, creating it on first call.

    Connections use dict_row and prepare_threshold=0 because LangGraph's
    PostgresSaver/PostgresStore require both (in addition to autocommit).
    Production consumers that read rows set their own per-cursor row_factory
    (e.g. class_row), so the dict_row default does not affect them.
    """
    global _pool
    if _pool is None:
        database_url = os.environ["DATABASE_URL"]
        _pool = psycopg_pool.ConnectionPool(
            database_url,
            kwargs={
                "autocommit": True,
                "row_factory": psycopg.rows.dict_row,
                "prepare_threshold": 0,
            },
            open=True,
        )
    return _pool


def enable_pgvector(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> None:
    """Enable the pgvector extension if not already installed."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
