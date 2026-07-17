import os

import psycopg
import psycopg_pool

_pool: psycopg_pool.ConnectionPool | None = None


def get_pool() -> psycopg_pool.ConnectionPool:
    """Return the singleton connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        database_url = os.environ["DATABASE_URL"]
        _pool = psycopg_pool.ConnectionPool(
            database_url,
            kwargs={"autocommit": True},
            open=True,
        )
    return _pool


def enable_pgvector(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> None:
    """Enable the pgvector extension if not already installed."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
