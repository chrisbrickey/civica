from pathlib import Path

import psycopg

from civica.db.pool import enable_pgvector, get_pool
from civica.ingest.embedder import EMBEDDING_DIMENSIONS

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _run_schema(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> None:
    """Enable pgvector and execute schema.sql against the given connection."""
    enable_pgvector(conn)
    raw = _SCHEMA_PATH.read_text()
    sql = "\n".join(
        line for line in raw.splitlines() if line.strip() and not line.strip().startswith("--")
    ).strip()
    sql = sql.replace("{embedding_dim}", str(EMBEDDING_DIMENSIONS))
    if sql:
        conn.execute(sql)


def apply_schema(conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None) -> None:
    """Enable pgvector and execute schema.sql.

    When connection is provided, the schema runs against that connection directly.
    When connection is None, a connection is checked out from the shared pool.
    """
    if conn is not None:
        _run_schema(conn)
        return

    pool = get_pool()
    with pool.connection() as pool_conn:
        _run_schema(pool_conn)
