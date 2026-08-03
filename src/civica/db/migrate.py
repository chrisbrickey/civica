from pathlib import Path

import psycopg

from civica.db.pool import enable_pgvector
from civica.db.session import run_on_connection
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS

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
    """Enable pgvector and execute schema.sql."""
    run_on_connection(_run_schema, conn)
