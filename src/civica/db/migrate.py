from pathlib import Path

import psycopg

from civica.db.pool import enable_pgvector
from civica.db.session import run_on_connection
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Advisory lock keys so concurrent DB setup requests (e.g., concurrent test workers) queue instead of deadlocking.
# The extension is shared by every schema. Tables are locked per schema so separate schemas run in parallel.
_EXTENSION_LOCK_KEY = 7_204_311_905
_SCHEMA_LOCK_CLASS = 7_204


def _run_schema(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> None:
    """Enable pgvector and execute schema.sql against the given connection."""
    raw = _SCHEMA_PATH.read_text()
    sql = "\n".join(
        line for line in raw.splitlines() if line.strip() and not line.strip().startswith("--")
    ).strip()
    sql = sql.replace("{embedding_dim}", str(EMBEDDING_DIMENSIONS))
    # Connections run in autocommit. So explicit transactions keep each transaction lock held until commit.
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_EXTENSION_LOCK_KEY,))
        enable_pgvector(conn)
    with conn.transaction():
        conn.execute(
            "SELECT pg_advisory_xact_lock(%s, hashtext(current_schema()))", (_SCHEMA_LOCK_CLASS,)
        )
        if sql:
            # schema.sql is a single multi-command string. The shared pool sets prepare_threshold=0, LangGraph requirement.
            # Postgres cannot prepare a statement containing multiple commands so we pass `prepare=False` to override
            # the default behavior of trying to prepare everything first.
            conn.execute(sql, prepare=False)


def apply_schema(conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None) -> None:
    """Enable pgvector and execute schema.sql."""
    run_on_connection(_run_schema, conn)
