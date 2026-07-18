from pathlib import Path

from civica.db.pool import enable_pgvector, get_pool

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def apply_schema() -> None:
    """Enable pgvector and execute schema.sql against the database."""
    pool = get_pool()
    with pool.connection() as conn:
        enable_pgvector(conn)
        raw = _SCHEMA_PATH.read_text()
        sql = "\n".join(
            line for line in raw.splitlines() if line.strip() and not line.strip().startswith("--")
        ).strip()
        if sql:
            conn.execute(sql)
