"""
Tests for civica.db.migrate public API.

These tests run against the local test database (civica_test via docker-compose).
They are NOT marked @pytest.mark.external because local docker Postgres is a normal
test dependency, not a remote service.
"""

from civica.db.migrate import apply_schema
from civica.db.pool import get_pool


def test_apply_schema_enables_vector_extension() -> None:
    """apply_schema() results in the vector extension being present in pg_extension.
    apply_schema() should enable the pgvector extension."""
    apply_schema()
    pool = get_pool()
    with pool.connection() as conn:
        result = conn.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    assert result is not None
    assert result[0] == "vector"


def test_apply_schema_is_idempotent() -> None:
    """Calling apply_schema() twice does not raise.
    apply_schema() is idempotent so calling it twice should not raise."""
    apply_schema()
    apply_schema()
