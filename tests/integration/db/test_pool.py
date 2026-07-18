"""
Tests for civica.db.pool public API.

These tests run against the local test database (civica_test via docker-compose).
They are NOT marked @pytest.mark.external because local docker Postgres is a normal
test dependency, not a remote service.
"""

from civica.db.pool import enable_pgvector, get_pool


def test_get_pool_returns_working_pool() -> None:
    """A connection checked out from get_pool() can execute a basic query.
    get_pool() returns a working pool; SELECT 1 succeeds through a checked-out connection."""
    pool = get_pool()
    with pool.connection() as conn:
        result = conn.execute("SELECT 1").fetchone()
    assert result is not None
    assert result[0] == 1


def test_enable_pgvector_is_idempotent() -> None:
    """Calling enable_pgvector twice on the same connection does not raise.
    enable_pgvector() is idempotent so calling it twice should not raise."""
    pool = get_pool()
    with pool.connection() as conn:
        enable_pgvector(conn)
        enable_pgvector(conn)
