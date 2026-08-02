"""
Integration-tier conftest.

Requires TEST_DATABASE_URL and redirects DATABASE_URL at it so integration
tests never touch the dev database. Unit tests do not load this fixture and
run without any database dependency.
"""

import os
import uuid
from collections.abc import Callable, Generator

import psycopg
import psycopg.rows
import pytest

from civica.db.migrate import apply_schema
from civica.ingestion.repository import ChunkRow


def _redirect_database_url_to_test_db() -> None:
    test_database_url = os.environ.get("TEST_DATABASE_URL")
    if not test_database_url:
        raise RuntimeError("TEST_DATABASE_URL must be set for integration tests")
    os.environ["DATABASE_URL"] = test_database_url


_redirect_database_url_to_test_db()

_DB_REACHABILITY_TIMEOUT_SECONDS = 2


@pytest.fixture(scope="session", autouse=True)
def _require_reachable_test_db() -> None:
    """Fail the integration tier fast when the test database is unreachable.

    Without this check, an unreachable database (e.g. the Docker container is
    not running) wastes the 30 second pool-checkout timeout in every single test.
    When this smoke test fails, the test file fails fast.
    """
    test_database_url = os.environ["TEST_DATABASE_URL"]
    try:
        psycopg.connect(
            test_database_url, connect_timeout=_DB_REACHABILITY_TIMEOUT_SECONDS
        ).close()
    except psycopg.OperationalError as error:
        pytest.fail(
            f"Cannot reach the test database at TEST_DATABASE_URL: {error}. "
            "Is Postgres running? Try: docker compose up -d",
            pytrace=False,
        )


@pytest.fixture()
def db_schema() -> Generator[psycopg.Connection[psycopg.rows.TupleRow], None, None]:
    """Yield a connection scoped to a fresh, isolated Postgres schema for each test.

    A dedicated connection is opened outside the shared pool (civica.db.pool.get_pool)
    to avoid the schema created for one test from being visible to another test.

    Flow: The test schema is created. apply_schema() runs against that schema.
    This private connection is yielded directly to the test.
    On teardown, the test schema is dropped and the private connection is closed.
    """
    test_database_url = os.environ["TEST_DATABASE_URL"]
    conn: psycopg.Connection[psycopg.rows.TupleRow] = psycopg.connect(
        test_database_url, autocommit=True
    )
    schema_name = f"test_{uuid.uuid4().hex}"
    try:
        conn.execute(f"CREATE SCHEMA {schema_name}")
        # public must stay on the search_path: the vector extension (and
        # therefore the VECTOR type) is installed in public, not per-schema.
        conn.execute(f"SET search_path TO {schema_name}, public")
        apply_schema(conn=conn)
        yield conn
    finally:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        conn.close()


@pytest.fixture()
def make_chunk_row() -> Callable[..., ChunkRow]:
    """Factory for ChunkRow test rows with generic defaults.

    Shared by the repository and retrieval test modules so the boilerplate
    fields (page_slug, section_id, chunk_index, text) are defined once;
    tests pass only the fields their assertions care about.
    """

    def _make(
        *,
        theme: str,
        embedding: list[float],
        content_hash: str,
        text: str = "sample-text",
        page_slug: str = "sample-page",
        section_id: str = "section-001",
        chunk_index: int = 0,
    ) -> ChunkRow:
        return ChunkRow(
            theme=theme,
            page_slug=page_slug,
            section_id=section_id,
            chunk_index=chunk_index,
            content_hash=content_hash,
            text=text,
            embedding=embedding,
        )

    return _make
