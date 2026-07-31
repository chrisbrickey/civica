"""
Integration-tier conftest.

Requires TEST_DATABASE_URL and redirects DATABASE_URL at it so integration
tests never touch the dev database. Unit tests do not load this fixture and
run without any database dependency.
"""

import os
import uuid
from collections.abc import Generator

import psycopg
import psycopg.rows
import pytest

from civica.db.migrate import apply_schema


def _redirect_database_url_to_test_db() -> None:
    test_database_url = os.environ.get("TEST_DATABASE_URL")
    if not test_database_url:
        raise RuntimeError("TEST_DATABASE_URL must be set for integration tests")
    os.environ["DATABASE_URL"] = test_database_url


_redirect_database_url_to_test_db()


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
