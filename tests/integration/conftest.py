"""
Integration-tier conftest.

Requires TEST_DATABASE_URL and redirects DATABASE_URL at it so integration
tests never touch the dev database. Unit tests do not load this fixture and
run without any database dependency.
"""

import os


def _redirect_database_url_to_test_db() -> None:
    test_database_url = os.environ.get("TEST_DATABASE_URL")
    if not test_database_url:
        raise RuntimeError("TEST_DATABASE_URL must be set for integration tests")
    os.environ["DATABASE_URL"] = test_database_url


_redirect_database_url_to_test_db()
