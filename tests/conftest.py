"""
Base conftest for all tests.

Loads .env at import time and points DATABASE_URL at the test database so
get_pool() never touches the dev database.
"""

import os

from dotenv import load_dotenv

# Load .env before any test module imports civica.db.pool
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

_test_database_url = os.environ.get("TEST_DATABASE_URL")
if not _test_database_url:
    raise RuntimeError("TEST_DATABASE_URL must be set for tests")

# Redirect DATABASE_URL so get_pool() connects to the test DB
os.environ["DATABASE_URL"] = _test_database_url
