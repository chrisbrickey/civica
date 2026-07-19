"""
Base conftest for all tests.

Loads .env at import time so downstream conftests and tests see the project's
environment variables. Database wiring lives in tests/integration/conftest.py.

Exposes the shared html_fixtures_dir fixture so tests at any depth resolve
fixture paths off the pytest rootdir rather than __file__ arithmetic.
"""

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


@pytest.fixture()
def html_fixtures_dir(pytestconfig: pytest.Config) -> Path:
    """Absolute path to the shared HTML fixtures directory."""
    return pytestconfig.rootpath / "tests" / "fixtures" / "html"
