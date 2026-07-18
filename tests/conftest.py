"""
Base conftest for all tests.

Loads .env at import time so downstream conftests and tests see the project's
environment variables. Database wiring lives in tests/integration/conftest.py.
"""

import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
