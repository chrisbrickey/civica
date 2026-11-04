"""Fixtures for the web app tests."""

from collections.abc import AsyncGenerator

import psycopg
import psycopg.rows
import pytest
from nicegui.testing import User, user_simulation

from civica.ui.app import register_pages
from civica.ui.deps import AppDeps
from tests.support.users import TEST_BCRYPT_ROUNDS


@pytest.fixture()
async def ui_user(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> AsyncGenerator[User, None]:
    """A simulated browser on the app, wired to the per-test schema connection."""
    async with user_simulation() as user:
        register_pages(AppDeps(conn=db_schema, bcrypt_rounds=TEST_BCRYPT_ROUNDS))
        yield user
