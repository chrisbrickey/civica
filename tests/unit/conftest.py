"""Fixtures shared by the unit tests."""

import pytest
from langgraph.store.memory import InMemoryStore


@pytest.fixture()
def store() -> InMemoryStore:
    """A fresh in-memory LangGraph store, so memory tests need no database."""
    return InMemoryStore()
