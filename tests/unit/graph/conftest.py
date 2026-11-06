"""Fixtures shared by the graph unit tests."""

import pytest
from langgraph.store.memory import InMemoryStore

from tests.support.planner import PlannerFakes, make_planner_fakes


@pytest.fixture()
def planner_fakes(store: InMemoryStore) -> PlannerFakes:
    return make_planner_fakes(store)
