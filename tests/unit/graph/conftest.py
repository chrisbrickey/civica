"""Fixtures shared by the graph unit tests."""

import pytest
from langgraph.store.memory import InMemoryStore

from tests.support.assessment import (
    FakeAnswerLogger,
    FakeChatModel,
    FakeQuestionStore,
    FakeRejectionStore,
    FakeRetriever,
)
from tests.support.planner import PlannerFakes


@pytest.fixture()
def planner_fakes(store: InMemoryStore) -> PlannerFakes:
    return PlannerFakes(
        store=store,
        retriever=FakeRetriever(),
        content_model=FakeChatModel(),
        question_store=FakeQuestionStore(),
        answer_logger=FakeAnswerLogger(),
        rejection_store=FakeRejectionStore(),
    )
