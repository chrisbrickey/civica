"""Fixtures for the web app tests."""

from collections.abc import AsyncGenerator, Generator

import httpx
import psycopg
import psycopg.rows
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from nicegui import core
from nicegui.testing import User, user_simulation

from civica.domain.user import UserId
from civica.graph.graph import GraphInput, GraphState, build_graph
from civica.graph.nodes import PlannerContext
from civica.memory.checkpointer import state_serializer
from civica.ui.deps import AppDeps
from civica.ui.pages import register_pages
from civica.ui.turns import TurnRunner
from civica.users.service import register
from tests.support.assessment import FakeChatModel
from tests.support.planner import (
    ControlledPlannerModel,
    PlannerFakes,
    make_planner_fakes,
    planner_deps,
)
from tests.support.ui import PASSWORD, PROMPT_INPUT_MARKER, USERNAME, sign_in
from tests.support.users import TEST_BCRYPT_ROUNDS

_CoachGraph = CompiledStateGraph[GraphState, PlannerContext, GraphInput, GraphState]


@pytest.fixture()
def planner_model() -> Generator[ControlledPlannerModel, None, None]:
    """The planner of the app under test. Tests set `script` before submitting a prompt."""
    model = ControlledPlannerModel()
    yield model
    model.release()


@pytest.fixture()
def content_model() -> FakeChatModel:
    """Writes lessons and questions. Tests set `replies` to script the lesson text."""
    return FakeChatModel()


@pytest.fixture()
def planner_fakes(content_model: FakeChatModel) -> PlannerFakes:
    return make_planner_fakes(InMemoryStore(), content_model)


@pytest.fixture()
def coach_graph(planner_fakes: PlannerFakes, planner_model: ControlledPlannerModel) -> _CoachGraph:
    """The real graph with only external services faked, on the production allow-list of state types."""
    saver = InMemorySaver(serde=state_serializer())
    return build_graph(planner_deps(planner_fakes, planner_model), checkpointer=saver)


@pytest.fixture()
def turn_runner(coach_graph: _CoachGraph) -> TurnRunner:
    return TurnRunner(coach_graph)


@pytest.fixture()
async def ui_user(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    coach_graph: _CoachGraph,
    turn_runner: TurnRunner,
) -> AsyncGenerator[User, None]:
    """A simulated browser on the app, wired to the per-test schema connection."""
    async with user_simulation() as user:
        register_pages(
            AppDeps(
                graph=coach_graph,
                turn_runner=turn_runner,
                conn=db_schema,
                bcrypt_rounds=TEST_BCRYPT_ROUNDS,
            )
        )
        yield user


@pytest.fixture()
def learner_id(db_schema: psycopg.Connection[psycopg.rows.TupleRow]) -> UserId:
    """The account `signed_in_user` signs in to."""
    return register(USERNAME, PASSWORD, conn=db_schema, rounds=TEST_BCRYPT_ROUNDS)


async def _sign_in_to_coach(user: User) -> None:
    await sign_in(user, USERNAME, PASSWORD)
    await user.should_see(marker=PROMPT_INPUT_MARKER)


@pytest.fixture()
async def signed_in_user(ui_user: User, learner_id: UserId) -> User:
    """A simulated browser already on the coach page."""
    await _sign_in_to_coach(ui_user)
    return ui_user


@pytest.fixture()
async def second_browser(signed_in_user: User) -> AsyncGenerator[User, None]:
    """Another simulated browser with its own session, signed in to the same account."""
    transport = httpx.ASGITransport(core.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        user = User(client)
        await _sign_in_to_coach(user)
        yield user
