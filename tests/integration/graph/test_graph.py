"""Integration tests for civica.graph.graph on the real checkpointer, store, and trace tables.

Runs on default_planner_deps() with only the chat models and the embedding client faked. No network call.

The default stores share the pool and the public schema of the test database,
which db_schema does not isolate, so each test uses its own user and thread and removes its rows.
"""

import logging
import uuid
from collections.abc import Callable, Generator
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from civica.db.migrate import apply_schema
from civica.db.pool import get_pool
from civica.domain.question import Question
from civica.domain.themes import DROITS_ET_DEVOIRS, VIVRE_DANS_LA_SOCIETE_FRANCAISE
from civica.domain.user import UserId
from civica.embeddings import embedder
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from civica.explanation.engine import Explanation
from civica.graph.graph import InputKind, build_graph
from civica.graph.nodes import PlannerContext
from civica.graph.planner import SUMMARY_KEY, HaltReason, ToolCall, default_planner_deps
from civica.ingestion.repository import EmbeddedChunk, upsert_chunks
from civica.memory.checkpointer import get_saver
from civica.memory.records import MemoryKind, SessionSummary, TopicMastery
from civica.memory.store import get_store
from civica.memory.writer import get, list_records
from civica.users.service import register
from tests.support.assessment import FakeChatModel, explanation_for, grouped_generation_reply
from tests.support.memory import FULL_MASTERY
from tests.support.planner import FakePlannerModel, final_message, tool_call
from tests.support.users import TEST_BCRYPT_ROUNDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SECRET = "sample-secret"
_LEARNER_MESSAGE = "sample-learner-message-001"
_SECOND_LEARNER_MESSAGE = "sample-learner-message-002"
_FINAL_TEXT = "sample-final-reply"
_SECOND_FINAL_TEXT = "sample-second-final-reply"
_FOCUS = "sample-focus"
_LESSON_TEXT = "sample-lesson-text"
_QUESTION_TEXT = "sample-graph-question-text"
_CHUNK_HASH = "sample-graph-chunk-hash"

_THEME = DROITS_ET_DEVOIRS
# Never seeded, so retrieval on this theme comes back empty.
_EMPTY_THEME = VIVRE_DANS_LA_SOCIETE_FRANCAISE

_READ_STATE = "read_learner_state"
_SUGGEST = "suggest_review_theme"
_RETRIEVE = "retrieve_corpus"
_TEACH = "teach"
_GENERATE_QUIZ = "generate_quiz"
_RECORD = "record_outcome"

# Identical query and chunk vectors give similarity 1, so the seeded chunk clears every floor.
_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)

_SERIALIZER_LOGGER = "langgraph.checkpoint.serde"

_TRACE_ROWS_SQL = """
SELECT tool_name, halt_reason, step_index FROM planner_trace
WHERE user_id = %s AND thread_id = %s
ORDER BY step_index
"""
_GAP_ROWS_SQL = "SELECT theme, query FROM retrieval_gaps WHERE query = %s"

_Graph = CompiledStateGraph[Any, PlannerContext, Any, Any]  # type: ignore[explicit-any]
_State = dict[str, Any]  # type: ignore[explicit-any]


class _FakeEmbeddingClient:
    """Stands in for the OpenAI client, the one network call behind the default retrievers."""

    def embed_query(self, _text: str) -> list[float]:
        return _VECTOR


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _execute(sql: str, *params: object) -> list[dict[str, object]]:
    with get_pool().connection() as conn:
        cursor = conn.execute(sql, params)
        rows: list[dict[str, object]] = cursor.fetchall() if cursor.description else []
        return rows


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(embedder._clients, EMBEDDING_MODEL, _FakeEmbeddingClient())


@pytest.fixture()
def user_id() -> Generator[UserId, None, None]:
    """A user in the public schema, removed with everything its turns wrote."""
    apply_schema()
    new_user = register(f"test-{uuid.uuid4().hex[:16]}", _SECRET, rounds=TEST_BCRYPT_ROUNDS)
    (last_gap,) = _execute("SELECT COALESCE(MAX(id), 0) AS id FROM retrieval_gaps")
    try:
        yield new_user
    finally:
        store = get_store()
        for kind in MemoryKind:
            namespace = (kind.value, str(new_user))
            for item in store.search(namespace, limit=100):
                store.delete(namespace, item.key)
        _execute("DELETE FROM planner_trace WHERE user_id = %s", new_user)
        _execute("DELETE FROM users WHERE user_id = %s", new_user)
        _execute("DELETE FROM retrieval_gaps WHERE id > %s", last_gap["id"])


@pytest.fixture()
def config() -> Generator[RunnableConfig, None, None]:
    thread_id = f"test-thread-{uuid.uuid4().hex}"
    try:
        yield {"configurable": {"thread_id": thread_id}}
    finally:
        get_saver().delete_thread(thread_id)


@pytest.fixture()
def corpus(make_chunk_row: Callable[..., EmbeddedChunk]) -> Generator[None, None, None]:
    """One chunk for _THEME in the public corpus, removed with the question built from it."""
    apply_schema()
    upsert_chunks([make_chunk_row(theme=_THEME, embedding=_VECTOR, content_hash=_CHUNK_HASH)])
    try:
        yield
    finally:
        _execute("DELETE FROM content_chunks WHERE content_hash = %s", _CHUNK_HASH)
        _execute("DELETE FROM generated_questions WHERE text = %s", _QUESTION_TEXT)


def _graph(model: FakePlannerModel) -> _Graph:
    """The production graph with only the chat models faked."""
    content_model = FakeChatModel(replies=(_LESSON_TEXT, grouped_generation_reply((1, _QUESTION_TEXT))))
    return build_graph(replace(default_planner_deps(), planner_model=model, content_model=content_model))


def _turn(
    graph: _Graph, user_id: UserId, config: RunnableConfig, message: str = _LEARNER_MESSAGE
) -> None:
    """One message turn, streamed the way the UI will run it."""
    list(
        graph.stream(
            {"kind": InputKind.MESSAGE, "learner_message": message},
            config=config,
            context=PlannerContext(user_id=user_id),
            subgraphs=True,
        )
    )


def _values(graph: _Graph, config: RunnableConfig) -> _State:
    values: _State = graph.get_state(config).values
    return values


def _scripted(*turns: AIMessage) -> FakePlannerModel:
    return FakePlannerModel(script=tuple(turns))


def _lesson_and_check() -> tuple[AIMessage, ...]:
    """A realistic coach turn: read state, teach, one check question, final message."""
    return (
        tool_call(_READ_STATE, "call-state"),
        tool_call(_TEACH, "call-teach", theme=_THEME.slug, focus=_FOCUS),
        tool_call(_GENERATE_QUIZ, "call-quiz", theme=_THEME.slug, n=1),
        final_message(_FINAL_TEXT),
    )


def _forbidden_write() -> AIMessage:
    return tool_call(
        _RECORD,
        "call-forbidden",
        kind="topic_mastery",
        key=_THEME.slug,
        value={"theme": _THEME.slug, "mastery": FULL_MASTERY},
    )


def _summary(user_id: UserId) -> str:
    record = get(user_id, SUMMARY_KEY, SessionSummary, get_store())
    assert record is not None, "memory_writer wrote no session summary"
    return record.summary


def _trace(user_id: UserId, config: RunnableConfig) -> list[dict[str, object]]:
    return _execute(_TRACE_ROWS_SQL, user_id, config["configurable"]["thread_id"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_graph_compiles_on_postgres(config: RunnableConfig) -> None:
    assert _values(build_graph(), config) == {}


@pytest.mark.usefixtures("corpus")
class TestCoachTurn:
    """A planner-backed turn runs end to end on the default deps."""

    def test_turn_result_survives_the_checkpoint_as_typed_objects(
        self, user_id: UserId, config: RunnableConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = _graph(_scripted(*_lesson_and_check()))

        with caplog.at_level(logging.WARNING):
            _turn(graph, user_id, config)
            values = _values(graph, config)

        assert values["reply"] == _FINAL_TEXT
        assert values["halt_reason"] == HaltReason.FINAL_MESSAGE
        assert isinstance(values["lesson"], Explanation)
        assert values["lesson"].text == _LESSON_TEXT
        assert values["lesson"].context_chunks[0].chunk.theme == _THEME
        (question,) = values["quiz_questions"]
        assert isinstance(question, Question)
        assert question.text == _QUESTION_TEXT
        assert question.explanation == explanation_for(_QUESTION_TEXT)
        assert all(isinstance(call, ToolCall) for call in values["tool_calls"])
        assert [r.message for r in caplog.records if r.name.startswith(_SERIALIZER_LOGGER)] == []

    def test_session_summary_row_is_written(self, user_id: UserId, config: RunnableConfig) -> None:
        _turn(_graph(_scripted(*_lesson_and_check())), user_id, config)

        assert _THEME.slug in _summary(user_id)

    def test_trace_rows_read_back_in_tool_call_order(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        _turn(_graph(_scripted(*_lesson_and_check())), user_id, config)

        rows = _trace(user_id, config)
        assert [row["tool_name"] for row in rows] == [_READ_STATE, _TEACH, _GENERATE_QUIZ, None]
        assert rows[-1]["halt_reason"] == HaltReason.FINAL_MESSAGE.value

    def test_next_turn_can_read_the_summary_of_the_last(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        model = _scripted(
            *_lesson_and_check(),
            tool_call(_READ_STATE, "call-next-state"),
            final_message(_SECOND_FINAL_TEXT),
        )
        graph = _graph(model)

        _turn(graph, user_id, config)
        summary = _summary(user_id)
        _turn(graph, user_id, config, _SECOND_LEARNER_MESSAGE)

        assert summary in model.tool_output("call-next-state")


class TestHaltedTurn:
    """memory_writer still runs when the planner does not end with a final message."""

    def test_step_bound_serves_a_fallback_and_writes_the_summary(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        graph = _graph(FakePlannerModel(repeat=tool_call(_SUGGEST, "call-forever")))

        _turn(graph, user_id, config)

        values = _values(graph, config)
        assert values["halt_reason"] == HaltReason.MAX_STEPS
        assert values["reply"] == values["lesson"].text
        assert HaltReason.MAX_STEPS.value in _summary(user_id)

    def test_forbidden_write_leaves_mastery_untouched_and_serves_a_fallback(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        graph = _graph(_scripted(_forbidden_write()))

        _turn(graph, user_id, config)

        values = _values(graph, config)
        assert list_records(user_id, TopicMastery, get_store()) == []
        assert _trace(user_id, config)[-1]["halt_reason"] == HaltReason.FORBIDDEN_WRITE.value
        assert values["reply"] == values["lesson"].text


class TestRetrievalGap:
    def test_empty_retrieval_writes_a_gap_row(self, user_id: UserId, config: RunnableConfig) -> None:
        query = f"sample-query-{uuid.uuid4().hex}"
        model = _scripted(
            tool_call(_RETRIEVE, "call-retrieve", theme=_EMPTY_THEME.slug, query=query),
            final_message(_FINAL_TEXT),
        )

        _turn(_graph(model), user_id, config)

        assert _execute(_GAP_ROWS_SQL, query) == [{"theme": _EMPTY_THEME.slug, "query": query}]


class TestThreadPersistence:
    """The thread survives between turns, while each turn starts clean."""

    def test_new_graph_on_the_same_thread_sees_the_saved_state(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        _turn(_graph(_scripted(final_message(_FINAL_TEXT))), user_id, config)

        restarted = _graph(_scripted(final_message(_SECOND_FINAL_TEXT)))

        assert _values(restarted, config)["reply"] == _FINAL_TEXT
        _turn(restarted, user_id, config, _SECOND_LEARNER_MESSAGE)
        assert _values(restarted, config)["reply"] == _SECOND_FINAL_TEXT

    def test_message_history_does_not_carry_into_the_next_turn(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        model = _scripted(final_message(_FINAL_TEXT), final_message(_SECOND_FINAL_TEXT))
        graph = _graph(model)

        _turn(graph, user_id, config)
        _turn(graph, user_id, config, _SECOND_LEARNER_MESSAGE)

        humans = [m.content for m in model.received[-1] if isinstance(m, HumanMessage)]
        assert humans == [_SECOND_LEARNER_MESSAGE]
        assert len(model.received[-1]) == len(model.received[0])
        assert "messages" not in _values(graph, config)
