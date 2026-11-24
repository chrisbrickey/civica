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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from civica.assessment.critic import LlmCritic
from civica.db.pool import get_pool
from civica.domain.question import Question
from civica.domain.themes import DROITS_ET_DEVOIRS, VIVRE_DANS_LA_SOCIETE_FRANCAISE
from civica.domain.user import UserId
from civica.embeddings import embedder
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from civica.explanation.engine import Explanation
from civica.graph.graph import InputKind, build_graph
from civica.graph.nodes import PlannerContext
from civica.graph.planner import (
    SUMMARY_KEY,
    HaltReason,
    PreviousExchange,
    ToolCall,
    default_planner_deps,
)
from civica.ingestion.repository import EmbeddedChunk, upsert_chunks
from civica.llm.usage import TurnUsage, UsageTally
from civica.memory.checkpointer import get_saver
from civica.memory.records import MemoryKind, MistakeEpisode, SessionSummary, TopicMastery
from civica.memory.store import get_store
from civica.memory.writer import get, list_records
from civica.users.service import register
from tests.support.answers import (
    ALREADY_ANSWERED_REPLY,
    EXPIRED_REPLY,
    UNKNOWN_QUESTION_ID,
    answer_input,
    answer_recap,
)
from tests.support.assessment import FakeChatModel, explanation_for, grouped_generation_reply, wrong_index
from tests.support.llm import CONTENT_USAGE, EXPLANATION_TAG, PLANNER_TAG, PLANNER_USAGE
from tests.support.logs import lines_from, only
from tests.support.memory import FULL_MASTERY
from tests.support.planner import (
    GENERATE_QUIZ,
    READ_STATE,
    RECORD_OUTCOME,
    RETRIEVE_CORPUS,
    SUGGEST_THEME,
    TEACH,
    FakePlannerModel,
    final_message,
    tool_call,
)
from tests.support.usage import counts_of, turn_usage
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

# One worker for this file: user_id cleans up retrieval_gaps by id range, which would delete other workers' rows.
pytestmark = pytest.mark.xdist_group("public_graph")

_THEME = DROITS_ET_DEVOIRS
# Never seeded, so retrieval on this theme comes back empty.
_EMPTY_THEME = VIVRE_DANS_LA_SOCIETE_FRANCAISE


# Identical query and chunk vectors give similarity 1, so the seeded chunk clears every floor.
_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)

_SERIALIZER_LOGGER = "langgraph.checkpoint.serde"
_GRAPH_LOGGER = "civica.graph.graph"
_ANSWER_SCORED = "answer_scored"
_RIGHT = "right"
_WRONG = "wrong"
_EXPIRED = "expired"
_ALREADY_ANSWERED = "already_answered"

_TRACE_ROWS_SQL = """
SELECT tool_name, halt_reason, step_index FROM planner_trace
WHERE user_id = %s AND thread_id = %s
ORDER BY step_index
"""
_GAP_ROWS_SQL = "SELECT theme, query FROM retrieval_gaps WHERE query = %s"
_ANSWER_ROWS_SQL = """
SELECT theme, question_id, chosen_index, correct_index, is_correct FROM quiz_answers
WHERE user_id = %s
ORDER BY id
"""

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
def user_id(public_schema: None) -> Generator[UserId, None, None]:
    """A user in the public schema, removed with everything its turns wrote."""
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
        _execute("DELETE FROM quiz_answers WHERE user_id = %s", new_user)
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
def corpus(
    public_schema: None, make_chunk_row: Callable[..., EmbeddedChunk]
) -> Generator[None, None, None]:
    """One chunk for _THEME in the public corpus, removed with the question built from it."""
    # A hash per test, so parallel workers do not delete each other's chunk.
    chunk_hash = f"{_CHUNK_HASH}-{uuid.uuid4().hex}"
    upsert_chunks([make_chunk_row(theme=_THEME, embedding=_VECTOR, content_hash=chunk_hash)])
    try:
        yield
    finally:
        _execute("DELETE FROM content_chunks WHERE content_hash = %s", chunk_hash)
        _execute("DELETE FROM generated_questions WHERE text = %s", _QUESTION_TEXT)


def _content_model() -> FakeChatModel:
    """Writes the lesson, then the quiz. Each reply reports CONTENT_USAGE under the explanation caller."""
    return FakeChatModel(
        replies=(_LESSON_TEXT, grouped_generation_reply((1, _QUESTION_TEXT))),
        tags=[EXPLANATION_TAG],
        usage=CONTENT_USAGE,
    )


@pytest.fixture()
def answer_content_model() -> Generator[FakeChatModel, None, None]:
    """Writes the one-question quiz; any further call fails.

    The question text is unique per test, so parallel tests never delete the bank row mastery reads."""
    text = f"{_QUESTION_TEXT}-{uuid.uuid4().hex}"
    try:
        yield FakeChatModel(
            replies=(grouped_generation_reply((1, text)),),
            tags=[EXPLANATION_TAG],
            usage=CONTENT_USAGE,
            fail_when_exhausted=True,
        )
    finally:
        _execute("DELETE FROM generated_questions WHERE text = %s", text)


def _graph(model: FakePlannerModel, content_model: FakeChatModel | None = None) -> _Graph:
    """The production graph with only the chat models faked. The critic stays off, as in every test deps."""
    deps = replace(
        default_planner_deps(), planner_model=model, content_model=content_model or _content_model(), critic=None
    )
    return build_graph(deps)


def _stream(
    graph: _Graph, user_id: UserId, config: RunnableConfig, graph_input: dict[str, object], tally: UsageTally | None
) -> None:
    """One turn, streamed the way the UI runs it."""
    counted = tally is not None
    list(
        graph.stream(
            graph_input,
            config={**config, "callbacks": [tally]} if counted else config,
            context=PlannerContext(user_id=user_id, usage=tally) if counted else PlannerContext(user_id=user_id),
            subgraphs=True,
        )
    )


def _turn(
    graph: _Graph,
    user_id: UserId,
    config: RunnableConfig,
    message: str = _LEARNER_MESSAGE,
    tally: UsageTally | None = None,
) -> None:
    """One message turn."""
    _stream(graph, user_id, config, {"kind": InputKind.MESSAGE, "learner_message": message}, tally)


def _answer(
    graph: _Graph,
    user_id: UserId,
    config: RunnableConfig,
    question_id: str,
    chosen_index: int,
    tally: UsageTally | None = None,
) -> None:
    """One answer turn for a checkpointed question."""
    _stream(graph, user_id, config, answer_input(question_id, chosen_index), tally)


def _values(graph: _Graph, config: RunnableConfig) -> _State:
    values: _State = graph.get_state(config).values
    return values


def _scripted(*turns: AIMessage) -> FakePlannerModel:
    """A planner model whose every reply reports PLANNER_USAGE under the planner caller."""
    return FakePlannerModel(script=tuple(turns), tags=[PLANNER_TAG], usage=PLANNER_USAGE)


def _lesson() -> tuple[AIMessage, ...]:
    """A turn with planner calls and one explanation call."""
    return (
        tool_call(READ_STATE, "call-state"),
        tool_call(TEACH, "call-teach", theme=_THEME.slug, focus=_FOCUS),
        final_message(_FINAL_TEXT),
    )


def _lesson_and_check() -> tuple[AIMessage, ...]:
    """A realistic coach turn: read state, teach, one check question, final message."""
    return (
        tool_call(READ_STATE, "call-state"),
        tool_call(TEACH, "call-teach", theme=_THEME.slug, focus=_FOCUS),
        tool_call(GENERATE_QUIZ, "call-quiz", theme=_THEME.slug, n=1),
        final_message(_FINAL_TEXT),
    )


def _quiz_only() -> tuple[AIMessage, ...]:
    """A turn that only asks one quiz question."""
    return (tool_call(GENERATE_QUIZ, "call-quiz", theme=_THEME.slug, n=1), final_message(_FINAL_TEXT))


def _quiz(user_id: UserId, config: RunnableConfig, content_model: FakeChatModel) -> Question:
    """Run a one-question quiz turn and return the question it checkpointed."""
    graph = _graph(_scripted(*_quiz_only()), content_model)
    _turn(graph, user_id, config)
    (question,) = _values(graph, config)["quiz_questions"]
    assert isinstance(question, Question)
    return question


def _answer_graph(content_model: FakeChatModel) -> _Graph:
    """A fresh graph for answer turns. Its planner has no script, so any planner call fails the test."""
    return _graph(_scripted(), content_model)


def _answer_rows(user_id: UserId) -> list[dict[str, object]]:
    return _execute(_ANSWER_ROWS_SQL, user_id)


def _mistakes(user_id: UserId) -> list[MistakeEpisode]:
    return list_records(user_id, MistakeEpisode, get_store())


def _mastery(user_id: UserId) -> TopicMastery | None:
    return get(user_id, _THEME.slug, TopicMastery, get_store())


def _forbidden_write() -> AIMessage:
    return tool_call(
        RECORD_OUTCOME,
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


def test_default_planner_deps_run_the_question_critic() -> None:
    # Answers move mastery, so production quizzes must pass the critic before a learner sees them.
    assert isinstance(default_planner_deps().critic, LlmCritic)


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
        assert [row["tool_name"] for row in rows] == [READ_STATE, TEACH, GENERATE_QUIZ, None]
        assert rows[-1]["halt_reason"] == HaltReason.FINAL_MESSAGE.value

    def test_next_turn_can_read_the_summary_of_the_last(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        model = _scripted(
            *_lesson_and_check(),
            tool_call(READ_STATE, "call-next-state"),
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
        graph = _graph(FakePlannerModel(repeat=tool_call(SUGGEST_THEME, "call-forever")))

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
            tool_call(RETRIEVE_CORPUS, "call-retrieve", theme=_EMPTY_THEME.slug, query=query),
            final_message(_FINAL_TEXT),
        )

        _turn(_graph(model), user_id, config)

        assert _execute(_GAP_ROWS_SQL, query) == [{"theme": _EMPTY_THEME.slug, "query": query}]


@pytest.mark.usefixtures("corpus")
class TestPreviousExchange:
    """The last exchange is checkpointed as a typed value and reaches a restarted graph's next turn."""

    def test_previous_exchange_survives_a_restart_on_the_same_thread(
        self, user_id: UserId, config: RunnableConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        _turn(_graph(_scripted(*_lesson_and_check())), user_id, config)
        model = _scripted(final_message(_SECOND_FINAL_TEXT))
        restarted = _graph(model)

        with caplog.at_level(logging.WARNING):
            saved = _values(restarted, config)["previous_exchange"]
            _turn(restarted, user_id, config, _SECOND_LEARNER_MESSAGE)

        assert isinstance(saved, PreviousExchange)
        assert (saved.learner_message, saved.reply) == (_LEARNER_MESSAGE, _FINAL_TEXT)
        assert saved.lesson_excerpt == _LESSON_TEXT
        assert saved.quiz_stems == (_QUESTION_TEXT,)
        context = "\n".join(str(m.content) for m in model.received[0])
        for text in (_LEARNER_MESSAGE, _FINAL_TEXT, _LESSON_TEXT, _QUESTION_TEXT):
            assert text in context
        assert [r.message for r in caplog.records if r.name.startswith(_SERIALIZER_LOGGER)] == []


@pytest.mark.usefixtures("corpus")
class TestTokenUsage:
    """memory_writer saves the turn's token totals in the checkpoint when the turn has a tally."""

    def test_turn_with_a_tally_saves_its_totals_per_caller_as_a_typed_object(
        self, user_id: UserId, config: RunnableConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        script = _lesson()
        content_model = _content_model()
        tally = UsageTally()

        with caplog.at_level(logging.WARNING):
            _turn(_graph(_scripted(*script), content_model), user_id, config, tally=tally)
            # A new graph has nothing but the checkpoint to read the totals from.
            saved = _values(_graph(_scripted()), config)["token_usage"]

        assert content_model.invocation_count == 1
        assert isinstance(saved, TurnUsage)
        assert saved == turn_usage(
            {PLANNER_TAG: counts_of(PLANNER_USAGE, len(script)), EXPLANATION_TAG: counts_of(CONTENT_USAGE)}
        )
        assert saved == tally.totals()
        assert [r.message for r in caplog.records if r.name.startswith(_SERIALIZER_LOGGER)] == []

    def test_turn_without_a_tally_saves_no_totals_and_does_not_error(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        graph = _graph(_scripted(*_lesson()))

        _turn(graph, user_id, config)

        values = _values(graph, config)
        assert values["reply"] == _FINAL_TEXT
        assert values["token_usage"] is None

    def test_second_turn_saves_only_its_own_totals(self, user_id: UserId, config: RunnableConfig) -> None:
        graph = _graph(_scripted(*_lesson(), final_message(_SECOND_FINAL_TEXT)))

        _turn(graph, user_id, config, tally=UsageTally())
        _turn(graph, user_id, config, _SECOND_LEARNER_MESSAGE, tally=UsageTally())

        # The second turn made one planner call and no content call.
        assert _values(graph, config)["token_usage"] == turn_usage({PLANNER_TAG: counts_of(PLANNER_USAGE)})

    def test_turn_without_a_tally_clears_the_totals_of_the_turn_before(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        graph = _graph(_scripted(*_lesson(), final_message(_SECOND_FINAL_TEXT)))

        _turn(graph, user_id, config, tally=UsageTally())
        _turn(graph, user_id, config, _SECOND_LEARNER_MESSAGE)

        assert _values(graph, config)["token_usage"] is None


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

    def test_only_the_previous_exchange_carries_into_the_next_turn(
        self, user_id: UserId, config: RunnableConfig
    ) -> None:
        model = _scripted(
            tool_call(READ_STATE, "call-state"),
            final_message(_FINAL_TEXT),
            final_message(_SECOND_FINAL_TEXT),
        )
        graph = _graph(model)

        _turn(graph, user_id, config)
        _turn(graph, user_id, config, _SECOND_LEARNER_MESSAGE)

        messages = model.received[-1]
        assert [type(m) for m in messages] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
        assert [m.content for m in messages if isinstance(m, HumanMessage)] == [
            _LEARNER_MESSAGE,
            _SECOND_LEARNER_MESSAGE,
        ]
        assert not any(isinstance(m, ToolMessage) for m in messages)
        assert "messages" not in _values(graph, config)


@pytest.mark.usefixtures("corpus")
class TestAnswerTurn:
    """An answer turn scores a checkpointed question without the planner, logs it, and moves mastery."""

    def test_next_turn_scores_the_question_the_quiz_turn_checkpointed(
        self,
        user_id: UserId,
        config: RunnableConfig,
        answer_content_model: FakeChatModel,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from civica.graph.answers import AnswerResult

        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        restarted = _answer_graph(content_model)

        with caplog.at_level(logging.WARNING):
            _answer(restarted, user_id, config, question.question_id, question.correct_index)
            values = _values(restarted, config)

        result = values["answer_result"]
        assert isinstance(result, AnswerResult)
        assert (result.question_id, result.chosen_index, result.is_correct) == (
            question.question_id,
            question.correct_index,
            True,
        )
        assert values["answered"] == {question.question_id: result}
        assert [r.message for r in caplog.records if r.name.startswith(_SERIALIZER_LOGGER)] == []

    @pytest.mark.parametrize("is_correct", [True, False], ids=["right", "wrong"])
    def test_answer_is_written_to_quiz_answers(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel, is_correct: bool
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        chosen = question.correct_index if is_correct else wrong_index(question)

        _answer(_answer_graph(content_model), user_id, config, question.question_id, chosen)

        assert _answer_rows(user_id) == [
            {
                "theme": _THEME.slug,
                "question_id": question.question_id,
                "chosen_index": chosen,
                "correct_index": question.correct_index,
                "is_correct": is_correct,
            }
        ]

    def test_wrong_answer_writes_a_mistake_episode_with_the_question_sources(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)

        _answer(_answer_graph(content_model), user_id, config, question.question_id, wrong_index(question))

        assert _mistakes(user_id) == [
            MistakeEpisode(theme=_THEME, question_id=question.question_id, sources=list(question.sources))
        ]

    def test_right_answer_writes_no_mistake_episode(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)

        _answer(_answer_graph(content_model), user_id, config, question.question_id, question.correct_index)

        assert _mistakes(user_id) == []

    @pytest.mark.parametrize("is_correct", [True, False], ids=["right", "wrong"])
    def test_answer_updates_topic_mastery_without_the_planner_recording_it(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel, is_correct: bool
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        trace_before = _trace(user_id, config)
        chosen = question.correct_index if is_correct else wrong_index(question)

        _answer(_answer_graph(content_model), user_id, config, question.question_id, chosen)

        saved = _mastery(user_id)
        assert saved is not None
        # The shared corpus sets the exact share, so a right answer only needs to earn some of the theme.
        if is_correct:
            assert 0.0 < saved.mastery <= 1.0
        else:
            assert saved.mastery == 0.0
        # The answer turn wrote no planner trace, so record_outcome never ran.
        assert _trace(user_id, config) == trace_before
        assert RECORD_OUTCOME not in [row["tool_name"] for row in trace_before]

    @pytest.mark.parametrize("is_correct", [True, False], ids=["right", "wrong"])
    def test_session_summary_describes_the_answer_not_the_quiz(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel, is_correct: bool
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        chosen = question.correct_index if is_correct else wrong_index(question)

        _answer(_answer_graph(content_model), user_id, config, question.question_id, chosen)

        assert _summary(user_id) == answer_recap(_THEME, is_correct)


@pytest.mark.usefixtures("corpus")
class TestAnswerThatIsNotScored:
    """An expired or repeated answer is told so, and writes no row, no mistake, and no mastery."""

    def test_answer_after_a_message_turn_cleared_the_quiz_has_expired(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        _turn(_graph(_scripted(final_message(_SECOND_FINAL_TEXT)), content_model), user_id, config)
        graph = _answer_graph(content_model)

        _answer(graph, user_id, config, question.question_id, wrong_index(question))

        values = _values(graph, config)
        assert values["reply"] == EXPIRED_REPLY
        assert values["answer_result"] is None
        self._assert_nothing_written(user_id)

    def test_answer_for_an_unknown_question_has_expired(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        graph = _answer_graph(content_model)

        _answer(graph, user_id, config, UNKNOWN_QUESTION_ID, wrong_index(question))

        assert _values(graph, config)["reply"] == EXPIRED_REPLY
        self._assert_nothing_written(user_id)

    def test_second_answer_to_the_same_question_is_already_answered(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        graph = _answer_graph(content_model)
        _answer(graph, user_id, config, question.question_id, wrong_index(question))
        first = _values(graph, config)["answer_result"]

        _answer(graph, user_id, config, question.question_id, question.correct_index)

        values = _values(graph, config)
        assert values["reply"] == ALREADY_ANSWERED_REPLY
        assert values["answered"] == {question.question_id: first}
        assert [row["is_correct"] for row in _answer_rows(user_id)] == [False]
        assert len(_mistakes(user_id)) == 1
        assert _mastery(user_id) == TopicMastery(theme=_THEME, mastery=0.0)

    def test_a_message_turn_clears_the_answered_results(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        _answer(_answer_graph(content_model), user_id, config, question.question_id, question.correct_index)
        graph = _graph(_scripted(final_message(_SECOND_FINAL_TEXT)), content_model)

        _turn(graph, user_id, config, _SECOND_LEARNER_MESSAGE)

        values = _values(graph, config)
        assert values["answered"] == {}
        assert values["quiz_questions"] == []

    @staticmethod
    def _assert_nothing_written(user_id: UserId) -> None:
        assert _answer_rows(user_id) == []
        assert _mistakes(user_id) == []
        assert _mastery(user_id) is None


@pytest.mark.usefixtures("corpus")
class TestAnswerScoredLine:
    """Each answer turn logs one answer_scored line saying whether the answer counted, with ids only."""

    @pytest.mark.parametrize(("is_correct", "outcome"), [(True, _RIGHT), (False, _WRONG)], ids=["right", "wrong"])
    def test_scored_answer_logs_whether_it_was_right(
        self,
        user_id: UserId,
        config: RunnableConfig,
        answer_content_model: FakeChatModel,
        caplog: pytest.LogCaptureFixture,
        is_correct: bool,
        outcome: str,
    ) -> None:
        question = _quiz(user_id, config, answer_content_model)
        chosen = question.correct_index if is_correct else wrong_index(question)

        with caplog.at_level(logging.INFO, logger=_GRAPH_LOGGER):
            _answer(_answer_graph(answer_content_model), user_id, config, question.question_id, chosen)

        assert only(lines_from(caplog.records, _GRAPH_LOGGER), _ANSWER_SCORED) == {
            "user_id": str(user_id), "question_id": question.question_id, "outcome": outcome,
        }

    def test_expired_answer_logs_expired(
        self,
        user_id: UserId,
        config: RunnableConfig,
        answer_content_model: FakeChatModel,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _quiz(user_id, config, answer_content_model)

        with caplog.at_level(logging.INFO, logger=_GRAPH_LOGGER):
            _answer(_answer_graph(answer_content_model), user_id, config, UNKNOWN_QUESTION_ID, 0)

        assert only(lines_from(caplog.records, _GRAPH_LOGGER), _ANSWER_SCORED) == {
            "user_id": str(user_id), "question_id": UNKNOWN_QUESTION_ID, "outcome": _EXPIRED,
        }

    def test_repeated_answer_logs_already_answered(
        self,
        user_id: UserId,
        config: RunnableConfig,
        answer_content_model: FakeChatModel,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        question = _quiz(user_id, config, answer_content_model)
        graph = _answer_graph(answer_content_model)
        _answer(graph, user_id, config, question.question_id, question.correct_index)

        with caplog.at_level(logging.INFO, logger=_GRAPH_LOGGER):
            _answer(graph, user_id, config, question.question_id, question.correct_index)

        assert only(lines_from(caplog.records, _GRAPH_LOGGER), _ANSWER_SCORED)["outcome"] == _ALREADY_ANSWERED


@pytest.mark.usefixtures("corpus")
class TestAnswerMakesNoModelCall:
    """An answer turn is checked against the checkpointed question alone; its stored explanation needs no call."""

    @pytest.mark.parametrize("is_correct", [True, False], ids=["right", "wrong"])
    def test_answer_turn_calls_neither_the_content_model_nor_the_planner(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel, is_correct: bool
    ) -> None:
        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        planner_model = _scripted()
        chosen = question.correct_index if is_correct else wrong_index(question)
        graph = _graph(planner_model, content_model)

        _answer(graph, user_id, config, question.question_id, chosen, UsageTally())

        # The one content call is the quiz turn's.
        assert content_model.invocation_count == 1
        assert planner_model.received == []
        assert _values(graph, config)["token_usage"].by_caller == {}

    @pytest.mark.parametrize("is_correct", [True, False], ids=["right", "wrong"])
    def test_answer_result_holds_the_score_and_the_theme(
        self, user_id: UserId, config: RunnableConfig, answer_content_model: FakeChatModel, is_correct: bool
    ) -> None:
        from civica.graph.answers import AnswerResult

        content_model = answer_content_model
        question = _quiz(user_id, config, content_model)
        graph = _answer_graph(content_model)
        chosen = question.correct_index if is_correct else wrong_index(question)

        _answer(graph, user_id, config, question.question_id, chosen)

        assert _values(graph, config)["answer_result"] == AnswerResult(
            question_id=question.question_id, theme=question.theme, chosen_index=chosen, is_correct=is_correct
        )
