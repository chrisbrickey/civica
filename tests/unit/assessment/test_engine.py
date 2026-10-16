"""Unit tests for civica.assessment.engine: quiz and mock exam generation.

No network calls. No DB spinup."""

import json
import socket
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from civica.assessment.engine import (
    PROMPT_VERSION,
    QuestionGenerationError,
    generate_mock_exam,
    generate_quiz,
)
from civica.assessment.scoring import MOCK_EXAM_TIME_LIMIT_SECONDS, MockExam, Quiz
from civica.domain.question import Question, QuestionKind
from civica.domain.themes import (
    EXAM_QUESTION_COUNTS,
    EXAM_SCENARIO_COUNTS,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE,
    Theme,
)
from civica.domain.source_ref import SourceRef
from civica.domain.user import UserId
from civica.retrieval.content import SearchFn
from tests.support.assessment import (
    OBJECTIVE_SECTION_IDS,
    OPTION_COUNT,
    SAMPLE_OPTIONS,
    FakeQueryAwareRetriever,
    FakeAnswerLogger,
    FakeQuestionStore,
    FakeRejectionStore,
    FakeRetriever,
    Slot,
    generated_item,
    grouped_generation_reply,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_ID = UserId(uuid.UUID(int=42))
_THEME = SYSTEME_INSTITUTIONNEL_ET_POLITIQUE
_QUIZ_SIZE = 3
_DEFAULT_QUIZ_SIZE = 5
_EXAM_SIZE = 40
_EXAM_KNOWLEDGE_TOTAL = 28
_EXAM_SCENARIO_TOTAL = 12
_THEME_COUNT = len(EXAM_QUESTION_COUNTS)

# Largest per-theme count in the exam; the fake model returns this many items by default.
_MAX_ITEMS_PER_CALL = max(EXAM_QUESTION_COUNTS.values())
_TOO_FEW_ITEMS = 2

_FEWER_SECTIONS_THAN_QUESTIONS = 2
_MORE_SECTIONS_THAN_QUESTIONS = 7
_CHUNKS_PER_SECTION = 2
# Exceeds every per-theme exam count, so each theme must drop its lower-ranked sections.
_MORE_SECTIONS_THAN_ANY_THEME = _MAX_ITEMS_PER_CALL + 2

# Course-objective sections ranked first and third, ahead of and between content sections.
_OBJECTIVE_RANKS = (0, 2)
_ALL_OBJECTIVE_SECTION_COUNT = 2

_MALFORMED_JSON = "this is not json ["

# Generation rules the system message must state verbatim; these strings are the spec.
_SELF_CONTAINED_RULE = (
    "Chaque question doit se comprendre seule : ne mentionne jamais le passage, le texte, "
    "la fiche ou le document."
)
_CIVIC_KNOWLEDGE_RULE = (
    "Interroge sur les connaissances civiques, jamais sur les fiches elles-memes "
    "(titres, objectifs, contenu du cours)."
)
_GENERATION_RULES = {
    "self-contained": _SELF_CONTAINED_RULE,
    "civic-knowledge": _CIVIC_KNOWLEDGE_RULE,
}

# The broad theme retrieved through several sub-topic queries.
_HISTOIRE = HISTOIRE_GEOGRAPHIE_ET_CULTURE
_HISTOIRE_EXAM_COUNT = EXAM_QUESTION_COUNTS[_HISTOIRE]
# Sub-topic queries the engine runs for Histoire; the interleaving expectations assume this many.
_HISTOIRE_QUERY_COUNT = 3
# One more than the query count, so the quiz reaches a second interleaving round.
_HISTOIRE_QUIZ_SIZE = _HISTOIRE_QUERY_COUNT + 1
# Two sub-topic queries of this size already cover every Histoire exam question.
_SECTIONS_PER_SUBTOPIC = _HISTOIRE_EXAM_COUNT // 2

# The reply shape's JSON example must name the group key literally.
_GROUP_KEY = '"group"'

# First exam theme: its last groups are scenario groups.
_EXAM_FIRST_THEME = next(iter(EXAM_QUESTION_COUNTS))
_EXAM_FIRST_COUNT = EXAM_QUESTION_COUNTS[_EXAM_FIRST_THEME]
_EXAM_FIRST_KINDS = [QuestionKind.KNOWLEDGE] * (
    _EXAM_FIRST_COUNT - EXAM_SCENARIO_COUNTS[_EXAM_FIRST_THEME]
) + [QuestionKind.SCENARIO] * EXAM_SCENARIO_COUNTS[_EXAM_FIRST_THEME]


def _item_text(call_index: int, item_index: int) -> str:
    return f"sample-generated-question-{call_index:02d}-{item_index:02d}"


# Each payload is a single well-formed JSON reply whose items break the required shape.
_WRONG_SHAPE_PAYLOADS = {
    "three-options": json.dumps(
        [{"text": "sample-text", "options": SAMPLE_OPTIONS[:3], "correct_index": 0}] * 5
    ),
    "index-out-of-range": json.dumps(
        [{"text": "sample-text", "options": SAMPLE_OPTIONS, "correct_index": 4}] * 5
    ),
    "missing-text": json.dumps([{"options": SAMPLE_OPTIONS, "correct_index": 0}] * 5),
    "object-not-array": json.dumps(generated_item("sample-text", 1)),
    "missing-group": json.dumps(
        [{"text": "sample-text", "options": SAMPLE_OPTIONS, "correct_index": 0}] * 5
    ),
    "group-not-int": json.dumps(
        [{**generated_item(f"sample-text-{i}", 1), "group": "sample-text"} for i in range(5)]
    ),
    "group-zero": json.dumps([generated_item(f"sample-text-{i}", 0) for i in range(5)]),
    # Lax validation would coerce these to valid groups 1..5; strict validation must reject them.
    "group-numeric-string": json.dumps(
        [{**generated_item(f"sample-text-{g}", g), "group": str(g)} for g in range(1, 6)]
    ),
    "group-bool": json.dumps(
        [{**generated_item("sample-text-1", 1), "group": True}]
        + [generated_item(f"sample-text-{g}", g) for g in range(2, 6)]
    ),
}

# Number of model calls when a reply leaves groups out: the first call plus one retry.
_CALLS_WITH_RETRY = 2

# Labels of the single group in a retry prompt; they tell the model which kind to write.
_RETRY_KNOWLEDGE_LABEL = "Groupe 1 (question de connaissance)"
_RETRY_SCENARIO_LABEL = "Groupe 1 (question de mise en situation)"


def _group_text(group: int) -> str:
    return f"sample-question-for-group-{group}"


def _retry_text(group: int) -> str:
    return f"sample-retry-question-for-group-{group}"


# ---------------------------------------------------------------------------
# Fakes (dependency injection, not monkeypatching)
# ---------------------------------------------------------------------------


class _FakeChatModel(BaseChatModel):  # type: ignore[explicit-any]
    """Fake chat model: replies with well-formed items naming groups 1, 2, ... in order.

    Item texts embed the call index, so each call yields distinct but reproducible questions.
    `items_by_call` overrides the item count per call; `replies` scripts exact replies by call
    index (later calls fall back to the default); `canned_text` replaces every reply.
    """

    items_per_call: int = _MAX_ITEMS_PER_CALL
    items_by_call: tuple[int, ...] = ()
    replies: tuple[str, ...] = ()
    canned_text: str | None = None

    _received: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def invocation_count(self) -> int:
        return len(self._received)

    def system_text(self, call_index: int) -> str:
        """The system message content of one call."""
        return "\n".join(
            str(m.content) for m in self._received[call_index] if isinstance(m, SystemMessage)
        )

    def human_text(self, call_index: int) -> str:
        """The human message content of one call."""
        return "\n".join(
            str(m.content) for m in self._received[call_index] if isinstance(m, HumanMessage)
        )

    def _reply_for(self, call_index: int) -> str:
        if self.canned_text is not None:
            return self.canned_text
        if call_index < len(self.replies):
            return self.replies[call_index]
        count = (
            self.items_by_call[call_index]
            if call_index < len(self.items_by_call)
            else self.items_per_call
        )
        return grouped_generation_reply(*((i + 1, _item_text(call_index, i)) for i in range(count)))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call_index = len(self._received)
        self._received.append(messages)
        message = AIMessage(content=self._reply_for(call_index))
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "sample-fake-chat-model"


@dataclass(frozen=True)
class _Fakes:
    """Every injected dependency of the engine; swap one with dataclasses.replace."""

    retriever: FakeRetriever
    chat_client: _FakeChatModel
    question_store: FakeQuestionStore
    rejection_store: FakeRejectionStore
    answer_logger: FakeAnswerLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fakes(answer_logger: FakeAnswerLogger) -> _Fakes:
    return _Fakes(
        retriever=FakeRetriever(),
        chat_client=_FakeChatModel(),
        question_store=FakeQuestionStore(),
        rejection_store=FakeRejectionStore(),
        answer_logger=answer_logger,
    )


@pytest.fixture()
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast instead of reaching a real model if a default client sneaks in."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("unit tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", _refuse)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quiz(
    fakes: _Fakes,
    n: int = _QUIZ_SIZE,
    *,
    theme: Theme = _THEME,
    retriever: SearchFn | None = None,
) -> Quiz:
    """A quiz on `theme`; `retriever` overrides the fakes' default retriever."""
    return generate_quiz(
        _USER_ID,
        theme,
        n,
        retriever=retriever if retriever is not None else fakes.retriever,
        chat_client=fakes.chat_client,
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
        critic=None,
        rejection_store=fakes.rejection_store,
    )


def _exam(fakes: _Fakes, *, retriever: SearchFn | None = None) -> MockExam:
    return generate_mock_exam(
        _USER_ID,
        retriever=retriever if retriever is not None else fakes.retriever,
        chat_client=fakes.chat_client,
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
        critic=None,
        rejection_store=fakes.rejection_store,
    )


def _quiz_with_default_critic(fakes: _Fakes) -> Quiz:
    """Like _quiz, but leaves the critic argument out so its default applies."""
    return generate_quiz(
        _USER_ID,
        _THEME,
        _QUIZ_SIZE,
        retriever=fakes.retriever,
        chat_client=fakes.chat_client,
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
        rejection_store=fakes.rejection_store,
    )


def _exam_with_default_critic(fakes: _Fakes) -> MockExam:
    """Like _exam, but leaves the critic argument out so its default applies."""
    return generate_mock_exam(
        _USER_ID,
        retriever=fakes.retriever,
        chat_client=fakes.chat_client,
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
        rejection_store=fakes.rejection_store,
    )


def _group_block(human_text: str, number: int) -> str:
    """The prompt text from the "Groupe {number} (" label up to the next group label."""
    start = human_text.index(f"Groupe {number} (")
    end = human_text.find(f"Groupe {number + 1} (", start)
    return human_text[start:] if end == -1 else human_text[start:end]


def _assert_well_typed(question: Question) -> None:
    assert isinstance(question, Question)
    assert question.text
    assert len(question.options) == OPTION_COUNT
    assert 0 <= question.correct_index < len(question.options)
    assert question.sources


# ---------------------------------------------------------------------------
# Tests: generate_quiz
# ---------------------------------------------------------------------------


class TestQuizSize:
    """generate_quiz returns exactly n well-typed questions on the requested theme."""

    def test_returns_exactly_n_questions(self, fakes: _Fakes) -> None:
        assert len(_quiz(fakes).questions) == _QUIZ_SIZE

    def test_defaults_to_five_questions(self, fakes: _Fakes) -> None:
        quiz = generate_quiz(
            _USER_ID,
            _THEME,
            retriever=fakes.retriever,
            chat_client=fakes.chat_client,
            question_store=fakes.question_store,
            answer_logger=fakes.answer_logger,
            critic=None,
            rejection_store=fakes.rejection_store,
        )

        assert len(quiz.questions) == _DEFAULT_QUIZ_SIZE

    def test_questions_are_well_typed_knowledge_items_on_the_theme(self, fakes: _Fakes) -> None:
        for question in _quiz(fakes).questions:
            _assert_well_typed(question)
            assert question.theme == _THEME
            assert question.kind is QuestionKind.KNOWLEDGE

    def test_quiz_belongs_to_the_requesting_user(self, fakes: _Fakes) -> None:
        assert _quiz(fakes).user_id == _USER_ID


class TestQuizRetrievalQuery:
    """With no learner question, the query is manufactured from the theme's French name."""

    def test_single_theme_scoped_query_built_from_display_name(self, fakes: _Fakes) -> None:
        _quiz(fakes)

        assert len(fakes.retriever.calls) == 1
        query, theme = fakes.retriever.calls[0]
        assert theme == _THEME
        assert query.strip()
        assert _THEME.display_name_fr in query


class TestQuizSources:
    """Each question cites exactly the one retrieved section it was written from."""

    def test_every_source_is_a_retrieved_section(self, fakes: _Fakes) -> None:
        questions = _quiz(fakes).questions

        retrieved = fakes.retriever.source_refs(_THEME)
        for question in questions:
            assert set(question.sources) <= retrieved

    def test_batch_spans_more_than_one_section(self, fakes: _Fakes) -> None:
        questions = _quiz(fakes).questions

        cited = {source for question in questions for source in question.sources}
        assert len(cited) > 1

    def test_each_question_cites_one_top_ranked_section_in_order(self, fakes: _Fakes) -> None:
        """Question i cites the i-th best matching section; lower-ranked sections are not cited."""
        fakes = replace(
            fakes,
            retriever=FakeRetriever(
                section_count=_MORE_SECTIONS_THAN_QUESTIONS, chunks_per_section=_CHUNKS_PER_SECTION
            ),
        )

        questions = _quiz(fakes).questions

        ranked = fakes.retriever.ranked_source_refs(_THEME)
        assert [q.sources for q in questions] == [(ref,) for ref in ranked[:_QUIZ_SIZE]]

    def test_fewer_sections_than_questions_still_fills_the_batch(self, fakes: _Fakes) -> None:
        """Sections are reused in rank order when too few, and each question still cites one."""
        fakes = replace(
            fakes, retriever=FakeRetriever(section_count=_FEWER_SECTIONS_THAN_QUESTIONS)
        )

        questions = _quiz(fakes, n=_DEFAULT_QUIZ_SIZE).questions

        assert len(questions) == _DEFAULT_QUIZ_SIZE
        ranked = fakes.retriever.ranked_source_refs(_THEME)
        assert [q.sources for q in questions] == [
            (ranked[i % len(ranked)],) for i in range(_DEFAULT_QUIZ_SIZE)
        ]


class TestQuizPassageGroups:
    """Each passage group sent to the model holds one section's chunks, grouped by section."""

    @pytest.fixture()
    def ranked_fakes(self, fakes: _Fakes) -> _Fakes:
        """Multi-chunk sections whose chunks interleave in retrieval order."""
        retriever = FakeRetriever(
            section_count=_MORE_SECTIONS_THAN_QUESTIONS, chunks_per_section=_CHUNKS_PER_SECTION
        )
        return replace(fakes, retriever=retriever)

    def test_each_group_holds_only_its_sections_chunks(self, ranked_fakes: _Fakes) -> None:
        _quiz(ranked_fakes)

        human_text = ranked_fakes.chat_client.human_text(0)
        retriever = ranked_fakes.retriever
        for section in range(_QUIZ_SIZE):
            block = _group_block(human_text, section + 1)
            for chunk in retriever.section_chunks(_THEME, section):
                assert chunk.chunk.text in block
            for other in range(_MORE_SECTIONS_THAN_QUESTIONS):
                if other != section:
                    for chunk in retriever.section_chunks(_THEME, other):
                        assert chunk.chunk.text not in block

    def test_lower_ranked_sections_are_not_sent(self, ranked_fakes: _Fakes) -> None:
        _quiz(ranked_fakes)

        human_text = ranked_fakes.chat_client.human_text(0)
        for section in range(_QUIZ_SIZE, _MORE_SECTIONS_THAN_QUESTIONS):
            for chunk in ranked_fakes.retriever.section_chunks(_THEME, section):
                assert chunk.chunk.text not in human_text


class TestQuizModelCalls:
    """Quiz questions are generated in batches, not one model call per question."""

    def test_fewer_model_calls_than_questions(self, fakes: _Fakes) -> None:
        _quiz(fakes)

        assert fakes.chat_client.invocation_count < _QUIZ_SIZE


class TestExtraItemsTruncated:
    """If the model over-delivers, items naming groups beyond n are ignored; groups 1..n are kept."""

    def test_keeps_first_n_groups_in_order(self, fakes: _Fakes) -> None:
        questions = _quiz(fakes).questions

        assert [q.text for q in questions] == [_item_text(0, i) for i in range(_QUIZ_SIZE)]


# ---------------------------------------------------------------------------
# Tests: generate_mock_exam
# ---------------------------------------------------------------------------


class TestMockExamStructure:
    """A mock exam mirrors the official format: 40 questions, 11/11/8/6/4, 28 knowledge + 12 scenario."""

    def test_has_forty_questions_in_official_theme_counts(self, fakes: _Fakes) -> None:
        questions = _exam(fakes).questions

        assert len(questions) == _EXAM_SIZE
        assert Counter(q.theme for q in questions) == EXAM_QUESTION_COUNTS

    def test_scenario_split_per_theme(self, fakes: _Fakes) -> None:
        questions = _exam(fakes).questions

        scenarios = Counter(q.theme for q in questions if q.kind is QuestionKind.SCENARIO)
        kinds = Counter(q.kind for q in questions)
        assert scenarios == EXAM_SCENARIO_COUNTS
        assert kinds[QuestionKind.KNOWLEDGE] == _EXAM_KNOWLEDGE_TOTAL
        assert kinds[QuestionKind.SCENARIO] == _EXAM_SCENARIO_TOTAL

    def test_questions_are_well_typed_and_sourced_from_their_theme(self, fakes: _Fakes) -> None:
        for question in _exam(fakes).questions:
            _assert_well_typed(question)
            assert set(question.sources) <= fakes.retriever.source_refs(question.theme)

    def test_each_question_cites_one_top_ranked_section(self, fakes: _Fakes) -> None:
        fakes = replace(
            fakes, retriever=FakeRetriever(section_count=_MORE_SECTIONS_THAN_ANY_THEME)
        )

        questions = _exam(fakes).questions

        for theme, count in EXAM_QUESTION_COUNTS.items():
            ranked = fakes.retriever.ranked_source_refs(theme)
            themed = [q for q in questions if q.theme == theme]
            assert [q.sources for q in themed] == [(ref,) for ref in ranked[:count]]

    def test_carries_the_official_time_limit(self, fakes: _Fakes) -> None:
        assert _exam(fakes).time_limit_seconds == MOCK_EXAM_TIME_LIMIT_SECONDS


class TestMockExamCalls:
    """Theme-scoped retrieval: one display-name query per theme, several for Histoire."""

    def test_theme_scoped_retrieval_per_theme(self, fakes: _Fakes) -> None:
        _exam(fakes)

        calls = fakes.retriever.calls
        per_theme = Counter(theme for _, theme in calls)
        assert set(per_theme) == set(EXAM_QUESTION_COUNTS)
        for query, theme in calls:
            assert theme is not None
            assert query.strip()
            if theme != _HISTOIRE:
                assert per_theme[theme] == 1
                assert theme.display_name_fr in query

    def test_fewer_model_calls_than_questions(self, fakes: _Fakes) -> None:
        _exam(fakes)

        assert fakes.chat_client.invocation_count < _EXAM_SIZE


# ---------------------------------------------------------------------------
# Tests: Histoire retrieved through several sub-topic queries
# ---------------------------------------------------------------------------


def _slot_sources(retriever: FakeQueryAwareRetriever, *slots: Slot) -> list[tuple[SourceRef, ...]]:
    """The expected one-section sources of questions written from these (query, rank) slots."""
    return [(retriever.slot_ref(slot),) for slot in slots]


def _histoire_quiz_sources(
    fakes: _Fakes, retriever: FakeQueryAwareRetriever, n: int = _HISTOIRE_QUIZ_SIZE
) -> list[tuple[SourceRef, ...]]:
    return [q.sources for q in _quiz(fakes, n, theme=_HISTOIRE, retriever=retriever).questions]


class TestHistoireSubTopicQueries:
    """Histoire is too broad for one query, so it is searched by several distinct sub-topics."""

    def test_quiz_runs_several_distinct_theme_scoped_queries(self, fakes: _Fakes) -> None:
        retriever = FakeQueryAwareRetriever()

        _quiz(fakes, theme=_HISTOIRE, retriever=retriever)

        queries = [query for query, _ in retriever.calls]
        assert len(queries) == _HISTOIRE_QUERY_COUNT
        assert all(theme == _HISTOIRE for _, theme in retriever.calls)
        assert all(query.strip() for query in queries)
        assert len(set(queries)) == len(queries)

    def test_other_theme_quiz_still_runs_one_query(self, fakes: _Fakes) -> None:
        retriever = FakeQueryAwareRetriever()

        _quiz(fakes, retriever=retriever)

        assert retriever.calls == [(retriever.queries(_THEME)[0], _THEME)]
        assert _THEME.display_name_fr in retriever.queries(_THEME)[0]


class TestHistoireSectionInterleaving:
    """Question i cites the i-th section round-robin by rank across the sub-topic searches.

    Expected sources are (query, rank) slots over three queries of three sections each."""

    def test_questions_take_sections_round_robin_by_rank(self, fakes: _Fakes) -> None:
        retriever = FakeQueryAwareRetriever()

        sources = _histoire_quiz_sources(fakes, retriever)

        assert sources == _slot_sources(retriever, (0, 0), (1, 0), (2, 0), (0, 1))

    def test_exhausted_query_is_skipped_in_later_rounds(self, fakes: _Fakes) -> None:
        # The first sub-topic search has a single section; the others have the default count.
        retriever = FakeQueryAwareRetriever(sections_per_query=(1,))

        sources = _histoire_quiz_sources(fakes, retriever)

        assert sources == _slot_sources(retriever, (0, 0), (1, 0), (2, 0), (1, 1))

    def test_shared_section_is_used_once_at_its_first_position(self, fakes: _Fakes) -> None:
        # Query 1's top section is the same as query 0's second section.
        retriever = FakeQueryAwareRetriever(shared_slots={(1, 0): (0, 1)})

        sources = _histoire_quiz_sources(fakes, retriever)

        # Slot (1, 0) carries section (0, 1), so (0, 1) is skipped when its own turn comes.
        assert sources == _slot_sources(retriever, (0, 0), (1, 0), (2, 0), (1, 1))
        assert sources[1] == _slot_sources(retriever, (0, 1))[0]

    @pytest.mark.parametrize("section_id", OBJECTIVE_SECTION_IDS)
    def test_objective_sections_are_dropped_from_sub_topic_results(
        self, fakes: _Fakes, section_id: str
    ) -> None:
        objective_slots = {(0, 0): section_id, (1, 1): section_id}
        retriever = FakeQueryAwareRetriever(objective_slots=objective_slots)

        sources = _histoire_quiz_sources(fakes, retriever)

        assert sources == _slot_sources(retriever, (1, 0), (2, 0), (0, 1), (2, 1))
        cited_ids = {ref.section_id for refs in sources for ref in refs}
        assert section_id not in cited_ids

    def test_no_usable_sub_topic_results_raises_without_calling_model(self, fakes: _Fakes) -> None:
        retriever = FakeQueryAwareRetriever(default_section_count=0)

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes, theme=_HISTOIRE, retriever=retriever)

        assert fakes.chat_client.invocation_count == 0
        assert fakes.question_store.saved_questions == []

    def test_mock_exam_histoire_questions_cite_distinct_sections(self, fakes: _Fakes) -> None:
        retriever = FakeQueryAwareRetriever(default_section_count=_SECTIONS_PER_SUBTOPIC)

        questions = _exam(fakes, retriever=retriever).questions

        cited = [q.sources for q in questions if q.theme == _HISTOIRE]
        assert len(cited) == _HISTOIRE_EXAM_COUNT
        assert len(set(cited)) == _HISTOIRE_EXAM_COUNT


# ---------------------------------------------------------------------------
# Tests: course-objective sections ignored
# ---------------------------------------------------------------------------


def _objective_retriever(section_id: str, section_count: int) -> FakeRetriever:
    """Multi-chunk sections where _OBJECTIVE_RANKS are course-objective sections."""
    return FakeRetriever(
        section_count=section_count,
        chunks_per_section=_CHUNKS_PER_SECTION,
        objective_sections=dict.fromkeys(_OBJECTIVE_RANKS, section_id),
    )


def _all_objective_retriever() -> FakeRetriever:
    """Every retrieved section is a course-objective section, mixing both ID variants."""
    return FakeRetriever(
        section_count=_ALL_OBJECTIVE_SECTION_COUNT,
        objective_sections=dict(enumerate(OBJECTIVE_SECTION_IDS)),
    )


@pytest.mark.parametrize("section_id", OBJECTIVE_SECTION_IDS)
class TestCourseObjectiveSectionsIgnored:
    """Course-objective intro sections are never sent or cited; next-ranked content takes their place."""

    def test_quiz_question_i_cites_the_i_th_content_section(
        self, fakes: _Fakes, section_id: str
    ) -> None:
        fakes = replace(
            fakes, retriever=_objective_retriever(section_id, _MORE_SECTIONS_THAN_QUESTIONS)
        )

        questions = _quiz(fakes).questions

        content = fakes.retriever.content_source_refs(_THEME)
        assert [q.sources for q in questions] == [(ref,) for ref in content[:_QUIZ_SIZE]]

    def test_quiz_prompt_omits_objective_text(self, fakes: _Fakes, section_id: str) -> None:
        fakes = replace(
            fakes, retriever=_objective_retriever(section_id, _MORE_SECTIONS_THAN_QUESTIONS)
        )

        _quiz(fakes)

        human_text = fakes.chat_client.human_text(0)
        for chunk in fakes.retriever.objective_chunks(_THEME):
            assert chunk.chunk.text not in human_text

    def test_mock_exam_cites_only_top_content_sections(
        self, fakes: _Fakes, section_id: str
    ) -> None:
        # Two extra sections so every theme still has enough content after the objectives drop.
        section_count = _MORE_SECTIONS_THAN_ANY_THEME + len(_OBJECTIVE_RANKS)
        fakes = replace(fakes, retriever=_objective_retriever(section_id, section_count))

        questions = _exam(fakes).questions

        for theme, count in EXAM_QUESTION_COUNTS.items():
            content = fakes.retriever.content_source_refs(theme)
            themed = [q for q in questions if q.theme == theme]
            assert [q.sources for q in themed] == [(ref,) for ref in content[:count]]


class TestOnlyCourseObjectiveSectionsRetrieved:
    """With nothing but course-objective sections, generation fails like having no material."""

    def test_quiz_raises_without_calling_model(self, fakes: _Fakes) -> None:
        fakes = replace(fakes, retriever=_all_objective_retriever())

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.chat_client.invocation_count == 0
        assert fakes.question_store.saved_questions == []

    def test_mock_exam_raises_without_calling_model(self, fakes: _Fakes) -> None:
        fakes = replace(fakes, retriever=_all_objective_retriever())

        with pytest.raises(QuestionGenerationError):
            _exam(fakes)

        assert fakes.chat_client.invocation_count == 0
        assert fakes.question_store.saved_questions == []


# ---------------------------------------------------------------------------
# Tests: generation prompt rules
# ---------------------------------------------------------------------------


class TestGenerationPromptRules:
    """Learners never see the passages, so questions must stand alone and test civic knowledge."""

    @pytest.mark.parametrize("rule", _GENERATION_RULES.values(), ids=_GENERATION_RULES.keys())
    def test_quiz_system_message_states_rule(self, fakes: _Fakes, rule: str) -> None:
        _quiz(fakes)

        assert rule in fakes.chat_client.system_text(0)

    @pytest.mark.parametrize("rule", _GENERATION_RULES.values(), ids=_GENERATION_RULES.keys())
    def test_every_mock_exam_system_message_states_rule(self, fakes: _Fakes, rule: str) -> None:
        _exam(fakes)

        calls = range(fakes.chat_client.invocation_count)
        assert len(calls) == _THEME_COUNT
        for call_index in calls:
            assert rule in fakes.chat_client.system_text(call_index)

    def test_system_message_reply_shape_names_the_group_key(self, fakes: _Fakes) -> None:
        _quiz(fakes)

        assert _GROUP_KEY in fakes.chat_client.system_text(0)


# ---------------------------------------------------------------------------
# Tests: critic off by default
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_network")
class TestCriticOffByDefault:
    """Without a critic argument, generation is a single pass: nothing is critiqued or rejected."""

    def test_quiz_returns_and_saves_every_generated_question(self, fakes: _Fakes) -> None:
        quiz = _quiz_with_default_critic(fakes)

        expected = [_item_text(0, i) for i in range(_QUIZ_SIZE)]
        assert [q.text for q in quiz.questions] == expected
        assert [q.text for q in fakes.question_store.saved_questions] == expected
        assert fakes.rejection_store.rejections == []

    def test_quiz_makes_one_model_call(self, fakes: _Fakes) -> None:
        _quiz_with_default_critic(fakes)

        assert fakes.chat_client.invocation_count == 1

    def test_mock_exam_returns_and_saves_all_forty_questions(self, fakes: _Fakes) -> None:
        exam = _exam_with_default_critic(fakes)

        assert len(exam.questions) == _EXAM_SIZE
        assert [q.question_id for q in fakes.question_store.saved_questions] == [
            q.question_id for q in exam.questions
        ]
        assert fakes.rejection_store.rejections == []

    def test_mock_exam_makes_one_model_call_per_theme(self, fakes: _Fakes) -> None:
        _exam_with_default_critic(fakes)

        assert fakes.chat_client.invocation_count == _THEME_COUNT


# ---------------------------------------------------------------------------
# Tests: persistence
# ---------------------------------------------------------------------------


class TestQuestionPersistence:
    """Every returned question is saved exactly once, stamped with the current prompt version."""

    def test_prompt_version_is_declared(self) -> None:
        assert PROMPT_VERSION.strip()

    def test_quiz_questions_saved_once_with_prompt_version(self, fakes: _Fakes) -> None:
        questions = _quiz(fakes).questions

        saved_ids = Counter(q.question_id for q in fakes.question_store.saved_questions)
        assert saved_ids == Counter(q.question_id for q in questions)
        assert all(count == 1 for count in saved_ids.values())
        assert fakes.question_store.saved_prompt_versions == {PROMPT_VERSION}

    def test_mock_exam_questions_saved_once_with_prompt_version(self, fakes: _Fakes) -> None:
        questions = _exam(fakes).questions

        saved_ids = Counter(q.question_id for q in fakes.question_store.saved_questions)
        assert saved_ids == Counter(q.question_id for q in questions)
        assert len(saved_ids) == _EXAM_SIZE
        assert all(count == 1 for count in saved_ids.values())
        assert fakes.question_store.saved_prompt_versions == {PROMPT_VERSION}

    def test_same_payload_yields_the_same_question_keys(self, fakes: _Fakes) -> None:
        """Stable content-hash keys are what make the store's upsert idempotent across runs."""
        first = _quiz(replace(fakes, chat_client=_FakeChatModel())).questions
        second = _quiz(replace(fakes, chat_client=_FakeChatModel())).questions

        assert [q.question_id for q in first] == [q.question_id for q in second]


# ---------------------------------------------------------------------------
# Tests: items matched to passage groups by number
# ---------------------------------------------------------------------------


def _scripted(fakes: _Fakes, *replies: str) -> _Fakes:
    return replace(fakes, chat_client=_FakeChatModel(replies=replies))


class TestItemsMatchedByGroupNumber:
    """Sources and kind follow the group each item names, not its position in the reply."""

    def test_quiz_reply_out_of_order_yields_questions_in_group_order(self, fakes: _Fakes) -> None:
        fakes = _scripted(
            fakes, grouped_generation_reply(*((g, _group_text(g)) for g in (3, 1, 2)))
        )

        questions = _quiz(fakes).questions

        ranked = fakes.retriever.ranked_source_refs(_THEME)
        assert [q.text for q in questions] == [_group_text(g) for g in (1, 2, 3)]
        assert [q.sources for q in questions] == [(ref,) for ref in ranked[:_QUIZ_SIZE]]

    def test_mock_exam_reply_out_of_order_keeps_kind_and_sources_per_group(
        self, fakes: _Fakes
    ) -> None:
        groups = range(1, _EXAM_FIRST_COUNT + 1)
        reversed_reply = grouped_generation_reply(*((g, _group_text(g)) for g in reversed(groups)))
        fakes = replace(
            _scripted(fakes, reversed_reply),
            retriever=FakeRetriever(section_count=_MORE_SECTIONS_THAN_ANY_THEME),
        )

        questions = _exam(fakes).questions

        themed = [q for q in questions if q.theme == _EXAM_FIRST_THEME]
        ranked = fakes.retriever.ranked_source_refs(_EXAM_FIRST_THEME)
        assert [q.text for q in themed] == [_group_text(g) for g in groups]
        assert [q.kind for q in themed] == _EXAM_FIRST_KINDS
        assert [q.sources for q in themed] == [(ref,) for ref in ranked[:_EXAM_FIRST_COUNT]]

    def test_out_of_range_group_is_ignored(self, fakes: _Fakes) -> None:
        out_of_range = _QUIZ_SIZE + 1
        fakes = _scripted(
            fakes,
            grouped_generation_reply(
                *((g, _group_text(g)) for g in (out_of_range, 1, 2, 3))
            ),
        )

        questions = _quiz(fakes).questions

        assert [q.text for q in questions] == [_group_text(g) for g in (1, 2, 3)]
        assert fakes.chat_client.invocation_count == 1

    def test_first_item_for_a_repeated_group_is_kept(self, fakes: _Fakes) -> None:
        fakes = _scripted(
            fakes,
            grouped_generation_reply(
                (1, _group_text(1)), (2, _group_text(2)), (2, _retry_text(2)), (3, _group_text(3))
            ),
        )

        questions = _quiz(fakes).questions

        assert [q.text for q in questions] == [_group_text(g) for g in (1, 2, 3)]
        assert fakes.chat_client.invocation_count == 1


class TestMissingGroupsRetried:
    """Groups the reply leaves out get exactly one more call, holding only those groups."""

    # Reply skipping group 2; the retry prompt numbers its single group 1.
    _SKIPS_GROUP_TWO = grouped_generation_reply((1, _group_text(1)), (3, _group_text(3)))
    _RETRY_FILLS_IT = grouped_generation_reply((1, _retry_text(2)))

    def test_missing_group_is_filled_from_the_retry(self, fakes: _Fakes) -> None:
        fakes = _scripted(fakes, self._SKIPS_GROUP_TWO, self._RETRY_FILLS_IT)

        questions = _quiz(fakes).questions

        ranked = fakes.retriever.ranked_source_refs(_THEME)
        assert [q.text for q in questions] == [_group_text(1), _retry_text(2), _group_text(3)]
        assert [q.sources for q in questions] == [(ref,) for ref in ranked[:_QUIZ_SIZE]]
        assert fakes.chat_client.invocation_count == _CALLS_WITH_RETRY

    def test_retry_prompt_holds_only_the_missing_groups_passages(self, fakes: _Fakes) -> None:
        fakes = _scripted(fakes, self._SKIPS_GROUP_TWO, self._RETRY_FILLS_IT)

        _quiz(fakes)

        retry_prompt = fakes.chat_client.human_text(1)
        for section in range(_QUIZ_SIZE):
            for chunk in fakes.retriever.section_chunks(_THEME, section):
                assert (chunk.chunk.text in retry_prompt) is (section == 1)

    def test_doubled_group_and_skipped_group_cite_the_right_sections(self, fakes: _Fakes) -> None:
        """The live failure: two items for group 2, none for group 3, so later items shifted."""
        fakes = _scripted(
            fakes,
            grouped_generation_reply(
                (1, _group_text(1)), (2, _group_text(2)), (2, _retry_text(2))
            ),
            grouped_generation_reply((1, _retry_text(3))),
        )

        questions = _quiz(fakes).questions

        ranked = fakes.retriever.ranked_source_refs(_THEME)
        assert [q.text for q in questions] == [_group_text(1), _group_text(2), _retry_text(3)]
        assert [q.sources for q in questions] == [(ref,) for ref in ranked[:_QUIZ_SIZE]]

    @staticmethod
    def _skips_last_scenario_group(fakes: _Fakes) -> _Fakes:
        """The first exam theme's reply skips its last (scenario) group; the retry fills it."""
        last = _EXAM_FIRST_COUNT
        return replace(
            _scripted(
                fakes,
                grouped_generation_reply(*((g, _group_text(g)) for g in range(1, last))),
                grouped_generation_reply((1, _retry_text(last))),
            ),
            retriever=FakeRetriever(section_count=_MORE_SECTIONS_THAN_ANY_THEME),
        )

    def test_retried_scenario_group_keeps_scenario_kind(self, fakes: _Fakes) -> None:
        fakes = self._skips_last_scenario_group(fakes)

        questions = _exam(fakes).questions

        themed = [q for q in questions if q.theme == _EXAM_FIRST_THEME]
        ranked = fakes.retriever.ranked_source_refs(_EXAM_FIRST_THEME)
        assert [q.kind for q in themed] == _EXAM_FIRST_KINDS
        assert themed[-1].text == _retry_text(_EXAM_FIRST_COUNT)
        assert themed[-1].sources == (ranked[_EXAM_FIRST_COUNT - 1],)

    def test_retry_prompt_asks_for_a_scenario_for_a_scenario_group(self, fakes: _Fakes) -> None:
        fakes = self._skips_last_scenario_group(fakes)

        _exam(fakes)

        retry_prompt = fakes.chat_client.human_text(1)
        assert _RETRY_SCENARIO_LABEL in retry_prompt
        assert _RETRY_KNOWLEDGE_LABEL not in retry_prompt

    def test_retry_prompt_asks_for_knowledge_for_a_knowledge_group(self, fakes: _Fakes) -> None:
        fakes = _scripted(fakes, self._SKIPS_GROUP_TWO, self._RETRY_FILLS_IT)

        _quiz(fakes)

        retry_prompt = fakes.chat_client.human_text(1)
        assert _RETRY_KNOWLEDGE_LABEL in retry_prompt
        assert _RETRY_SCENARIO_LABEL not in retry_prompt

    def test_still_missing_after_retry_raises_and_saves_nothing(self, fakes: _Fakes) -> None:
        fakes = _scripted(fakes, self._SKIPS_GROUP_TWO, grouped_generation_reply())

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.chat_client.invocation_count == _CALLS_WITH_RETRY
        assert fakes.question_store.saved_questions == []


# ---------------------------------------------------------------------------
# Tests: error paths
# ---------------------------------------------------------------------------


class TestNoRetrievedMaterial:
    """Without grounding passages the model is never called and nothing is saved."""

    def test_quiz_raises_without_calling_model(self, fakes: _Fakes) -> None:
        fakes = replace(fakes, retriever=FakeRetriever(section_count=0))

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.chat_client.invocation_count == 0
        assert fakes.question_store.saved_questions == []

    def test_mock_exam_raises_without_calling_model(self, fakes: _Fakes) -> None:
        fakes = replace(fakes, retriever=FakeRetriever(section_count=0))

        with pytest.raises(QuestionGenerationError):
            _exam(fakes)

        assert fakes.chat_client.invocation_count == 0
        assert fakes.question_store.saved_questions == []


class TestInvalidModelReply:
    """A reply that cannot be turned into the requested questions raises and saves nothing."""

    def test_malformed_json_raises(self, fakes: _Fakes) -> None:
        fakes = replace(fakes, chat_client=_FakeChatModel(canned_text=_MALFORMED_JSON))

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.question_store.saved_questions == []

    @pytest.mark.parametrize("payload", _WRONG_SHAPE_PAYLOADS.values(), ids=_WRONG_SHAPE_PAYLOADS.keys())
    def test_wrong_item_shape_raises(self, fakes: _Fakes, payload: str) -> None:
        fakes = replace(fakes, chat_client=_FakeChatModel(canned_text=payload))

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.question_store.saved_questions == []

    def test_too_few_items_raises(self, fakes: _Fakes) -> None:
        """Too few items, then nothing on the retry: still short, so it raises."""
        fakes = replace(fakes, chat_client=_FakeChatModel(items_by_call=(_TOO_FEW_ITEMS, 0)))

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.chat_client.invocation_count == _CALLS_WITH_RETRY
        assert fakes.question_store.saved_questions == []

    def test_mock_exam_failing_on_last_theme_saves_nothing(self, fakes: _Fakes) -> None:
        """A half-built exam must not leak its earlier themes into the bank."""
        healthy_calls = (_MAX_ITEMS_PER_CALL,) * (_THEME_COUNT - 1)
        # The last theme returns nothing on its first call and on its retry.
        fakes = replace(
            fakes, chat_client=_FakeChatModel(items_by_call=(*healthy_calls, 0, 0))
        )

        with pytest.raises(QuestionGenerationError):
            _exam(fakes)

        assert fakes.question_store.saved_questions == []


# ---------------------------------------------------------------------------
# Tests: scoring through the injected logger
# ---------------------------------------------------------------------------


class TestGeneratedAssessmentLogsAnswers:
    """The returned Quiz and MockExam log every scored answer through the injected logger."""

    def test_quiz_score_logs_each_answer(self, fakes: _Fakes) -> None:
        quiz = _quiz(fakes)

        quiz.score([q.correct_index for q in quiz.questions])

        calls = fakes.answer_logger.calls
        assert [call.question_id for call in calls] == [q.question_id for q in quiz.questions]
        assert all(call.user_id == _USER_ID for call in calls)

    def test_mock_exam_score_logs_all_forty_answers(self, fakes: _Fakes) -> None:
        exam = _exam(fakes)

        result = exam.score([q.correct_index for q in exam.questions])

        calls = fakes.answer_logger.calls
        assert len(calls) == _EXAM_SIZE
        assert [call.question_id for call in calls] == [q.question_id for q in exam.questions]
        assert all(call.is_correct for call in calls)
        assert result.passed is True
