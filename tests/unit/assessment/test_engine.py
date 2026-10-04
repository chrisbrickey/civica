"""Unit tests for civica.assessment.engine: quiz and mock exam generation.

No network calls. No DB spinup."""

import json
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from civica.assessment.engine import (
    PROMPT_VERSION,
    QuestionGenerationError,
    generate_mock_exam,
    generate_quiz,
)
from civica.assessment.scoring import MOCK_EXAM_TIME_LIMIT_SECONDS, MockExam, Quiz
from civica.domain.chunk import Chunk
from civica.domain.question import Question, QuestionKind
from civica.domain.source_ref import SourceRef
from civica.domain.themes import (
    EXAM_QUESTION_COUNTS,
    EXAM_SCENARIO_COUNTS,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE,
    Theme,
)
from civica.domain.user import UserId
from civica.retrieval.content import ContentChunk
from tests.support.assessment import (
    OPTION_COUNT,
    SAMPLE_OPTIONS,
    RecordingAnswerLogger,
    RecordingQuestionStore,
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

_SECTION_COUNT = 3
_FEWER_SECTIONS_THAN_QUESTIONS = 2

_SAMPLE_CORRECT_INDEX = 1

_MALFORMED_JSON = "this is not json ["


def _item(text: str) -> dict[str, object]:
    return {"text": text, "options": SAMPLE_OPTIONS, "correct_index": _SAMPLE_CORRECT_INDEX}


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
    "object-not-array": json.dumps(_item("sample-text")),
}


# ---------------------------------------------------------------------------
# Fakes (dependency injection, not monkeypatching)
# ---------------------------------------------------------------------------


def _make_content_chunk(theme: Theme, index: int) -> ContentChunk:
    return ContentChunk(
        chunk=Chunk(
            theme=theme,
            page_slug=f"sample-page-{index:03d}",
            section_id=f"section-{index:03d}",
            chunk_index=0,
            content_hash=f"sample-hash-{theme.slug}-{index:03d}",
            text=f"sample passage {index} about {theme.slug}.",
        ),
        similarity=0.9,
    )


class _RecordingRetriever:
    """Fake SearchFn: records every (query, theme), returns chunks from distinct sections of that theme."""

    def __init__(self, section_count: int = _SECTION_COUNT) -> None:
        self.section_count = section_count
        self.calls: list[tuple[str, Theme | None]] = []

    def __call__(self, query: str, theme: Theme | None) -> list[ContentChunk]:
        self.calls.append((query, theme))
        assert theme is not None, "assessment retrieval must always be scoped to a theme"
        return [_make_content_chunk(theme, i) for i in range(self.section_count)]

    def source_refs(self, theme: Theme) -> set[SourceRef]:
        return {
            SourceRef(page_slug=c.chunk.page_slug, section_id=c.chunk.section_id)
            for c in (_make_content_chunk(theme, i) for i in range(self.section_count))
        }


class _RecordingChatModel(BaseChatModel):  # type: ignore[explicit-any]
    """Fake chat model: replies with a JSON array of well-formed multiple choice items.

    Item texts embed the call index, so each call yields distinct but reproducible questions.
    `items_by_call` overrides the item count per call; `canned_text` replaces the reply entirely.
    """

    items_per_call: int = _MAX_ITEMS_PER_CALL
    items_by_call: tuple[int, ...] = ()
    canned_text: str | None = None

    _received: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def invocation_count(self) -> int:
        return len(self._received)

    def _reply_for(self, call_index: int) -> str:
        if self.canned_text is not None:
            return self.canned_text
        count = (
            self.items_by_call[call_index]
            if call_index < len(self.items_by_call)
            else self.items_per_call
        )
        return json.dumps([_item(_item_text(call_index, i)) for i in range(count)])

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
        return "sample-recording-chat-model"


@dataclass(frozen=True)
class _Fakes:
    """Every injected dependency of the engine; swap one with dataclasses.replace."""

    retriever: _RecordingRetriever
    chat_client: _RecordingChatModel
    question_store: RecordingQuestionStore
    answer_logger: RecordingAnswerLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fakes(answer_logger: RecordingAnswerLogger) -> _Fakes:
    return _Fakes(
        retriever=_RecordingRetriever(),
        chat_client=_RecordingChatModel(),
        question_store=RecordingQuestionStore(),
        answer_logger=answer_logger,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quiz(fakes: _Fakes, n: int = _QUIZ_SIZE) -> Quiz:
    return generate_quiz(
        _USER_ID,
        _THEME,
        n,
        retriever=fakes.retriever,
        chat_client=fakes.chat_client,
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
    )


def _exam(fakes: _Fakes) -> MockExam:
    return generate_mock_exam(
        _USER_ID,
        retriever=fakes.retriever,
        chat_client=fakes.chat_client,
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
    )


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
    """sources come from retrieved chunks, and a batch draws on more than one section."""

    def test_every_source_is_a_retrieved_section(self, fakes: _Fakes) -> None:
        questions = _quiz(fakes).questions

        retrieved = fakes.retriever.source_refs(_THEME)
        for question in questions:
            assert set(question.sources) <= retrieved

    def test_batch_spans_more_than_one_section(self, fakes: _Fakes) -> None:
        questions = _quiz(fakes).questions

        cited = {source for question in questions for source in question.sources}
        assert len(cited) > 1

    def test_fewer_sections_than_questions_still_fills_the_batch(self, fakes: _Fakes) -> None:
        """Sections are reused round-robin when there are fewer of them than questions."""
        fakes = replace(
            fakes, retriever=_RecordingRetriever(section_count=_FEWER_SECTIONS_THAN_QUESTIONS)
        )

        questions = _quiz(fakes, n=_DEFAULT_QUIZ_SIZE).questions

        assert len(questions) == _DEFAULT_QUIZ_SIZE
        cited = {source for question in questions for source in question.sources}
        assert cited == fakes.retriever.source_refs(_THEME)


class TestQuizModelCalls:
    """Quiz questions are generated in batches, not one model call per question."""

    def test_fewer_model_calls_than_questions(self, fakes: _Fakes) -> None:
        _quiz(fakes)

        assert fakes.chat_client.invocation_count < _QUIZ_SIZE


class TestExtraItemsTruncated:
    """If the model over-delivers, the first n items are kept."""

    def test_keeps_first_n_items_in_order(self, fakes: _Fakes) -> None:
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

    def test_carries_the_official_time_limit(self, fakes: _Fakes) -> None:
        assert _exam(fakes).time_limit_seconds == MOCK_EXAM_TIME_LIMIT_SECONDS


class TestMockExamCalls:
    """One retrieval per theme; questions are generated in batches, not one call per question."""

    def test_one_theme_scoped_retrieval_per_theme(self, fakes: _Fakes) -> None:
        _exam(fakes)

        calls = fakes.retriever.calls
        assert Counter(theme for _, theme in calls) == Counter(EXAM_QUESTION_COUNTS.keys())
        for query, theme in calls:
            assert theme is not None
            assert theme.display_name_fr in query

    def test_fewer_model_calls_than_questions(self, fakes: _Fakes) -> None:
        _exam(fakes)

        assert fakes.chat_client.invocation_count < _EXAM_SIZE


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
        first = _quiz(replace(fakes, chat_client=_RecordingChatModel())).questions
        second = _quiz(replace(fakes, chat_client=_RecordingChatModel())).questions

        assert [q.question_id for q in first] == [q.question_id for q in second]


# ---------------------------------------------------------------------------
# Tests: error paths
# ---------------------------------------------------------------------------


class TestNoRetrievedMaterial:
    """Without grounding passages the model is never called and nothing is saved."""

    def test_quiz_raises_without_calling_model(self, fakes: _Fakes) -> None:
        fakes = replace(fakes, retriever=_RecordingRetriever(section_count=0))

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.chat_client.invocation_count == 0
        assert fakes.question_store.saved_questions == []

    def test_mock_exam_raises_without_calling_model(self, fakes: _Fakes) -> None:
        fakes = replace(fakes, retriever=_RecordingRetriever(section_count=0))

        with pytest.raises(QuestionGenerationError):
            _exam(fakes)

        assert fakes.chat_client.invocation_count == 0
        assert fakes.question_store.saved_questions == []


class TestInvalidModelReply:
    """A reply that cannot be turned into the requested questions raises and saves nothing."""

    def test_malformed_json_raises(self, fakes: _Fakes) -> None:
        fakes = replace(fakes, chat_client=_RecordingChatModel(canned_text=_MALFORMED_JSON))

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.question_store.saved_questions == []

    @pytest.mark.parametrize("payload", _WRONG_SHAPE_PAYLOADS.values(), ids=_WRONG_SHAPE_PAYLOADS.keys())
    def test_wrong_item_shape_raises(self, fakes: _Fakes, payload: str) -> None:
        fakes = replace(fakes, chat_client=_RecordingChatModel(canned_text=payload))

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.question_store.saved_questions == []

    def test_too_few_items_raises(self, fakes: _Fakes) -> None:
        fakes = replace(fakes, chat_client=_RecordingChatModel(items_per_call=_TOO_FEW_ITEMS))

        with pytest.raises(QuestionGenerationError):
            _quiz(fakes)

        assert fakes.question_store.saved_questions == []

    def test_mock_exam_failing_on_last_theme_saves_nothing(self, fakes: _Fakes) -> None:
        """A half-built exam must not leak its earlier themes into the bank."""
        healthy_calls = (_MAX_ITEMS_PER_CALL,) * (_THEME_COUNT - 1)
        fakes = replace(fakes, chat_client=_RecordingChatModel(items_by_call=(*healthy_calls, 0)))

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
