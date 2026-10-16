"""External test for civica.assessment.engine: the real model honours the grouped reply shape.

Requires ANTHROPIC_API_KEY and network access. Run with `uv run pytest -m external`.
"""

import uuid

import pytest

from civica.assessment.engine import generate_quiz
from civica.domain.chunk import Chunk
from civica.domain.source_ref import SourceRef
from civica.domain.themes import PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE, Theme
from civica.domain.user import UserId
from civica.retrieval.content import ContentChunk
from tests.support.assessment import OPTION_COUNT, FakeAnswerLogger, FakeQuestionStore

_USER_ID = UserId(uuid.UUID(int=42))
_SAMPLE_THEME = PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE

# Two short civic passages, so the model must label which group each question came from.
_SAMPLE_PASSAGES = (
    "La devise de la République française est « Liberté, Égalité, Fraternité ».",
    "Le drapeau français est bleu, blanc et rouge.",
)
_QUIZ_SIZE = len(_SAMPLE_PASSAGES)


def _chunk(index: int) -> ContentChunk:
    return ContentChunk(
        chunk=Chunk(
            theme=_SAMPLE_THEME,
            page_slug=f"sample-page-{index:03d}",
            section_id=f"section-{index:03d}",
            chunk_index=0,
            content_hash=f"sample-hash-{index:03d}",
            text=_SAMPLE_PASSAGES[index],
        ),
        similarity=0.9,
    )


_STUB_CHUNKS = [_chunk(i) for i in range(_QUIZ_SIZE)]


def _stub_retriever(query: str, theme: Theme | None) -> list[ContentChunk]:
    """Fixed retriever result, so no DB spinup is required."""
    return _STUB_CHUNKS


@pytest.mark.external
def test_generate_quiz_returns_one_question_per_group_from_real_model() -> None:
    """Real API call: generation succeeds only if every item names a valid group."""
    question_store = FakeQuestionStore()

    quiz = generate_quiz(
        _USER_ID,
        _SAMPLE_THEME,
        _QUIZ_SIZE,
        retriever=_stub_retriever,
        question_store=question_store,
        answer_logger=FakeAnswerLogger(),
    )

    expected_sources = [
        (SourceRef(page_slug=c.chunk.page_slug, section_id=c.chunk.section_id),)
        for c in _STUB_CHUNKS
    ]
    assert [q.sources for q in quiz.questions] == expected_sources
    for question in quiz.questions:
        assert question.text.strip()
        assert len(question.options) == OPTION_COUNT
        assert 0 <= question.correct_index < OPTION_COUNT
    assert question_store.saved_questions == list(quiz.questions)
