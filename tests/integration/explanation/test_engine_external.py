"""Integration tests for civica.explanation.engine."""

import pytest

from civica.domain.chunk import Chunk
from civica.domain.themes import PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE
from civica.explanation.engine import explain
from civica.retrieval.content import ContentChunk

_SAMPLE_THEME = PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE
_SAMPLE_QUESTION = "What is the motto of the Republic?"

_SAMPLE_PASSAGE = (
    "La devise de la République française est « Liberté, Égalité, Fraternité »."
)

# Case-insensitive substrings that would indicate the model refused or declined to answer
_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "insufficient official material",
)

# Substrings that only appear if raw content blocks were stringified instead of read as text.
_UNEXTRACTED_BLOCK_MARKERS = ("'type':", '"type":')


def _stub_retriever(_query: str, _theme: object) -> list[ContentChunk]:
    """Fixed retriever result: one short French civic passage."""
    return [
        ContentChunk(
            chunk=Chunk(
                theme=_SAMPLE_THEME,
                page_slug="sample-page",
                section_id="sample-section",
                chunk_index=0,
                content_hash="sample-hash",
                text=_SAMPLE_PASSAGE,
            ),
            similarity=0.9,
        )
    ]


@pytest.mark.external
def test_explain_returns_grounded_explanation_from_real_model() -> None:
    """Real API call with a stubbed retriever, so no DB spinup is required."""

    stub_chunks = _stub_retriever(_SAMPLE_QUESTION, _SAMPLE_THEME)

    result = explain(
        _SAMPLE_QUESTION,
        _SAMPLE_THEME,
        retriever=lambda query, theme: stub_chunks,
    )

    assert result.text.strip() != ""

    lowered_text = result.text.lower()
    for marker in _REFUSAL_MARKERS:
        assert marker not in lowered_text

    # A stringified block list is non-empty and refusal-free, so assert on it directly.
    for marker in _UNEXTRACTED_BLOCK_MARKERS:
        assert marker not in result.text

    assert result.context_chunks == tuple(stub_chunks)
