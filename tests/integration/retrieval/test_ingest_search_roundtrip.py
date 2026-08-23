"""Integration test proving ingest() and search() share one embedding space."""

import json
from pathlib import Path

import psycopg
import psycopg.rows
import pytest

from civica.domain.themes import DROITS_ET_DEVOIRS
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS, Embedder
from civica.retrieval import content
from civica.scripts import ingest_corpus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THEME = DROITS_ET_DEVOIRS
_FAKE_MODEL = "test-roundtrip-embedding-model"

_SECTION_ID = "section-001"
_SECTION_HEADING = "sample-heading"

_ALPHA_PAGE_SLUG = "sample-page-alpha"
_ALPHA_TITLE = "sample-title-alpha"
_ALPHA_SEED_WORD = "alpha-topic"
_ALPHA_TEXT = f"Sample corpus text about {_ALPHA_SEED_WORD} for retrieval testing."

_BETA_PAGE_SLUG = "sample-page-beta"
_BETA_TITLE = "sample-title-beta"
_BETA_SEED_WORD = "beta-topic"
_BETA_TEXT = f"Sample corpus text about {_BETA_SEED_WORD} for retrieval testing."

_SEED_WORDS = (_ALPHA_SEED_WORD, _BETA_SEED_WORD)
_AXIS_BY_SEED_WORD = {word: index for index, word in enumerate(_SEED_WORDS)}

_QUERY_TEXT = f"A query about {_BETA_SEED_WORD}."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vector_for_text(text: str) -> list[float]:
    """Map text to a unit vector on the axis of whichever seed word it contains.

    Both embed (batch) and embed_query route through this helper, so the
    ingested chunk vectors and the query vector live in the same fake space.
    """
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for word in _SEED_WORDS:
        if word in text:
            vector[_AXIS_BY_SEED_WORD[word]] = 1.0
    return vector


def _seeded_embedder() -> Embedder:
    """One Embedder whose batch and single-text paths both use _vector_for_text."""

    def _embed(texts: list[str]) -> list[list[float]]:
        return [_vector_for_text(text) for text in texts]

    def _embed_query(text: str) -> list[float]:
        return _vector_for_text(text)

    return Embedder(model=_FAKE_MODEL, embed=_embed, embed_query=_embed_query)


def _write_page(corpus_root: Path, slug: str, title: str, text: str) -> None:
    """Write one minimal normalized-page JSON under corpus_root."""
    page = {
        "theme": _THEME.slug,
        "slug": slug,
        "title": title,
        "source_url": "https://example.com",
        "description": "sample-description",
        "sections": [{"id": _SECTION_ID, "heading": _SECTION_HEADING, "text": text}],
    }
    (corpus_root / f"{slug}.json").write_text(json.dumps(page), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def corpus_root(tmp_path: Path) -> Path:
    """A corpus directory holding two distinguishable pages."""
    _write_page(tmp_path, _ALPHA_PAGE_SLUG, _ALPHA_TITLE, _ALPHA_TEXT)
    _write_page(tmp_path, _BETA_PAGE_SLUG, _BETA_TITLE, _BETA_TEXT)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestThenSearchRoundtrip:
    """A chunk ingested under one embedder is findable by search() using that embedder."""

    def test_search_ranks_the_semantically_matching_ingested_chunk_first(
        self,
        corpus_root: Path,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        embedder = _seeded_embedder()

        ingest_corpus.ingest(corpus_root, embedder=embedder, conn=db_schema)

        pending = ingest_corpus.collect_pending_chunks(corpus_root, embedder=embedder)
        expected_match = next(
            chunk for chunk in pending if chunk.page_slug == _BETA_PAGE_SLUG
        )

        results = content.search(
            query=_QUERY_TEXT, k=2, conn=db_schema, embedder=embedder
        )

        assert results[0].chunk.page_slug == _BETA_PAGE_SLUG
        assert results[0].chunk.text == expected_match.text
        assert results[0].similarity > results[1].similarity
