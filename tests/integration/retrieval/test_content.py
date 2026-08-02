"""
Integration tests for civica.retrieval.content.

Exercises search() against a real (isolated, per-test) Postgres schema.
Query embeddings are stubbed by monkeypatching (no network calls, not 'external').
Corpus embeddings are fixed, hand-picked fake vectors (not real OpenAI output)
chosen so their cosine similarity to the stubbed query vector is hand-computable.
"""

from collections.abc import Callable

import psycopg
import psycopg.rows
import pytest

from civica.domain.themes import DROITS_ET_DEVOIRS, HISTOIRE_GEOGRAPHIE_ET_CULTURE, Theme
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS
from civica.ingestion.repository import EmbeddedChunk, upsert_chunks
from civica.retrieval import content

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THEME_A = DROITS_ET_DEVOIRS
_THEME_B = HISTOIRE_GEOGRAPHIE_ET_CULTURE

_DEFAULT_QUERY_TEXT = "sample-query"

_SIMILARITY_TOLERANCE = 1e-2  # HNSW index compares half-precision halfvec casts

# The stubbed query embedding lives on axis 0; embeddings built on that axis
# match the query exactly, embeddings on any other axis are orthogonal to it.
_QUERY_AXIS = 0
_ORTHOGONAL_AXIS = 1


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def _unit_vector(axis: int) -> list[float]:
    """A unit vector at EMBEDDING_DIMENSIONS with a single 1.0 at `axis`."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[axis] = 1.0
    return vector


def _combine_vectors(*vectors: list[float]) -> list[float]:
    """Component-wise sum of same-length vectors, left unnormalized."""
    return [sum(components) for components in zip(*vectors)]


def _matching_embedding() -> list[float]:
    """An embedding identical to the stubbed query vector (cosine similarity 1.0)."""
    return _unit_vector(_QUERY_AXIS)


def _orthogonal_embedding() -> list[float]:
    """An embedding orthogonal to the stubbed query vector (cosine similarity 0.0)."""
    return _unit_vector(_ORTHOGONAL_AXIS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_query_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[list[float]], None]:
    """Stub content.embed_query with the default query vector; return an override.

    Patches the reference inside civica.retrieval.content (not the definition
    in civica.embeddings.embedder) so search() never makes a real network
    call to the embeddings API. The default stub is applied at setup; tests
    needing a different query vector call the returned function.
    """

    def _stub(vector: list[float]) -> None:
        monkeypatch.setattr(content, "embed_query", lambda text: vector)

    _stub(_unit_vector(_QUERY_AXIS))
    return _stub


@pytest.fixture()
def seed(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> Callable[[list[EmbeddedChunk]], None]:
    """Upsert chunk rows on the per-test schema connection."""

    def _seed(rows: list[EmbeddedChunk]) -> None:
        upsert_chunks(rows, conn=db_schema)

    return _seed


@pytest.fixture()
def run_search(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    stub_query_embedding: Callable[[list[float]], None],
) -> Callable[..., list[content.ContentChunk]]:
    """Run content.search with the default query text on the per-test schema.

    Depends on stub_query_embedding so no test can hit the embeddings API by
    forgetting to request the stub.
    """

    def _run(theme: Theme | None = None, k: int = 5) -> list[content.ContentChunk]:
        return content.search(query=_DEFAULT_QUERY_TEXT, theme=theme, k=k, conn=db_schema)

    return _run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrderingBySimilarity:
    """search() returns seeded chunks ordered by descending similarity."""

    def test_returns_chunks_ordered_by_descending_similarity(
        self,
        make_chunk_row: Callable[..., EmbeddedChunk],
        seed: Callable[[list[EmbeddedChunk]], None],
        run_search: Callable[..., list[content.ContentChunk]],
    ) -> None:
        # cosine similarity to the stubbed query vector:
        #   exact_match:      identical vector            -> 1.0
        #   partial_match:    query + orthogonal, unnormalized -> 1/sqrt(2) ~= 0.707
        #   orthogonal_match: no overlap with the query   -> 0.0
        exact_match = make_chunk_row(
            theme=_THEME_A, embedding=_matching_embedding(), content_hash="hash-exact"
        )
        partial_match = make_chunk_row(
            theme=_THEME_A,
            embedding=_combine_vectors(_matching_embedding(), _orthogonal_embedding()),
            content_hash="hash-partial",
        )
        orthogonal_match = make_chunk_row(
            theme=_THEME_A,
            embedding=_orthogonal_embedding(),
            content_hash="hash-orthogonal",
        )
        seed([orthogonal_match, exact_match, partial_match])

        results = run_search(k=3)

        assert [result.chunk.content_hash for result in results] == [
            exact_match.chunk.content_hash,
            partial_match.chunk.content_hash,
            orthogonal_match.chunk.content_hash,
        ]
        assert results[0].similarity > results[1].similarity > results[2].similarity


class TestThemeFilter:
    """The theme filter is a hard filter, not just an ordering preference."""

    def test_theme_filter_excludes_more_similar_chunk_from_other_theme(
        self,
        make_chunk_row: Callable[..., EmbeddedChunk],
        seed: Callable[[list[EmbeddedChunk]], None],
        run_search: Callable[..., list[content.ContentChunk]],
    ) -> None:
        # The theme-A chunk is the closer match to the query, but the filter
        # requests theme B, so only the (less similar) theme-B chunk returns.
        more_similar_other_theme = make_chunk_row(
            theme=_THEME_A, embedding=_matching_embedding(), content_hash="hash-theme-a"
        )
        less_similar_target_theme = make_chunk_row(
            theme=_THEME_B,
            embedding=_orthogonal_embedding(),
            content_hash="hash-theme-b",
        )
        seed([more_similar_other_theme, less_similar_target_theme])

        results = run_search(theme=_THEME_B)

        assert [result.chunk.content_hash for result in results] == [
            less_similar_target_theme.chunk.content_hash
        ]
        assert all(result.chunk.theme == _THEME_B for result in results)


class TestSimilarityComputation:
    """similarity equals 1 - cosine_distance, verified with hand-computable vectors."""

    def test_orthogonal_vectors_have_zero_similarity(
        self,
        make_chunk_row: Callable[..., EmbeddedChunk],
        seed: Callable[[list[EmbeddedChunk]], None],
        run_search: Callable[..., list[content.ContentChunk]],
    ) -> None:
        chunk = make_chunk_row(
            theme=_THEME_A,
            embedding=_orthogonal_embedding(),
            content_hash="hash-orthogonal",
        )
        seed([chunk])

        results = run_search(k=1)

        assert results[0].similarity == pytest.approx(0.0, abs=_SIMILARITY_TOLERANCE)

    def test_identical_vectors_have_full_similarity(
        self,
        make_chunk_row: Callable[..., EmbeddedChunk],
        seed: Callable[[list[EmbeddedChunk]], None],
        run_search: Callable[..., list[content.ContentChunk]],
    ) -> None:
        chunk = make_chunk_row(
            theme=_THEME_A,
            embedding=_matching_embedding(),
            content_hash="hash-identical",
        )
        seed([chunk])

        results = run_search(k=1)

        assert results[0].similarity == pytest.approx(1.0, abs=_SIMILARITY_TOLERANCE)


class TestResultLimit:
    """k caps the number of rows returned."""

    def test_k_limits_number_of_results(
        self,
        make_chunk_row: Callable[..., EmbeddedChunk],
        seed: Callable[[list[EmbeddedChunk]], None],
        run_search: Callable[..., list[content.ContentChunk]],
    ) -> None:
        rows = [
            make_chunk_row(
                theme=_THEME_A,
                embedding=_unit_vector(axis),
                content_hash=f"hash-{axis}",
            )
            for axis in range(4)
        ]
        seed(rows)

        results = run_search(k=2)

        assert len(results) == 2


class TestFieldRoundTrip:
    """Returned ContentChunk fields round-trip what was seeded."""

    def test_fields_round_trip_seeded_values(
        self,
        make_chunk_row: Callable[..., EmbeddedChunk],
        seed: Callable[[list[EmbeddedChunk]], None],
        run_search: Callable[..., list[content.ContentChunk]],
    ) -> None:
        seeded = make_chunk_row(
            theme=_THEME_A,
            embedding=_matching_embedding(),
            content_hash="hash-round-trip",
            text="sample-text-round-trip",
            page_slug="sample-page-round-trip",
            section_id="section-round-trip",
            chunk_index=7,
        )
        seed([seeded])

        results = run_search(k=1)

        assert len(results) == 1
        result = results[0]
        assert result.chunk.text == seeded.chunk.text
        assert isinstance(result.chunk.theme, Theme)
        assert result.chunk.theme.slug == seeded.chunk.theme.slug
        assert result.chunk.page_slug == seeded.chunk.page_slug
        assert result.chunk.section_id == seeded.chunk.section_id
        assert result.chunk.chunk_index == seeded.chunk.chunk_index
        assert result.chunk.content_hash == seeded.chunk.content_hash
