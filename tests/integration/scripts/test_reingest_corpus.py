"""
Integration tests for civica.scripts.ingest_corpus against an existing Postgres schema.

These tests cover the scenario where ingestion is triggered after
a previous version of the corpus was ingested and persisted.
"""

import json
import os
from collections.abc import Generator
from pathlib import Path

import psycopg
import psycopg.rows
import pytest

from civica.domain.themes import DROITS_ET_DEVOIRS
from civica.embeddings.embedder import (
    DEFAULT_EMBEDDER,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    Embedder,
    EmbedBatchFn,
)
from civica.scripts import ingest_corpus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THEME = DROITS_ET_DEVOIRS
_PAGE_SLUG = "sample-page"
_ADDED_PAGE_SLUG = "another-sample-page"
_SECTION_ID = "section-001"
_SECTION_HEADING = "sample-heading"
_SECTION_TEXT = "sample corpus text for incremental ingestion"
_ADDED_SECTION_TEXT = "different sample text"
_EXPECTED_CHUNK_COUNT = 1
_OTHER_MODEL = "sample-other-embedding-model"
_MODEL_A = "sample-model-a"
_MODEL_B = "sample-model-b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Return one fixed, deterministic vector per input text."""
    return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]


def _fake_embedder(embed: EmbedBatchFn, model: str = EMBEDDING_MODEL) -> Embedder:
    """Build an Embedder around a fake embed function, keyed to `model`.

    ingest() never calls embed_query, so it is stubbed to fail loudly if it ever is.
    """

    def _unreachable_embed_query(text: str) -> list[float]:
        raise AssertionError("ingest() must not call embed_query")

    return Embedder(model=model, embed=embed, embed_query=_unreachable_embed_query)


def _write_page(corpus_root: Path, slug: str, text: str) -> None:
    """Write one minimal normalized-page JSON under corpus_root."""
    page = {
        "theme": _THEME.slug,
        "slug": slug,
        "title": "sample-title",
        "source_url": "https://example.com",
        "description": "sample-description",
        "sections": [{"id": _SECTION_ID, "heading": _SECTION_HEADING, "text": text}],
    }
    (corpus_root / f"{slug}.json").write_text(json.dumps(page), encoding="utf-8")


def _count_rows(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> int:
    """Return the number of rows currently in content_chunks."""
    with conn.cursor(row_factory=psycopg.rows.scalar_row) as cursor:
        return int(cursor.execute("SELECT count(*) FROM content_chunks").fetchone())


def _stored_hashes(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> set[str]:
    """Return every content_hash currently persisted in content_chunks."""
    with conn.cursor(row_factory=psycopg.rows.scalar_row) as cursor:
        hashes: list[str] = cursor.execute("SELECT content_hash FROM content_chunks").fetchall()
    return set(hashes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def corpus_root(tmp_path: Path) -> Path:
    """A corpus directory holding one minimal normalized-page JSON."""
    _write_page(tmp_path, _PAGE_SLUG, _SECTION_TEXT)
    return tmp_path


@pytest.fixture()
def pooled_style_conn(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> Generator[psycopg.Connection[psycopg.rows.TupleRow], None, None]:
    """Yield a connection into the test schema configured like a pooled one.

    The shared pool hands out dict_row connections.
    That is a LangGraph requirement, which is covered by test_pool.py.
    """
    schema_row = db_schema.execute("SELECT current_schema()").fetchone()
    assert schema_row is not None
    conn = psycopg.connect(
        os.environ["TEST_DATABASE_URL"],
        autocommit=True,
        row_factory=psycopg.rows.dict_row,
        prepare_threshold=0,
    )
    try:
        conn.execute(f"SET search_path TO {schema_row[0]}, public")
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIncrementalIngestion:
    """Re-ingesting an unchanged corpus skips the work already done."""

    def test_first_ingest_writes_every_chunk(
        self,
        corpus_root: Path,
        pooled_style_conn: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        written = ingest_corpus.ingest(
            corpus_root, embedder=_fake_embedder(_fake_embed), conn=pooled_style_conn
        )

        assert written == _EXPECTED_CHUNK_COUNT
        assert _count_rows(pooled_style_conn) == _EXPECTED_CHUNK_COUNT

    def test_hashes_are_computed_from_the_injected_embedder_not_the_module_default(
        self,
        corpus_root: Path,
        pooled_style_conn: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        """content_hash is computed from the same embedder that produces the vectors."""
        other_embedder = _fake_embedder(_fake_embed, model=_OTHER_MODEL)
        assert other_embedder.model != DEFAULT_EMBEDDER.model

        ingest_corpus.ingest(corpus_root, embedder=other_embedder, conn=pooled_style_conn)

        expected_hashes = {
            chunk.content_hash
            for chunk in ingest_corpus.collect_pending_chunks(
                corpus_root, embedder=other_embedder
            )
        }
        default_hashes = {
            chunk.content_hash
            for chunk in ingest_corpus.collect_pending_chunks(
                corpus_root, embedder=DEFAULT_EMBEDDER
            )
        }
        stored_hashes = _stored_hashes(pooled_style_conn)

        assert stored_hashes == expected_hashes
        assert stored_hashes.isdisjoint(default_hashes)

    def test_second_ingest_of_unchanged_corpus_embeds_nothing(
        self,
        corpus_root: Path,
        pooled_style_conn: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        """Already-ingested hashes are read back, so no chunk is re-embedded."""
        ingest_corpus.ingest(
            corpus_root, embedder=_fake_embedder(_fake_embed), conn=pooled_style_conn
        )

        def _fail_if_called(texts: list[str]) -> list[list[float]]:
            raise AssertionError(f"re-embedded {len(texts)} already-ingested chunk(s)")

        written = ingest_corpus.ingest(
            corpus_root, embedder=_fake_embedder(_fail_if_called), conn=pooled_style_conn
        )

        assert written == 0
        assert _count_rows(pooled_style_conn) == _EXPECTED_CHUNK_COUNT

    def test_second_ingest_with_default_embedder_writes_zero_new_rows(
        self,
        corpus_root: Path,
        pooled_style_conn: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        """Re-running ingest() with the production default embedder embeds nothing new."""
        assert DEFAULT_EMBEDDER.model == EMBEDDING_MODEL
        ingest_corpus.ingest(
            corpus_root, embedder=_fake_embedder(_fake_embed), conn=pooled_style_conn
        )

        written = ingest_corpus.ingest(
            corpus_root, embedder=DEFAULT_EMBEDDER, conn=pooled_style_conn
        )

        assert written == 0
        assert _count_rows(pooled_style_conn) == _EXPECTED_CHUNK_COUNT

    def test_grown_corpus_embeds_only_the_added_chunk(
        self,
        corpus_root: Path,
        pooled_style_conn: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        """A corpus that gained a page embeds the new chunk and keeps the rest."""
        ingest_corpus.ingest(
            corpus_root, embedder=_fake_embedder(_fake_embed), conn=pooled_style_conn
        )
        _write_page(corpus_root, _ADDED_PAGE_SLUG, _ADDED_SECTION_TEXT)

        embedded_batches: list[list[str]] = []

        def _recording_embed(texts: list[str]) -> list[list[float]]:
            embedded_batches.append(texts)
            return _fake_embed(texts)

        written = ingest_corpus.ingest(
            corpus_root, embedder=_fake_embedder(_recording_embed), conn=pooled_style_conn
        )

        assert written == 1
        assert [len(batch) for batch in embedded_batches] == [1]
        assert _count_rows(pooled_style_conn) == _EXPECTED_CHUNK_COUNT + 1


class TestModelSwap:
    """The table can never hold two embedding model's vectors at once."""

    def test_reingest_under_a_new_model_prunes_the_old_models_rows(
        self,
        corpus_root: Path,
        pooled_style_conn: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        """Re-ingesting the same corpus under model B replaces model A's rows."""
        ingest_corpus.ingest(
            corpus_root,
            embedder=_fake_embedder(_fake_embed, model=_MODEL_A),
            conn=pooled_style_conn,
        )

        ingest_corpus.ingest(
            corpus_root,
            embedder=_fake_embedder(_fake_embed, model=_MODEL_B),
            conn=pooled_style_conn,
        )

        expected_hashes = {
            chunk.content_hash
            for chunk in ingest_corpus.collect_pending_chunks(
                corpus_root, embedder=_fake_embedder(_fake_embed, model=_MODEL_B)
            )
        }

        assert _count_rows(pooled_style_conn) == _EXPECTED_CHUNK_COUNT
        assert _stored_hashes(pooled_style_conn) == expected_hashes
