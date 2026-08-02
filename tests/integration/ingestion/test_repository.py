"""
Integration tests for civica.ingestion.repository.

Exercises upsert_chunks() against a real (isolated, per-test) Postgres schema
via the db_schema fixture. Embeddings are fixed, deterministic fake vectors
(not real OpenAI output) since these tests only need to prove the upsert/read
contract of the content_chunks table.
"""

from collections.abc import Callable

import psycopg
import psycopg.rows
import pytest
from pydantic import ValidationError

from civica.domain.chunk import Chunk
from civica.domain.themes import DROITS_ET_DEVOIRS, HISTOIRE_GEOGRAPHIE_ET_CULTURE
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS
from civica.ingestion.repository import EmbeddedChunk, delete_chunks_not_in, upsert_chunks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THEME_A = DROITS_ET_DEVOIRS
_THEME_B = HISTOIRE_GEOGRAPHIE_ET_CULTURE


def _fake_embedding(seed: float) -> list[float]:
    """A fixed, deterministic 3072-dim vector distinguished only by its last entry."""
    return [0.0] * (EMBEDDING_DIMENSIONS - 1) + [seed]


def _select_rows(
    conn: psycopg.Connection[psycopg.rows.TupleRow], theme: str | None = None
) -> list[tuple[str, str]]:
    """Return (content_hash, text) rows, optionally filtered by theme."""
    if theme is None:
        cursor = conn.execute("SELECT content_hash, text FROM content_chunks")
    else:
        cursor = conn.execute(
            "SELECT content_hash, text FROM content_chunks WHERE theme = %s", (theme,)
        )
    return [(row[0], row[1]) for row in cursor.fetchall()]


def _select_hashes(
    conn: psycopg.Connection[psycopg.rows.TupleRow], theme: str | None = None
) -> set[str]:
    """Return the set of content hashes, optionally filtered by theme."""
    return {content_hash for content_hash, _ in _select_rows(conn, theme)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def three_seeded_chunks(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    make_chunk_row: Callable[..., EmbeddedChunk],
) -> list[EmbeddedChunk]:
    """Upsert three distinct chunk rows and return them in insertion order."""
    rows = [
        make_chunk_row(
            theme=_THEME_A,
            chunk_index=index,
            content_hash=f"hash-prune-{index:03d}",
            text=f"sample-text-{index:03d}",
            embedding=_fake_embedding(float(index)),
        )
        for index in range(3)
    ]
    upsert_chunks(rows, conn=db_schema)
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpsertIdempotency:
    """Re-upserting the same content_hash updates the row in place rather than duplicating it."""

    def test_upsert_with_same_hash_and_changed_text_updates_single_row(
        self,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
        make_chunk_row: Callable[..., EmbeddedChunk],
    ) -> None:
        original_row = make_chunk_row(
            theme=_THEME_A,
            content_hash="hash-001",
            text="sample-text-original",
            embedding=_fake_embedding(1.0),
        )
        updated_row = original_row.model_copy(update={
            "embedding": _fake_embedding(2.0),
            "chunk": original_row.chunk.model_copy(update={"text": "sample-text-updated"}),
        })

        upsert_chunks([original_row], conn=db_schema)
        upsert_chunks([updated_row], conn=db_schema)

        rows = _select_rows(db_schema)
        assert len(rows) == 1
        assert rows[0] == (updated_row.chunk.content_hash, updated_row.chunk.text)


class TestMultiRowInsertAcrossThemes:
    """A theme filter query returns only chunks belonging to that theme."""

    def test_theme_filter_returns_only_matching_theme_rows(
        self,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
        make_chunk_row: Callable[..., EmbeddedChunk],
    ) -> None:
        rows_theme_a = [
            make_chunk_row(
                theme=_THEME_A,
                chunk_index=index,
                content_hash=f"hash-a-{index:03d}",
                embedding=_fake_embedding(float(index)),
            )
            for index in range(2)
        ]
        rows_theme_b = [
            make_chunk_row(
                theme=_THEME_B,
                content_hash="hash-b-001",
                embedding=_fake_embedding(3.0),
            ),
        ]

        upsert_chunks(rows_theme_a + rows_theme_b, conn=db_schema)

        assert _select_hashes(db_schema, _THEME_A.slug) == {
            row.chunk.content_hash for row in rows_theme_a
        }
        assert _select_hashes(db_schema, _THEME_B.slug) == {
            row.chunk.content_hash for row in rows_theme_b
        }


class TestDeleteChunksNotIn:
    """delete_chunks_not_in() prunes content_chunks down to an exact set of hashes."""

    def test_deletes_rows_whose_hash_is_not_in_keep_set(
        self,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
        three_seeded_chunks: list[EmbeddedChunk],
    ) -> None:
        kept, dropped = three_seeded_chunks[:2], three_seeded_chunks[2]

        deleted_count = delete_chunks_not_in(
            [row.chunk.content_hash for row in kept], conn=db_schema
        )

        remaining_hashes = _select_hashes(db_schema)
        assert deleted_count == 1
        assert remaining_hashes == {row.chunk.content_hash for row in kept}
        assert dropped.chunk.content_hash not in remaining_hashes

    def test_deletes_nothing_when_keep_set_covers_all_existing_hashes(
        self,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
        three_seeded_chunks: list[EmbeddedChunk],
    ) -> None:
        all_hashes = [row.chunk.content_hash for row in three_seeded_chunks]

        deleted_count = delete_chunks_not_in(all_hashes, conn=db_schema)

        assert deleted_count == 0
        assert _select_hashes(db_schema) == set(all_hashes)

    def test_raises_value_error_and_deletes_nothing_when_keep_set_is_empty(
        self,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
        three_seeded_chunks: list[EmbeddedChunk],
    ) -> None:
        with pytest.raises(ValueError):
            delete_chunks_not_in([], conn=db_schema)

        assert _select_hashes(db_schema) == {
            row.chunk.content_hash for row in three_seeded_chunks
        }


class TestEmbeddingDimensionValidation:
    """EmbeddedChunk rejects embeddings of the wrong length before any DB write."""

    def _make_valid_chunk(self) -> Chunk:
        return Chunk(
            theme=_THEME_A,
            page_slug="sample-page",
            section_id="section-001",
            chunk_index=0,
            content_hash="hash-validation",
            text="sample-text",
        )

    def test_raises_validation_error_when_embedding_is_too_short(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddedChunk(
                chunk=self._make_valid_chunk(),
                embedding=[0.0] * (EMBEDDING_DIMENSIONS - 1),
            )

    def test_raises_validation_error_when_embedding_is_too_long(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddedChunk(
                chunk=self._make_valid_chunk(),
                embedding=[0.0] * (EMBEDDING_DIMENSIONS + 1),
            )
