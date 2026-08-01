"""
Integration tests for civica.ingest.repository.

Exercises upsert_chunks() against a real (isolated, per-test) Postgres schema
via the db_schema fixture. Embeddings are fixed, deterministic fake vectors
(not real OpenAI output) since these tests only need to prove the upsert/read
contract of the content_chunks table.
"""

import psycopg
import psycopg.rows
import pytest

from civica.embeddings.embedder import EMBEDDING_DIMENSIONS
from civica.ingest.repository import ChunkRow, delete_chunks_not_in, upsert_chunks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THEME_A = "droits-et-devoirs"
_THEME_B = "histoire-geographie-et-culture"

_PAGE_SLUG = "page-001"
_SECTION_ID = "section-001"

_PRUNE_HASH_1 = "hash-prune-001"
_PRUNE_HASH_2 = "hash-prune-002"
_PRUNE_HASH_3 = "hash-prune-003"


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


def _upsert_three_chunks(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> tuple[str, str, str]:
    """Upsert three distinct chunk rows and return their content hashes in insertion order."""
    rows = [
        ChunkRow(
            theme=_THEME_A,
            page_slug=_PAGE_SLUG,
            section_id=_SECTION_ID,
            chunk_index=index,
            content_hash=content_hash,
            text=f"sample-text-{content_hash}",
            embedding=_fake_embedding(float(index)),
        )
        for index, content_hash in enumerate((_PRUNE_HASH_1, _PRUNE_HASH_2, _PRUNE_HASH_3))
    ]
    upsert_chunks(rows, conn=conn)
    return (_PRUNE_HASH_1, _PRUNE_HASH_2, _PRUNE_HASH_3)


class TestUpsertIdempotency:
    """Re-upserting the same content_hash updates the row in place rather than duplicating it."""

    def test_upsert_with_same_hash_and_changed_text_updates_single_row(
        self, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
    ) -> None:
        content_hash = "hash-001"
        original_row = ChunkRow(
            theme=_THEME_A,
            page_slug=_PAGE_SLUG,
            section_id=_SECTION_ID,
            chunk_index=0,
            content_hash=content_hash,
            text="sample-text-original",
            embedding=_fake_embedding(1.0),
        )
        updated_row = ChunkRow(
            theme=_THEME_A,
            page_slug=_PAGE_SLUG,
            section_id=_SECTION_ID,
            chunk_index=0,
            content_hash=content_hash,
            text="sample-text-updated",
            embedding=_fake_embedding(2.0),
        )

        upsert_chunks([original_row], conn=db_schema)
        upsert_chunks([updated_row], conn=db_schema)

        rows = _select_rows(db_schema)
        assert len(rows) == 1
        assert rows[0] == (content_hash, "sample-text-updated")


class TestMultiRowInsertAcrossThemes:
    """A theme filter query returns only chunks belonging to that theme."""

    def test_theme_filter_returns_only_matching_theme_rows(
        self, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
    ) -> None:
        rows_theme_a = [
            ChunkRow(
                theme=_THEME_A,
                page_slug=_PAGE_SLUG,
                section_id=_SECTION_ID,
                chunk_index=0,
                content_hash="hash-a-001",
                text="sample-text-a-001",
                embedding=_fake_embedding(1.0),
            ),
            ChunkRow(
                theme=_THEME_A,
                page_slug=_PAGE_SLUG,
                section_id=_SECTION_ID,
                chunk_index=1,
                content_hash="hash-a-002",
                text="sample-text-a-002",
                embedding=_fake_embedding(2.0),
            ),
        ]
        rows_theme_b = [
            ChunkRow(
                theme=_THEME_B,
                page_slug=_PAGE_SLUG,
                section_id=_SECTION_ID,
                chunk_index=0,
                content_hash="hash-b-001",
                text="sample-text-b-001",
                embedding=_fake_embedding(3.0),
            ),
        ]

        upsert_chunks(rows_theme_a + rows_theme_b, conn=db_schema)

        theme_a_hashes = {content_hash for content_hash, _ in _select_rows(db_schema, _THEME_A)}
        theme_b_hashes = {content_hash for content_hash, _ in _select_rows(db_schema, _THEME_B)}

        assert theme_a_hashes == {"hash-a-001", "hash-a-002"}
        assert theme_b_hashes == {"hash-b-001"}


class TestDeleteChunksNotIn:
    """delete_chunks_not_in() prunes content_chunks down to an exact set of hashes."""

    def test_deletes_rows_whose_hash_is_not_in_keep_set(
        self, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
    ) -> None:
        hash_1, hash_2, hash_3 = _upsert_three_chunks(db_schema)

        deleted_count = delete_chunks_not_in([hash_1, hash_2], conn=db_schema)

        remaining_hashes = {content_hash for content_hash, _ in _select_rows(db_schema)}
        assert deleted_count == 1
        assert remaining_hashes == {hash_1, hash_2}
        assert hash_3 not in remaining_hashes

    def test_deletes_nothing_when_keep_set_covers_all_existing_hashes(
        self, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
    ) -> None:
        hash_1, hash_2, hash_3 = _upsert_three_chunks(db_schema)

        deleted_count = delete_chunks_not_in([hash_1, hash_2, hash_3], conn=db_schema)

        remaining_hashes = {content_hash for content_hash, _ in _select_rows(db_schema)}
        assert deleted_count == 0
        assert remaining_hashes == {hash_1, hash_2, hash_3}

    def test_raises_value_error_and_deletes_nothing_when_keep_set_is_empty(
        self, db_schema: psycopg.Connection[psycopg.rows.TupleRow]
    ) -> None:
        _upsert_three_chunks(db_schema)

        with pytest.raises(ValueError):
            delete_chunks_not_in([], conn=db_schema)

        remaining_hashes = {content_hash for content_hash, _ in _select_rows(db_schema)}
        assert remaining_hashes == {_PRUNE_HASH_1, _PRUNE_HASH_2, _PRUNE_HASH_3}
