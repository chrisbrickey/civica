"""
Repository for corpus content chunks.

Persists chunked, embedded corpus text into the content_chunks table.
Upserts on content_hash so re-ingesting unchanged source material never duplicates rows.
"""

from collections.abc import Collection, Iterable

import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel, ConfigDict, Field

from civica.db.pool import get_pool
from civica.domain.chunk import Chunk
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS

_UPSERT_SQL = """
INSERT INTO content_chunks (
    content_hash, theme, page_slug, section_id, chunk_index, text, embedding
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (content_hash) DO UPDATE SET
    theme = EXCLUDED.theme,
    page_slug = EXCLUDED.page_slug,
    section_id = EXCLUDED.section_id,
    chunk_index = EXCLUDED.chunk_index,
    text = EXCLUDED.text,
    embedding = EXCLUDED.embedding
"""

_DELETE_NOT_IN_SQL = "DELETE FROM content_chunks WHERE content_hash != ALL(%s)"


class EmbeddedChunk(BaseModel):  # type: ignore[explicit-any]
    """A Chunk plus its embedding vector, ready to persist."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    embedding: list[float] = Field(min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS)


def _upsert_on_connection(
    rows: Iterable[EmbeddedChunk], conn: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    register_vector(conn)
    params = [
        (
            row.chunk.content_hash,
            row.chunk.theme.slug,
            row.chunk.page_slug,
            row.chunk.section_id,
            row.chunk.chunk_index,
            row.chunk.text,
            row.embedding,
        )
        for row in rows
    ]
    if not params:
        return
    with conn.cursor() as cursor:
        cursor.executemany(_UPSERT_SQL, params)


def upsert_chunks(
    rows: Iterable[EmbeddedChunk], conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None
) -> None:
    """Insert or update content chunks, keyed by content_hash.

    When conn is provided, the upsert runs on that connection directly.
    When conn is None, a connection is checked out from the shared pool.
    """
    if conn is not None:
        _upsert_on_connection(rows, conn)
        return

    pool = get_pool()
    with pool.connection() as pool_conn:
        _upsert_on_connection(rows, pool_conn)


def _delete_not_in_on_connection(
    keep_hashes: Collection[str], conn: psycopg.Connection[psycopg.rows.TupleRow]
) -> int:
    with conn.cursor() as cursor:
        cursor.execute(_DELETE_NOT_IN_SQL, (list(keep_hashes),))
        return cursor.rowcount


def delete_chunks_not_in(
    keep_hashes: Collection[str],
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
) -> int:
    """Delete every content_chunks row whose content_hash is not in keep_hashes.

    Returns the number of rows deleted. Raises ValueError if keep_hashes is
    empty, before touching the database, so a full wipe never happens
    implicitly. Resetting the table to empty is a manual operation, not
    something this function will do on your behalf.

    When conn is provided, the delete runs on that connection directly.
    When conn is None, a connection is checked out from the shared pool.
    """
    if not keep_hashes:
        raise ValueError("keep_hashes must not be empty; refusing to wipe content_chunks")

    if conn is not None:
        return _delete_not_in_on_connection(keep_hashes, conn)

    pool = get_pool()
    with pool.connection() as pool_conn:
        return _delete_not_in_on_connection(keep_hashes, pool_conn)
