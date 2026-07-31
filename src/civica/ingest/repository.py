"""
Repository for corpus content chunks.

Persists chunked, embedded corpus text into the content_chunks table.
Upserts on content_hash so re-ingesting unchanged source material never duplicates rows.
"""

from collections.abc import Collection, Iterable
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from civica.db.pool import get_pool

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


@dataclass(frozen=True)
class ChunkRow:
    """A single embedded corpus chunk ready to be persisted."""

    theme: str
    page_slug: str
    section_id: str
    chunk_index: int
    content_hash: str
    text: str
    embedding: list[float]


def _upsert_on_connection(
    rows: Iterable[ChunkRow], conn: psycopg.Connection[psycopg.rows.TupleRow]
) -> None:
    register_vector(conn)
    params = [
        (
            row.content_hash,
            row.theme,
            row.page_slug,
            row.section_id,
            row.chunk_index,
            row.text,
            row.embedding,
        )
        for row in rows
    ]
    if not params:
        return
    with conn.cursor() as cursor:
        cursor.executemany(_UPSERT_SQL, params)


def upsert_chunks(
    rows: Iterable[ChunkRow], conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None
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
