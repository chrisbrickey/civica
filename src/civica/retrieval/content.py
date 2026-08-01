"""Semantic-search API for the official exam corpus."""

from dataclasses import dataclass

import psycopg
import psycopg.rows
from pgvector.psycopg import register_vector

from civica.db.pool import get_pool
from civica.domain.themes import Theme
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS, embed_query

# The HNSW index is built on embedding::halfvec(EMBEDDING_DIMENSIONS).
# So the query side must cast through the same halfvec(N) to hit that index.
_SEARCH_SQL_TEMPLATE = """
SELECT
    text,
    theme,
    page_slug,
    section_id,
    chunk_index,
    content_hash,
    embedding::halfvec({dim}) <=> %s::halfvec({dim}) AS distance
FROM content_chunks
{where_clause}
ORDER BY distance ASC
LIMIT %s
"""

_THEME_WHERE_CLAUSE = "WHERE theme = %s"

@dataclass(frozen=True)
class ContentChunk:
    """A single corpus chunk returned by search(), with its similarity to the query."""

    text: str
    theme: Theme
    page_slug: str
    section_id: str
    chunk_index: int
    content_hash: str
    similarity: float

def _search_on_connection(
    query: str,
    theme: Theme | None,
    k: int,
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> list[ContentChunk]:
    register_vector(conn)
    query_vector = embed_query(query)

    where_clause = _THEME_WHERE_CLAUSE if theme is not None else ""
    sql = _SEARCH_SQL_TEMPLATE.format(dim=EMBEDDING_DIMENSIONS, where_clause=where_clause)

    params: list[object] = [query_vector]
    if theme is not None:
        params.append(theme.slug)
    params.append(k)

    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    return [
        ContentChunk(
            text=row[0],
            theme=Theme.from_slug(row[1]),
            page_slug=row[2],
            section_id=row[3],
            chunk_index=row[4],
            content_hash=row[5],
            similarity=1 - row[6],
        )
        for row in rows
    ]

def search(
    query: str,
    theme: Theme | None = None,
    k: int = 5,
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
) -> list[ContentChunk]:
    """Semantic search over the official corpus, ranked by descending similarity.

    Embeds query using the universal embedder (same used for ingestion).
    Ranks content_chunks rows by cosine similarity (1 - cosine distance) via the
    pgvector halfvec HNSW index.

    When theme is provided, results are hard-filtered (not a ranking preference)
    to that theme only. This inserts some deterministic behavior, reducing the
    probability of responses diverging from official study material.
    NB: If a theme filter is applied, less than k chunks may be returned.

    When conn is provided, the search runs on that connection directly.
    When conn is None, a connection is checked out from the shared pool.
    """
    if conn is not None:
        return _search_on_connection(query, theme, k, conn)

    pool = get_pool()
    with pool.connection() as pool_conn:
        return _search_on_connection(query, theme, k, pool_conn)
