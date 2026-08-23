"""Semantic-search API for the official exam corpus."""

import psycopg
import psycopg.rows
from pgvector.psycopg import register_vector
from pydantic import BaseModel, ConfigDict

from civica.db.session import run_on_connection
from civica.domain.chunk import Chunk
from civica.domain.themes import Theme
from civica.embeddings.embedder import DEFAULT_EMBEDDER, EMBEDDING_DIMENSIONS, Embedder

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

class ContentChunk(BaseModel):  # type: ignore[explicit-any]
    """A single corpus Chunk returned by search(), with its similarity to the query."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    similarity: float

class _SearchRow(BaseModel):  # type: ignore[explicit-any]
    """One raw content_chunks row as returned by the search query.

    Columns map by name via psycopg class_row, so field access is typed rather
    than positional. `theme` is the stored slug string and `distance` is cosine
    distance, both DB-native; hydration into the domain shape (a Theme instance
    and similarity = 1 - distance) happens in to_content_chunk.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    theme: str
    page_slug: str
    section_id: str
    chunk_index: int
    content_hash: str
    distance: float

    def to_content_chunk(self) -> ContentChunk:
        return ContentChunk(
            chunk=Chunk(
                theme=Theme.from_slug(self.theme),
                page_slug=self.page_slug,
                section_id=self.section_id,
                chunk_index=self.chunk_index,
                content_hash=self.content_hash,
                text=self.text,
            ),
            similarity=1 - self.distance,
        )

def _search_on_connection(
    query: str,
    theme: Theme | None,
    k: int,
    embedder: Embedder,
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> list[ContentChunk]:
    register_vector(conn)
    query_vector = embedder.embed_query(query)

    where_clause = _THEME_WHERE_CLAUSE if theme is not None else ""
    sql = _SEARCH_SQL_TEMPLATE.format(dim=EMBEDDING_DIMENSIONS, where_clause=where_clause)

    params: list[object] = [query_vector]
    if theme is not None:
        params.append(theme.slug)
    params.append(k)

    with conn.cursor(row_factory=psycopg.rows.class_row(_SearchRow)) as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    return [row.to_content_chunk() for row in rows]

def search(
    query: str,
    theme: Theme | None = None,
    k: int = 5,
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
    *,
    embedder: Embedder = DEFAULT_EMBEDDER,
) -> list[ContentChunk]:
    """Semantic search over the official corpus, ranked by descending similarity.

    Embeds query using the given embedder (same one used for ingestion, by default).
    Ranks content_chunks rows by cosine similarity (1 - cosine distance) via the
    pgvector halfvec HNSW index.

    When theme is provided, results are hard-filtered (not a ranking preference)
    to that theme only. This inserts some deterministic behavior, reducing the
    probability of responses diverging from official study material.
    NB: If a theme filter is applied, less than k chunks may be returned.
    """
    return run_on_connection(
        lambda connection: _search_on_connection(query, theme, k, embedder, connection), conn
    )
