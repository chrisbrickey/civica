"""
Ingest the normalized JSON corpus into the content_chunks pgvector table.

- Reads data/corpus/thematic_sheets/<theme>/<page-slug>.json.
- Splits each section into chunks.
- Prefixes each window with the page title and section heading (to colocate chunks of the same official topic)
- Embeds new chunks, upserting rows keyed by content_hash.

Idempotent and incremental:
Re-running skips chunks already present in the table and only embeds new or changed content.
After embedding, stale rows (content removed or reformatted at the source) are pruned.

Entrypoint: main() runnable via:
    uv run python -m civica.scripts.ingest_corpus
"""

import hashlib
import json
import logging
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import psycopg
import psycopg.rows
from dotenv import load_dotenv

from civica.db.session import run_on_connection
from civica.domain.chunk import Chunk
from civica.domain.themes import Theme
from civica.ingestion.chunker import chunk_section
from civica.embeddings.embedder import DEFAULT_EMBEDDER, Embedder
from civica.ingestion.repository import EmbeddedChunk, delete_chunks_not_in, upsert_chunks
from civica.scripts.normalize_thematic_sheets import NormalizedPage

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path("data/corpus/thematic_sheets")
EMBED_BATCH_SIZE = 256
CONTEXT_SEPARATOR = " - "


@dataclass(frozen=True)
class PendingChunk:
    """A chunk that is ready to embed: an EmbeddedChunk minus its embedding."""

    theme: str
    page_slug: str
    section_id: str
    chunk_index: int
    content_hash: str
    text: str


def _contextualized(title: str, heading: str, chunk: str) -> str:
    """Prefix a chunk with its page title and section heading, when present."""
    prefix = CONTEXT_SEPARATOR.join(part for part in (title, heading) if part)
    return f"{prefix}\n\n{chunk}" if prefix else chunk


# NUL cannot appear in an embedding-model name, so it unambiguously separates
# the model identity from the text (no model name can spoof the boundary).
_HASH_FIELD_SEPARATOR = "\x00"


def _content_hash(text: str, embedding_model: str) -> str:
    """sha256 over the embedding model identity plus NFC-normalized UTF-8 text.

    NFC normalization keeps accent-encoding drift from duplicating rows.

    Folding the embedding model into the hash means that swapping to a different
    embedding model (with same dimensions) changes every hash. That is important
    because the embedding step of the ingestion pipeline skips unchanged hashes
    to reduce unnecessary cost. If the embedding model changes, we want the hash
    to also change so that re-embedding is triggered.
    """
    normalized = unicodedata.normalize("NFC", text)
    payload = f"{embedding_model}{_HASH_FIELD_SEPARATOR}{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pending_chunks_for_page(page: NormalizedPage, embedder: Embedder) -> list[PendingChunk]:
    """Chunk every section of one normalized page. Raises ValueError on unknown theme."""
    theme = Theme.from_slug(page["theme"])
    pending: list[PendingChunk] = []
    for section in page["sections"]:
        for index, window in enumerate(chunk_section(section["text"])):
            text = _contextualized(page["title"], section["heading"], window)
            pending.append(
                PendingChunk(
                    theme=theme.slug,
                    page_slug=page["slug"],
                    section_id=section["id"],
                    chunk_index=index,
                    content_hash=_content_hash(text, embedder.model),
                    text=text,
                )
            )
    return pending


def collect_pending_chunks(corpus_root: Path, *, embedder: Embedder) -> list[PendingChunk]:
    """Read every corpus JSON and return its chunks, deduplicated by content_hash."""
    by_hash: dict[str, PendingChunk] = {}
    page_count = 0
    for json_path in sorted(corpus_root.rglob("*.json")):
        page: NormalizedPage = json.loads(json_path.read_text(encoding="utf-8"))
        page_count += 1
        for chunk in _pending_chunks_for_page(page, embedder):
            by_hash.setdefault(chunk.content_hash, chunk)
    logger.info("Collected %d chunk(s) from %d page(s).", len(by_hash), page_count)
    return list(by_hash.values())


def _existing_hashes_on_connection(conn: psycopg.Connection[psycopg.rows.TupleRow]) -> set[str]:
    # scalar_row because pooled connections default to dict_row which cannot be accessed by index number
    with conn.cursor(row_factory=psycopg.rows.scalar_row) as cursor:
        hashes: list[str] = cursor.execute("SELECT content_hash FROM content_chunks").fetchall()
    return set(hashes)


def _existing_hashes(conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None) -> set[str]:
    """Return the content hashes already ingested."""
    return run_on_connection(_existing_hashes_on_connection, conn)


def ingest(
    corpus_root: Path,
    *,
    embedder: Embedder = DEFAULT_EMBEDDER,
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
) -> int:
    """Chunk, embed, and upsert everything new under corpus_root. Returns rows written.
    """
    pending = collect_pending_chunks(corpus_root, embedder=embedder)
    existing = _existing_hashes(conn)
    new_chunks = [chunk for chunk in pending if chunk.content_hash not in existing]
    logger.info(
        "%d chunk(s) already ingested; %d new chunk(s) to embed.",
        len(pending) - len(new_chunks),
        len(new_chunks),
    )

    written = 0
    for start in range(0, len(new_chunks), EMBED_BATCH_SIZE):
        batch = new_chunks[start : start + EMBED_BATCH_SIZE]
        embeddings = embedder.embed([chunk.text for chunk in batch])
        rows = [
            EmbeddedChunk(
                chunk=Chunk(
                    theme=Theme.from_slug(chunk.theme),
                    page_slug=chunk.page_slug,
                    section_id=chunk.section_id,
                    chunk_index=chunk.chunk_index,
                    content_hash=chunk.content_hash,
                    text=chunk.text,
                ),
                embedding=embedding,
            )
            for chunk, embedding in zip(batch, embeddings, strict=True)
        ]
        upsert_chunks(rows, conn)
        written += len(rows)
        logger.info("Upserted %d/%d chunk(s).", written, len(new_chunks))

    if pending:
        keep_hashes = {chunk.content_hash for chunk in pending}
        deleted = delete_chunks_not_in(keep_hashes, conn)
        logger.info("Pruned %d stale row(s) no longer present in the corpus.", deleted)
    else:
        logger.warning("Corpus yielded zero chunks; skipping prune to avoid emptying the table.")

    logger.info("Ingestion complete: %d new row(s) written.", written)
    return written


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    load_dotenv()
    ingest(DEFAULT_CORPUS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
