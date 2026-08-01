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

from dotenv import load_dotenv

from civica.db.pool import get_pool
from civica.domain.themes import Theme
from civica.ingest.chunker import chunk_section
from civica.embeddings.embedder import embed
from civica.ingest.repository import ChunkRow, delete_chunks_not_in, upsert_chunks
from civica.scripts.normalize_thematic_sheets import NormalizedPage

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path("data/corpus/thematic_sheets")
EMBED_BATCH_SIZE = 256
CONTEXT_SEPARATOR = " - "


@dataclass(frozen=True)
class PendingChunk:
    """A chunk that is ready to embed: a ChunkRow minus its embedding."""

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


def _content_hash(text: str) -> str:
    """sha256 over NFC-normalized UTF-8 so accent encoding drift never duplicates rows."""
    normalized = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _pending_chunks_for_page(page: NormalizedPage) -> list[PendingChunk]:
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
                    content_hash=_content_hash(text),
                    text=text,
                )
            )
    return pending


def collect_pending_chunks(corpus_root: Path) -> list[PendingChunk]:
    """Read every corpus JSON and return its chunks, deduplicated by content_hash."""
    by_hash: dict[str, PendingChunk] = {}
    page_count = 0
    for json_path in sorted(corpus_root.rglob("*.json")):
        page: NormalizedPage = json.loads(json_path.read_text(encoding="utf-8"))
        page_count += 1
        for chunk in _pending_chunks_for_page(page):
            by_hash.setdefault(chunk.content_hash, chunk)
    logger.info("Collected %d chunk(s) from %d page(s).", len(by_hash), page_count)
    return list(by_hash.values())


def _existing_hashes() -> set[str]:
    """Return the content hashes already ingested."""
    with get_pool().connection() as conn:
        rows = conn.execute("SELECT content_hash FROM content_chunks").fetchall()
    return {row[0] for row in rows}


def ingest(corpus_root: Path) -> int:
    """Chunk, embed, and upsert everything new under corpus_root. Returns rows written."""
    pending = collect_pending_chunks(corpus_root)
    existing = _existing_hashes()
    new_chunks = [chunk for chunk in pending if chunk.content_hash not in existing]
    logger.info(
        "%d chunk(s) already ingested; %d new chunk(s) to embed.",
        len(pending) - len(new_chunks),
        len(new_chunks),
    )

    written = 0
    for start in range(0, len(new_chunks), EMBED_BATCH_SIZE):
        batch = new_chunks[start : start + EMBED_BATCH_SIZE]
        embeddings = embed([chunk.text for chunk in batch])
        rows = [
            ChunkRow(
                theme=chunk.theme,
                page_slug=chunk.page_slug,
                section_id=chunk.section_id,
                chunk_index=chunk.chunk_index,
                content_hash=chunk.content_hash,
                text=chunk.text,
                embedding=embedding,
            )
            for chunk, embedding in zip(batch, embeddings, strict=True)
        ]
        upsert_chunks(rows)
        written += len(rows)
        logger.info("Upserted %d/%d chunk(s).", written, len(new_chunks))

    if pending:
        keep_hashes = {chunk.content_hash for chunk in pending}
        deleted = delete_chunks_not_in(keep_hashes)
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
