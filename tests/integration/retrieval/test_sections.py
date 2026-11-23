"""Integration tests for civica.retrieval.sections against a real, isolated Postgres schema."""

from collections.abc import Callable

import psycopg
import psycopg.rows

from civica.domain.source_ref import SourceRef
from civica.domain.themes import DROITS_ET_DEVOIRS, HISTOIRE_GEOGRAPHIE_ET_CULTURE, Theme
from civica.embeddings.embedder import EMBEDDING_DIMENSIONS
from civica.ingestion.repository import EmbeddedChunk, upsert_chunks
from civica.retrieval.sections import OBJECTIVE_SECTION_IDS, theme_sections

_THEME = DROITS_ET_DEVOIRS
_OTHER_THEME = HISTOIRE_GEOGRAPHIE_ET_CULTURE
_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
_PAGE = "sample-page"
_SECTION_A = "section-a"
_SECTION_B = "section-b"
_SHORT_TEXT = "sample-text"
_LONG_TEXT = "sample-longer-text"


def test_returns_each_theme_section_with_its_content_size(
    make_chunk_row: Callable[..., EmbeddedChunk],
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> None:
    def chunk(theme: Theme, section_id: str, text: str, index: int) -> EmbeddedChunk:
        return make_chunk_row(
            theme=theme, embedding=_VECTOR, content_hash=f"hash-{section_id}-{index}-{theme}",
            text=text, page_slug=_PAGE, section_id=section_id, chunk_index=index,
        )

    objective_id = sorted(OBJECTIVE_SECTION_IDS)[0]
    upsert_chunks(
        [
            chunk(_THEME, _SECTION_A, _SHORT_TEXT, 0),
            chunk(_THEME, _SECTION_A, _LONG_TEXT, 1),
            chunk(_THEME, _SECTION_B, _SHORT_TEXT, 0),
            chunk(_THEME, objective_id, _LONG_TEXT, 0),
            chunk(_OTHER_THEME, _SECTION_A, _LONG_TEXT, 0),
        ],
        conn=db_schema,
    )

    assert theme_sections(_THEME, conn=db_schema) == {
        SourceRef(page_slug=_PAGE, section_id=_SECTION_A): len(_SHORT_TEXT) + len(_LONG_TEXT),
        SourceRef(page_slug=_PAGE, section_id=_SECTION_B): len(_SHORT_TEXT),
    }


def test_a_theme_with_no_chunks_has_no_sections(db_schema: psycopg.Connection[psycopg.rows.TupleRow]) -> None:
    assert theme_sections(_THEME, conn=db_schema) == {}
