"""The corpus sections of a theme, each with its content size."""

from typing import Final, Protocol

import psycopg
import psycopg.rows
from pydantic import BaseModel

from civica.db.session import run_on_connection
from civica.domain.source_ref import SourceRef
from civica.domain.themes import Theme

# Course-objective intros carry no civic facts to test, so they are not part of what a learner masters.
OBJECTIVE_SECTION_IDS: Final = frozenset({"objectif-de-la-fiche", "objectifs-de-la-fiche"})

_THEME_SECTIONS_SQL = """
SELECT page_slug, section_id, SUM(LENGTH(text)) AS size
FROM content_chunks
WHERE theme = %s AND NOT (section_id = ANY(%s))
GROUP BY page_slug, section_id
"""


class _SectionRow(BaseModel):  # type: ignore[explicit-any]
    page_slug: str
    section_id: str
    size: int


class ThemeSections(Protocol):
    """Reads a theme's sections and their sizes."""

    def __call__(self, theme: Theme) -> dict[SourceRef, int]: ...


def _theme_sections_on_connection(
    theme: Theme, conn: psycopg.Connection[psycopg.rows.TupleRow]
) -> dict[SourceRef, int]:
    with conn.cursor(row_factory=psycopg.rows.class_row(_SectionRow)) as cursor:
        cursor.execute(_THEME_SECTIONS_SQL, (theme.slug, list(OBJECTIVE_SECTION_IDS)))
        rows = cursor.fetchall()
    return {SourceRef(page_slug=row.page_slug, section_id=row.section_id): row.size for row in rows}


def theme_sections(
    theme: Theme, conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None
) -> dict[SourceRef, int]:
    """Each section of the theme, minus course objectives, with its content size in characters."""
    return run_on_connection(lambda connection: _theme_sections_on_connection(theme, connection), conn)
