"""Integration tests for civica.assessment.store: the question_rejections log in Postgres.

Exercises the real schema via the db_schema fixture. No network calls.
"""

import psycopg
import psycopg.rows
import pytest

from civica.assessment.store import PostgresRejectionStore, Rejection
from civica.domain.themes import DROITS_ET_DEVOIRS, VIVRE_DANS_LA_SOCIETE_FRANCAISE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPT_VERSION = "sample-prompt-v1"

_ROWS_SQL = """
SELECT prompt_version, theme, reason, attempt, rejected_at
FROM question_rejections
ORDER BY id
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> PostgresRejectionStore:
    return PostgresRejectionStore()


@pytest.fixture()
def rejections() -> tuple[Rejection, ...]:
    return (
        Rejection(
            prompt_version=_PROMPT_VERSION,
            theme=DROITS_ET_DEVOIRS,
            reason="sample-reason-001",
            attempt=1,
        ),
        Rejection(
            prompt_version=_PROMPT_VERSION,
            theme=VIVRE_DANS_LA_SOCIETE_FRANCAISE,
            reason="sample-reason-002",
            attempt=2,
        ),
    )


def _rows(
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> list[dict[str, object]]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(_ROWS_SQL)
        return cursor.fetchall()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRejectionRows:
    """Each rejection becomes one timestamped row keyed by theme slug and prompt version."""

    def test_rows_match_rejections(
        self,
        store: PostgresRejectionStore,
        rejections: tuple[Rejection, ...],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        store.save(rejections, conn=db_schema)

        rows = _rows(db_schema)
        assert [
            (row["prompt_version"], row["theme"], row["reason"], row["attempt"]) for row in rows
        ] == [(r.prompt_version, r.theme.slug, r.reason, r.attempt) for r in rejections]
        assert all(row["rejected_at"] is not None for row in rows)

    def test_saving_twice_appends(
        self,
        store: PostgresRejectionStore,
        rejections: tuple[Rejection, ...],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        """The log is append-only: a repeated rejection is a second data point, not a duplicate."""
        store.save(rejections, conn=db_schema)
        store.save(rejections, conn=db_schema)

        assert len(_rows(db_schema)) == 2 * len(rejections)

    def test_empty_sequence_writes_nothing(
        self,
        store: PostgresRejectionStore,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        store.save([], conn=db_schema)

        assert _rows(db_schema) == []
