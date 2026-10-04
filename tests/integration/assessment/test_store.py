"""Integration tests for civica.assessment.store: the generated_questions bank in Postgres.

Exercises the real schema via the db_schema fixture. No network calls.
"""

import psycopg
import psycopg.rows
import pytest

from civica.assessment.store import PostgresQuestionStore
from civica.domain.question import Question, QuestionKind
from civica.domain.source_ref import SourceRef
from civica.domain.themes import VIVRE_DANS_LA_SOCIETE_FRANCAISE
from tests.support.assessment import SAMPLE_OPTIONS, make_question

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPT_VERSION = "sample-prompt-v1"
_NEXT_PROMPT_VERSION = "sample-prompt-v2"
_THEME = VIVRE_DANS_LA_SOCIETE_FRANCAISE
_CORRECT_INDEX = 3
_SOURCES = (
    SourceRef(page_slug="sample-page-001", section_id="section-001"),
    SourceRef(page_slug="sample-page-002", section_id="section-002"),
)

_ROWS_SQL = """
SELECT question_id, theme, kind, text, options, correct_index, sources, prompt_version
FROM generated_questions
ORDER BY question_id
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> PostgresQuestionStore:
    return PostgresQuestionStore()


@pytest.fixture()
def questions() -> tuple[Question, ...]:
    return (
        make_question(
            text="sample-stored-question-001",
            theme=_THEME,
            kind=QuestionKind.SCENARIO,
            correct_index=_CORRECT_INDEX,
            sources=_SOURCES,
        ),
        make_question(text="sample-stored-question-002", theme=_THEME),
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


class TestIdempotentSave:
    """Saving the same questions twice leaves exactly one row per question."""

    def test_second_save_does_not_duplicate_rows(
        self,
        store: PostgresQuestionStore,
        questions: tuple[Question, ...],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        store.save(questions, _PROMPT_VERSION, conn=db_schema)
        store.save(questions, _PROMPT_VERSION, conn=db_schema)

        rows = _rows(db_schema)
        assert sorted(str(row["question_id"]) for row in rows) == sorted(
            q.question_id for q in questions
        )


class TestPromptVersionStamp:
    """Each row carries the prompt version that generated it, updated on re-save."""

    def test_row_carries_prompt_version(
        self,
        store: PostgresQuestionStore,
        questions: tuple[Question, ...],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        store.save(questions, _PROMPT_VERSION, conn=db_schema)

        assert {row["prompt_version"] for row in _rows(db_schema)} == {_PROMPT_VERSION}

    def test_resave_with_new_prompt_version_updates_stamp(
        self,
        store: PostgresQuestionStore,
        questions: tuple[Question, ...],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        store.save(questions, _PROMPT_VERSION, conn=db_schema)
        store.save(questions, _NEXT_PROMPT_VERSION, conn=db_schema)

        rows = _rows(db_schema)
        assert len(rows) == len(questions)
        assert {row["prompt_version"] for row in rows} == {_NEXT_PROMPT_VERSION}


class TestFieldFidelity:
    """Every multiple choice question (MCQ) field round-trips through the table."""

    def test_stored_row_matches_question(
        self,
        store: PostgresQuestionStore,
        questions: tuple[Question, ...],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        question = questions[0]

        store.save([question], _PROMPT_VERSION, conn=db_schema)

        assert _rows(db_schema) == [
            {
                "question_id": question.question_id,
                "theme": _THEME.slug,
                "kind": QuestionKind.SCENARIO.value,
                "text": question.text,
                "options": list(SAMPLE_OPTIONS),
                "correct_index": _CORRECT_INDEX,
                "sources": [source.model_dump() for source in _SOURCES],
                "prompt_version": _PROMPT_VERSION,
            }
        ]
