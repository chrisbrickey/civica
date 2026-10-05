"""Persisted question bank (generated_questions) and critic rejection log (question_rejections)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import psycopg
import psycopg.rows
from psycopg.types.json import Jsonb

from civica.db.session import run_on_connection
from civica.domain.question import Question
from civica.domain.themes import Theme

_UPSERT_SQL = """
INSERT INTO generated_questions (
    question_id, theme, kind, text, options, correct_index, sources, prompt_version
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (question_id) DO UPDATE SET
    theme = EXCLUDED.theme,
    kind = EXCLUDED.kind,
    text = EXCLUDED.text,
    options = EXCLUDED.options,
    correct_index = EXCLUDED.correct_index,
    sources = EXCLUDED.sources,
    prompt_version = EXCLUDED.prompt_version
"""


class QuestionStore(Protocol):
    """Saves generated questions, stamped with the prompt version that produced them."""

    def save(self, questions: Sequence[Question], prompt_version: str) -> None: ...


def _save_on_connection(
    questions: Sequence[Question],
    prompt_version: str,
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> None:
    params = [
        (
            q.question_id,
            q.theme.slug,
            q.kind.value,
            q.text,
            Jsonb(list(q.options)),
            q.correct_index,
            Jsonb([source.model_dump() for source in q.sources]),
            prompt_version,
        )
        for q in questions
    ]
    if not params:
        return
    with conn.cursor() as cursor:
        cursor.executemany(_UPSERT_SQL, params)


class PostgresQuestionStore:
    """QuestionStore backed by the generated_questions table (idempotent upsert)."""

    def save(
        self,
        questions: Sequence[Question],
        prompt_version: str,
        conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
    ) -> None:
        run_on_connection(
            lambda connection: _save_on_connection(questions, prompt_version, connection), conn
        )


_INSERT_REJECTION_SQL = """
INSERT INTO question_rejections (prompt_version, theme, reason, attempt)
VALUES (%s, %s, %s, %s)
"""


@dataclass(frozen=True)
class Rejection:
    """One question the critic rejected, with the refine attempt (1 = first pass) that rejected it."""

    prompt_version: str
    theme: Theme
    reason: str
    attempt: int


class RejectionStore(Protocol):
    """Appends critic rejections to a log for prompt-version quality comparison."""

    def save(self, rejections: Sequence[Rejection]) -> None: ...


def _save_rejections_on_connection(
    rejections: Sequence[Rejection],
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> None:
    params = [(r.prompt_version, r.theme.slug, r.reason, r.attempt) for r in rejections]
    if not params:
        return
    with conn.cursor() as cursor:
        cursor.executemany(_INSERT_REJECTION_SQL, params)


class PostgresRejectionStore:
    """RejectionStore backed by the question_rejections table (append-only)."""

    def save(
        self,
        rejections: Sequence[Rejection],
        conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
    ) -> None:
        run_on_connection(lambda connection: _save_rejections_on_connection(rejections, connection), conn)
