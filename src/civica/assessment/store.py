"""Persisted question bank: the generated_questions table, keyed by question content hash."""

from collections.abc import Sequence
from typing import Protocol

import psycopg
import psycopg.rows
from psycopg.types.json import Jsonb

from civica.db.session import run_on_connection
from civica.domain.question import Question

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
