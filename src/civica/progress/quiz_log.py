"""Append-only quiz-answer log, kept in plain SQL outside LangGraph.

Every quiz answer, right or wrong, is recorded as one `quiz_answers` row in the database.
This module manages that logging, which fascilitates scoring and mistake-driven review.

The quiz_log table holds raw per-answer stats. This is independent from
(but related to) a mistake_episode store record, which captures the corpus
source context of a wrong answer for re-teaching.
"""

from datetime import datetime

import psycopg
import psycopg.rows
from pydantic import BaseModel, ConfigDict, Field, field_validator

from civica.db.session import run_on_connection
from civica.domain.source_ref import SourceRef
from civica.domain.themes import Theme
from civica.domain.user import UserId

_INSERT_SQL = """
INSERT INTO quiz_answers (
    user_id, theme, question_id, chosen_index, correct_index, is_correct
)
VALUES (%s, %s, %s, %s, %s, %s)
"""

# `id DESC` is a deterministic tiebreak:
#   - answered_at defaults to NOW() - transaction start time - so answers persisted together could share a timestamp
#   - the monotonic BIGSERIAL id persists an order of when the user selected an answer
_RECENT_MISTAKES_SQL = """
SELECT id, user_id, theme, question_id, chosen_index, correct_index, is_correct, answered_at
FROM quiz_answers
WHERE user_id = %s AND is_correct = false
ORDER BY answered_at DESC, id DESC
LIMIT %s
"""

# An answer whose question is not in generated_questions has no sources, so it is excluded.
_THEME_ANSWERS_SQL = """
SELECT a.is_correct, q.sources
FROM quiz_answers a
JOIN generated_questions q ON q.question_id = a.question_id
WHERE a.user_id = %s AND a.theme = %s
ORDER BY a.answered_at, a.id
"""


class QuizAnswerRow(BaseModel):  # type: ignore[explicit-any]
    """One `quiz_answers` row, with `theme` rehydrated to a `Theme`."""

    model_config = ConfigDict(frozen=True)

    id: int
    user_id: UserId
    theme: Theme
    question_id: str
    chosen_index: int = Field(ge=0)
    correct_index: int = Field(ge=0)
    is_correct: bool
    answered_at: datetime

    @field_validator("theme", mode="before")
    @classmethod
    def _validate_theme(cls, value: object) -> object:
        if isinstance(value, str):
            return Theme.from_slug(value)
        return value


class SourcedAnswer(BaseModel):  # type: ignore[explicit-any]
    """Whether one answer was correct, with the corpus sections its question cites."""

    model_config = ConfigDict(frozen=True)

    is_correct: bool
    sources: list[SourceRef]


def _log_answer_on_connection(
    user_id: UserId,
    theme: Theme,
    question_id: str,
    chosen_index: int,
    correct_index: int,
    is_correct: bool,
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> None:
    conn.execute(
        _INSERT_SQL,
        (user_id, theme.slug, question_id, chosen_index, correct_index, is_correct),
    )


def log_answer(
    user_id: UserId,
    theme: Theme,
    question_id: str,
    chosen_index: int,
    correct_index: int,
    is_correct: bool,
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
) -> None:
    """Append one quiz answer to the log, writing `theme.slug` into `theme`.

    Raises psycopg.errors.ForeignKeyViolation when `user_id` is not a
    registered user, since `quiz_answers.user_id` references `users(user_id)`.
    """
    run_on_connection(
        lambda connection: _log_answer_on_connection(
            user_id,
            theme,
            question_id,
            chosen_index,
            correct_index,
            is_correct,
            connection,
        ),
        conn,
    )


def _recent_mistakes_on_connection(
    user_id: UserId,
    limit: int,
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> list[QuizAnswerRow]:
    with conn.cursor(row_factory=psycopg.rows.class_row(QuizAnswerRow)) as cursor:
        cursor.execute(_RECENT_MISTAKES_SQL, (user_id, limit))
        return cursor.fetchall()


def recent_mistakes(
    user_id: UserId,
    limit: int = 20,
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
) -> list[QuizAnswerRow]:
    """Return the learner's most recent incorrect answers, newest first.

    Filters to `is_correct = false` and orders by recency, capped at `limit`.
    Scoped to the given `user_id`.
    """
    return run_on_connection(
        lambda connection: _recent_mistakes_on_connection(user_id, limit, connection),
        conn,
    )


def _theme_answers_on_connection(
    user_id: UserId,
    theme: Theme,
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> list[SourcedAnswer]:
    with conn.cursor(row_factory=psycopg.rows.class_row(SourcedAnswer)) as cursor:
        cursor.execute(_THEME_ANSWERS_SQL, (user_id, theme.slug))
        return cursor.fetchall()


def theme_answers(
    user_id: UserId,
    theme: Theme,
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
) -> list[SourcedAnswer]:
    """Return the learner's answers on one theme, oldest first, each with its question's sources."""
    return run_on_connection(
        lambda connection: _theme_answers_on_connection(user_id, theme, connection),
        conn,
    )
