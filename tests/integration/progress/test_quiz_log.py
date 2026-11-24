"""Integration tests for civica.progress.quiz_log.

Exercises the Postgres schema.
quiz_answers.user_id has a foreign key to users(user_id), so a real users row
is registered first and its UserId is reused for every write.

No network calls.
"""

import uuid
from collections.abc import Callable

import psycopg
import psycopg.rows
import pytest

from civica.assessment.store import PostgresQuestionStore
from civica.domain.question import Question
from civica.domain.source_ref import SourceRef
from civica.domain.themes import DROITS_ET_DEVOIRS, HISTOIRE_GEOGRAPHIE_ET_CULTURE, Theme
from civica.domain.user import UserId
from civica.progress.quiz_log import log_answer, recent_mistakes
from civica.users.service import register
from tests.support.assessment import make_question
from tests.support.users import TEST_BCRYPT_ROUNDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USERNAME = "test-user"
_SECRET = "sample-secret"

_THEME = DROITS_ET_DEVOIRS
_OTHER_THEME = HISTOIRE_GEOGRAPHIE_ET_CULTURE

# For an incorrect answer the chosen index differs from the correct index;
# for a correct answer the two coincide.
_CORRECT_INDEX = 1
_WRONG_INDEX = 2

_QUESTION_ID = "question-042"

# A five-answer sequence, logged in list order. Two correct answers are
# interleaved so recent_mistakes must both exclude them and respect the limit.
_ANSWER_SEQUENCE: list[tuple[str, bool]] = [
    ("question-001", False),
    ("question-002", True),
    ("question-003", False),
    ("question-004", False),
    ("question-005", False),
]
_MISTAKE_LIMIT = 3
# The three most recent incorrect answers, most recent first.
_EXPECTED_RECENT_MISTAKE_IDS = ["question-005", "question-004", "question-003"]
_PROMPT_VERSION = "sample-prompt-version"
_SOURCE_A = SourceRef(page_slug="sample-page", section_id="section-a")
_SOURCE_B = SourceRef(page_slug="sample-page", section_id="section-b")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def user_id(db_schema: psycopg.Connection[psycopg.rows.TupleRow]) -> UserId:
    """Register a real user and return its UserId for the FK-bound writes.

    Injects the lowered bcrypt cost factor to improve test velocity.
    """
    return register(_USERNAME, _SECRET, conn=db_schema, rounds=TEST_BCRYPT_ROUNDS)


@pytest.fixture()
def log_one(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> Callable[..., None]:
    """Log a single answer against the per-test schema connection."""

    def _log(
        *,
        user_id: UserId,
        theme: Theme = _THEME,
        question_id: str,
        is_correct: bool,
    ) -> None:
        log_answer(
            user_id=user_id,
            theme=theme,
            question_id=question_id,
            chosen_index=_CORRECT_INDEX if is_correct else _WRONG_INDEX,
            correct_index=_CORRECT_INDEX,
            is_correct=is_correct,
            conn=db_schema,
        )

    return _log


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecentMistakes:
    """recent_mistakes returns only wrong answers, newest first, up to the limit."""

    def test_returns_three_most_recent_incorrect_in_reverse_chronological_order(
        self,
        user_id: UserId,
        log_one: Callable[..., None],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        for question_id, is_correct in _ANSWER_SEQUENCE:
            log_one(user_id=user_id, question_id=question_id, is_correct=is_correct)

        rows = recent_mistakes(user_id, limit=_MISTAKE_LIMIT, conn=db_schema)

        assert [row.question_id for row in rows] == _EXPECTED_RECENT_MISTAKE_IDS
        assert all(row.is_correct is False for row in rows)

    def test_empty_list_for_fresh_user(
        self,
        user_id: UserId,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        assert recent_mistakes(user_id, conn=db_schema) == []


class TestThemeAnswers:
    """theme_answers returns one theme's answers with their question's sources, oldest first."""

    @staticmethod
    def _save(db_schema: psycopg.Connection[psycopg.rows.TupleRow], *questions: Question) -> None:
        PostgresQuestionStore().save(questions, _PROMPT_VERSION, conn=db_schema)

    def test_returns_the_themes_answers_oldest_first_with_their_sources(
        self,
        user_id: UserId,
        log_one: Callable[..., None],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        from civica.progress.quiz_log import SourcedAnswer, theme_answers

        first = make_question(text="sample-question-1", sources=(_SOURCE_A,))
        second = make_question(text="sample-question-2", sources=(_SOURCE_A, _SOURCE_B))
        other = make_question(text="sample-question-3", theme=_OTHER_THEME)
        self._save(db_schema, first, second, other)
        log_one(user_id=user_id, question_id=first.question_id, is_correct=True)
        log_one(user_id=user_id, theme=_OTHER_THEME, question_id=other.question_id, is_correct=True)
        log_one(user_id=user_id, question_id=second.question_id, is_correct=False)

        assert theme_answers(user_id, _THEME, conn=db_schema) == [
            SourcedAnswer(is_correct=True, sources=[_SOURCE_A]),
            SourcedAnswer(is_correct=False, sources=[_SOURCE_A, _SOURCE_B]),
        ]

    def test_empty_list_for_a_theme_with_no_answers(
        self,
        user_id: UserId,
        log_one: Callable[..., None],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        from civica.progress.quiz_log import theme_answers

        other = make_question(theme=_OTHER_THEME)
        self._save(db_schema, other)
        log_one(user_id=user_id, theme=_OTHER_THEME, question_id=other.question_id, is_correct=True)

        assert theme_answers(user_id, _THEME, conn=db_schema) == []


class TestFieldFidelity:
    """A returned QuizAnswerRow matches exactly what was logged."""

    def test_returned_row_matches_logged_fields(
        self,
        user_id: UserId,
        log_one: Callable[..., None],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        log_one(
            user_id=user_id,
            theme=_OTHER_THEME,
            question_id=_QUESTION_ID,
            is_correct=False,
        )

        rows = recent_mistakes(user_id, conn=db_schema)

        assert len(rows) == 1
        row = rows[0]
        # theme is written as a slug and rehydrated back to the same Theme.
        assert row.theme == _OTHER_THEME
        assert row.theme == Theme.from_slug(_OTHER_THEME.slug)
        assert row.question_id == _QUESTION_ID
        assert row.chosen_index == _WRONG_INDEX
        assert row.correct_index == _CORRECT_INDEX
        assert row.is_correct is False


class TestForeignKey:
    """Logging against a user that does not exist violates the FK constraint."""

    def test_unknown_user_raises_foreign_key_violation(
        self,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        unknown_user_id = UserId(uuid.uuid4())

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            log_answer(
                user_id=unknown_user_id,
                theme=_THEME,
                question_id=_QUESTION_ID,
                chosen_index=_WRONG_INDEX,
                correct_index=_CORRECT_INDEX,
                is_correct=False,
                conn=db_schema,
            )
