"""Integration test: scoring a mock exam writes one quiz_answers row per answer.

Wires the real log_answer into MockExam against the db_schema connection.
quiz_answers.user_id has a foreign key to users, so a real user is registered first.

No network calls.
"""

import psycopg
import psycopg.rows
import pytest

from civica.assessment.scoring import MockExam
from civica.domain.question import Question
from civica.domain.themes import Theme
from civica.domain.user import UserId
from civica.progress.quiz_log import log_answer
from civica.users.service import register
from tests.support.assessment import answers_with_correct_count, make_mock_exam_questions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USERNAME = "test-user"
_SECRET = "sample-secret"
# bcrypt cost factor: lowered for tests, matching the users-service tests.
_TEST_BCRYPT_ROUNDS = 4

_EXAM_SIZE = 40
_CORRECT_COUNT = 31

_COUNT_SQL = "SELECT is_correct, COUNT(*) FROM quiz_answers WHERE user_id = %s GROUP BY is_correct"
_QUESTION_IDS_SQL = "SELECT question_id FROM quiz_answers WHERE user_id = %s"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def user_id(db_schema: psycopg.Connection[psycopg.rows.TupleRow]) -> UserId:
    return register(_USERNAME, _SECRET, conn=db_schema, rounds=_TEST_BCRYPT_ROUNDS)


@pytest.fixture()
def exam_questions() -> tuple[Question, ...]:
    return make_mock_exam_questions()


@pytest.fixture()
def mock_exam(
    user_id: UserId,
    exam_questions: tuple[Question, ...],
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> MockExam:
    def _log_to_test_schema(
        user_id: UserId,
        theme: Theme,
        question_id: str,
        chosen_index: int,
        correct_index: int,
        is_correct: bool,
    ) -> None:
        log_answer(
            user_id, theme, question_id, chosen_index, correct_index, is_correct, conn=db_schema
        )

    return MockExam(user_id=user_id, questions=exam_questions, answer_logger=_log_to_test_schema)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMockExamWritesQuizAnswers:
    """All 40 answers from a scored mock exam land in quiz_answers."""

    def test_forty_rows_with_matching_correctness(
        self,
        mock_exam: MockExam,
        exam_questions: tuple[Question, ...],
        user_id: UserId,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        result = mock_exam.score(answers_with_correct_count(exam_questions, _CORRECT_COUNT))

        counts: dict[bool, int] = dict(db_schema.execute(_COUNT_SQL, (user_id,)).fetchall())
        assert counts == {True: _CORRECT_COUNT, False: _EXAM_SIZE - _CORRECT_COUNT}
        assert result.total_correct == _CORRECT_COUNT
        assert result.passed is False

    def test_rows_reference_each_question_id(
        self,
        mock_exam: MockExam,
        exam_questions: tuple[Question, ...],
        user_id: UserId,
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        mock_exam.score(answers_with_correct_count(exam_questions, _CORRECT_COUNT))

        rows = db_schema.execute(_QUESTION_IDS_SQL, (user_id,)).fetchall()
        assert sorted(row[0] for row in rows) == sorted(q.question_id for q in exam_questions)
