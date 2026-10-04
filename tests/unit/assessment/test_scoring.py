"""Unit tests for civica.assessment.scoring: Quiz and MockExam scoring and answer logging.

No network calls. No DB spinup."""

import uuid

import pytest

from civica.assessment.scoring import (
    MOCK_EXAM_TIME_LIMIT_SECONDS,
    PASS_MARK,
    MockExam,
    MockExamResult,
    Quiz,
    QuizResult,
)
from civica.domain.question import Question
from civica.domain.themes import EXAM_QUESTION_COUNTS, HISTOIRE_GEOGRAPHIE_ET_CULTURE
from civica.domain.user import UserId
from tests.support.assessment import (
    LoggedAnswer,
    RecordingAnswerLogger,
    answers_with_correct_count,
    make_mock_exam_questions,
    make_question,
    wrong_index,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_ID = UserId(uuid.UUID(int=42))
_QUIZ_THEME = HISTOIRE_GEOGRAPHIE_ET_CULTURE
_EXAM_SIZE = 40
_EXPECTED_PASS_MARK = 32
_EXPECTED_TIME_LIMIT_SECONDS = 2700
_JUST_FAILING_SCORE = 31
_JUST_PASSING_SCORE = 32
_PERFECT_SCORE = 40

# Quiz answers: right, wrong, right.
_QUIZ_CORRECTNESS = (True, False, True)
_QUIZ_CORRECT_TOTAL = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def quiz_questions() -> tuple[Question, ...]:
    return tuple(
        make_question(text=f"sample-quiz-question-{i:03d}", theme=_QUIZ_THEME, correct_index=i)
        for i in range(len(_QUIZ_CORRECTNESS))
    )


@pytest.fixture()
def quiz_answers(quiz_questions: tuple[Question, ...]) -> list[int]:
    return [
        q.correct_index if correct else wrong_index(q)
        for q, correct in zip(quiz_questions, _QUIZ_CORRECTNESS, strict=True)
    ]


@pytest.fixture()
def quiz(quiz_questions: tuple[Question, ...], answer_logger: RecordingAnswerLogger) -> Quiz:
    return Quiz(user_id=_USER_ID, questions=quiz_questions, answer_logger=answer_logger)


@pytest.fixture()
def exam_questions() -> tuple[Question, ...]:
    return make_mock_exam_questions()


@pytest.fixture()
def mock_exam(
    exam_questions: tuple[Question, ...], answer_logger: RecordingAnswerLogger
) -> MockExam:
    return MockExam(user_id=_USER_ID, questions=exam_questions, answer_logger=answer_logger)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExamConstants:
    """The official exam rules: pass at 32/40, 45 minutes."""

    def test_pass_mark_is_thirty_two(self) -> None:
        assert PASS_MARK == _EXPECTED_PASS_MARK

    def test_time_limit_is_forty_five_minutes(self) -> None:
        assert MOCK_EXAM_TIME_LIMIT_SECONDS == _EXPECTED_TIME_LIMIT_SECONDS

    def test_mock_exam_carries_the_time_limit(self, mock_exam: MockExam) -> None:
        assert mock_exam.time_limit_seconds == _EXPECTED_TIME_LIMIT_SECONDS


class TestQuizScore:
    """Quiz.score counts correct answers overall and per theme."""

    def test_counts_correct_answers(self, quiz: Quiz, quiz_answers: list[int]) -> None:
        result = quiz.score(quiz_answers)

        assert isinstance(result, QuizResult)
        assert result.total_correct == _QUIZ_CORRECT_TOTAL
        assert result.per_theme_scores[_QUIZ_THEME] == _QUIZ_CORRECT_TOTAL


class TestQuizAnswerLogging:
    """Every quiz answer is logged once, with the facts needed for a quiz_answers row."""

    def test_logs_one_entry_per_answer_with_full_detail(
        self,
        quiz: Quiz,
        quiz_questions: tuple[Question, ...],
        quiz_answers: list[int],
        answer_logger: RecordingAnswerLogger,
    ) -> None:
        quiz.score(quiz_answers)

        expected = [
            LoggedAnswer(
                user_id=_USER_ID,
                theme=q.theme,
                question_id=q.question_id,
                chosen_index=chosen,
                correct_index=q.correct_index,
                is_correct=correct,
            )
            for q, chosen, correct in zip(
                quiz_questions, quiz_answers, _QUIZ_CORRECTNESS, strict=True
            )
        ]
        assert answer_logger.calls == expected


class TestMockExamPassMark:
    """passed flips exactly at the 31/32 boundary."""

    @pytest.mark.parametrize(
        ("correct_count", "expected_passed"),
        [
            (_JUST_FAILING_SCORE, False),
            (_JUST_PASSING_SCORE, True),
            (_PERFECT_SCORE, True),
        ],
    )
    def test_passed_reflects_pass_mark(
        self,
        mock_exam: MockExam,
        exam_questions: tuple[Question, ...],
        correct_count: int,
        expected_passed: bool,
    ) -> None:
        result = mock_exam.score(answers_with_correct_count(exam_questions, correct_count))

        assert isinstance(result, MockExamResult)
        assert result.total_correct == correct_count
        assert result.passed is expected_passed


class TestMockExamPerThemeScores:
    """per_theme_scores reports correct answers for each of the five themes."""

    def test_perfect_exam_scores_full_count_per_theme(
        self, mock_exam: MockExam, exam_questions: tuple[Question, ...]
    ) -> None:
        result = mock_exam.score(answers_with_correct_count(exam_questions, _PERFECT_SCORE))

        assert result.per_theme_scores == EXAM_QUESTION_COUNTS

    def test_all_wrong_exam_scores_zero_for_every_theme(
        self, mock_exam: MockExam, exam_questions: tuple[Question, ...]
    ) -> None:
        result = mock_exam.score(answers_with_correct_count(exam_questions, 0))

        assert result.total_correct == 0
        assert result.passed is False
        assert result.per_theme_scores == {theme: 0 for theme in EXAM_QUESTION_COUNTS}


class TestMockExamAnswerLogging:
    """All 40 mock exam answers are logged: the densest mastery signal must not be dropped."""

    def test_logs_forty_entries_matching_each_answer(
        self,
        mock_exam: MockExam,
        exam_questions: tuple[Question, ...],
        answer_logger: RecordingAnswerLogger,
    ) -> None:
        answers = answers_with_correct_count(exam_questions, _JUST_PASSING_SCORE)

        mock_exam.score(answers)

        assert len(answer_logger.calls) == _EXAM_SIZE
        for call, question, chosen in zip(answer_logger.calls, exam_questions, answers, strict=True):
            assert call.user_id == _USER_ID
            assert call.theme == question.theme
            assert call.question_id == question.question_id
            assert call.chosen_index == chosen
            assert call.correct_index == question.correct_index
            assert call.is_correct is (chosen == question.correct_index)
        assert sum(call.is_correct for call in answer_logger.calls) == _JUST_PASSING_SCORE


class TestAnswerCountMismatch:
    """A partial or oversized answer sheet is rejected before anything is logged."""

    @pytest.mark.parametrize("delta", [-1, 1])
    def test_quiz_rejects_wrong_length_and_logs_nothing(
        self,
        quiz: Quiz,
        quiz_answers: list[int],
        answer_logger: RecordingAnswerLogger,
        delta: int,
    ) -> None:
        answers = quiz_answers[:delta] if delta < 0 else [*quiz_answers, 0]

        with pytest.raises(ValueError):
            quiz.score(answers)

        assert answer_logger.calls == []

    @pytest.mark.parametrize("delta", [-1, 1])
    def test_mock_exam_rejects_wrong_length_and_logs_nothing(
        self,
        mock_exam: MockExam,
        exam_questions: tuple[Question, ...],
        answer_logger: RecordingAnswerLogger,
        delta: int,
    ) -> None:
        full = answers_with_correct_count(exam_questions, _PERFECT_SCORE)
        answers = full[:delta] if delta < 0 else [*full, 0]

        with pytest.raises(ValueError):
            mock_exam.score(answers)

        assert answer_logger.calls == []
