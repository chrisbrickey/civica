"""Quiz and mock exam scoring. All scored answers are logged through one shared path."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from civica.domain.question import Question
from civica.domain.themes import Theme
from civica.domain.user import UserId

PASS_MARK = 32
MOCK_EXAM_TIME_LIMIT_SECONDS = 45 * 60


class AnswerLogger(Protocol):
    """Records one scored answer (production: progress.quiz_log.log_answer)."""

    def __call__(
        self,
        user_id: UserId,
        theme: Theme,
        question_id: str,
        chosen_index: int,
        correct_index: int,
        is_correct: bool,
    ) -> None: ...


@dataclass(frozen=True)
class QuizResult:
    total_correct: int
    per_theme_scores: dict[Theme, int]


@dataclass(frozen=True)
class MockExamResult(QuizResult):
    @property
    def passed(self) -> bool:
        return self.total_correct >= PASS_MARK


def _score_and_log(
    user_id: UserId,
    questions: Sequence[Question],
    answers: Sequence[int],
    answer_logger: AnswerLogger,
) -> tuple[int, dict[Theme, int]]:
    """Validate the answer count, then log and tally every answer. The only logging path."""
    if len(answers) != len(questions):
        raise ValueError(f"Expected {len(questions)} answers, got {len(answers)}")

    per_theme = {question.theme: 0 for question in questions}
    for question, chosen in zip(questions, answers, strict=True):
        is_correct = chosen == question.correct_index
        answer_logger(
            user_id, question.theme, question.question_id, chosen, question.correct_index, is_correct
        )
        if is_correct:
            per_theme[question.theme] += 1
    return sum(per_theme.values()), per_theme


@dataclass(frozen=True)
class Quiz:
    user_id: UserId
    questions: tuple[Question, ...]
    answer_logger: AnswerLogger

    def score(self, answers: Sequence[int]) -> QuizResult:
        total, per_theme = _score_and_log(self.user_id, self.questions, answers, self.answer_logger)
        return QuizResult(total_correct=total, per_theme_scores=per_theme)


@dataclass(frozen=True)
class MockExam:
    user_id: UserId
    questions: tuple[Question, ...]
    answer_logger: AnswerLogger
    time_limit_seconds: int = MOCK_EXAM_TIME_LIMIT_SECONDS

    def score(self, answers: Sequence[int]) -> MockExamResult:
        total, per_theme = _score_and_log(self.user_id, self.questions, answers, self.answer_logger)
        return MockExamResult(total_correct=total, per_theme_scores=per_theme)
