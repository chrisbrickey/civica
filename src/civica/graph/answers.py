"""Answer scoring and mastery update for one submitted choice."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from civica.assessment.scoring import Quiz
from civica.domain.question import Question
from civica.domain.themes import Theme
from civica.domain.user import UserId
from civica.graph.mastery import mastery_from_answers
from civica.graph.planner import PlannerDeps
from civica.memory.records import MistakeEpisode, TopicMastery
from civica.memory.writer import put

EXPIRED_REPLY: Final = "That question has expired."
ALREADY_ANSWERED_REPLY: Final = "You have already answered that question."


@dataclass(frozen=True)
class AnswerResult:
    """The scored outcome of one answered question."""

    question_id: str
    theme: Theme
    chosen_index: int
    is_correct: bool


def answer_recap(result: AnswerResult) -> str:
    """The session summary an answer turn saves."""
    return f"Answered 1 on {result.theme.slug}: {'right' if result.is_correct else 'wrong'}."


def find_question(questions: Sequence[Question], question_id: str) -> Question | None:
    """The checkpointed question with this id, or None when it is not in the current quiz."""
    return next((question for question in questions if question.question_id == question_id), None)


def score_choice(deps: PlannerDeps, user_id: UserId, question: Question, chosen_index: int) -> AnswerResult:
    """Score and log one choice through a one-question Quiz; a wrong choice also records a mistake."""
    # Quiz.score wants an answer for every question, so a one-question quiz scores just this one.
    outcome = Quiz(user_id, (question,), deps.answer_logger).score([chosen_index])
    is_correct = outcome.total_correct == 1
    if not is_correct:
        key = f"{question.question_id}-{datetime.now(UTC).isoformat()}"
        episode = MistakeEpisode(theme=question.theme, question_id=question.question_id, sources=list(question.sources))
        put(user_id, key, episode, deps.store)
    return AnswerResult(question.question_id, question.theme, chosen_index, is_correct)


def record_mastery(deps: PlannerDeps, user_id: UserId, theme: Theme) -> None:
    """Recompute the theme's mastery from the answer log and save it; no answers means nothing to save."""
    mastery = mastery_from_answers(deps.theme_answers(user_id, theme), deps.theme_sections(theme))
    if mastery is not None:
        put(user_id, theme.slug, TopicMastery(theme=theme, mastery=mastery), deps.store)
