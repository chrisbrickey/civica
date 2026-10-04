"""Shared builders and recording fakes for assessment tests (unit and integration)."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from civica.domain.question import Question, QuestionKind
from civica.domain.source_ref import SourceRef
from civica.domain.themes import DROITS_ET_DEVOIRS, EXAM_QUESTION_COUNTS, Theme
from civica.domain.user import UserId

SAMPLE_OPTIONS = ("option-a", "option-b", "option-c", "option-d")
SAMPLE_SOURCE = SourceRef(page_slug="sample-page", section_id="section-001")
OPTION_COUNT = len(SAMPLE_OPTIONS)


def make_question(
    *,
    text: str = "sample-question-text",
    theme: Theme = DROITS_ET_DEVOIRS,
    kind: QuestionKind = QuestionKind.KNOWLEDGE,
    correct_index: int = 0,
    sources: tuple[SourceRef, ...] = (SAMPLE_SOURCE,),
) -> Question:
    return Question(
        theme=theme,
        kind=kind,
        text=text,
        options=SAMPLE_OPTIONS,
        correct_index=correct_index,
        sources=sources,
    )


def make_mock_exam_questions() -> tuple[Question, ...]:
    """Forty distinct questions in the official per-theme counts, correct index cycling 0..3."""
    questions: list[Question] = []
    for theme, count in EXAM_QUESTION_COUNTS.items():
        for i in range(count):
            questions.append(
                make_question(
                    text=f"sample-question-{theme.slug}-{i:03d}",
                    theme=theme,
                    correct_index=len(questions) % OPTION_COUNT,
                )
            )
    return tuple(questions)


def wrong_index(question: Question) -> int:
    """An option index that is never the correct one."""
    return (question.correct_index + 1) % OPTION_COUNT


def answers_with_correct_count(questions: Sequence[Question], correct_count: int) -> list[int]:
    """Answer the first `correct_count` questions correctly and the rest wrongly."""
    return [
        q.correct_index if i < correct_count else wrong_index(q) for i, q in enumerate(questions)
    ]


@dataclass(frozen=True)
class LoggedAnswer:
    user_id: UserId
    theme: Theme
    question_id: str
    chosen_index: int
    correct_index: int
    is_correct: bool


@dataclass
class RecordingAnswerLogger:
    """Fake AnswerLogger: records every answer instead of writing quiz_answers rows."""

    calls: list[LoggedAnswer] = field(default_factory=list)

    def __call__(
        self,
        user_id: UserId,
        theme: Theme,
        question_id: str,
        chosen_index: int,
        correct_index: int,
        is_correct: bool,
    ) -> None:
        self.calls.append(
            LoggedAnswer(user_id, theme, question_id, chosen_index, correct_index, is_correct)
        )


@dataclass
class RecordingQuestionStore:
    """Fake QuestionStore: records each save call as (questions, prompt_version)."""

    saves: list[tuple[tuple[Question, ...], str]] = field(default_factory=list)

    def save(self, questions: Sequence[Question], prompt_version: str) -> None:
        self.saves.append((tuple(questions), prompt_version))

    @property
    def saved_questions(self) -> list[Question]:
        return [q for questions, _ in self.saves for q in questions]

    @property
    def saved_prompt_versions(self) -> set[str]:
        return {version for _, version in self.saves}
