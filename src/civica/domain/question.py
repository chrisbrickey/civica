"""Question domain model: one multiple-choice exam question with its grounding sources."""

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from civica.domain.source_ref import SourceRef
from civica.domain.themes import Theme


class QuestionKind(StrEnum):
    """The two question styles on the official exam."""

    KNOWLEDGE = "knowledge"
    SCENARIO = "scenario"


class Question(BaseModel):  # type: ignore[explicit-any]
    """An immutable multiple choice question (MCQ) with exactly four options and one correct answer.
    This is the same as the official exam format."""

    model_config = ConfigDict(frozen=True)

    theme: Theme
    kind: QuestionKind
    text: str = Field(min_length=1)
    options: tuple[str, str, str, str]
    correct_index: int = Field(ge=0, le=3)
    sources: tuple[SourceRef, ...]

    @property
    def question_id(self) -> str:
        """Content hash of the question text: the bank key and the quiz_answers key."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()
