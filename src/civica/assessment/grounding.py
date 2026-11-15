"""How well a generated quiz was grounded in retrieved passages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionSimilarity:
    """The highest and lowest chunk similarity behind one question."""

    question_id: str
    highest: float
    lowest: float
    chunks: int | None = None  # optional for backwards compatability


@dataclass(frozen=True)
class QuizGrounding:
    chunks_retrieved: int
    max_chunks: int
    minimum_similarity: float
    questions: tuple[QuestionSimilarity, ...]

    def __post_init__(self) -> None:
        # The checkpoint serializer restores tuples as lists; keep the field a tuple.
        object.__setattr__(self, "questions", tuple(self.questions))
