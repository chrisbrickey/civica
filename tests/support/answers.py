"""Expected replies and inputs of answer turns, shared by the graph and coach page tests."""

from typing import Final

from civica.domain.themes import Theme
from civica.graph.graph import InputKind

EXPIRED_REPLY: Final = "That question has expired."
ALREADY_ANSWERED_REPLY: Final = "You have already answered that question."
UNKNOWN_QUESTION_ID: Final = "sample-unknown-question-id"
VIA_ANSWER: Final = "answer"


def answer_message(number: int) -> str:
    """The display message of an answer turn for the 1-based question `number`."""
    return f"Answered question {number}"


def answer_recap(theme: Theme, is_correct: bool) -> str:
    """The session summary memory_writer saves after an answer turn."""
    return f"Answered 1 on {theme.slug}: {'right' if is_correct else 'wrong'}."


def answer_input(question_id: str, chosen_index: int, number: int = 1) -> dict[str, object]:
    """A graph input answering one checkpointed question."""
    return {
        "kind": InputKind.ANSWER,
        "learner_message": answer_message(number),
        "question_id": question_id,
        "chosen_index": chosen_index,
    }
