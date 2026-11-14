"""Unit tests for civica.memory.checkpointer's state serializer."""

from civica.assessment.grounding import QuestionSimilarity, QuizGrounding
from civica.memory.checkpointer import state_serializer

_GROUNDING = QuizGrounding(
    chunks_retrieved=42,
    max_chunks=80,
    minimum_similarity=0.25,
    questions=(
        QuestionSimilarity(question_id="sample-question-id-001", highest=0.75, lowest=0.5, chunks=2),
        # Recorded before chunk counts existed.
        QuestionSimilarity(question_id="sample-question-id-002", highest=0.5, lowest=0.5),
    ),
)


def test_quiz_grounding_survives_a_round_trip_through_the_state_serializer() -> None:
    serializer = state_serializer()

    assert serializer.loads_typed(serializer.dumps_typed(_GROUNDING)) == _GROUNDING
