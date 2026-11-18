"""Names of the tools the planner model may call."""

from enum import StrEnum


class PlannerTool(StrEnum):
    """The name of each tool the planner model may call."""

    SUGGEST_REVIEW_THEME = "suggest_review_theme"
    READ_LEARNER_STATE = "read_learner_state"
    RETRIEVE_CORPUS = "retrieve_corpus"
    TEACH = "teach"
    GENERATE_QUIZ = "generate_quiz"
    RECORD_OUTCOME = "record_outcome"
