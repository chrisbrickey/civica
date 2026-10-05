"""Errors shared by the question generation engine and the critic."""


class QuestionGenerationError(Exception):
    """Raised when questions cannot be generated, so nothing is shown or saved."""
