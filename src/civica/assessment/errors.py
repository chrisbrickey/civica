"""Errors shared by the question generation engine and the critic."""

from collections.abc import Sequence

from civica.domain.themes import Theme


class QuestionGenerationError(Exception):
    """Raised when questions cannot be generated, so nothing is displayed or persisted."""


class InsufficientMaterialError(QuestionGenerationError):
    """Raised when retrieval yields no testable passages for a theme.

    Stores the queries that yielded no testable passages."""

    def __init__(self, theme: Theme, queries: Sequence[str]) -> None:
        super().__init__(f"No testable corpus passages retrieved for theme {theme.slug!r}")
        self.theme = theme
        self.queries = tuple(queries)
