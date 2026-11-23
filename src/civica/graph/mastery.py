"""Derived mastery: the content-weighted share of a theme's sections the learner has earned."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from civica.domain.source_ref import SourceRef
from civica.domain.themes import Theme
from civica.domain.user import UserId
from civica.progress.quiz_log import SourcedAnswer


class ThemeAnswers(Protocol):
    """Reads a learner's answers on a theme, oldest first."""

    def __call__(self, user_id: UserId, theme: Theme) -> Sequence[SourcedAnswer]: ...


def mastery_from_answers(answers: Sequence[SourcedAnswer], sections: Mapping[SourceRef, int]) -> float | None:
    """Size-weighted average of section credit, from answers in the order given; None with no answers or sections."""
    total_size = sum(sections.values())
    if not answers or total_size == 0:
        return None
    credit: dict[SourceRef, float] = {}
    for answer in answers:
        if not answer.sources:
            continue
        # An answer is worth 1.0 split across its sections; a wrong one takes that share away, kept within 0 to 1.
        share = 1 / len(answer.sources)
        change = share if answer.is_correct else -share
        for source in answer.sources:
            if source in sections:
                credit[source] = min(1.0, max(0.0, credit.get(source, 0.0) + change))
    return sum(sections[source] * earned for source, earned in credit.items()) / total_size
