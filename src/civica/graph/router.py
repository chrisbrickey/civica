"""Deterministic review-theme heuristic the planner may call as a tool."""

from langgraph.store.base import BaseStore

from civica.domain.themes import EXAM_QUESTION_COUNTS, Theme
from civica.domain.user import UserId
from civica.memory.records import MistakeEpisode, TopicMastery
from civica.memory.writer import list_records


def suggest_review_theme(user_id: UserId, *, store: BaseStore | None = None) -> Theme:
    """Pick the theme with the highest weight x (1 - mastery), then mistakes, then exam order."""
    total = sum(EXAM_QUESTION_COUNTS.values())
    mistake_slugs = {
        episode.theme.slug for episode in list_records(user_id, MistakeEpisode, store)
    }

    # One list read instead of a request per theme
    recorded = {record.theme: record.mastery for record in list_records(user_id, TopicMastery, store)}

    def key(theme: Theme) -> tuple[float, bool]:
        mastery = recorded.get(theme, 0.0)
        weight = EXAM_QUESTION_COUNTS[theme] / total
        return weight * (1 - mastery), theme.slug in mistake_slugs

    return max(EXAM_QUESTION_COUNTS, key=key)
