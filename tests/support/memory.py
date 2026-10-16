"""Shared mastery values and seeding helpers for tests over the long-term memory store."""

from collections.abc import Mapping

from langgraph.store.base import BaseStore

from civica.domain.themes import (
    DROITS_ET_DEVOIRS,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE,
    PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE,
    VIVRE_DANS_LA_SOCIETE_FRANCAISE,
    Theme,
)
from civica.domain.user import UserId
from civica.memory.records import MistakeEpisode, TopicMastery
from civica.memory.writer import put

FULL_MASTERY = 1.0
HALF_MASTERY = 0.5

# Weights 11, 11, 8, 6, 4 (of 40). Priorities x40: 1.1, 2.2, 4.8, 3.0, 4.0.
# The review heuristic picks HISTOIRE, though it is neither the heaviest nor the least mastered theme.
WEIGHTED_MASTERY: dict[Theme, float] = {
    PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE: 0.9,
    DROITS_ET_DEVOIRS: 0.8,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE: 0.4,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE: 0.5,
    VIVRE_DANS_LA_SOCIETE_FRANCAISE: 0.0,
}
WEIGHTED_CHOICE = HISTOIRE_GEOGRAPHIE_ET_CULTURE


def seed_mastery(store: BaseStore, user_id: UserId, mastery: Mapping[Theme, float]) -> None:
    """Write one TopicMastery record per theme for the user."""
    for theme, value in mastery.items():
        put(user_id, theme.slug, TopicMastery(theme=theme, mastery=value), store)


def seed_mistake(store: BaseStore, user_id: UserId, theme: Theme) -> MistakeEpisode:
    """Write one MistakeEpisode for the theme under a key unique to that theme."""
    episode = MistakeEpisode(theme=theme, question_id=f"q-{theme.slug}")
    put(user_id, f"mistake-{episode.question_id}", episode, store)
    return episode
