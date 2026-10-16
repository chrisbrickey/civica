"""Unit tests for civica.graph.router.suggest_review_theme.

priority(theme) = weight(theme) x (1 - mastery(theme)), weight from the official
exam distribution. Ties go to a theme with a mistake, then to official order.
Memory is seeded into an injected in-memory store; no database.
"""

from langgraph.store.memory import InMemoryStore

from civica.domain.themes import (
    DROITS_ET_DEVOIRS,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE,
    PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE,
    VIVRE_DANS_LA_SOCIETE_FRANCAISE,
    Theme,
)
from civica.graph.router import suggest_review_theme
from tests.support.memory import (
    FULL_MASTERY,
    HALF_MASTERY,
    WEIGHTED_CHOICE,
    WEIGHTED_MASTERY,
    seed_mastery,
    seed_mistake,
)
from tests.support.users import OTHER_USER_ID, USER_ID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PRINCIPES and DROITS tie at equal mastery; every other theme is mastered.
TIED_PAIR_MASTERY: dict[Theme, float] = {
    PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE: HALF_MASTERY,
    DROITS_ET_DEVOIRS: HALF_MASTERY,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE: FULL_MASTERY,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE: FULL_MASTERY,
    VIVRE_DANS_LA_SOCIETE_FRANCAISE: FULL_MASTERY,
}


# ---------------------------------------------------------------------------
# Priority = weight x (1 - mastery)
# ---------------------------------------------------------------------------


def test_returns_highest_priority_theme_from_weight_and_mastery(
    store: InMemoryStore,
) -> None:
    seed_mastery(store, USER_ID, WEIGHTED_MASTERY)

    assert suggest_review_theme(USER_ID, store=store) == WEIGHTED_CHOICE


def test_new_user_with_empty_store_gets_highest_weight_theme(
    store: InMemoryStore,
) -> None:
    assert (
        suggest_review_theme(USER_ID, store=store)
        == PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE
    )


def test_full_mastery_drops_theme_to_zero_priority(store: InMemoryStore) -> None:
    seed_mastery(
        store, USER_ID, {PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE: FULL_MASTERY}
    )

    assert suggest_review_theme(USER_ID, store=store) == DROITS_ET_DEVOIRS


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------


def test_tied_theme_with_mistake_episode_wins(store: InMemoryStore) -> None:
    seed_mistake(store, USER_ID, DROITS_ET_DEVOIRS)

    assert suggest_review_theme(USER_ID, store=store) == DROITS_ET_DEVOIRS


def test_tied_theme_with_mistake_wins_at_partial_mastery(
    store: InMemoryStore,
) -> None:
    seed_mastery(store, USER_ID, TIED_PAIR_MASTERY)
    seed_mistake(store, USER_ID, DROITS_ET_DEVOIRS)

    assert suggest_review_theme(USER_ID, store=store) == DROITS_ET_DEVOIRS


def test_mistake_does_not_outrank_a_higher_priority_theme(
    store: InMemoryStore,
) -> None:
    seed_mistake(store, USER_ID, VIVRE_DANS_LA_SOCIETE_FRANCAISE)

    assert (
        suggest_review_theme(USER_ID, store=store)
        == PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE
    )


def test_full_tie_without_mistakes_returns_earliest_official_theme(
    store: InMemoryStore,
) -> None:
    seed_mastery(store, USER_ID, TIED_PAIR_MASTERY)

    assert (
        suggest_review_theme(USER_ID, store=store)
        == PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE
    )


# ---------------------------------------------------------------------------
# Per-user isolation
# ---------------------------------------------------------------------------


def test_other_users_memory_does_not_affect_result(store: InMemoryStore) -> None:
    seed_mastery(
        store, OTHER_USER_ID, {PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE: FULL_MASTERY}
    )
    seed_mistake(store, OTHER_USER_ID, DROITS_ET_DEVOIRS)

    assert (
        suggest_review_theme(USER_ID, store=store)
        == PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE
    )
