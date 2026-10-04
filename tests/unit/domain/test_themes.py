"""Unit tests for civica.domain.themes: Theme pydantic model and module constants."""

import pytest

from civica.domain.themes import (
    DROITS_ET_DEVOIRS,
    EXAM_QUESTION_COUNTS,
    EXAM_SCENARIO_COUNTS,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE,
    PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE,
    THEMES_BY_SLUG,
    VIVRE_DANS_LA_SOCIETE_FRANCAISE,
    Theme,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_SLUG = "droits-et-devoirs"
UNKNOWN_SLUG = "not-a-real-theme"

# Slug -> module constant for the five official themes; each slug string is
# defined here exactly once and every test derives from this mapping.
EXPECTED_CONSTANTS_BY_SLUG = {
    "principes-et-valeurs-de-la-republique": PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE,
    "droits-et-devoirs": DROITS_ET_DEVOIRS,
    "histoire-geographie-et-culture": HISTOIRE_GEOGRAPHIE_ET_CULTURE,
    "systeme-institutionnel-et-politique": SYSTEME_INSTITUTIONNEL_ET_POLITIQUE,
    "vivre-dans-la-societe-francaise": VIVRE_DANS_LA_SOCIETE_FRANCAISE,
}

# Official mock exam format: 40 questions, 12 of them scenario questions.
# These are written out explicitly for these tests so that it catches an accidental, breaking change to the production values.
EXAM_TOTAL_QUESTIONS = 40
EXAM_TOTAL_SCENARIOS = 12
EXPECTED_QUESTION_COUNTS = {
    PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE: 11,
    DROITS_ET_DEVOIRS: 11,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE: 8,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE: 6,
    VIVRE_DANS_LA_SOCIETE_FRANCAISE: 4,
}
EXPECTED_SCENARIO_COUNTS = {
    PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE: 3,
    DROITS_ET_DEVOIRS: 3,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE: 2,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE: 2,
    VIVRE_DANS_LA_SOCIETE_FRANCAISE: 2,
}

# ---------------------------------------------------------------------------
# Theme.from_slug - happy path
# ---------------------------------------------------------------------------


def test_from_slug_happy_path() -> None:
    assert Theme.from_slug(KNOWN_SLUG) is DROITS_ET_DEVOIRS


# ---------------------------------------------------------------------------
# Theme.from_slug - unknown slug
# ---------------------------------------------------------------------------


def test_from_slug_unknown_slug_raises_value_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        Theme.from_slug(UNKNOWN_SLUG)
    message = str(exc_info.value)
    assert UNKNOWN_SLUG in message
    for slug in THEMES_BY_SLUG:
        assert slug in message


# ---------------------------------------------------------------------------
# THEMES_BY_SLUG shape
# ---------------------------------------------------------------------------


def test_themes_by_slug_contains_exactly_the_five_official_slugs() -> None:
    assert set(THEMES_BY_SLUG) == set(EXPECTED_CONSTANTS_BY_SLUG)


def test_themes_by_slug_keys_match_value_slug_fields() -> None:
    for key, theme in THEMES_BY_SLUG.items():
        assert key == theme.slug


# ---------------------------------------------------------------------------
# All 5 module constants are present in THEMES_BY_SLUG
# ---------------------------------------------------------------------------


def test_all_module_constants_in_themes_by_slug() -> None:
    for slug, constant in EXPECTED_CONSTANTS_BY_SLUG.items():
        assert THEMES_BY_SLUG[slug] is constant


# ---------------------------------------------------------------------------
# Validation: empty slug rejected
# ---------------------------------------------------------------------------


def test_empty_slug_raises_validation_error() -> None:
    with pytest.raises(ValueError):
        Theme(slug="", display_name_fr="Some Label")


# ---------------------------------------------------------------------------
# Validation: empty display_name_fr rejected
# ---------------------------------------------------------------------------


def test_empty_display_name_fr_raises_validation_error() -> None:
    with pytest.raises(ValueError):
        Theme(slug="valid-slug", display_name_fr="")


# ---------------------------------------------------------------------------
# Frozen model: mutation raises
# ---------------------------------------------------------------------------


def test_frozen_theme_raises_on_mutation() -> None:
    with pytest.raises(ValueError):
        DROITS_ET_DEVOIRS.slug = "mutated"


# ---------------------------------------------------------------------------
# Official exam weighting
# ---------------------------------------------------------------------------


class TestExamQuestionCounts:
    """EXAM_QUESTION_COUNTS is the official 11/11/8/6/4 theme weighting."""

    def test_matches_official_counts(self) -> None:
        assert EXAM_QUESTION_COUNTS == EXPECTED_QUESTION_COUNTS

    def test_sums_to_forty(self) -> None:
        assert sum(EXAM_QUESTION_COUNTS.values()) == EXAM_TOTAL_QUESTIONS


class TestExamScenarioCounts:
    """EXAM_SCENARIO_COUNTS splits the 40 into 28 knowledge and 12 scenario questions."""

    def test_matches_agreed_split(self) -> None:
        assert EXAM_SCENARIO_COUNTS == EXPECTED_SCENARIO_COUNTS

    def test_sums_to_twelve(self) -> None:
        assert sum(EXAM_SCENARIO_COUNTS.values()) == EXAM_TOTAL_SCENARIOS

    def test_never_exceeds_the_theme_question_count(self) -> None:
        for theme, scenarios in EXAM_SCENARIO_COUNTS.items():
            assert 0 <= scenarios <= EXAM_QUESTION_COUNTS[theme]
