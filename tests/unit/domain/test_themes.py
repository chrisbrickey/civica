"""Unit tests for civica.domain.themes: Theme pydantic model and module constants."""

import pytest

from civica.domain.themes import (
    DROITS_ET_DEVOIRS,
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