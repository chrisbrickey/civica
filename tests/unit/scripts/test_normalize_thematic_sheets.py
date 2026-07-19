"""
Unit tests for civica.scripts.normalize_thematic_sheets.

Each test builds a raw/ tree in tmp_path, invokes normalize_all(), and asserts
against the produced corpus/ tree. Fixtures live under tests/fixtures/html/.
"""

import json
from pathlib import Path

import pytest

from civica.scripts.normalize_thematic_sheets import normalize_all

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KNOWN_THEME = "principes-et-valeurs-de-la-republique"
_UNKNOWN_THEME = "some-invalid-theme-slug"

_LEAF_SUBPATH = Path("laicite/histoire-de-la-laicite")
_NESTED_LEAF_SUBPATH = Path("laicite/histoire-de-la-laicite")
_FLAT_LEAF_SUBPATH = Path("la-langue-de-la-republique")

_EXPECTED_LEAF_SLUG = "laicite__histoire-de-la-laicite"
_EXPECTED_FLAT_LEAF_SLUG = "la-langue-de-la-republique"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def leaf_html(html_fixtures_dir: Path) -> bytes:
    return (html_fixtures_dir / "normalize_leaf_page.html").read_bytes()


@pytest.fixture()
def listing_html(html_fixtures_dir: Path) -> bytes:
    return (html_fixtures_dir / "normalize_listing_page.html").read_bytes()


@pytest.fixture()
def expected_leaf_json(html_fixtures_dir: Path) -> bytes:
    return (html_fixtures_dir / "normalize_leaf_page.expected.json").read_bytes()


@pytest.fixture()
def raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    return root


@pytest.fixture()
def corpus_root(tmp_path: Path) -> Path:
    return tmp_path / "corpus"


def _write_page(raw_root: Path, relative_dir: Path, html: bytes) -> Path:
    dest = raw_root / relative_dir / "index.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(html)
    return dest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Byte-for-byte JSON match against the expected fixture."""

    def test_produced_json_matches_expected_fixture_byte_for_byte(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
        expected_leaf_json: bytes,
    ) -> None:
        _write_page(raw_root, Path(_KNOWN_THEME) / _LEAF_SUBPATH, leaf_html)

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        produced = corpus_root / _KNOWN_THEME / f"{_EXPECTED_LEAF_SLUG}.json"
        assert produced.exists(), f"Expected {produced} to be written"
        assert produced.read_bytes() == expected_leaf_json, (
            "Produced JSON does not match expected fixture byte-for-byte.\n"
            f"Produced:\n{produced.read_text()}\n"
            f"Expected:\n{expected_leaf_json.decode()}"
        )


class TestUnknownTheme:
    """Unknown theme slugs raise a clear error rather than being written to a wrong bucket."""

    def test_unknown_theme_raises_value_error(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
    ) -> None:
        _write_page(raw_root, Path(_UNKNOWN_THEME) / _LEAF_SUBPATH, leaf_html)

        with pytest.raises(ValueError, match=_UNKNOWN_THEME):
            normalize_all(raw_root=raw_root, corpus_root=corpus_root)

    def test_unknown_theme_does_not_write_any_json(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
    ) -> None:
        _write_page(raw_root, Path(_UNKNOWN_THEME) / _LEAF_SUBPATH, leaf_html)

        with pytest.raises(ValueError):
            normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        assert not corpus_root.exists() or not any(corpus_root.rglob("*.json")), (
            "No JSON should be written when the theme is unknown."
        )


class TestListingPagesAreSkipped:
    """Intermediate index/listing pages (no cmsfr-block-paragraph blocks) are skipped."""

    def test_theme_root_index_is_not_written(
        self,
        raw_root: Path,
        corpus_root: Path,
        listing_html: bytes,
    ) -> None:
        _write_page(raw_root, Path(_KNOWN_THEME), listing_html)

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        assert not corpus_root.exists() or not any(corpus_root.rglob("*.json")), (
            "Listing page should not produce any JSON output."
        )

    def test_subtheme_index_is_not_written(
        self,
        raw_root: Path,
        corpus_root: Path,
        listing_html: bytes,
        leaf_html: bytes,
    ) -> None:
        _write_page(raw_root, Path(_KNOWN_THEME) / "laicite", listing_html)
        _write_page(
            raw_root,
            Path(_KNOWN_THEME) / "laicite" / "histoire-de-la-laicite",
            leaf_html,
        )

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        subtheme_output = corpus_root / _KNOWN_THEME / "laicite.json"
        assert not subtheme_output.exists(), (
            "Subtheme index page should be skipped, but a JSON was written."
        )

        leaf_output = corpus_root / _KNOWN_THEME / f"{_EXPECTED_LEAF_SLUG}.json"
        assert leaf_output.exists(), (
            "The nested leaf page should still be normalized."
        )


class TestSlugFlattening:
    """Nested subtheme directories are flattened into the page slug with '__'."""

    def test_two_level_leaf_uses_flat_slug(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
    ) -> None:
        _write_page(raw_root, Path(_KNOWN_THEME) / _FLAT_LEAF_SUBPATH, leaf_html)

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        produced = corpus_root / _KNOWN_THEME / f"{_EXPECTED_FLAT_LEAF_SLUG}.json"
        assert produced.exists(), f"Expected {produced} to be written for a two-level leaf"

    def test_three_level_leaf_uses_double_underscore_slug(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
    ) -> None:
        _write_page(raw_root, Path(_KNOWN_THEME) / _NESTED_LEAF_SUBPATH, leaf_html)

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        produced = corpus_root / _KNOWN_THEME / f"{_EXPECTED_LEAF_SLUG}.json"
        assert produced.exists(), (
            f"Expected {produced} (flattened subtheme + page) to be written"
        )


class TestAsciiThemeNormalization:
    """Theme slugs containing diacritics on disk (e.g. cedilla) map to the canonical ASCII slug."""

    def test_cedilla_theme_slug_is_normalized_to_ascii(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
    ) -> None:
        _write_page(
            raw_root,
            Path("vivre-dans-la-societe-française") / "une-fiche-sample",
            leaf_html,
        )

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        ascii_dir = corpus_root / "vivre-dans-la-societe-francaise"
        assert ascii_dir.exists(), (
            "Expected the theme directory to be normalized to ASCII (no cedilla)."
        )
        written = list(ascii_dir.glob("*.json"))
        assert len(written) == 1, f"Expected exactly one JSON; got {written}"


class TestDeterministicOutput:
    """Re-running normalize_all produces byte-identical output."""

    def test_second_run_produces_identical_bytes(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
    ) -> None:
        _write_page(raw_root, Path(_KNOWN_THEME) / _LEAF_SUBPATH, leaf_html)

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)
        first = (corpus_root / _KNOWN_THEME / f"{_EXPECTED_LEAF_SLUG}.json").read_bytes()

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)
        second = (corpus_root / _KNOWN_THEME / f"{_EXPECTED_LEAF_SLUG}.json").read_bytes()

        assert first == second


class TestSlugNormalization:
    """Path segments with ligatures or consecutive dashes are cleaned up in the page slug."""

    def test_ligature_in_page_segment_folds_to_ascii(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
    ) -> None:
        _write_page(
            raw_root,
            Path(_KNOWN_THEME) / "un-systeme-aerien-au-cœur-du-reseau-mondial",
            leaf_html,
        )

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        produced = (
            corpus_root
            / _KNOWN_THEME
            / "un-systeme-aerien-au-coeur-du-reseau-mondial.json"
        )
        assert produced.exists(), (
            f"Ligature 'œ' should fold to 'oe' in the slug; expected {produced}"
        )

    def test_consecutive_dashes_in_segment_collapse_to_one(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
    ) -> None:
        _write_page(
            raw_root,
            Path(_KNOWN_THEME) / "some-page--with-double-dash",
            leaf_html,
        )

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        produced = corpus_root / _KNOWN_THEME / "some-page-with-double-dash.json"
        assert produced.exists(), (
            f"Consecutive dashes in a segment should collapse; expected {produced}"
        )


class TestJsonIsValid:
    """Emitted JSON parses and has the documented shape."""

    def test_parsed_shape_contains_expected_top_level_keys(
        self,
        raw_root: Path,
        corpus_root: Path,
        leaf_html: bytes,
    ) -> None:
        _write_page(raw_root, Path(_KNOWN_THEME) / _LEAF_SUBPATH, leaf_html)

        normalize_all(raw_root=raw_root, corpus_root=corpus_root)

        produced = corpus_root / _KNOWN_THEME / f"{_EXPECTED_LEAF_SLUG}.json"
        parsed = json.loads(produced.read_text())

        assert set(parsed) == {
            "theme",
            "slug",
            "title",
            "source_url",
            "description",
            "sections",
        }
        assert parsed["theme"] == _KNOWN_THEME
        assert parsed["slug"] == _EXPECTED_LEAF_SLUG
        assert isinstance(parsed["sections"], list)
        assert len(parsed["sections"]) >= 1
        for section in parsed["sections"]:
            assert set(section) == {"id", "heading", "text"}
