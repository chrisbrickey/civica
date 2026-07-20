"""
Unit tests for civica.scripts.verify_corpus_coverage.

Each test builds a raw/ and corpus/ tree under tmp_path,
invokes verify_all(), and asserts on the returned VerificationReport.
Fixtures are inlined so each test is self-contained.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from civica.scripts.verify_corpus_coverage import (
    VerificationReport,
    log_report,
    verify_all,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KNOWN_THEME = "principes-et-valeurs-de-la-republique"
_UNKNOWN_THEME = "bogus-theme-slug"
_SAMPLE_SLUG = "sample-page"

_LEAF_HTML = """<!DOCTYPE html>
<html lang="fr">
<body>
<main>
<h1>Titre exemple</h1>
<div class="fr-container cmsfr-block-item_grid">
  <div class="fr-card fr-card--grey">
    <div class="fr-card__body"><div class="fr-card__content">
      <h2 class="fr-card__title">Objectif de la fiche :</h2>
      <p class="fr-card__desc">Comprendre le fonctionnement du test de verification.</p>
    </div></div>
  </div>
</div>
<div class="fr-container cmsfr-block-paragraph">
  <h3>Une section reelle</h3>
  <p>Le contenu de la section pour valider la couverture par le verificateur.</p>
</div>
</main>
</body>
</html>
"""

_LISTING_HTML = (
    '<!DOCTYPE html><html><body><main>'
    '<div id="posts-list"><h2>Pages</h2></div>'
    '</main></body></html>'
)

_MATCHING_JSON: dict[str, Any] = {
    "theme": _KNOWN_THEME,
    "slug": _SAMPLE_SLUG,
    "title": "Titre exemple",
    "source_url": "https://example.com/",
    "description": "",
    "sections": [
        {
            "id": "objectif-de-la-fiche",
            "heading": "Objectif de la fiche :",
            "text": "Comprendre le fonctionnement du test de verification.",
        },
        {
            "id": "une-section-reelle",
            "heading": "Une section reelle",
            "text": "Le contenu de la section pour valider la couverture par le verificateur.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_raw(raw_root: Path, theme: str, slug: str, html: str) -> Path:
    dest = raw_root / theme / slug / "index.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    return dest


def _write_corpus(corpus_root: Path, theme: str, slug: str, payload: dict[str, Any]) -> Path:
    dest = corpus_root / theme / f"{slug}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    return root


@pytest.fixture()
def corpus_root(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCleanRun:
    """A leaf page with matching JSON produces zero issues."""

    def test_leaf_with_matching_json_reports_no_issues(
        self, raw_root: Path, corpus_root: Path
    ) -> None:
        _write_raw(raw_root, _KNOWN_THEME, _SAMPLE_SLUG, _LEAF_HTML)
        _write_corpus(corpus_root, _KNOWN_THEME, _SAMPLE_SLUG, _MATCHING_JSON)

        report = verify_all(raw_root=raw_root, corpus_root=corpus_root)

        assert report.total_issues == 0
        assert report.unknown_classes == set()
        assert report.leaf_missing_json == []
        assert report.heading_gaps == []
        assert report.text_shortfalls == []
        assert report.problems == []


class TestMissingJson:
    """A raw leaf page with no corresponding JSON is flagged."""

    def test_leaf_without_json_is_recorded(
        self, raw_root: Path, corpus_root: Path
    ) -> None:
        html_path = _write_raw(raw_root, _KNOWN_THEME, _SAMPLE_SLUG, _LEAF_HTML)

        report = verify_all(raw_root=raw_root, corpus_root=corpus_root)

        assert html_path in report.leaf_missing_json
        assert report.total_issues == 1


class TestHeadingGap:
    """A JSON that is missing a section heading present in HTML is flagged."""

    def test_json_with_no_sections_flags_all_html_headings(
        self, raw_root: Path, corpus_root: Path
    ) -> None:
        _write_raw(raw_root, _KNOWN_THEME, _SAMPLE_SLUG, _LEAF_HTML)
        empty = {**_MATCHING_JSON, "sections": []}
        _write_corpus(corpus_root, _KNOWN_THEME, _SAMPLE_SLUG, empty)

        report = verify_all(raw_root=raw_root, corpus_root=corpus_root)

        assert len(report.heading_gaps) == 1
        _, missing = report.heading_gaps[0]
        assert "objectif de la fiche :" in missing
        assert "une section reelle" in missing


class TestTextShortfall:
    """A JSON with drastically less text than the source HTML is flagged."""

    def test_truncated_json_is_flagged(
        self, raw_root: Path, corpus_root: Path
    ) -> None:
        long_html = _LEAF_HTML.replace(
            "Le contenu de la section pour valider la couverture par le verificateur.",
            "Contenu long. " * 40,
        )
        _write_raw(raw_root, _KNOWN_THEME, _SAMPLE_SLUG, long_html)
        truncated = {
            **_MATCHING_JSON,
            "sections": [
                {
                    "id": "objectif-de-la-fiche",
                    "heading": "Objectif de la fiche :",
                    "text": "court",
                },
                {
                    "id": "une-section-reelle",
                    "heading": "Une section reelle",
                    "text": "court",
                },
            ],
        }
        _write_corpus(corpus_root, _KNOWN_THEME, _SAMPLE_SLUG, truncated)

        report = verify_all(raw_root=raw_root, corpus_root=corpus_root)

        assert len(report.text_shortfalls) == 1
        html_path, html_chars, json_chars = report.text_shortfalls[0]
        assert html_chars > json_chars


class TestUnknownBlockClass:
    """A cmsfr-block-* class not on the allowlist is surfaced for review."""

    def test_unknown_class_is_recorded(
        self, raw_root: Path, corpus_root: Path
    ) -> None:
        html_with_unknown = _LEAF_HTML.replace(
            "</main>",
            '<div class="cmsfr-block-carousel">Widget nouveau</div></main>',
        )
        _write_raw(raw_root, _KNOWN_THEME, _SAMPLE_SLUG, html_with_unknown)
        _write_corpus(corpus_root, _KNOWN_THEME, _SAMPLE_SLUG, _MATCHING_JSON)

        report = verify_all(raw_root=raw_root, corpus_root=corpus_root)

        assert "cmsfr-block-carousel" in report.unknown_classes


class TestListingPageIgnored:
    """Non-leaf pages (no content blocks) are not verified at all."""

    def test_listing_page_produces_no_findings(
        self, raw_root: Path, corpus_root: Path
    ) -> None:
        _write_raw(raw_root, _KNOWN_THEME, _SAMPLE_SLUG, _LISTING_HTML)

        report = verify_all(raw_root=raw_root, corpus_root=corpus_root)

        assert report.total_issues == 0


class TestUnknownTheme:
    """A leaf page under an unknown theme directory is flagged in problems."""

    def test_unknown_theme_appears_in_problems(
        self, raw_root: Path, corpus_root: Path
    ) -> None:
        _write_raw(raw_root, _UNKNOWN_THEME, _SAMPLE_SLUG, _LEAF_HTML)

        report = verify_all(raw_root=raw_root, corpus_root=corpus_root)

        assert len(report.problems) == 1
        assert _UNKNOWN_THEME in report.problems[0]


class TestReportAggregation:
    """VerificationReport.total_issues sums all failure categories."""

    def test_multiple_failure_categories_sum_correctly(
        self, raw_root: Path, corpus_root: Path
    ) -> None:
        # One leaf missing JSON, one leaf with empty JSON (heading gap + text shortfall).
        _write_raw(raw_root, _KNOWN_THEME, "missing-json-page", _LEAF_HTML)
        _write_raw(raw_root, _KNOWN_THEME, "empty-json-page", _LEAF_HTML)
        empty = {**_MATCHING_JSON, "sections": []}
        _write_corpus(corpus_root, _KNOWN_THEME, "empty-json-page", empty)

        report = verify_all(raw_root=raw_root, corpus_root=corpus_root)

        assert len(report.leaf_missing_json) == 1
        assert len(report.heading_gaps) == 1
        assert len(report.text_shortfalls) == 1
        assert report.total_issues == 3


class TestVerificationReportDefaults:
    """VerificationReport is empty by default and computes total_issues correctly."""

    def test_default_report_has_zero_issues(self) -> None:
        report = VerificationReport()
        assert report.total_issues == 0
        assert report.unknown_classes == set()
        assert report.leaf_missing_json == []


class TestLogReport:
    """log_report emits INFO summaries on a clean report and WARNINGs when there are findings."""

    def test_clean_report_emits_info_records_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="civica.scripts.verify_corpus_coverage")

        log_report(VerificationReport())

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == [], f"Clean report should emit no WARNINGs; got: {warnings!r}"
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "Unknown cmsfr-block-* classes" in text
        assert "Leaf pages with NO corresponding JSON" in text
        assert "heading(s) present in HTML but missing from JSON" in text
        assert "Other problems" in text

    def test_clean_report_emits_pass_summary_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="civica.scripts.verify_corpus_coverage")

        log_report(VerificationReport())

        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "RESULT: PASS" in text
        assert "no issues found" in text

    def test_report_with_findings_emits_warnings_for_each_category(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="civica.scripts.verify_corpus_coverage")

        report = VerificationReport(
            unknown_classes={"cmsfr-block-carousel"},
            leaf_missing_json=[Path("/tmp/raw/foo/index.html")],
            heading_gaps=[(Path("/tmp/raw/bar/index.html"), ["missing-heading-x"])],
            text_shortfalls=[(Path("/tmp/raw/baz/index.html"), 1000, 400)],
            problems=["theme resolution failed"],
        )

        log_report(report)

        warnings_text = "\n".join(
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        )
        assert "cmsfr-block-carousel" in warnings_text
        assert "/tmp/raw/foo/index.html" in warnings_text
        assert "missing-heading-x" in warnings_text
        assert "/tmp/raw/baz/index.html" in warnings_text
        assert "theme resolution failed" in warnings_text

    def test_report_with_findings_emits_fail_summary_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="civica.scripts.verify_corpus_coverage")

        report = VerificationReport(
            leaf_missing_json=[Path("/tmp/raw/foo/index.html")],
            problems=["theme resolution failed"],
        )

        log_report(report)

        warnings_text = "\n".join(
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        )
        assert "RESULT: FAIL" in warnings_text
        assert "2 issue(s) found" in warnings_text

    def test_no_print_output_on_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reporting must go through the logger, not print(); stdout stays empty."""
        log_report(VerificationReport(problems=["boom"]))

        captured = capsys.readouterr()
        assert captured.out == ""
