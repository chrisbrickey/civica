"""
Verify that the normalized JSON corpus covers everything in the raw HTML.

For each raw leaf page:
  1. Enumerate every h3/h4 heading inside a content block (cmsfr-block-paragraph or
     cmsfr-block-imageandtext) plus the fr-card--grey objective card title.
  2. Load the corresponding JSON and collect its section headings.
  3. Diff: report headings present in the HTML but missing from the JSON.
  4. Sample a lower-bound char-count check: sum the visible prose in each HTML
     block, sum the JSON section text + headings, and warn if the JSON is
     materially shorter than what the HTML held.

Also flags:
  - Raw leaf pages that include an unknown cmsfr-block-* class (not on the allowlist)
  - Raw leaf pages that produced no JSON
  - Raw pages under an unknown theme directory

Entrypoint: main() runnable via:
    uv run python -m civica.scripts.verify_corpus_coverage
"""

from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from bs4 import BeautifulSoup, Tag

from civica.scripts.normalize_thematic_sheets import (
    CONTENT_BLOCK_CLASSES,
    DEFAULT_CORPUS_DIR,
    DEFAULT_RAW_DIR,
    NormalizedPage,
    _page_slug_from_segments,
    _relative_segments,
    _strip_noise,
    _theme_from_segments,
)

logger = logging.getLogger(__name__)

TEXT_SHORTFALL_RATIO = 0.85

KNOWN_BLOCK_CLASSES = frozenset(
    {
        "cmsfr-block-paragraph",
        "cmsfr-block-imageandtext",
        "cmsfr-block-image-and-text",
        "cmsfr-block-item_grid",
        "cmsfr-block-callout",
        "cmsfr-block-transcription",
        "cmsfr-block-image",
        "cmsfr-block-separator",
        "cmsfr-block-link",
    }
)


@dataclass
class VerificationReport:
    unknown_classes: set[str] = field(default_factory=set)
    leaf_missing_json: list[Path] = field(default_factory=list)
    heading_gaps: list[tuple[Path, list[str]]] = field(default_factory=list)
    text_shortfalls: list[tuple[Path, int, int]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return (
            len(self.leaf_missing_json)
            + len(self.heading_gaps)
            + len(self.text_shortfalls)
            + len(self.problems)
        )


def _normalize_for_compare(s: str) -> str:
    """Collapse whitespace + lowercase + strip accents so headings compare cleanly."""
    nfd = unicodedata.normalize("NFD", s)
    ascii_str = "".join(c for c in nfd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


def _html_headings(soup: BeautifulSoup) -> list[str]:
    """Return the ordered list of headings this page SHOULD produce."""
    out: list[str] = []
    grey_card = soup.find(class_="fr-card--grey")
    if isinstance(grey_card, Tag):
        title = grey_card.find(class_="fr-card__title")
        if isinstance(title, Tag):
            out.append(title.get_text(separator=" ", strip=True))
    for block in soup.find_all(class_=list(CONTENT_BLOCK_CLASSES)):
        if not isinstance(block, Tag):
            continue
        heading = block.find(["h2", "h3", "h4"])
        out.append(
            heading.get_text(separator=" ", strip=True) if isinstance(heading, Tag) else ""
        )
    return out


def _html_content_charcount(soup: BeautifulSoup) -> int:
    """Sum visible chars across content blocks, mirroring the normalizer's stripping."""
    total = 0
    for block in soup.find_all(class_=list(CONTENT_BLOCK_CLASSES)):
        if not isinstance(block, Tag):
            continue
        block_copy = BeautifulSoup(str(block), "html.parser")
        root = block_copy.find(class_=list(CONTENT_BLOCK_CLASSES))
        if not isinstance(root, Tag):
            continue
        _strip_noise(root)
        text = re.sub(r"\s+", " ", root.get_text(separator=" ", strip=True)).strip()
        total += len(text)
    grey = soup.find(class_="fr-card--grey")
    if isinstance(grey, Tag):
        for d in grey.find_all(class_="fr-card__desc"):
            total += len(re.sub(r"\s+", " ", d.get_text(separator=" ", strip=True)).strip())
    return total


def _json_headings(payload: NormalizedPage) -> list[str]:
    return [s["heading"] for s in payload["sections"]]


def _json_charcount(payload: NormalizedPage) -> int:
    """Sum heading + text so we compare like-for-like against the raw block."""
    return sum(len(s["heading"]) + len(s["text"]) for s in payload["sections"])


def _is_leaf(soup: BeautifulSoup) -> bool:
    return soup.find(class_=list(CONTENT_BLOCK_CLASSES)) is not None


def _collect_unknown_classes(soup: BeautifulSoup) -> set[str]:
    found: set[str] = set()
    for tag in soup.find_all(class_=re.compile(r"^cmsfr-block-")):
        if not isinstance(tag, Tag):
            continue
        raw = tag.get("class")
        classes: list[str] = [raw] if isinstance(raw, str) else [c for c in (raw or []) if isinstance(c, str)]
        for cls in classes:
            if cls.startswith("cmsfr-block-") and cls not in KNOWN_BLOCK_CLASSES:
                found.add(cls)
    return found


def verify_all(*, raw_root: Path, corpus_root: Path) -> VerificationReport:
    """Walk raw_root, compare each leaf page against its corpus_root JSON, return findings."""
    banner = "=" * 60
    logger.info(banner)
    logger.info("CORPUS COVERAGE VERIFICATION (this may take a few minutes)")
    logger.info(banner)

    report = VerificationReport()

    if not raw_root.exists():
        return report

    for html_path in sorted(raw_root.rglob("index.html")):
        soup = BeautifulSoup(html_path.read_bytes(), "html.parser")
        if not _is_leaf(soup):
            continue

        report.unknown_classes.update(_collect_unknown_classes(soup))

        segments = _relative_segments(html_path, raw_root)
        try:
            theme = _theme_from_segments(segments)
            slug = _page_slug_from_segments(segments)
        except ValueError as exc:
            report.problems.append(f"cannot resolve theme/slug for {html_path}: {exc}")
            continue
        json_path = corpus_root / theme.slug / f"{slug}.json"

        if not json_path.exists():
            report.leaf_missing_json.append(html_path)
            continue

        payload = cast(NormalizedPage, json.loads(json_path.read_text()))
        html_hs = [_normalize_for_compare(h) for h in _html_headings(soup)]
        json_hs = {_normalize_for_compare(h) for h in _json_headings(payload)}

        missing = [h for h in html_hs if h and h not in json_hs]
        if missing:
            report.heading_gaps.append((html_path, missing))

        html_chars = _html_content_charcount(soup)
        json_chars = _json_charcount(payload)
        if html_chars > 0 and json_chars < html_chars * TEXT_SHORTFALL_RATIO:
            report.text_shortfalls.append((html_path, html_chars, json_chars))

    return report


def log_report(report: VerificationReport) -> None:
    """Emit each category of finding at INFO, or WARNING when the category is non-empty."""
    banner = "=" * 60
    logger.info(banner)
    logger.info("VERIFICATION RESULTS")
    logger.info(banner)

    unknown_level = logging.WARNING if report.unknown_classes else logging.INFO
    logger.log(
        unknown_level,
        "Unknown cmsfr-block-* classes (not handled): %d",
        len(report.unknown_classes),
    )
    for cls in sorted(report.unknown_classes):
        logger.warning("  %s", cls)

    missing_level = logging.WARNING if report.leaf_missing_json else logging.INFO
    logger.log(
        missing_level,
        "Leaf pages with NO corresponding JSON: %d",
        len(report.leaf_missing_json),
    )
    for path in report.leaf_missing_json:
        logger.warning("  %s", path)

    gap_level = logging.WARNING if report.heading_gaps else logging.INFO
    logger.log(
        gap_level,
        "Pages with heading(s) present in HTML but missing from JSON: %d",
        len(report.heading_gaps),
    )
    for gap_path, missing_headings in report.heading_gaps[:20]:
        logger.warning("  %s", gap_path)
        for heading in missing_headings:
            logger.warning("    - missing heading: %r", heading)
    if len(report.heading_gaps) > 20:
        logger.warning("  ... and %d more", len(report.heading_gaps) - 20)

    shortfall_level = logging.WARNING if report.text_shortfalls else logging.INFO
    logger.log(
        shortfall_level,
        "Pages where JSON text < %d%% of HTML content chars: %d",
        int(TEXT_SHORTFALL_RATIO * 100),
        len(report.text_shortfalls),
    )
    for short_path, html_chars, json_chars in report.text_shortfalls[:20]:
        ratio = json_chars / html_chars if html_chars else 0
        logger.warning(
            "  %s  html=%d json=%d ratio=%.2f%%",
            short_path,
            html_chars,
            json_chars,
            ratio * 100,
        )
    if len(report.text_shortfalls) > 20:
        logger.warning("  ... and %d more", len(report.text_shortfalls) - 20)

    problem_level = logging.WARNING if report.problems else logging.INFO
    logger.log(problem_level, "Other problems: %d", len(report.problems))
    for problem in report.problems:
        logger.warning("  %s", problem)

    logger.info(banner)
    if report.total_issues:
        logger.warning("RESULT: FAIL - %d issue(s) found", report.total_issues)
    else:
        logger.info("RESULT: PASS - no issues found")
    logger.info(banner)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    report = verify_all(raw_root=DEFAULT_RAW_DIR, corpus_root=DEFAULT_CORPUS_DIR)
    log_report(report)
    return 1 if report.total_issues else 0


if __name__ == "__main__":
    sys.exit(main())
