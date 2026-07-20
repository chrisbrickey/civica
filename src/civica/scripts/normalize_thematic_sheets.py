"""
Normalize captured ministry raw HTML into JSON records.

Reads leaf pages under data/raw/thematic_sheets/<theme>/[<subtheme>/]<page>/index.html
and writes one JSON per page to data/corpus/thematic_sheets/<theme>/<page-slug>.json.

Skips intermediate index pages (theme roots and subtheme listing pages that only display links to children).
Detection: a leaf page has at least one cmsfr-block-paragraph content block; a listing page does not.

Slug names:
  - Flattens nested subtheme paths into page slugs with a double-underscore separator;
    e.g., laicite/histoire-de-la-laicite -> laicite__histoire-de-la-laicite.
  - Uses only standard english ASCII by stripping diacritics and replacing ligatures.

Verification:
  After normalization completes, cross-checks the JSON against the raw HTML
  to ensure no information was dropped (via verify_corpus_coverage.py).

Entrypoint: main() runnable via:
    uv run python -m civica.scripts.normalize_thematic_sheets
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote

from bs4 import BeautifulSoup, Tag

from civica.domain.themes import Theme

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/thematic_sheets")
DEFAULT_CORPUS_DIR = Path("data/corpus/thematic_sheets")
BASE_URL = "https://formation-civique.interieur.gouv.fr"
URL_PREFIX = "/fiches-par-thematiques/"
SLUG_SEPARATOR = "__"
CONTENT_BLOCK_CLASSES = ("cmsfr-block-paragraph", "cmsfr-block-imageandtext")
NOISE_SELECTORS = ("figcaption",)
NOISE_CLASSES = ("fr-sr-only",)
# Photo credits are embedded inline as <p><i>Source photo : ...</i></p> in ~100 pages.
NOISE_TEXT_PREFIXES = ("Source photo",)

_LIGATURES = {
    "œ": "oe",
    "Œ": "OE",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
}


class Section(TypedDict):
    id: str
    heading: str
    text: str


class NormalizedPage(TypedDict):
    theme: str
    slug: str
    title: str
    source_url: str
    description: str
    sections: list[Section]


def _strip_diacritics(s: str) -> str:
    """Fold ligatures then NFD-normalize and drop combining marks.

    Ligatures (`œ`, `æ`, `ß`) don't decompose under NFD/NFKD, so we replace them
    explicitly first; then Unicode-normalize to split accented Latin into base +
    combining marks, and finally drop the combining marks.
    """
    for lig, repl in _LIGATURES.items():
        s = s.replace(lig, repl)
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def _slugify(s: str) -> str:
    """Return an ASCII lowercase slug: alphanumerics preserved, everything else becomes '-'."""
    ascii_str = _strip_diacritics(s).lower()
    hyphenated = re.sub(r"[^a-z0-9]+", "-", ascii_str)
    return hyphenated.strip("-")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_title(soup: BeautifulSoup) -> str:
    """Return the page's main <h1> text (whitespace-collapsed), or empty string."""
    main = soup.find("main")
    if not isinstance(main, Tag):
        return ""
    h1 = main.find("h1")
    if not isinstance(h1, Tag):
        return ""
    return _normalize_whitespace(h1.get_text(separator=" ", strip=True))


def _extract_description(soup: BeautifulSoup) -> str:
    """Return the meta description content, or empty string."""
    meta = soup.find("meta", attrs={"name": "description"})
    if not isinstance(meta, Tag):
        return ""
    content = meta.get("content", "")
    if content is None:
        return ""
    if isinstance(content, list):
        content = " ".join(content)
    return _normalize_whitespace(content)


def _objective_section(soup: BeautifulSoup) -> Section | None:
    """Return the 'Objectif de la fiche' card (fr-card--grey) as a section, if present."""
    card = soup.find(class_="fr-card--grey")
    if not isinstance(card, Tag):
        return None
    title_el = card.find(class_="fr-card__title")
    heading = (
        _normalize_whitespace(title_el.get_text(separator=" ", strip=True))
        if isinstance(title_el, Tag)
        else ""
    )
    descs = card.find_all(class_="fr-card__desc")
    text = _normalize_whitespace(
        " ".join(d.get_text(separator=" ", strip=True) for d in descs)
    )
    if not heading and not text:
        return None
    section_id = _slugify(heading) or "objectif"
    return Section(id=section_id, heading=heading, text=text)


def _strip_noise(block: Tag) -> None:
    """Remove accessibility captions, image captions, and photo-credit lines in-place."""
    for selector in NOISE_SELECTORS:
        for el in block.find_all(selector):
            el.extract()
    for cls in NOISE_CLASSES:
        for el in block.find_all(class_=cls):
            el.extract()
    for p in block.find_all("p"):
        text = p.get_text(strip=True)
        if any(text.startswith(prefix) for prefix in NOISE_TEXT_PREFIXES):
            p.extract()


def _content_sections(soup: BeautifulSoup) -> list[Section]:
    """Return one Section per content block (paragraph or imageandtext), in document order."""
    sections: list[Section] = []
    seen_ids: dict[str, int] = {}
    for block in soup.find_all(class_=list(CONTENT_BLOCK_CLASSES)):
        if not isinstance(block, Tag):
            continue
        _strip_noise(block)
        heading_el = block.find(["h2", "h3", "h4"])
        heading = (
            _normalize_whitespace(heading_el.get_text(separator=" ", strip=True))
            if isinstance(heading_el, Tag)
            else ""
        )
        if isinstance(heading_el, Tag):
            heading_el.extract()
        text = _normalize_whitespace(block.get_text(separator=" ", strip=True))
        if not text and not heading:
            continue
        base_id = _slugify(heading) or f"section-{len(sections) + 1:02d}"
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        section_id = base_id if seen_ids[base_id] == 1 else f"{base_id}-{seen_ids[base_id]}"
        sections.append(Section(id=section_id, heading=heading, text=text))
    return sections


def _is_leaf_page(soup: BeautifulSoup) -> bool:
    """A page is a leaf if it has at least one content block (paragraph or imageandtext)."""
    return soup.find(class_=list(CONTENT_BLOCK_CLASSES)) is not None


def _relative_segments(html_path: Path, raw_root: Path) -> list[str]:
    """Path segments between raw_root and index.html (exclusive of index.html)."""
    return list(html_path.relative_to(raw_root).parent.parts)


def _theme_from_segments(segments: list[str]) -> Theme:
    if not segments:
        raise ValueError("Page path has no theme segment.")
    return Theme.from_slug(_slugify(segments[0]))


def _page_slug_from_segments(segments: list[str]) -> str:
    """Flatten all segments after the theme with SLUG_SEPARATOR, slug-normalized.

    Each segment is slugified independently so ligatures fold to ASCII and any
    non-alphanumeric runs within a segment collapse to a single hyphen.
    """
    tail = [_slugify(seg) for seg in segments[1:]]
    if not tail:
        raise ValueError("Cannot compute page slug for a theme-root page.")
    return SLUG_SEPARATOR.join(tail)


def _build_source_url(segments: list[str]) -> str:
    """Construct the canonical ministry URL for a page from its on-disk path segments."""
    quoted = "/".join(quote(seg) for seg in segments)
    return f"{BASE_URL}{URL_PREFIX}{quoted}/"


def normalize_page(html_path: Path, raw_root: Path) -> NormalizedPage | None:
    """Parse one captured index.html into a NormalizedPage, or None if it is not a leaf.

    Raises ValueError if the theme segment does not resolve to a known Theme.
    """
    segments = _relative_segments(html_path, raw_root)
    if len(segments) < 2:
        return None

    theme = _theme_from_segments(segments)
    slug = _page_slug_from_segments(segments)

    html = html_path.read_bytes()
    soup = BeautifulSoup(html, "html.parser")

    if not _is_leaf_page(soup):
        return None

    sections: list[Section] = []
    objective = _objective_section(soup)
    if objective is not None:
        sections.append(objective)
    sections.extend(_content_sections(soup))

    return NormalizedPage(
        theme=theme.slug,
        slug=slug,
        title=_extract_title(soup),
        source_url=_build_source_url(segments),
        description=_extract_description(soup),
        sections=sections,
    )


def _dump_json(page: NormalizedPage) -> bytes:
    text = json.dumps(page, ensure_ascii=False, indent=2) + "\n"
    return text.encode("utf-8")


def _validate_all_themes(raw_root: Path) -> None:
    """Fail-fast: verify every top-level directory under raw_root maps to a known theme."""
    for entry in sorted(raw_root.iterdir()):
        if entry.is_dir():
            _theme_from_segments([entry.name])


def normalize_all(*, raw_root: Path, corpus_root: Path) -> list[Path]:
    """Walk raw_root, normalize each leaf page, and write JSON under corpus_root.

    Returns the list of written JSON file paths in write order. Skipped listing pages
    are logged at INFO level. Raises ValueError if any theme directory is unknown.
    """
    if not raw_root.exists():
        logger.info("Raw directory %s does not exist; nothing to normalize.", raw_root)
        return []

    _validate_all_themes(raw_root)

    written: list[Path] = []
    for html_path in sorted(raw_root.rglob("index.html")):
        page = normalize_page(html_path, raw_root)
        if page is None:
            logger.info("Skipping non-leaf page: %s", html_path.relative_to(raw_root))
            continue

        dest = corpus_root / page["theme"] / f"{page['slug']}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_dump_json(page))
        written.append(dest)
        logger.info("Wrote %s", dest)

    logger.info("Normalized %d page(s).", len(written))
    return written


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Transformation: Transform the raw HTML to normalized JSON and persist
    normalize_all(raw_root=DEFAULT_RAW_DIR, corpus_root=DEFAULT_CORPUS_DIR)

    # Import verification script internals here to avoid a circular imports
    from civica.scripts.verify_corpus_coverage import log_report, verify_all

    # Verification: Cross check the resulting JSON against the raw HTML
    report = verify_all(raw_root=DEFAULT_RAW_DIR, corpus_root=DEFAULT_CORPUS_DIR)
    log_report(report)
    return 1 if report.total_issues else 0


if __name__ == "__main__":
    sys.exit(main())
