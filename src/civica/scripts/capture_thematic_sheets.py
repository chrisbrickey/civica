"""
Capture ministry thematic sheets in data/raw/thematic_sheets/.

One-shot sitemap-driven capture of the Civic Education Fact Sheets hosted
by France's Ministry of the Interior. Each in-scope sitemap URL is fetched
(rate-limited) and persisted as HTML under data/raw/thematic_sheets/<path>/index.html,
which mirrors the URL hierarchy under /fiches-par-thematiques/.
A second phase re-parses the fetched HTML (no additional network calls)
to surface any discrepancy between the site's link graph and the sitemap.

Entrypoint: main() runnable via:
    uv run python -m civica.scripts.capture_thematic_sheets
"""

import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

INDEX_URL = "https://formation-civique.interieur.gouv.fr/fiches-par-thematiques/"
SITEMAP_URL = "https://formation-civique.interieur.gouv.fr/sitemap.xml"
ALLOWED_PATH_PREFIX = "/fiches-par-thematiques/"
DEFAULT_OUT_DIR = Path("data/raw/thematic_sheets")
_INDEX_HOST = urlparse(INDEX_URL).netloc
_SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"

# Permitted by target security as of 2026-07-19. Communicates identity and intention of the requests.
DEFAULT_USER_AGENT = (
    "civica/0.1 (+https://github.com/chrisbrickey/civica; "
    "amateur french naturalization civics exam prep)"
)


def _parse_sitemap_urls(xml_bytes: bytes) -> list[str]:
    """Parse a sitemap XML and return in-scope production-host URLs.

    Each <loc> has its scheme and netloc replaced with the production host.
    Only URLs whose path starts with ALLOWED_PATH_PREFIX are kept.
    Order is preserved; duplicates are removed.
    """
    root = ET.fromstring(xml_bytes.decode("utf-8"))
    seen: set[str] = set()
    result: list[str] = []

    for loc_el in root.iter(f"{{{_SITEMAP_NS}}}loc"):
        raw = (loc_el.text or "").strip()
        if not raw:
            continue
        parsed = urlparse(raw)
        # Rewrite the host to the canonical production host
        production = parsed._replace(scheme="https", netloc=_INDEX_HOST)
        url = production.geturl()

        if not production.path.startswith(ALLOWED_PATH_PREFIX):
            continue

        if url in seen:
            continue

        seen.add(url)
        result.append(url)

    return result


def _relative_path_for(url: str) -> Path:
    """Return the on-disk relative path for a URL by stripping ALLOWED_PATH_PREFIX.

    The index URL (path == ALLOWED_PATH_PREFIX) returns Path('.').
    A leaf URL like /fiches-par-thematiques/theme/sub/page/ returns
    Path('theme/sub/page'). Percent-encoded characters in path segments
    are decoded so filenames use the natural Unicode form.
    """
    path = urlparse(url).path
    # Strip the prefix and any trailing slash
    remainder = path[len(ALLOWED_PATH_PREFIX):].rstrip("/")
    if not remainder:
        return Path(".")
    return Path(unquote(remainder))


def _discover_in_scope_links(html: bytes, base_url: str) -> list[str]:
    """Parse HTML and return absolute URLs that are in-scope for crawling.

    In-scope means: same host as INDEX_URL and path starts with
    ALLOWED_PATH_PREFIX and is not the index URL itself.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    result: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if not isinstance(href, str):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        if parsed.netloc != _INDEX_HOST:
            continue

        if not parsed.path.startswith(ALLOWED_PATH_PREFIX):
            continue

        # Normalise: strip query/fragment for deduplication
        normalised = parsed._replace(query="", fragment="").geturl()
        if normalised in seen:
            continue

        seen.add(normalised)
        result.append(normalised)

    return result


def _write_if_changed(dest: Path, content: bytes) -> bool:
    """Write content to destination only when the content differs from what is already on disk.

    Skipping the write when bytes are identical preserves the file's mtime,
    which is how the idempotency test detects that no rewrite occurred.

    Returns True if the file was written, False if it was left untouched.
    """
    if dest.exists() and dest.read_bytes() == content:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return True


def capture(
    *,
    client: httpx.Client,
    out_dir: Path,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Crawl the sitemap and persist each in-scope page to out_dir.

    Phase 1: sitemap-driven fetch.
      - GET SITEMAP_URL and parse all in-scope URLs.
      - For each URL: sleep, GET, on failure print WARN and continue;
        on success write to out_dir/<relative-path>/index.html.

    Phase 2: BFS cross-check (no additional HTTP).
      - Parse links from every fetched page.
      - Warn about URLs discovered via links but absent from the sitemap.
      - Warn about sitemap URLs that no fetched page links to.

    Args:
        client:   already-constructed httpx.Client allows injection of mock transport.
        out_dir:  root output directory; files are persisted under the URL hierarchy.
        sleep_fn: callable used for rate-limiting; default is time.sleep.
                  Inject a no-op in tests to avoid sleeping.
    """
    # Phase 1: sitemap-driven fetch
    logger.info("Capturing thematic sheets. This will take 5-10 minutes (approx 1 request/second).")
    logger.info("Phase 1: Fetching pages")
    logger.info("Fetching sitemap...")
    sitemap_response = client.get(SITEMAP_URL)
    sitemap_response.raise_for_status()

    urls = _parse_sitemap_urls(sitemap_response.content)
    total = len(urls)
    logger.info("Sitemap parsed: %d URL(s) to fetch.", total)

    fetched: dict[str, bytes] = {}
    failed: list[str] = []
    width = len(str(total))

    for i, url in enumerate(urls, start=1):
        sleep_fn(1.0)
        display_path = urlparse(url).path
        prefix = f"[{i:>{width}}/{total}] GET {display_path}"
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("%s -> FAILED (HTTP %d)", prefix, exc.response.status_code)
            failed.append(url)
            continue
        except Exception as exc:
            logger.warning("%s -> FAILED (%s)", prefix, exc)
            failed.append(url)
            continue

        content = response.content
        rel = _relative_path_for(url)
        dest = out_dir / rel / "index.html"
        wrote = _write_if_changed(dest, content)
        logger.info("%s -> %s", prefix, "wrote" if wrote else "unchanged")
        fetched[url] = content

    if failed:
        logger.warning("%d URL(s) failed to fetch:", len(failed))
        for u in failed:
            logger.warning("  %s", u)

    # Phase 2: BFS cross-check (no HTTP)
    logger.info("Phase 2: Cross-checking against page links")
    discovered: set[str] = set()
    for url, content in fetched.items():
        for link in _discover_in_scope_links(content, url):
            discovered.add(link)

    sitemap_set = set(urls)
    missing_from_sitemap = discovered - sitemap_set
    unreferenced_in_sitemap = sitemap_set - discovered

    if missing_from_sitemap:
        logger.warning(
            "%d URL(s) found in pages but absent from sitemap:",
            len(missing_from_sitemap),
        )
        for u in sorted(missing_from_sitemap):
            logger.warning("  %s", u)

    if unreferenced_in_sitemap:
        logger.warning(
            "%d URL(s) in sitemap not linked from any fetched page:",
            len(unreferenced_in_sitemap),
        )
        for u in sorted(unreferenced_in_sitemap):
            logger.warning("  %s", u)

    if not missing_from_sitemap and not unreferenced_in_sitemap:
        logger.info(
            "Cross-check complete: %d link(s) discovered across %d fetched page(s); "
            "no discrepancies with sitemap.",
            len(discovered),
            len(fetched),
        )
    else:
        logger.info(
            "Cross-check complete: %d link(s) discovered across %d fetched page(s).",
            len(discovered),
            len(fetched),
        )


def main() -> None:
    """Wire real dependencies and run the capture."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client:
        capture(client=client, out_dir=DEFAULT_OUT_DIR, sleep_fn=time.sleep)

    logger.info("Capture complete. Pages written to %s/", DEFAULT_OUT_DIR)


if __name__ == "__main__":
    main()
