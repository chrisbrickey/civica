"""
Unit tests for civica.scripts.capture_thematic_sheets.

Uses httpx.MockTransport to eschew a network call and html test fixtures.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from civica.scripts.capture_thematic_sheets import capture, resolve_user_agent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Held as a literal rather than read from the module, so that a rename of the production
# constant's value will result in test failure instead of remaining silent.
_USER_AGENT_ENV_VAR = "CIVICA_USER_AGENT"
_SAMPLE_OVERRIDE_USER_AGENT = "sample-fork/0.1 (+https://example.com/sample-fork)"

_BASE_URL = "https://formation-civique.interieur.gouv.fr"
_SITEMAP_PATH = "/sitemap.xml"
_INDEX_PATH = "/fiches-par-thematiques/"
_ALPHA_PATH = "/fiches-par-thematiques/theme-sample/subtheme-sample/fiche-sample-alpha/"
_BETA_PATH = "/fiches-par-thematiques/theme-sample/subtheme-sample/fiche-sample-beta/"
_ORPHAN_PATH = "/fiches-par-thematiques/theme-sample/subtheme-sample/fiche-sample-orphan/"
_UNLISTED_EXTRA_PATH = "/fiches-par-thematiques/theme-sample/subtheme-sample/fiche-unlisted-extra/"

# Mirrored on-disk relative paths (strip /fiches-par-thematiques/ prefix)
_ALPHA_REL_PATH = Path("theme-sample/subtheme-sample/fiche-sample-alpha")
_BETA_REL_PATH = Path("theme-sample/subtheme-sample/fiche-sample-beta")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transport(
    responses: dict[str, bytes],
    fetched_urls: list[str] | None = None,
) -> httpx.MockTransport:
    """Build an httpx.MockTransport that returns the given bytes per path.

    Returns 404 for any path not in the responses dict so accidental fetches
    of out-of-scope URLs are detectable via test assertions on written files.
    When fetched_urls is given, every requested URL is appended to it so tests
    can assert on exactly what was fetched and in which order.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if fetched_urls is not None:
            fetched_urls.append(str(request.url))
        path = request.url.path
        if path in responses:
            return httpx.Response(200, content=responses[path])
        return httpx.Response(404, content=b"Not Found")

    return httpx.MockTransport(handler)


def _make_client(transport: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=transport, base_url=_BASE_URL)


def _no_sleep(seconds: float) -> None:
    """Drop-in replacement for time.sleep that does nothing."""


def _written_file(out_dir: Path, rel_path: Path) -> Path:
    """On-disk location capture() mirrors an in-scope page to."""
    return out_dir / rel_path / "index.html"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sitemap_bytes(html_fixtures_dir: Path) -> bytes:
    return (html_fixtures_dir / "thematic_sitemap.xml").read_bytes()


@pytest.fixture()
def index_bytes(html_fixtures_dir: Path) -> bytes:
    return (html_fixtures_dir / "thematic_index.html").read_bytes()


@pytest.fixture()
def alpha_bytes(html_fixtures_dir: Path) -> bytes:
    return (html_fixtures_dir / "fiche_sample_alpha.html").read_bytes()


@pytest.fixture()
def beta_bytes(html_fixtures_dir: Path) -> bytes:
    return (html_fixtures_dir / "fiche_sample_beta.html").read_bytes()


@pytest.fixture()
def alpha_v2_bytes(html_fixtures_dir: Path) -> bytes:
    return (html_fixtures_dir / "fiche_sample_alpha_v2.html").read_bytes()


@pytest.fixture()
def orphan_bytes() -> bytes:
    return b"<html><body><h1>Orphan page</h1></body></html>"


@pytest.fixture()
def unlisted_extra_bytes(html_fixtures_dir: Path) -> bytes:
    return (html_fixtures_dir / "fiche_with_unlisted_link.html").read_bytes()


@pytest.fixture()
def sleep_calls() -> list[float]:
    """Collects the delay values passed to the injected sleep function."""
    return []


@pytest.fixture()
def recording_sleep(sleep_calls: list[float]) -> Callable[[float], None]:
    def _sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    return _sleep


@pytest.fixture()
def standard_responses(
    sitemap_bytes: bytes,
    index_bytes: bytes,
    alpha_bytes: bytes,
    beta_bytes: bytes,
    orphan_bytes: bytes,
) -> dict[str, bytes]:
    """Minimal response map for the four sitemap-listed in-scope URLs."""
    return {
        _SITEMAP_PATH: sitemap_bytes,
        _INDEX_PATH: index_bytes,
        _ALPHA_PATH: alpha_bytes,
        _BETA_PATH: beta_bytes,
        _ORPHAN_PATH: orphan_bytes,
    }


@pytest.fixture()
def run_capture(tmp_path: Path) -> Callable[..., None]:
    """Run capture() against a mocked transport, writing into tmp_path.

    Accepts the response map plus optional fetched_urls (to record requests)
    and sleep_fn (defaults to the no-op sleep).
    """

    def _run(
        responses: dict[str, bytes],
        *,
        fetched_urls: list[str] | None = None,
        sleep_fn: Callable[[float], None] = _no_sleep,
    ) -> None:
        transport = _make_transport(responses, fetched_urls)
        with _make_client(transport) as client:
            capture(client=client, out_dir=tmp_path, sleep_fn=sleep_fn)

    return _run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Crawling via the sitemap writes the correct files at mirrored paths."""

    def test_child_pages_are_written_to_expected_paths(
        self,
        tmp_path: Path,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   Crawling via a stubbed sitemap with alpha and beta pages.
                    Each page should be written at out_dir/<relative-path>/index.html,
                    mirroring the URL hierarchy under /fiches-par-thematiques/.
        """
        run_capture(standard_responses)

        alpha_file = _written_file(tmp_path, _ALPHA_REL_PATH)
        beta_file = _written_file(tmp_path, _BETA_REL_PATH)
        assert alpha_file.exists(), f"Expected {alpha_file} to be written"
        assert beta_file.exists(), f"Expected {beta_file} to be written"

    def test_index_is_written_to_out_dir_root(
        self,
        tmp_path: Path,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   The thematic index URL itself is in the sitemap.
                    It should land at out_dir/index.html (not in a subdirectory).
        """
        run_capture(standard_responses)

        index_file = tmp_path / "index.html"
        assert index_file.exists(), f"Expected {index_file} to be written"

    def test_file_contents_match_response_bytes_exactly(
        self,
        tmp_path: Path,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   File contents are verbatim HTTP response bodies.
                    Each written file should contain exactly the bytes from the HTTP response.
        """
        run_capture(standard_responses)

        alpha_file = _written_file(tmp_path, _ALPHA_REL_PATH)
        beta_file = _written_file(tmp_path, _BETA_REL_PATH)
        assert alpha_file.read_bytes() == standard_responses[_ALPHA_PATH]
        assert beta_file.read_bytes() == standard_responses[_BETA_PATH]


class TestScopeEnforcement:
    """Only in-scope URLs from the sitemap are fetched and written."""

    def test_external_domain_link_is_not_fetched_or_written(
        self,
        tmp_path: Path,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   An HTML page links to an external domain.
                    No file should be written for the external URL, and it must not be fetched.
                    The sitemap must be the first thing fetched (drives URL list).
        """
        fetched_urls: list[str] = []

        run_capture(standard_responses, fetched_urls=fetched_urls)

        assert not any(
            "external-site.example.com" in url for url in fetched_urls
        ), f"External URL was fetched; all fetched: {fetched_urls}"
        assert not (tmp_path / "page").exists()
        # Sitemap must be the URL source (not BFS from index HTML alone)
        assert fetched_urls[0].endswith("sitemap.xml"), (
            f"First fetch should be sitemap.xml; got: {fetched_urls[0]}"
        )

    def test_out_of_scope_sitemap_entry_is_not_fetched_or_written(
        self,
        tmp_path: Path,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   The sitemap contains an out-of-scope entry (/mentions-legales/).
                    It must not be fetched and no file should be written for it.
                    The sitemap must be the first thing fetched.
        """
        fetched_urls: list[str] = []

        run_capture(standard_responses, fetched_urls=fetched_urls)

        assert not any(
            "mentions-legales" in url for url in fetched_urls
        ), f"Out-of-scope path was fetched; all fetched: {fetched_urls}"
        assert not (tmp_path / "mentions-legales").exists()
        # Sitemap must be the URL source
        assert fetched_urls[0].endswith("sitemap.xml"), (
            f"First fetch should be sitemap.xml; got: {fetched_urls[0]}"
        )


class TestIdempotency:
    """Running capture() twice produces deterministic results."""

    def test_unchanged_response_does_not_overwrite_file(
        self,
        tmp_path: Path,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   Running capture() twice with identical responses.
                    The on-disk file mtime must not change on the second run.
        """
        run_capture(standard_responses)

        alpha_file = _written_file(tmp_path, _ALPHA_REL_PATH)
        content_after_first_run = alpha_file.read_bytes()
        stat_before = alpha_file.stat()

        run_capture(standard_responses)

        stat_after = alpha_file.stat()

        assert alpha_file.read_bytes() == content_after_first_run
        assert stat_before.st_mtime_ns == stat_after.st_mtime_ns, (
            "File was re-written even though content was unchanged; "
            "capture() must skip the write when bytes are identical."
        )

    def test_changed_response_overwrites_file(
        self,
        tmp_path: Path,
        standard_responses: dict[str, bytes],
        alpha_v2_bytes: bytes,
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   Running capture() twice when a page's content has changed.
                    The on-disk file for alpha should be updated to the new bytes.
        """
        run_capture(standard_responses)

        alpha_file = _written_file(tmp_path, _ALPHA_REL_PATH)
        assert alpha_file.read_bytes() == standard_responses[_ALPHA_PATH]

        run_capture({**standard_responses, _ALPHA_PATH: alpha_v2_bytes})

        assert alpha_file.read_bytes() == alpha_v2_bytes, (
            "File was not updated even though the remote content changed."
        )


class TestRateLimit:
    """capture() applies rate-limiting between requests."""

    def test_sleep_is_called_between_requests(
        self,
        standard_responses: dict[str, bytes],
        sleep_calls: list[float],
        recording_sleep: Callable[[float], None],
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   Rate-limiting between HTTP requests driven by the sitemap.
                    Given a sitemap with N in-scope URLs, sleep is called at least N-1 times
                    with a delay >= 1.0 second between each fetch.
        """
        run_capture(standard_responses, sleep_fn=recording_sleep)

        # sitemap has 4 in-scope URLs (index + alpha + beta + orphan); expect >= 3 sleeps
        assert len(sleep_calls) >= len(standard_responses) - 2, (
            f"Expected at least {len(standard_responses) - 2} sleep calls, "
            f"got {len(sleep_calls)}: {sleep_calls}"
        )
        assert all(delay >= 1.0 for delay in sleep_calls), (
            f"All sleep delays must be >= 1.0 second; got {sleep_calls}"
        )


class TestSitemapParsing:
    """Sitemap fetch, localhost rewrite, and scope filtering."""

    def test_localhost_urls_are_rewritten_to_production_host(
        self,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   Sitemap <loc> values use http://localhost:8383/ as the host.
                    capture() must fetch the sitemap, rewrite localhost to the production host,
                    and never issue a request to localhost:8383.
        """
        fetched_urls: list[str] = []

        run_capture(standard_responses, fetched_urls=fetched_urls)

        assert not any(
            "localhost" in url for url in fetched_urls
        ), f"localhost was fetched; all fetched: {fetched_urls}"
        assert any(
            "formation-civique.interieur.gouv.fr" in url for url in fetched_urls
        ), f"No production-host request found; all fetched: {fetched_urls}"
        # The sitemap itself must have been fetched
        assert any(
            "sitemap.xml" in url for url in fetched_urls
        ), f"sitemap.xml was never fetched; all fetched: {fetched_urls}"

    def test_out_of_scope_sitemap_entry_is_filtered_before_any_fetch(
        self,
        tmp_path: Path,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
    ) -> None:
        """
        Scenario:   Sitemap contains /mentions-legales/ (out-of-scope path).
                    capture() must not request that URL, must not write any file for it,
                    and must have fetched the sitemap to discover the URL list.
        """
        fetched_urls: list[str] = []

        run_capture(standard_responses, fetched_urls=fetched_urls)

        assert not any(
            "mentions-legales" in url for url in fetched_urls
        ), f"Out-of-scope URL was fetched; all fetched: {fetched_urls}"
        assert not (tmp_path / "mentions-legales").exists()
        # Confirm the sitemap was the source (not just HTML BFS)
        assert any(
            "sitemap.xml" in url for url in fetched_urls
        ), f"sitemap.xml was never fetched; all fetched: {fetched_urls}"

    def test_sitemap_constant_is_defined_on_module(self) -> None:
        """
        Scenario:   The module exports SITEMAP_URL pointing to the production sitemap.
        """
        from civica.scripts import capture_thematic_sheets

        assert hasattr(capture_thematic_sheets, "SITEMAP_URL"), (
            "Module must define SITEMAP_URL constant"
        )
        assert "sitemap.xml" in capture_thematic_sheets.SITEMAP_URL
        assert "formation-civique.interieur.gouv.fr" in capture_thematic_sheets.SITEMAP_URL


class TestBFSCrossCheck:
    """Phase 2 cross-check: WARN lines for sitemap vs. discovered link discrepancies."""

    def test_warns_about_link_found_in_page_but_absent_from_sitemap(
        self,
        standard_responses: dict[str, bytes],
        unlisted_extra_bytes: bytes,
        run_capture: Callable[..., None],
        warning_text: Callable[[], str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Scenario:   A fetched page links to an in-scope URL not listed in the sitemap.
                    capture() must log a WARNING record naming that URL (missing_from_sitemap).
        """
        caplog.set_level(logging.INFO)
        # Replace alpha with a page that links to an unlisted URL
        run_capture({**standard_responses, _ALPHA_PATH: unlisted_extra_bytes})

        warnings = warning_text()
        assert warnings, f"Expected WARNING records; got: {caplog.text!r}"
        assert _UNLISTED_EXTRA_PATH in warnings or "fiche-unlisted-extra" in warnings, (
            f"Expected unlisted URL to appear in WARNING records; got: {warnings!r}"
        )

    def test_warns_about_sitemap_url_not_linked_from_any_page(
        self,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
        warning_text: Callable[[], str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Scenario:   The sitemap lists fiche-sample-orphan but no fetched page links to it.
                    capture() must log a WARNING record naming that URL (unreferenced_in_sitemap).
        """
        caplog.set_level(logging.INFO)
        run_capture(standard_responses)

        warnings = warning_text()
        assert warnings, f"Expected WARNING records for orphan URL; got: {caplog.text!r}"
        assert "fiche-sample-orphan" in warnings, (
            f"Expected orphan URL to appear in WARNING records; got: {warnings!r}"
        )


class TestFetchFailureHandling:
    """Individual page fetch failures are logged and do not halt the run."""

    def test_404_for_one_url_does_not_halt_capture(
        self,
        tmp_path: Path,
        standard_responses: dict[str, bytes],
        run_capture: Callable[..., None],
        warning_text: Callable[[], str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Scenario:   One in-scope sitemap URL returns 404.
                    Other pages are still written, a WARNING record names the failing URL,
                    and capture() returns normally without raising.
        """
        caplog.set_level(logging.INFO)
        # Dropping beta from the response map makes the transport 404 it while
        # the sitemap still lists it.
        responses = {path: body for path, body in standard_responses.items() if path != _BETA_PATH}

        # Must not raise
        run_capture(responses)

        warnings = warning_text()
        assert warnings, f"Expected WARNING records for failing URL; got: {caplog.text!r}"
        assert "fiche-sample-beta" in warnings, (
            f"Expected failing URL to appear in WARNING records; got: {warnings!r}"
        )

        # Alpha still written despite beta failure
        alpha_file = _written_file(tmp_path, _ALPHA_REL_PATH)
        assert alpha_file.exists(), f"Alpha file should still be written; not found at {alpha_file}"

        # Beta must not be written
        beta_file = _written_file(tmp_path, _BETA_REL_PATH)
        assert not beta_file.exists(), "Beta file should NOT be written after a 404"


class TestUserAgentResolution:
    """resolve_user_agent() prefers the CIVICA_USER_AGENT env variable override but falls back to the default."""

    @pytest.mark.parametrize(
        "env",
        [
            pytest.param({}, id="unset"),
            pytest.param({_USER_AGENT_ENV_VAR: ""}, id="blank"),
            pytest.param({_USER_AGENT_ENV_VAR: "   \t\n"}, id="whitespace-only"),
        ],
    )
    def test_falls_back_to_default_when_override_is_unusable(self, env: dict[str, str]) -> None:
        """Scenario: The override is absent, empty, or whitespace-only. Each yields the built-in default."""
        from civica.scripts import capture_thematic_sheets

        result = resolve_user_agent(env)

        assert result == capture_thematic_sheets.DEFAULT_USER_AGENT, (
            f"Expected fallback to the default User-Agent; got {result!r}"
        )

    @pytest.mark.parametrize(
        "raw_value",
        [
            pytest.param(_SAMPLE_OVERRIDE_USER_AGENT, id="exact"),
            pytest.param(f"  {_SAMPLE_OVERRIDE_USER_AGENT}  ", id="padded"),
        ],
    )
    def test_override_replaces_the_default_and_is_trimmed(self, raw_value: str) -> None:
        """Scenario: A fork sets CIVICA_USER_AGENT. Its value is returned in place of the default."""
        result = resolve_user_agent({_USER_AGENT_ENV_VAR: raw_value})

        assert result == _SAMPLE_OVERRIDE_USER_AGENT, (
            f"Expected the override User-Agent, trimmed; got {result!r}"
        )
