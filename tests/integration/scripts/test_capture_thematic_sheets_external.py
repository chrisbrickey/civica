"""
External contract test for capture_thematic_sheets script that scrapes a website.

Verifies the real ministry index URL is reachable and contains at least
one link matching the allowed path prefix.

Does NOT write anything to disk and does NOT perform a full crawl.

Run with: uv run pytest -m external
"""

import pytest

import httpx

from civica.scripts.capture_thematic_sheets import (
    ALLOWED_PATH_PREFIX,
    DEFAULT_USER_AGENT,
    INDEX_URL,
)


@pytest.mark.external
def test_ministry_index_is_reachable_and_contains_in_scope_links() -> None:
    """
    Scenario:   Ministry thematic index is accessible and well-formed.
                Given the real ministry site is available online,
                when a GET request is sent to the thematic index URL...
                then the response should be 200 OK with HTML content.
                And the HTML should contain at least one link whose href starts with the allowed path prefix.
    """
    from bs4 import BeautifulSoup

    response = httpx.get(
        INDEX_URL,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        follow_redirects=True,
        timeout=15.0,
    )
    assert response.status_code == 200, (
        f"Expected HTTP 200 from {INDEX_URL}, got {response.status_code}"
    )

    content_type = response.headers.get("content-type", "")
    assert "html" in content_type.lower(), (
        f"Expected HTML content-type, got: {content_type!r}"
    )

    soup = BeautifulSoup(response.content, "html.parser")
    in_scope_links = [
        a["href"]
        for a in soup.find_all("a", href=True)
        if isinstance(a["href"], str) and a["href"].startswith(ALLOWED_PATH_PREFIX)
        and a["href"] != ALLOWED_PATH_PREFIX  # exclude the index itself
    ]
    assert len(in_scope_links) >= 1, (
        f"Expected at least one link under {ALLOWED_PATH_PREFIX!r} "
        f"on the index page; found none."
    )
