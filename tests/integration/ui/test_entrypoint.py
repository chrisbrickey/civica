"""
Integration tests for the real web app entrypoint (python -m civica.ui.app).

NiceGUI runs app.py as __main__ in its simulation mode, so the production
ui.run(...) wiring is exercised without starting a server.
"""

from pathlib import Path

import pytest
from nicegui.testing import user_simulation

from civica.ui import settings
from civica.ui.settings import MissingSessionSecret

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_VAR = "CIVICA_SESSION_SECRET"
_SAMPLE_SECRET = "sample-session-secret"

# 14 days in seconds; written out so the test does not just echo SESSION_MAX_AGE.
_FOURTEEN_DAYS_IN_SECONDS = 1_209_600

# Differs from Starlette's own 14-day default, so it proves SESSION_MAX_AGE reaches the cookie.
_SAMPLE_MAX_AGE = 3600


@pytest.fixture()
def entrypoint(pytestconfig: pytest.Config) -> Path:
    return pytestconfig.rootpath / "src" / "civica" / "ui" / "app.py"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_session_cookie_lasts_fourteen_days(
    entrypoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _SAMPLE_SECRET)

    async with user_simulation(main_file=entrypoint) as user:
        response = await user.http_client.get("/login")

    assert f"Max-Age={_FOURTEEN_DAYS_IN_SECONDS}" in response.headers["set-cookie"]


async def test_session_cookie_uses_configured_max_age(
    entrypoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _SAMPLE_SECRET)
    monkeypatch.setattr(settings, "SESSION_MAX_AGE", _SAMPLE_MAX_AGE)

    async with user_simulation(main_file=entrypoint) as user:
        response = await user.http_client.get("/login")

    assert f"Max-Age={_SAMPLE_MAX_AGE}" in response.headers["set-cookie"]


async def test_refuses_to_start_without_session_secret(
    entrypoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, "")

    with pytest.raises(MissingSessionSecret):
        async with user_simulation(main_file=entrypoint):
            pass
