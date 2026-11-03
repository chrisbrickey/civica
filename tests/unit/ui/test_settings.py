"""Unit tests for civica.ui.settings: the session secret read from the environment."""

import pytest

from civica.ui.settings import MissingSessionSecret, session_secret

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_VAR = "CIVICA_SESSION_SECRET"
_SAMPLE_SECRET = "sample-session-secret"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_secret_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_VAR, _SAMPLE_SECRET)

    assert session_secret() == _SAMPLE_SECRET


@pytest.mark.parametrize("value", [None, ""], ids=["missing", "empty"])
def test_missing_or_empty_secret_raises_with_guidance(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(_ENV_VAR, value)

    with pytest.raises(MissingSessionSecret) as error:
        session_secret()

    assert _ENV_VAR in str(error.value)
    assert "See README for instructions on how to generate and store this secret" in str(error.value)
