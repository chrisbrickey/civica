"""Session settings for the web app."""

import os
from datetime import timedelta

_SESSION_SECRET_ENV_VAR = "CIVICA_SESSION_SECRET"

# How long a login lasts. This is passed to NiceGUI's session middleware instead of relying on defaults.
SESSION_MAX_AGE = int(timedelta(days=14).total_seconds())


class MissingSessionSecret(Exception):
    """Raised when CIVICA_SESSION_SECRET is missing or empty."""


def session_secret() -> str:
    """Return the secret that signs the browser's login cookie."""
    secret = os.environ.get(_SESSION_SECRET_ENV_VAR)
    if not secret:
        raise MissingSessionSecret(
            f"{_SESSION_SECRET_ENV_VAR} must be set to run the web app. "
            f"See README for instructions on how to generate and store this secret."
        )
    return secret
