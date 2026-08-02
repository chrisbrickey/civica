"""
Shared fixtures for script unit tests.

The capture/normalize/verify script tests all build synthetic data trees under
tmp_path and assert on log output; the fixtures for both concerns live here so
each value and helper is defined once.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture()
def raw_root(tmp_path: Path) -> Path:
    """Root of a synthetic data/raw/ tree."""
    root = tmp_path / "raw"
    root.mkdir()
    return root


@pytest.fixture()
def corpus_root(tmp_path: Path) -> Path:
    """Root of a synthetic data/corpus/ tree."""
    root = tmp_path / "corpus"
    root.mkdir()
    return root


@pytest.fixture()
def warning_text(caplog: pytest.LogCaptureFixture) -> Callable[[], str]:
    """Return a function that joins all WARNING messages captured so far."""

    def _text() -> str:
        return "\n".join(
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    return _text
