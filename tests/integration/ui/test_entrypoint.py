"""
Integration tests for the real web app entrypoint (python -m civica.ui.app).

NiceGUI runs app.py as __main__ in its simulation mode, so the production
ui.run(...) wiring is exercised without starting a server.
"""

import importlib
import logging
import re
import runpy
import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from nicegui.testing import user_simulation
from nicegui.testing.general import nicegui_reset_globals

from civica.ui import settings
from civica.ui.deps import AppDeps
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

_SIMULATION_ENV_VAR = "NICEGUI_USER_SIMULATION"
_RELOAD_RUN_NAME = "__mp_main__"

_SAMPLE_LOGGER_NAME = "sample.logger"
_SAMPLE_INFO_MESSAGE = "sample-info-message"
_SAMPLE_DEBUG_MESSAGE = "sample-debug-message"

# The scripts/ format (time, level, message) with the logger name before the message.
_EXPECTED_LINE = re.compile(
    rf"^\d{{2}}:\d{{2}}:\d{{2}} INFO .*{re.escape(_SAMPLE_LOGGER_NAME)}.* {_SAMPLE_INFO_MESSAGE}$",
    re.MULTILINE,
)

_DEV_FLAG = "--dev"
_HELP_FLAG = "--help"
_UNKNOWN_FLAG = "--sample-unknown-flag"
_DEV_MODE_ON_LINE = "Developer mode on"
_DEV_MODE_OFF_LINE = "Developer mode off"

# Each launch as its command-line arguments, with whether it turns developer mode on.
_LAUNCHES = [pytest.param([], False, id="plain"), pytest.param([_DEV_FLAG], True, id="dev")]


@pytest.fixture()
def entrypoint(pytestconfig: pytest.Config) -> Path:
    return pytestconfig.rootpath / "src" / "civica" / "ui" / "app.py"


@pytest.fixture()
def launch_with(entrypoint: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[Sequence[str]], None]:
    """Sets the command line the entrypoint will read when it next runs."""

    def set_arguments(arguments: Sequence[str]) -> None:
        monkeypatch.setattr(sys, "argv", [str(entrypoint), *arguments])

    return set_arguments


@pytest.fixture(autouse=True)
def plain_launch(launch_with: Callable[[Sequence[str]], None]) -> None:
    """Under pytest sys.argv holds the test runner's arguments, so every test starts from a plain launch."""
    launch_with([])


@pytest.fixture()
def registered_deps(monkeypatch: pytest.MonkeyPatch) -> list[AppDeps]:
    """The dependencies each launch registers its pages with, recorded on the way through."""
    registered: list[AppDeps] = []
    # Looked up now, not at import: NiceGUI drops the page modules after each simulation, so the entrypoint
    # imports whichever copy is current.
    pages = importlib.import_module("civica.ui.pages")
    register: Callable[[AppDeps], None] = pages.register_pages

    def record_and_register(deps: AppDeps) -> None:
        registered.append(deps)
        register(deps)

    monkeypatch.setattr(pages, "register_pages", record_and_register)
    return registered


@contextmanager
def _bare_root_logger() -> Generator[None, None, None]:
    """Remove pytest's root handlers, which make basicConfig a no-op, then put everything back.

    Not a fixture: pytest adds its handlers again at the start of each phase, so this must run in the test body."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers[:] = []
    root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _assert_logging_is_configured(capsys: pytest.CaptureFixture[str]) -> None:
    """Write real records through the root logger and check what reaches the output."""
    capsys.readouterr()
    sample_logger = logging.getLogger(_SAMPLE_LOGGER_NAME)
    sample_logger.info(_SAMPLE_INFO_MESSAGE)
    sample_logger.debug(_SAMPLE_DEBUG_MESSAGE)

    output = capsys.readouterr().err
    assert _EXPECTED_LINE.search(output), f"no line in the expected format in: {output!r}"
    assert _SAMPLE_DEBUG_MESSAGE not in output


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


async def test_startup_configures_logging_under_main(
    entrypoint: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_ENV_VAR, _SAMPLE_SECRET)

    with _bare_root_logger():
        async with user_simulation(main_file=entrypoint):
            _assert_logging_is_configured(capsys)


def test_startup_configures_logging_under_auto_reload(
    entrypoint: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_ENV_VAR, _SAMPLE_SECRET)
    monkeypatch.setenv(_SIMULATION_ENV_VAR, "true")

    # user_simulation always runs main_file as __main__, so run the reload name the same way it does.
    with _bare_root_logger(), nicegui_reset_globals():
        runpy.run_path(str(entrypoint), run_name=_RELOAD_RUN_NAME)
        _assert_logging_is_configured(capsys)


async def test_session_secret_never_reaches_the_logs(
    entrypoint: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(_ENV_VAR, _SAMPLE_SECRET)

    with caplog.at_level(logging.DEBUG):
        async with user_simulation(main_file=entrypoint) as user:
            await user.http_client.get("/login")

    assert caplog.records
    assert _SAMPLE_SECRET not in caplog.text


# ---------------------------------------------------------------------------
# Developer mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("arguments", "dev_mode"), _LAUNCHES)
def test_only_the_dev_flag_asks_for_developer_mode(arguments: list[str], dev_mode: bool) -> None:
    from civica.ui.app import parse_args

    assert parse_args(arguments).dev is dev_mode


@pytest.mark.parametrize(("arguments", "dev_mode"), _LAUNCHES)
async def test_launch_registers_the_app_with_developer_mode_only_under_the_dev_flag(
    entrypoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch_with: Callable[[Sequence[str]], None],
    registered_deps: list[AppDeps],
    arguments: list[str],
    dev_mode: bool,
) -> None:
    monkeypatch.setenv(_ENV_VAR, _SAMPLE_SECRET)
    launch_with(arguments)

    async with user_simulation(main_file=entrypoint):
        pass

    assert [deps.dev_mode for deps in registered_deps] == [dev_mode]


@pytest.mark.parametrize(
    ("arguments", "logged", "not_logged"),
    [
        pytest.param([], _DEV_MODE_OFF_LINE, _DEV_MODE_ON_LINE, id="plain"),
        pytest.param([_DEV_FLAG], _DEV_MODE_ON_LINE, _DEV_MODE_OFF_LINE, id="dev"),
    ],
)
async def test_startup_logs_whether_developer_mode_is_on(
    entrypoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch_with: Callable[[Sequence[str]], None],
    caplog: pytest.LogCaptureFixture,
    arguments: list[str],
    logged: str,
    not_logged: str,
) -> None:
    monkeypatch.setenv(_ENV_VAR, _SAMPLE_SECRET)
    launch_with(arguments)

    with caplog.at_level(logging.INFO):
        async with user_simulation(main_file=entrypoint):
            pass

    info_lines = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert info_lines.count(logged) == 1
    assert not_logged not in info_lines


def test_help_works_without_a_session_secret(
    entrypoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch_with: Callable[[Sequence[str]], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(_ENV_VAR, "")
    launch_with([_HELP_FLAG])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(entrypoint), run_name="__main__")

    assert stopped.value.code == 0
    assert _DEV_FLAG in capsys.readouterr().out


def test_unknown_argument_stops_the_app_with_usage_help(
    entrypoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch_with: Callable[[Sequence[str]], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No secret either, so a launch that ignored its arguments stops here instead of starting a server.
    monkeypatch.setenv(_ENV_VAR, "")
    launch_with([_UNKNOWN_FLAG])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(entrypoint), run_name="__main__")

    assert stopped.value.code not in (0, None)
    usage = capsys.readouterr().err
    assert "usage:" in usage
    assert _UNKNOWN_FLAG in usage
