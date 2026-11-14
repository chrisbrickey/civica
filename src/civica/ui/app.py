"""NiceGUI web app entrypoint. Run with: uv run python -m civica.ui.app"""

import argparse
import logging
from collections.abc import Sequence

from dotenv import load_dotenv
from nicegui import ui

from civica.ui.deps import default_app_deps
from civica.ui.pages import register_pages
from civica.ui.settings import SESSION_MAX_AGE, session_secret

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command line: one flag, --dev. Reads sys.argv when no list is given."""
    parser = argparse.ArgumentParser(description="Run the Civica web app.")
    parser.add_argument("--dev", action="store_true", help="show the developer panel beside each coach response")
    return parser.parse_args(argv)


if __name__ in {"__main__", "__mp_main__"}:  # __mp_main__ is needed for NiceGUI's auto-reload
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    load_dotenv()
    args = parse_args()  # before the secret, so --help and bad arguments need no secret
    logger.info("Developer mode %s", "on" if args.dev else "off")
    storage_secret = session_secret()  # before building deps, so a missing secret fails fast
    register_pages(default_app_deps(dev_mode=args.dev))
    ui.run(
        storage_secret=storage_secret,
        session_middleware_kwargs={"max_age": SESSION_MAX_AGE},
        title="Civica",
        favicon="📘",
    )
