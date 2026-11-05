"""The web app's pages.

Each page exposes the same register(deps) function, which provides the dependencies required.
This allows different dependencies to be injected by tests (if desired) and results in
routes being created when called rather than on import."""

from civica.ui.auth import install_auth_middleware
from civica.ui.deps import AppDeps
from civica.ui.pages import account, home


def register_pages(deps: AppDeps) -> None:
    """Install auth and define the app's pages. Tests call this with their own dependencies."""
    install_auth_middleware()
    account.register(deps)
    home.register()
